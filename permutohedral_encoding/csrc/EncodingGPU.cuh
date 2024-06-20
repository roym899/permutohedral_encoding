#pragma once

#include <torch/torch.h>

#define BLOCK_SIZE 128
#define BLOCK_SIZE_BACK 128
#define BLOCK_SIZE_DOUBLE_BACK 128

// atomicAdd for half precision
__device__ __forceinline__ void atomicAdd(c10::Half* address, c10::Half val) {
  unsigned int* address_as_ui = reinterpret_cast<unsigned int*>(
      reinterpret_cast<char*>(address) - (reinterpret_cast<size_t>(address) & 2)
  );
  unsigned int old = *address_as_ui;
  unsigned int assumed;

  do {
    assumed = old;
    unsigned short hsum = reinterpret_cast<size_t>(address) & 2 ? (old >> 16) : (old & 0xffff);
    hsum += val;
    old = reinterpret_cast<size_t>(address) & 2 ? (old & 0xffff) | (hsum << 16) : (old & 0xffff0000) | hsum;
    old = atomicCAS(address_as_ui, assumed, old);

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);
}

template <int pos_dim>
/* Hash function used in this implementation. A simple base conversion. */
__forceinline__ __device__ unsigned int hash(const int* const key) {
  unsigned int k = 0;

  for (int i = 0; i < pos_dim; ++i) {
    k += key[i];
    k = k * 2531011;
  }
  return k;
}

__forceinline__ __device__ int modHash(const unsigned int& n, const int& capacity) { return (n % capacity); }

template <int pos_dim>
__forceinline__ __device__ int idx_hash_with_collision(const int* const key, const int& capacity) {
  int h = modHash(hash<pos_dim>(key), capacity);
  return h;
}

template <int pos_dim, int val_dim, typename scalar_t>
__global__ void __launch_bounds__(BLOCK_SIZE) forward_gpu(
    const int nr_positions, const int lattice_capacity, const int nr_resolutions,
    const int nr_resolutions_extra,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> positions,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> features,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> scale_factor,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> random_shift_monolithic,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> anneal_window,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> outs, const bool concat_points,
    const scalar_t points_scaling, const bool require_features_grad, const bool require_positions_grad
) {
  const int batch_idx = blockIdx.z;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nr_positions) {
    return;
  }

  const uint32_t level = blockIdx.y;

  scalar_t pos[pos_dim];
  for (int i = 0; i < pos_dim; ++i) {
    pos[i] = positions[batch_idx][idx][i];
  }

  // if we are in one of the extra resolutions and we are also concatenating the
  // points, then do so
  int idx_extra_level = level - nr_resolutions;
  if (idx_extra_level >= 0) {
    // we are in one of the extra levels
    // check if this extra level that we are in is within the bound of the
    // pos_dim we can have for example 2 extra levels with 2 val dim each, so a
    // total of 4 more dimensions. But our pos dim =3 so the last val dim is
    // just 0

    for (int i = 0; i < val_dim; ++i) {
      // for the first resolution this is 0 and 1 , for
      // the other resolution it will be 2 and 3.
      int position_dimension_to_copy_from = i + idx_extra_level * val_dim;

      if (position_dimension_to_copy_from < pos_dim) {
        outs[batch_idx][level][i][idx] = pos[position_dimension_to_copy_from] * points_scaling;
      } else {
        // we are in the 4 dimensions but we have only posdim=3 so we
        // just put a zero here
        outs[batch_idx][level][i][idx] = scalar_t{0.0};
      }
    }
    return;  // nothing further to do for the concatenated dimensions
  }

  // embed position vectors on d+1 plane
  // see Adams et al (2010) p. 5 (scale_factor includes alpha_is, see
  // Encoding.cuh)
  scalar_t elevated[pos_dim + 1];
  scalar_t sm{0.0};

  for (int i = pos_dim; i > 0; --i) {
    scalar_t cf = (pos[i - 1] + random_shift_monolithic[level][i - 1]) * scale_factor[level][i - 1];
    elevated[i] = sm - i * cf;
    sm += cf;
  }
  elevated[0] = sm;

  // Find the closest 0-colored simplex through rounding
  // Adams et al. (2010) p. 4
  // Conway et al., 1998 p. 447-448 (Algorithm 3, Step 2)
  // greedily search for the closest zero-colored lattice point
  int rem0[pos_dim + 1];  // closest remainder-0 point
  int sum{0};
  scalar_t factor{scalar_t{1.0} / (pos_dim + 1)};

  for (int i = 0; i <= pos_dim; ++i) {
    scalar_t v = elevated[i] * factor;
    // find nearest multiples of (pos_dim + 1)
    scalar_t up = ceil(v) * (pos_dim + 1);
    scalar_t down = floor(v) * (pos_dim + 1);
    if (up - elevated[i] < elevated[i] - down) {  // up is closer
      rem0[i] = (int)up;
    } else {  // down is closer
      rem0[i] = (int)down;
    }
    sum += rem0[i];
  }
  sum /= pos_dim + 1;

  // Find the simplex we are in and store it in rank (where rank describes what
  // position coordinate i has in the sorted order of the features values)
  // Conway et al., 1998 p. 447-448 (Algorithm 3, Step 3)
  int rank[pos_dim + 1]{};

  for (int i = 0; i < pos_dim; ++i) {
    scalar_t di = elevated[i] - rem0[i];


    for (int j = i + 1; j <= pos_dim; ++j)
      if (di < elevated[j] - rem0[j])
        ++rank[i];
      else
        ++rank[j];
  }

  // If the point doesn't lie on the plane (sum != 0) bring it back
  // Conway et al., 1998 p. 447-448 (Algorithm 3, Step 4)

  for (int i = 0; i <= pos_dim; ++i) {
    rank[i] += sum;
    if (rank[i] < 0) {
      rank[i] += pos_dim + 1;
      rem0[i] += pos_dim + 1;
    } else if (rank[i] > pos_dim) {
      rank[i] -= pos_dim + 1;
      rem0[i] -= pos_dim + 1;
    }
  }

  // Compute the barycentric coordinates
  // Baek et al., 2009 (Proposition 4.2)
  scalar_t barycentric[pos_dim + 2]{};

  for (int i = 0; i <= pos_dim; ++i) {
    scalar_t delta = (elevated[i] - rem0[i]) * factor;
    // NOTE
    //   1. the original implementation (below) is somehow significantly slower
    //   for float16. Could not find an explanation why.
    //   2. for original implementation float32 was faster than float16; with
    //   this float16 is faster than float32.
    for (int j = 0; j <= pos_dim; ++j) {
      if (j != rank[i]) continue;
      barycentric[pos_dim - j] += delta;
      barycentric[pos_dim + 1 - j] -= delta;
    }
  }
  barycentric[0] += scalar_t{1} + barycentric[pos_dim + 1];

  /* // Original Implementation */
  /* scalar_t barycentric[pos_dim + 2]{}; */
  /*  */
  /* for (int i = 0; i <= pos_dim; ++i) { */
  /*   scalar_t delta = (elevated[i] - rem0[i]) * factor; */
  /*   barycentric[pos_dim - rank[i]] += delta; */
  /*   barycentric[pos_dim + 1 - rank[i]] -= delta; */
  /* } */
  /* barycentric[0] += scalar_t{1.0} + barycentric[pos_dim + 1]; */

  // here we accumulate the values and the homogeneous term
  scalar_t val_hom_vec[2] = {};

  scalar_t w_lvl = anneal_window[level];

  int key[pos_dim];

  for (int remainder = 0; remainder <= pos_dim; remainder++) {

    for (int i = 0; i < pos_dim; ++i) {
      key[i] = rem0[i] + remainder;
      if (rank[i] > pos_dim - remainder) key[i] -= (pos_dim + 1);
    }

    int idx_val = idx_hash_with_collision<pos_dim>(key, lattice_capacity);

    scalar_t w = barycentric[remainder] * w_lvl;

    val_hom_vec[0] = val_hom_vec[0] + features[batch_idx][level][idx_val][0] * w;
    val_hom_vec[1] = val_hom_vec[1] + features[batch_idx][level][idx_val][1] * w;
  }

  outs[batch_idx][level][0][idx] = val_hom_vec[0];
  outs[batch_idx][level][1][idx] = val_hom_vec[1];
}

template <int pos_dim, int val_dim, typename scalar_t>
__global__ void __launch_bounds__(BLOCK_SIZE_BACK) backward_gpu(
    const int nr_positions, const int lattice_capacity,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> features,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> positions,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> scale_factor,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> random_shift_monolithic,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> anneal_window,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad_outs,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad_features,
    const bool concat_points
) {
  const int batch_idx = blockIdx.z;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nr_positions) {
    return;
  }

  const uint32_t level = blockIdx.y;

  scalar_t pos[pos_dim];
  for (int i = 0; i < pos_dim; ++i) {
    pos[i] = positions[batch_idx][idx][i];
  }

  scalar_t elevated[pos_dim + 1];
  scalar_t sm{};

  for (int i = pos_dim; i > 0; i--) {
    scalar_t cf = (pos[i - 1] + random_shift_monolithic[level][i - 1]) * scale_factor[level][i - 1];
    elevated[i] = sm - i * cf;
    sm += cf;
  }
  elevated[0] = sm;

  int rem0[pos_dim + 1];
  int sum{0};
  scalar_t factor{scalar_t{1.0} / (pos_dim + 1)};

  for (int i = 0; i <= pos_dim; ++i) {
    scalar_t v = elevated[i] * factor;
    scalar_t up = ceil(v) * (pos_dim + 1);
    scalar_t down = floor(v) * (pos_dim + 1);
    if (up - elevated[i] < elevated[i] - down) {
      rem0[i] = (int)up;
    } else {
      rem0[i] = (int)down;
    }
    sum += rem0[i];
  }
  sum /= pos_dim + 1;

  int rank[pos_dim + 1]{};

  for (int i = 0; i < pos_dim; ++i) {
    scalar_t di = elevated[i] - rem0[i];


    for (int j = i + 1; j <= pos_dim; ++j)
      if (di < elevated[j] - rem0[j])
        ++rank[i];
      else
        ++rank[j];
  }


  for (int i = 0; i <= pos_dim; ++i) {
    rank[i] += sum;
    if (rank[i] < 0) {
      rank[i] += pos_dim + 1;
      rem0[i] += pos_dim + 1;
    } else if (rank[i] > pos_dim) {
      rank[i] -= pos_dim + 1;
      rem0[i] -= pos_dim + 1;
    }
  }

  scalar_t barycentric[pos_dim + 2]{};

  for (int i = 0; i <= pos_dim; ++i) {
    scalar_t delta = (elevated[i] - rem0[i]) * factor;
    for (int j = 0; j <= pos_dim; ++j) {
      if (j != rank[i]) continue;
      barycentric[pos_dim - j] += delta;
      barycentric[pos_dim + 1 - j] -= delta;
    }
  }
  barycentric[0] += scalar_t{1.0} + barycentric[pos_dim + 1];

  scalar_t grad_cur_outs[] = {grad_outs[batch_idx][level][0][idx], grad_outs[batch_idx][level][1][idx]};

  scalar_t w_lvl = anneal_window[level];

  int key[pos_dim];

  for (int remainder = 0; remainder <= pos_dim; remainder++) {

    for (int i = 0; i < pos_dim; ++i) {
      key[i] = rem0[i] + remainder;
      if (rank[i] > pos_dim - remainder) key[i] -= (pos_dim + 1);
    }

    // Retrieve pointer to the value at this vertex.
    int idx_val = idx_hash_with_collision<pos_dim>(key, lattice_capacity);

    scalar_t w = barycentric[remainder] * w_lvl;

    atomicAdd(&grad_features[batch_idx][level][idx_val][0], grad_cur_outs[0] * w);
    atomicAdd(&grad_features[batch_idx][level][idx_val][1], grad_cur_outs[1] * w);
  }
}

template <int pos_dim, int val_dim, typename scalar_t>
__global__ void __launch_bounds__(BLOCK_SIZE_BACK)

    backward_gpu_only_pos(
        const int nr_positions, const int lattice_capacity,
        const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> features,
        const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> positions,
        const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> scale_factor,
        const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> random_shift_monolithic,
        const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> anneal_window,
        const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad_outs,
        torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_positions,
        const bool concat_points, const bool require_features_grad, const bool require_positions_grad
    ) {
  const int batch_idx = blockIdx.z;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nr_positions) {
    return;
  }

  const uint32_t level = blockIdx.y;

  scalar_t pos[pos_dim];
  for (int i = 0; i < pos_dim; ++i) {
    pos[i] = positions[batch_idx][idx][i];
  }

  scalar_t elevated[pos_dim + 1];
  scalar_t sm{};

  for (int i = pos_dim; i > 0; i--) {
    scalar_t cf = (pos[i - 1] + random_shift_monolithic[level][i - 1]) * scale_factor[level][i - 1];
    elevated[i] = sm - i * cf;
    sm += cf;
  }
  elevated[0] = sm;

  int rem0[pos_dim + 1];
  int sum{0};
  scalar_t factor{scalar_t{1.0} / (pos_dim + 1)};

  for (int i = 0; i <= pos_dim; ++i) {
    scalar_t v = elevated[i] * factor;
    // find nearest multiples of (pos_dim + 1)
    scalar_t up = ceil(v) * (pos_dim + 1);
    scalar_t down = floor(v) * (pos_dim + 1);
    if (up - elevated[i] < elevated[i] - down) {  // up is closer
      rem0[i] = (int)up;
    } else {  // down is closer
      rem0[i] = (int)down;
    }
    sum += rem0[i];
  }
  sum /= pos_dim + 1;

  int rank[pos_dim + 1]{};

  for (int i = 0; i < pos_dim; ++i) {
    scalar_t di = elevated[i] - rem0[i];


    for (int j = i + 1; j <= pos_dim; ++j)
      if (di < elevated[j] - rem0[j])
        ++rank[i];
      else
        ++rank[j];
  }


  for (int i = 0; i <= pos_dim; ++i) {
    rank[i] += sum;
    if (rank[i] < 0) {
      rank[i] += pos_dim + 1;
      rem0[i] += pos_dim + 1;
    } else if (rank[i] > pos_dim) {
      rank[i] -= pos_dim + 1;
      rem0[i] -= pos_dim + 1;
    }
  }

  scalar_t barycentric[pos_dim + 2]{};

  for (int i = 0; i <= pos_dim; ++i) {
    scalar_t delta = (elevated[i] - rem0[i]) * factor;
    for (int j = 0; j <= pos_dim; ++j) {
      if (j != rank[i]) continue;
      barycentric[pos_dim - j] += delta;
      barycentric[pos_dim + 1 - j] -= delta;
    }
  }
  barycentric[0] += scalar_t{1.0} + barycentric[pos_dim + 1];

  scalar_t grad_cur_outs[] = {grad_outs[batch_idx][level][0][idx], grad_outs[batch_idx][level][1][idx]};

  scalar_t w_lvl = anneal_window[level];

  int key[pos_dim];

  scalar_t dL_dbarycentric[pos_dim + 2]{};
  for (int remainder = 0; remainder <= pos_dim; ++remainder) {

    for (int i = 0; i < pos_dim; ++i) {
      key[i] = rem0[i] + remainder;
      if (rank[i] > pos_dim - remainder) key[i] -= (pos_dim + 1);
    }
    int idx_val = idx_hash_with_collision<pos_dim>(key, lattice_capacity);

    const scalar_t* fv = &features[batch_idx][level][idx_val][0];
    dL_dbarycentric[remainder] += fv[0] * w_lvl * grad_cur_outs[0];
    dL_dbarycentric[remainder] += fv[1] * w_lvl * grad_cur_outs[1];
  }

  dL_dbarycentric[pos_dim + 1] += dL_dbarycentric[0];

  scalar_t dL_delevated[pos_dim + 1]{};

  for (int i = 0; i <= pos_dim; ++i) {
    dL_delevated[i] += dL_dbarycentric[pos_dim - rank[i]] * factor;
    dL_delevated[i] -= dL_dbarycentric[pos_dim + 1 - rank[i]] * factor;
  }

  scalar_t dL_dP[pos_dim]{};

  for (int i = 0; i < pos_dim; ++i) {

    for (int j = 0; j <= i; ++j) {
      dL_dP[i] += dL_delevated[j] * scale_factor[level][i];
    }
  }

  for (int i = 0; i < pos_dim; ++i) {
    dL_dP[i] -= dL_delevated[i + 1] * scale_factor[level][i] * (i + 1);
  }


  for (int i = 0; i < pos_dim; ++i) {
    atomicAdd(&grad_positions[batch_idx][idx][i], dL_dP[i]);
  }
}

//double back
template <int pos_dim, int val_dim, typename scalar_t>
__global__ void __launch_bounds__(BLOCK_SIZE_DOUBLE_BACK) double_backward_gpu(
    const int nr_positions, const int lattice_capacity, const int nr_resolutions,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_grad_positions,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad_grad_features,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> features,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> positions,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> scale_factor,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> random_shift_monolithic,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> anneal_window,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad_outs,
    const bool concat_points,
    //output
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad_grad_outs,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad_features,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_positions
) {
  const int batch_idx = blockIdx.z;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nr_positions) {
    return;
  }

  const uint32_t level = blockIdx.y;

  // a kernel is responsible for a single level of a single position of a single batch element

  if (level >= nr_resolutions) {
    grad_grad_outs[batch_idx][level][0][idx] = 0;
    grad_grad_outs[batch_idx][level][1][idx] = 0;
    return;
  }

  scalar_t pos[pos_dim];
  for (int i = 0; i < pos_dim; i++) {
    pos[i] = positions[batch_idx][idx][i];
  }

  scalar_t elevated[pos_dim + 1];
  scalar_t sm = 0;

  for (int i = pos_dim; i > 0; i--) {
    scalar_t cf = (pos[i - 1] + random_shift_monolithic[level][i - 1]) * scale_factor[level][i - 1];
    elevated[i] = sm - i * cf;
    sm += cf;
  }
  elevated[0] = sm;

  int rem0[pos_dim + 1];
  int rank[pos_dim + 1]{0};
  scalar_t factor{scalar_t{1.0} / (pos_dim + 1)};
  int sum = 0;

  for (int i = 0; i <= pos_dim; i++) {
    scalar_t v = elevated[i] * (1.0f / (pos_dim + 1));
    scalar_t up = ceil(v) * (pos_dim + 1);
    scalar_t down = floor(v) * (pos_dim + 1);
    if (up - elevated[i] < elevated[i] - down) {
      rem0[i] = (int)up;
    } else {
      rem0[i] = (int)down;
    }
    sum += rem0[i];
  }
  sum /= pos_dim + 1;


  for (int i = 0; i < pos_dim; i++) {
    double di = elevated[i] - rem0[i];
    for (int j = i + 1; j <= pos_dim; j++)
      if (di < elevated[j] - rem0[j])
        rank[i]++;
      else
        rank[j]++;
  }


  for (int i = 0; i <= pos_dim; i++) {
    rank[i] += sum;
    if (rank[i] < 0) {
      rank[i] += pos_dim + 1;
      rem0[i] += pos_dim + 1;
    } else if (rank[i] > pos_dim) {
      rank[i] -= pos_dim + 1;
      rem0[i] -= pos_dim + 1;
    }
  }

  scalar_t barycentric[pos_dim + 2]{0.0f};

  for (int i = 0; i <= pos_dim; i++) {
    scalar_t delta = (elevated[i] - rem0[i]) * (1.0f / (pos_dim + 1));
    barycentric[pos_dim - rank[i]] += delta;
    barycentric[pos_dim + 1 - rank[i]] -= delta;
  }
  barycentric[0] += 1.0f + barycentric[pos_dim + 1];

  // until here the code is the same as the forward pass

  scalar_t w_lvl = anneal_window[level];

  scalar_t grad_outs_cur[val_dim];

  for (int j = 0; j < val_dim; j++) {
    grad_outs_cur[j] = grad_outs[batch_idx][level][j][idx];
  }

  scalar_t grad_grad_positions_cur[pos_dim];

  for (int j = 0; j < pos_dim; j++) {
    grad_grad_positions_cur[j] = grad_grad_positions[batch_idx][idx][j];
  }

  // 1
  scalar_t dl2__ddl1_de[pos_dim + 1]{0.0f};
  for (int i = 0; i < pos_dim; i++) {
    scalar_t grad = grad_grad_positions_cur[i] * scale_factor[level][i];

    for (int j = 0; j <= i; j++) {
      dl2__ddl1_de[j] += grad;
    }
  }

  for (int i = 0; i < pos_dim; i++) {
    dl2__ddl1_de[i + 1] -= grad_grad_positions_cur[i] * scale_factor[level][i] * (i + 1);
  }

  // 2
  scalar_t dl2__ddl1_db[pos_dim + 2]{0.0f};
  for (int i = 0; i <= pos_dim; i++) {
    dl2__ddl1_db[pos_dim - rank[i]] += dl2__ddl1_de[i] * factor;
    dl2__ddl1_db[pos_dim + 1 - rank[i]] -= dl2__ddl1_de[i] * factor;
  }
  dl2__ddl1_db[0] += dl2__ddl1_db[pos_dim + 1];

  // 3, a
  int key[pos_dim];
  scalar_t grad_grad_outs_cur[val_dim]{0.0f};

  scalar_t dl2_db[pos_dim + 2]{0.0f};
  for (int remainder = 0; remainder <= pos_dim; remainder++) {

    for (int i = 0; i < pos_dim; i++) {
      key[i] = rem0[i] + remainder;
      if (rank[i] > pos_dim - remainder) key[i] -= (pos_dim + 1);
    }
    int idx_val = idx_hash_with_collision<pos_dim>(key, lattice_capacity);

    const scalar_t* fv = &features[batch_idx][level][idx_val][0];
    atomicAdd(
        &grad_features[batch_idx][level][idx_val][0], dl2__ddl1_db[remainder] * w_lvl * grad_outs_cur[0]
    );
    atomicAdd(
        &grad_features[batch_idx][level][idx_val][1], dl2__ddl1_db[remainder] * w_lvl * grad_outs_cur[1]
    );

    dl2_db[remainder] += w_lvl * (grad_outs_cur[0] * grad_grad_features[batch_idx][level][idx_val][0] +
                                  grad_outs_cur[1] * grad_grad_features[batch_idx][level][idx_val][1]);

    scalar_t w = barycentric[remainder] * w_lvl;

    grad_grad_outs_cur[0] +=
        dl2__ddl1_db[remainder] * w_lvl * fv[0] + grad_grad_features[batch_idx][level][idx_val][0] * w;
    grad_grad_outs_cur[1] +=
        dl2__ddl1_db[remainder] * w_lvl * fv[1] + grad_grad_features[batch_idx][level][idx_val][1] * w;
  }
  grad_grad_outs[batch_idx][level][0][idx] = grad_grad_outs_cur[0];
  grad_grad_outs[batch_idx][level][1][idx] = grad_grad_outs_cur[1];

  scalar_t dl2_de[pos_dim + 1]{0.0f};
  dl2_db[pos_dim + 1] += dl2_db[0];
  for (int i = 0; i <= pos_dim; ++i) {
    for (int j = 0; j <= pos_dim; ++j) {
      if (j != rank[i]) continue;
      dl2_de[i] += factor * dl2_db[pos_dim - j];
      dl2_de[i] -= factor * dl2_db[pos_dim + 1 - j];
    }
  }

  scalar_t dl2_dpos[pos_dim]{0.0f};
  scalar_t acc = dl2_de[0];
  for (int i = 1; i <= pos_dim; ++i) {
    dl2_dpos[i - 1] = (acc - i * dl2_de[i]) * scale_factor[level][i - 1];
    acc += dl2_de[i];
  }

  for (int i = 0; i < pos_dim; ++i) {
    atomicAdd(&grad_positions[batch_idx][idx][i], dl2_dpos[i]);
  }
}
