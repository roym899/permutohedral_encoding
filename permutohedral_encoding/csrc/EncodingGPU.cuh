#pragma once

#include <torch/torch.h>

#define BLOCK_SIZE 128
#define BLOCK_SIZE_BACK 128
#define BLOCK_SIZE_DOUBLE_BACK 128

template <int pos_dim>
/* Hash function used in this implementation. A simple base conversion. */
__forceinline__ __device__ unsigned int hash(const int* const key) {
  unsigned int k = 0;
#pragma unroll
  for (int i = 0; i < pos_dim; i++) {
    k += key[i];
    k = k * 2531011;
  }
  return k;
}

__forceinline__ __device__ int modHash(const unsigned int& n,
                                       const int& capacity) {
  return (n % capacity);
}

template <int pos_dim>
__forceinline__ __device__ int idx_hash_with_collision(const int* const key,
                                                       const int& capacity) {
  int h = modHash(hash<pos_dim>(key), capacity);
  return h;
}

template <int pos_dim, int val_dim, typename scalar_t>
__global__ void __launch_bounds__(
    BLOCK_SIZE)  // since the block size is known at compile time we can specify
                 // it to the kernel and therefore cuda doesnt need to use
                 // heuristics based on code complexity to minimize registry
                 // usage
    forward_gpu(
        const int nr_positions, const int lattice_capacity,
        const int nr_resolutions, const int nr_resolutions_extra,
        const torch::PackedTensorAccessor32<scalar_t, 3,
                                            torch::RestrictPtrTraits>
            positions,
        torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits>
            lattice_values_monolithic,
        const torch::PackedTensorAccessor32<scalar_t, 2,
                                            torch::RestrictPtrTraits>
            scale_factor,
        const torch::PackedTensorAccessor32<scalar_t, 2,
                                            torch::RestrictPtrTraits>
            random_shift_monolithic,
        const torch::PackedTensorAccessor32<scalar_t, 1,
                                            torch::RestrictPtrTraits>
            anneal_window,
        torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits>
            sliced_values_monolithic,
        const bool concat_points, const scalar_t points_scaling,
        const bool require_lattice_values_grad,
        const bool require_positions_grad) {
  const int batch_idx = blockIdx.z;
  int idx = blockIdx.x * blockDim.x +
            threadIdx.x;  // each thread will deal with a new value

  if (idx >= nr_positions) {  // don't go out of bounds
    return;
  }

  const uint32_t level =
      blockIdx.y;  // <- the level is the same for all threads

  scalar_t pos[pos_dim];
  for (int i = 0; i < pos_dim; ++i) {
    pos[i] = positions[batch_idx][idx][i];
  }

  // if we are in one of the extra resolutions and we are also concating the
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
        sliced_values_monolithic[batch_idx][level][i][idx] =
            pos[position_dimension_to_copy_from] * points_scaling;
      } else {
        // we are in the 4 dimensions but we have only posdim=3 so we
        // just put a zero here
        sliced_values_monolithic[batch_idx][level][i][idx] = scalar_t{0.0};
      }
    }
    return;  // nothing further to do for the concatenated dimensions
  }

  // embed position vectors on d+1 plane
  // see Adams et al (2010) p. 5 (scale_factor includes alpha_is, see
  // Encoding.cuh)
  scalar_t elevated[pos_dim + 1];
  scalar_t sm{0.0};
#pragma unroll
  for (int i = pos_dim; i > 0; --i) {
    scalar_t cf = (pos[i - 1] + random_shift_monolithic[level][i - 1]) *
                  scale_factor[level][i - 1];
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
#pragma unroll
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
#pragma unroll
  for (int i = 0; i < pos_dim; ++i) {
    scalar_t di = elevated[i] - rem0[i];

#pragma unroll
    for (int j = i + 1; j <= pos_dim; ++j)
      if (di < elevated[j] - rem0[j])
        ++rank[i];
      else
        ++rank[j];
  }

  // If the point doesn't lie on the plane (sum != 0) bring it back
  // Conway et al., 1998 p. 447-448 (Algorithm 3, Step 4)
#pragma unroll
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

  // Compute the barycentric coordinates
  // Baek et al., 2009 (Proposition 4.2)
  scalar_t barycentric[pos_dim + 2]{};
#pragma unroll
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
  /* #pragma unroll */
  /* for (int i = 0; i <= pos_dim; i++) { */
  /*   scalar_t delta = (elevated[i] - rem0[i]) * factor; */
  /*   barycentric[pos_dim - rank[i]] += delta; */
  /*   barycentric[pos_dim + 1 - rank[i]] -= delta; */
  /* } */
  /* barycentric[0] += scalar_t{1.0} + barycentric[pos_dim + 1]; */

  // here we accumulate the values and the homogeneous term
  scalar_t val_hom_vec[2] = {};

  scalar_t w_lvl = anneal_window[level];

  int key[pos_dim];
#pragma unroll
  for (int remainder = 0; remainder <= pos_dim; remainder++) {
    // Compute the location of the lattice point explicitly (all but
    // the last coordinate - it's redundant because they sum to zero)
#pragma unroll
    for (int i = 0; i < pos_dim; i++) {
      key[i] = rem0[i] + remainder;
      if (rank[i] > pos_dim - remainder) key[i] -= (pos_dim + 1);
    }

    // Retrieve pointer to the value at this vertex.
    int idx_val = idx_hash_with_collision<pos_dim>(key, lattice_capacity);

    // if the vertex exists accumulate its value weighted by the barycentric
    // weight (accumulates also the homogeneous coordinate)
    scalar_t w = barycentric[remainder] * w_lvl;

    val_hom_vec[0] =
        val_hom_vec[0] +
        lattice_values_monolithic[batch_idx][level][idx_val][0] * w;
    val_hom_vec[1] =
        val_hom_vec[1] +
        lattice_values_monolithic[batch_idx][level][idx_val][1] * w;
  }

  sliced_values_monolithic[batch_idx][level][0][idx] = val_hom_vec[0];
  sliced_values_monolithic[batch_idx][level][1][idx] = val_hom_vec[1];
}

template <int pos_dim, int val_dim, typename scalar_t>
__global__ void __launch_bounds__(
    BLOCK_SIZE_BACK)  // since the block size is known at compile time we can
                      // specify it to the kernel and therefore cuda doesnt need
                      // to use heuristics based on code complexity to minimize
                      // registry usage
    backward_gpu(
        const int nr_positions, const int lattice_capacity,
        const torch::PackedTensorAccessor32<scalar_t, 4,
                                            torch::RestrictPtrTraits>
            lattice_values_monolithic,
        const torch::PackedTensorAccessor32<scalar_t, 3,
                                            torch::RestrictPtrTraits>
            positions,
        const torch::PackedTensorAccessor32<scalar_t, 2,
                                            torch::RestrictPtrTraits>
            scale_factor,
        const torch::PackedTensorAccessor32<scalar_t, 2,
                                            torch::RestrictPtrTraits>
            random_shift_monolithic,
        const torch::PackedTensorAccessor32<scalar_t, 1,
                                            torch::RestrictPtrTraits>
            anneal_window,
        const torch::PackedTensorAccessor32<scalar_t, 4,
                                            torch::RestrictPtrTraits>
            grad_sliced_values_monolithic,
        torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits>
            lattice_values_monolithic_grad,
        const bool concat_points) {
  // lattice_values_monolithic refers to the values that the lattice had in the
  // forward pass. it has size m_hash_table_capcity x (val_dim+1)
  // grad_sliced_values is the gradient of the loss with respect to the sliced
  // out values which has size nr_positions x val_dim

  // code should be the same as forward (without concatenating positions in the
  // beginning) until last loop
  const int batch_idx = blockIdx.z;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nr_positions) {
    return;
  }

  const uint32_t level = blockIdx.y;

  scalar_t pos[pos_dim];
  for (int i = 0; i < pos_dim; i++) {
    pos[i] = positions[batch_idx][idx][i];
  }

  scalar_t elevated[pos_dim + 1];
  scalar_t sm{};
#pragma unroll
  for (int i = pos_dim; i > 0; i--) {
    scalar_t cf = (pos[i - 1] + random_shift_monolithic[level][i - 1]) *
                  scale_factor[level][i - 1];
    elevated[i] = sm - i * cf;
    sm += cf;
  }
  elevated[0] = sm;

  int rem0[pos_dim + 1];
  int sum{0};
  scalar_t factor{scalar_t{1.0} / (pos_dim + 1)};
#pragma unroll
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
#pragma unroll
  for (int i = 0; i < pos_dim; ++i) {
    scalar_t di = elevated[i] - rem0[i];

#pragma unroll
    for (int j = i + 1; j <= pos_dim; ++j)
      if (di < elevated[j] - rem0[j])
        ++rank[i];
      else
        ++rank[j];
  }

#pragma unroll
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

  scalar_t barycentric[pos_dim + 2]{};
#pragma unroll
  for (int i = 0; i <= pos_dim; ++i) {
    scalar_t delta = (elevated[i] - rem0[i]) * factor;
    for (int j = 0; j <= pos_dim; ++j) {
      if (j != rank[i]) continue;
      barycentric[pos_dim - j] += delta;
      barycentric[pos_dim + 1 - j] -= delta;
    }
  }
  barycentric[0] += scalar_t{1.0} + barycentric[pos_dim + 1];

  scalar_t grad_sliced_val_cur[] = {
      grad_sliced_values_monolithic[batch_idx][level][0][idx],
      grad_sliced_values_monolithic[batch_idx][level][1][idx]};

  scalar_t w_lvl = anneal_window[level];

  int key[pos_dim];
#pragma unroll
  for (int remainder = 0; remainder <= pos_dim; remainder++) {
#pragma unroll
    for (int i = 0; i < pos_dim; i++) {
      key[i] = rem0[i] + remainder;
      if (rank[i] > pos_dim - remainder) key[i] -= (pos_dim + 1);
    }

    // Retrieve pointer to the value at this vertex.
    int idx_val = idx_hash_with_collision<pos_dim>(key, lattice_capacity);

    scalar_t w = barycentric[remainder] * w_lvl;

    /* TODO maybe can change layout to use half2 here instead
     * for now just use float for atomicAdd
     * see also
     * https://github.com/NVlabs/tiny-cuda-nn/blob/28ca991f99b44d10387d73077c07ccfdd7f96275/include/tiny-cuda-nn/encodings/grid.h#L652-L665
     */

    atomicAdd(&lattice_values_monolithic_grad[batch_idx][level][0][idx_val],
              grad_sliced_val_cur[0] * w);
    atomicAdd(&lattice_values_monolithic_grad[batch_idx][level][1][idx_val],
              grad_sliced_val_cur[1] * w);
  }
}

template <int pos_dim, int val_dim, typename scalar_t>
__global__ void __launch_bounds__(
    BLOCK_SIZE_BACK)  // since the block size is known at compile time we can
                      // specify it to the kernel and therefore cuda doesnt need
                      // to use heuristics based on code complexity to minimize
                      // registry usage
    backward_gpu_only_pos(
        const int nr_positions, const int lattice_capacity,
        const torch::PackedTensorAccessor32<scalar_t, 4,
                                            torch::RestrictPtrTraits>
            lattice_values_monolithic,
        const torch::PackedTensorAccessor32<scalar_t, 3,
                                            torch::RestrictPtrTraits>
            positions,
        const torch::PackedTensorAccessor32<scalar_t, 2,
                                            torch::RestrictPtrTraits>
            scale_factor,
        const torch::PackedTensorAccessor32<scalar_t, 2,
                                            torch::RestrictPtrTraits>
            random_shift_monolithic,
        const torch::PackedTensorAccessor32<scalar_t, 1,
                                            torch::RestrictPtrTraits>
            anneal_window,
        const torch::PackedTensorAccessor32<scalar_t, 4,
                                            torch::RestrictPtrTraits>
            grad_sliced_values_monolithic,
        torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits>
            positions_grad,
        const bool concat_points, const bool require_lattice_values_grad,
        const bool require_positions_grad) {
  // values_vertices refers to the values that the lattice had in the forward
  // pass. it has size m_hash_table_capcity x (val_dim+1) grad_sliced_values is
  // the gradient of the loss with respect to the sliced out values which has
  // size nr_positions x val_dim
  const int batch_idx = blockIdx.z;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nr_positions) {
    return;
  }

  const uint32_t level = blockIdx.y;

  scalar_t pos[pos_dim];
  for (int i = 0; i < pos_dim; i++) {
    pos[i] = positions[batch_idx][idx][i];
  }

  scalar_t elevated[pos_dim + 1];
  scalar_t sm{};
#pragma unroll
  for (int i = pos_dim; i > 0; i--) {
    scalar_t cf = (pos[i - 1] + random_shift_monolithic[level][i - 1]) *
                  scale_factor[level][i - 1];
    elevated[i] = sm - i * cf;
    sm += cf;
  }
  elevated[0] = sm;

  int rem0[pos_dim + 1];
  int sum{0};
  scalar_t factor{scalar_t{1.0} / (pos_dim + 1)};
#pragma unroll
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
#pragma unroll
  for (int i = 0; i < pos_dim; ++i) {
    scalar_t di = elevated[i] - rem0[i];

#pragma unroll
    for (int j = i + 1; j <= pos_dim; ++j)
      if (di < elevated[j] - rem0[j])
        ++rank[i];
      else
        ++rank[j];
  }

#pragma unroll
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

  scalar_t barycentric[pos_dim + 2]{};
#pragma unroll
  for (int i = 0; i <= pos_dim; ++i) {
    scalar_t delta = (elevated[i] - rem0[i]) * factor;
    for (int j = 0; j <= pos_dim; ++j) {
      if (j != rank[i]) continue;
      barycentric[pos_dim - j] += delta;
      barycentric[pos_dim + 1 - j] -= delta;
    }
  }
  barycentric[0] += scalar_t{1.0} + barycentric[pos_dim + 1];

  scalar_t grad_sliced_val_cur[] = {
      grad_sliced_values_monolithic[batch_idx][level][0][idx],
      grad_sliced_values_monolithic[batch_idx][level][1][idx]};

  scalar_t w_lvl = anneal_window[level];

  int key[pos_dim];

  // We have from upstrema grad the dL/dS which is the derivative of the loss
  // wrt to the sliced value If we require positions grad we want to obtain
  // dL/dPos dL/dPos = dL/dS *dS/dB * dB/dE * dE/dPos We need dS/dB which is
  // the derivative of the sliced value wrt to the barycentric coords We need
  // dB/dE which is the derivative of the barycentric wrt to the elevated
  // value We need dE/dP which is the derivative of the elevated wrt to the
  // position in xyz
  // dL/dB  = dL/dS *dS/dB
  // foward pass is just S=B0*WLvl*V0 + B1*WLvl*V1 etc
  // so dS/dB0 is just W*V0
  scalar_t dL_dbarycentric[pos_dim + 2]{};
  for (int remainder = 0; remainder <= pos_dim; remainder++) {
#pragma unroll
    // Compute the location of the lattice point explicitly (all but
    // the last coordinate - it's redundant because they sum to zero)
    for (int i = 0; i < pos_dim; i++) {
      key[i] = rem0[i] + remainder;
      if (rank[i] > pos_dim - remainder) key[i] -= (pos_dim + 1);
    }
    // Retrieve pointer to the value at this vertex.
    int idx_val = idx_hash_with_collision<pos_dim>(key, lattice_capacity);

    // Load the value for this vertex
    const scalar_t* fv = &lattice_values_monolithic[batch_idx][level][idx_val][0];
    // add to the dL_d_barycentric
    dL_dbarycentric[remainder] += fv[0] * w_lvl * grad_sliced_val_cur[0];
    dL_dbarycentric[remainder] += fv[1] * w_lvl * grad_sliced_val_cur[1];
  }

  // dL/dE  = dL/dB *dB/dE
  // In the forward pass of computing B from E there is this wraparound line
  // of barycentric[0] += 1.0 + barycentric[pos_dim + 1];
  // barycentric[0] = barycentric[0]+ 1.0 + barycentric[pos_dim + 1];
  // I think this means that the gradient of also added to
  // barycentric{pos_dim+1}
  // TODO check for correctness here
  dL_dbarycentric[pos_dim + 1] +=
      dL_dbarycentric[0];  // order here is important btw, we first add B0 to
                           // B5 and only afterwards we double B0
  // dL_dbarycentric[0]=dL_dbarycentric[0]*2;
  // Now we need to accumulate gradient into elevated from from each
  // barycentric that the particlar elevated affected
  scalar_t dL_delevated[pos_dim + 1]{};
#pragma unroll
  for (int i = 0; i <= pos_dim; i++) {
    dL_delevated[i] +=
        dL_dbarycentric[pos_dim - rank[i]] * (1.0f / (pos_dim + 1));
    dL_delevated[i] -=
        dL_dbarycentric[pos_dim + 1 - rank[i]] * (1.0f / (pos_dim + 1));
  }

  // dL/dPos = dL/dE * dE/dPos
  scalar_t dL_dPos[pos_dim]{};
  // I unrolles the loop that computes E from P and I got some local
  // derivatives like dEx/dPx=Sx  dEx/dPy=Sy dEy/dPx=-Sx  dEy/dPy=Sy
  // dEy/dPz=Sz dEz/dPy=-2Sy  dEz/dPz=Sz dEw/dPz=-3Sz So we just accumulate
  // these values inot dL_dPos
  // x
  // dL_dPos[0]= dL_delevated[0]* scale_factor[level][0] +
  //             dL_delevated[1]* (-scale_factor[level][0]);
  // //y
  // dL_dPos[1]= dL_delevated[0]* scale_factor[level][1] +
  //             dL_delevated[1]* scale_factor[level][1] +
  //             dL_delevated[2]* (-2*scale_factor[level][1]);
  // //z
  // dL_dPos[2]= dL_delevated[0]* scale_factor[level][2] +
  //             dL_delevated[1]* scale_factor[level][2] +
  //             dL_delevated[2]* scale_factor[level][2] +
  //             dL_delevated[3]* (-3*scale_factor[level][2]);
  // do it in a loop so as to support various pos_dims
  for (int i = 0; i < pos_dim; i++) {
#pragma unroll
    for (int j = 0; j <= i; j++) {
      dL_dPos[i] += dL_delevated[j] * scale_factor[level][i];
      // dL_dPos[i]+=dL_delevated[j]*scale_factor_constant[level*pos_dim + i];
    }
  }
#pragma unroll
  for (int i = 0; i < pos_dim; i++) {
    dL_dPos[i] -= dL_delevated[i + 1] * scale_factor[level][i] * (i + 1);
    // dL_dPos[i]-=dL_delevated[i+1] * scale_factor_constant[level*pos_dim +
    // i] * (i+1);
  }
// finish
// atomicAdd(&positions_grad[idx][0], dL_dPos[0]  );
// atomicAdd(&positions_grad[idx][1], dL_dPos[1]  );
// atomicAdd(&positions_grad[idx][2], dL_dPos[2]  );
#pragma unroll
  for (int i = 0; i < pos_dim; i++) {
    // atomicAdd(&positions_grad[idx][i], dL_dPos[i]  );
    atomicAdd(&positions_grad[batch_idx][i][idx], dL_dPos[i]);
  }
  // Cannot be done like this because the sums into the positions grad may
  // come from multiple levels so they need to be atomic
  // positions_grad[idx][0]=dL_dPos[0];
  // positions_grad[idx][1]=dL_dPos[1];
  // positions_grad[idx][2]=dL_dPos[2];

  // positions_grad[level][idx][0]=dL_dPos[0];
  // positions_grad[level][idx][1]=dL_dPos[1];
  // positions_grad[level][idx][2]=dL_dPos[2];
  // #pragma unroll
  // for(int i=0; i<pos_dim; i++){
  //     positions_grad[level][idx][i]=dL_dPos[i];
  // }
}

// double back
template <int pos_dim, int val_dim>
__global__ void __launch_bounds__(
    BLOCK_SIZE_DOUBLE_BACK)  // since the block size is known at compile time we
                             // can specify it to the kernel and therefore cuda
                             // doesnt need to use heuristics based on code
                             // complexity to minimize registry usage
    double_backward_from_positions_gpu(
        const int nr_positions, const int lattice_capacity,
        const int nr_resolutions,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits>
            double_positions_grad,
        const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits>
            lattice_values_monolithic,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits>
            positions,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits>
            scale_factor,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits>
            random_shift_monolithic,
        const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits>
            anneal_window,
        const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits>
            grad_sliced_values_monolithic,
        const bool concat_points,
        // output
        torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits>
            grad_grad_sliced_values_monolithic,
        torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits>
            lattice_values_monolithic_grad) {
  // values_vertices refers to the values that the lattice had in the forward
  // pass. it has size m_hash_table_capcity x (val_dim+1) grad_sliced_values is
  // the gradient of the loss with respect to the sliced out values which has
  // size nr_positions x val_dim
  const int idx = blockIdx.x * blockDim.x +
                  threadIdx.x;  // each thread will deal with one position
  if (idx >= nr_positions) {
    return;
  }

  const uint32_t level =
      blockIdx.y;  // <- the level is the same for all threads

  if (level >= nr_resolutions) {
    // we are in one of the extra resolutions so we just write zero in the grad
    // sliced grad
    grad_grad_sliced_values_monolithic[level][0][idx] = 0;
    grad_grad_sliced_values_monolithic[level][1][idx] = 0;
    return;
  }

  float pos[pos_dim];
  for (int i = 0; i < pos_dim; i++) {
    pos[i] = positions[idx][i];
  }

  float elevated[pos_dim + 1];
  float sm = 0;
#pragma unroll
  for (int i = pos_dim; i > 0; i--) {
    // float cf = (pos[i-1] +random_shift_constant[level*pos_dim + i-1]  ) *
    // scale_factor_constant[level*pos_dim + i-1];
    float cf = (pos[i - 1] + random_shift_monolithic[level][i - 1]) *
               scale_factor[level][i - 1];
    elevated[i] = sm - i * cf;
    sm += cf;
  }
  elevated[0] = sm;

  int rem0[pos_dim + 1];
  int rank[pos_dim + 1]{};

  // Find the closest 0-colored simplex through rounding
  // greedily search for the closest zero-colored lattice point
  int sum = 0;
#pragma unroll
  for (int i = 0; i <= pos_dim; i++) {
    float v = elevated[i] * (1.0f / (pos_dim + 1));
    float up = ceil(v) * (pos_dim + 1);
    float down = floor(v) * (pos_dim + 1);
    if (up - elevated[i] < elevated[i] - down) {
      rem0[i] = (int)up;
    } else {
      rem0[i] = (int)down;
    }
    sum += rem0[i];
  }
  sum /= pos_dim + 1;

// Find the simplex we are in and store it in rank (where rank describes what
// position coordinate i has in the sorted order of the features values)
#pragma unroll
  for (int i = 0; i < pos_dim; i++) {
    double di = elevated[i] - rem0[i];
    for (int j = i + 1; j <= pos_dim; j++)
      if (di < elevated[j] - rem0[j])
        rank[i]++;
      else
        rank[j]++;
  }

// If the point doesn't lie on the plane (sum != 0) bring it back
#pragma unroll
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

  float barycentric[pos_dim + 2]{0.0f};
// Compute the barycentric coordinates (p.10 in [Adams et al. 2010])
#pragma unroll
  for (int i = 0; i <= pos_dim; i++) {
    float delta = (elevated[i] - rem0[i]) * (1.0f / (pos_dim + 1));
    barycentric[pos_dim - rank[i]] += delta;
    barycentric[pos_dim + 1 - rank[i]] -= delta;
  }
  // Wrap around
  barycentric[0] += 1.0f + barycentric[pos_dim + 1];

  float w_lvl = anneal_window[level];

  // get the value at the position
  float grad_sliced_val_cur[val_dim];
#pragma unroll
  for (int j = 0; j < val_dim; j++) {
    grad_sliced_val_cur[j] = grad_sliced_values_monolithic[level][j][idx];
  }

  // get eh gradient at the curent position
  float grad_p_cur[pos_dim];
#pragma unroll
  for (int j = 0; j < pos_dim; j++) {
    grad_p_cur[j] = double_positions_grad[idx][j];
  }

  int key[pos_dim];

  // We have upstream gradient dL/dPos which is double_positions_grad
  // we want dL/dV and dL/dS, so we want to push the gradient into
  // lattice_values_monolithic_grad    grad_grad_sliced_values_monolithic
  // dL/dS = dL/dP * dP/dE * dE/dB * dB/dS
  // dL/dV = dL/dP * dP/dE * dE/dB * dB/dV
  // STARTING
  // dP/dE
  float dL_delevated[pos_dim + 1]{0.0f};
  //-------hardocded for 3 positions----------
  // dL_delevated[0] =   grad_p_cur[0] * scale_factor[level][0] +
  //                     grad_p_cur[1] * scale_factor[level][1] +
  //                     grad_p_cur[2] * scale_factor[level][2];
  // dL_delevated[1] =   grad_p_cur[0] * (-scale_factor[level][0]) +
  //                     grad_p_cur[1] * scale_factor[level][1] +
  //                     grad_p_cur[2] * scale_factor[level][2];
  // dL_delevated[2] =   grad_p_cur[1] * (-2*scale_factor[level][1]) +
  //                     grad_p_cur[2] * scale_factor[level][2];
  // dL_delevated[3] =   grad_p_cur[2] * (-3*scale_factor[level][2]);
  //------doing it so that it support all pos_dims--------
  // in the forward pass we do:
  // for(int i=0; i<pos_dim; i++){
  //     for(int j=0; j<=i; j++){
  //         dL_dPos[i]+=dL_delevated[j]*scale_factor[level][i];
  //     }
  // }
  // for(int i=0; i<pos_dim; i++){
  //     dL_dPos[i]-=dL_delevated[i+1] * scale_factor[level][i] * (i+1);
  // }
  // so the gradient from grad_p_cur[i] will go into each elevated <= i.
  // Afterwards we have another loop which passes the gradient from
  // grad_p_cur[i] into elevated[i+1]
  for (int i = 0; i < pos_dim; i++) {
    float grad = grad_p_cur[i] * scale_factor[level][i];
// float grad=grad_p_cur[i]*scale_factor_constant[level*pos_dim + i];
#pragma unroll
    for (int j = 0; j <= i; j++) {
      dL_delevated[j] += grad;
    }
  }
#pragma unroll
  for (int i = 0; i < pos_dim; i++) {
    dL_delevated[i + 1] -= grad_p_cur[i] * scale_factor[level][i] * (i + 1);
    // dL_delevated[i+1]-=grad_p_cur[i] * scale_factor_constant[level*pos_dim +
    // i] * (i+1);
  }

  // dE/dB
  float dL_dbarycentric[pos_dim + 2]{0.0f};
  // in the forward pass we did:
  // dL_dbarycentric[pos_dim + 1] += dL_dbarycentric[0]; //order here is
  // important btw, we first add B0 to B5 and only afterwards we double B0
  // dL_dbarycentric[0]=dL_dbarycentric[0]*2;
  // float dL_delevated[pos_dim + 1]{0};
  // #pragma unroll
  // for (int i = 0; i <= pos_dim; i++) {
  //     dL_delevated[i]+=  dL_dbarycentric[pos_dim - rank[i]] * (1.0 / (pos_dim
  //     + 1)); dL_delevated[i]-=  dL_dbarycentric[pos_dim + 1 - rank[i]] * (1.0
  //     / (pos_dim + 1));
  // }
  // So now we do this
  for (int i = 0; i <= pos_dim; i++) {
    dL_dbarycentric[pos_dim - rank[i]] +=
        dL_delevated[i] * (1.0f / (pos_dim + 1));
    dL_dbarycentric[pos_dim + 1 - rank[i]] -=
        dL_delevated[i] * (1.0f / (pos_dim + 1));
  }
  // dL_dbarycentric[0]=dL_dbarycentric[0]*2;
  dL_dbarycentric[0] += dL_dbarycentric[pos_dim + 1];
  // push gradient into values_lattice and grad_sliced
  float grad_grad_sliced_val_cur[val_dim]{0.0f};
  for (int remainder = 0; remainder <= pos_dim; remainder++) {
// Compute the location of the lattice point explicitly (all but
// the last coordinate - it's redundant because they sum to zero)
#pragma unroll
    for (int i = 0; i < pos_dim; i++) {
      key[i] = rem0[i] + remainder;
      if (rank[i] > pos_dim - remainder) key[i] -= (pos_dim + 1);
    }
    // Retrieve pointer to the value at this vertex.
    int idx_val = idx_hash_with_collision<pos_dim>(key, lattice_capacity);

    // Load the value for this vertex
    const float* fv = &lattice_values_monolithic[level][idx_val][0];
    const float2 val_lattice_vertex = reinterpret_cast<const float2*>(fv)[0];

    // atomicAdd(&lattice_values_monolithic_grad[level][idx_val][0],
    // dL_dbarycentric[remainder]* w_lvl * grad_sliced_val_cur[0]  );
    // atomicAdd(&lattice_values_monolithic_grad[level][idx_val][1],
    // dL_dbarycentric[remainder]* w_lvl * grad_sliced_val_cur[1]  );
    atomicAdd(&lattice_values_monolithic_grad[level][0][idx_val],
              dL_dbarycentric[remainder] * w_lvl * grad_sliced_val_cur[0]);
    atomicAdd(&lattice_values_monolithic_grad[level][1][idx_val],
              dL_dbarycentric[remainder] * w_lvl * grad_sliced_val_cur[1]);

    // push gradient into grad_sliced_val_cur
    // grad_sliced_val_cur add towards all the barycentric coord, so in the
    // backward pass the gradient from b0 to all the grad_sliced_val_cur
    grad_grad_sliced_val_cur[0] +=
        dL_dbarycentric[remainder] * w_lvl * val_lattice_vertex.x;
    grad_grad_sliced_val_cur[1] +=
        dL_dbarycentric[remainder] * w_lvl * val_lattice_vertex.y;
  }
  // finish the accumulation of grad_grad_sliced
  grad_grad_sliced_values_monolithic[level][0][idx] =
      grad_grad_sliced_val_cur[0];
  grad_grad_sliced_values_monolithic[level][1][idx] =
      grad_grad_sliced_val_cur[1];
}

// double back
template <int pos_dim, int val_dim>
__global__ void __launch_bounds__(
    BLOCK_SIZE_DOUBLE_BACK)  // since the block size is known at compile time we
                             // can specify it to the kernel and therefore cuda
                             // doesnt need to use heuristics based on code
                             // complexity to minimize registry usage
    double_backward_from_positions_gpu_1(
        const int nr_positions, const int lattice_capacity,
        const int nr_resolutions,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits>
            double_positions_grad,
        const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits>
            lattice_values_monolithic,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits>
            positions,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits>
            scale_factor,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits>
            random_shift_monolithic,
        const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits>
            anneal_window,
        const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits>
            grad_sliced_values_monolithic,
        const bool concat_points,
        // output
        torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits>
            grad_grad_sliced_values_monolithic,
        torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits>
            lattice_values_monolithic_grad) {
  // values_vertices refers to the values that the lattice had in the forward
  // pass. it has size m_hash_table_capcity x (val_dim+1) grad_sliced_values is
  // the gradient of the loss with respect to the sliced out values which has
  // size nr_positions x val_dim
  const int idx = blockIdx.x * blockDim.x +
                  threadIdx.x;  // each thread will deal with one position
  if (idx >= nr_positions) {
    return;
  }

  const uint32_t level =
      blockIdx.y;  // <- the level is the same for all threads

  if (level >= nr_resolutions) {
    // we are in one of the extra resolutions so we just write zero in the grad
    // sliced grad
    // grad_grad_sliced_values_monolithic[level][0][idx]=0;
    // grad_grad_sliced_values_monolithic[level][1][idx]=0;
    // return;
  }

  float pos[pos_dim];
  for (int i = 0; i < pos_dim; i++) {
    pos[i] = positions[idx][i];
  }

  float elevated[pos_dim + 1];
  float sm = 0;
#pragma unroll
  for (int i = pos_dim; i > 0; i--) {
    // float cf = (pos[i-1] +random_shift_constant[level*pos_dim + i-1]  ) *
    // scale_factor_constant[level*pos_dim + i-1];
    float cf = (pos[i - 1] + random_shift_monolithic[level][i - 1]) *
               scale_factor[level][i - 1];
    elevated[i] = sm - i * cf;
    sm += cf;
  }
  elevated[0] = sm;

  int rem0[pos_dim + 1];
  int rank[pos_dim + 1]{};

  // Find the closest 0-colored simplex through rounding
  // greedily search for the closest zero-colored lattice point
  int sum = 0;
#pragma unroll
  for (int i = 0; i <= pos_dim; i++) {
    float v = elevated[i] * (1.0f / (pos_dim + 1));
    float up = ceil(v) * (pos_dim + 1);
    float down = floor(v) * (pos_dim + 1);
    if (up - elevated[i] < elevated[i] - down) {
      rem0[i] = (int)up;
    } else {
      rem0[i] = (int)down;
    }
    sum += rem0[i];
  }
  sum /= pos_dim + 1;

// Find the simplex we are in and store it in rank (where rank describes what
// position coordinate i has in the sorted order of the features values)
#pragma unroll
  for (int i = 0; i < pos_dim; i++) {
    double di = elevated[i] - rem0[i];
    for (int j = i + 1; j <= pos_dim; j++)
      if (di < elevated[j] - rem0[j])
        rank[i]++;
      else
        rank[j]++;
  }

// If the point doesn't lie on the plane (sum != 0) bring it back
#pragma unroll
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

  float barycentric[pos_dim + 2]{0.0f};
// Compute the barycentric coordinates (p.10 in [Adams etal 2010])
#pragma unroll
  for (int i = 0; i <= pos_dim; i++) {
    float delta = (elevated[i] - rem0[i]) * (1.0f / (pos_dim + 1));
    barycentric[pos_dim - rank[i]] += delta;
    barycentric[pos_dim + 1 - rank[i]] -= delta;
  }
  // Wrap around
  barycentric[0] += 1.0f + barycentric[pos_dim + 1];

  float w_lvl = anneal_window[level];

  // get the value at the position
  float grad_sliced_val_cur[val_dim];
#pragma unroll
  for (int j = 0; j < val_dim; j++) {
    grad_sliced_val_cur[j] = grad_sliced_values_monolithic[level][j][idx];
  }

  // //get eh gradient at the curent position
  float grad_p_cur[pos_dim];
#pragma unroll
  for (int j = 0; j < pos_dim; j++) {
    grad_p_cur[j] = double_positions_grad[idx][j];
  }

  int key[pos_dim];

  // We have upstream gradient dL/dPos which is double_positions_grad
  // we want dL/dV and dL/dS, so we want to push the gradient into
  // lattice_values_monolithic_grad    grad_grad_sliced_values_monolithic
  // dL/dS = dL/dP * dP/dE * dE/dB * dB/dS
  // dL/dV = dL/dP * dP/dE * dE/dB * dB/dV
  // STARTING
  // dP/dE
  float dL_delevated[pos_dim + 1]{0.0f};
  //-------hardocded for 3 positions----------
  // dL_delevated[0] =   grad_p_cur[0] * scale_factor[level][0] +
  //                     grad_p_cur[1] * scale_factor[level][1] +
  //                     grad_p_cur[2] * scale_factor[level][2];
  // dL_delevated[1] =   grad_p_cur[0] * (-scale_factor[level][0]) +
  //                     grad_p_cur[1] * scale_factor[level][1] +
  //                     grad_p_cur[2] * scale_factor[level][2];
  // dL_delevated[2] =   grad_p_cur[1] * (-2*scale_factor[level][1]) +
  //                     grad_p_cur[2] * scale_factor[level][2];
  // dL_delevated[3] =   grad_p_cur[2] * (-3*scale_factor[level][2]);
  //------doing it so that it support all pos_dims--------
  // in the forward pass we do:
  // for(int i=0; i<pos_dim; i++){
  //     for(int j=0; j<=i; j++){
  //         dL_dPos[i]+=dL_delevated[j]*scale_factor[level][i];
  //     }
  // }
  // for(int i=0; i<pos_dim; i++){
  //     dL_dPos[i]-=dL_delevated[i+1] * scale_factor[level][i] * (i+1);
  // }
  // so the gradient from grad_p_cur[i] will go into each elevated <= i.
  // Afterwards we have another loop which passes the gradient from
  // grad_p_cur[i] into elevated[i+1]
  for (int i = 0; i < pos_dim; i++) {
    float grad = grad_p_cur[i] * scale_factor[level][i];
// float grad=grad_p_cur[i]*scale_factor_constant[level*pos_dim + i];
#pragma unroll
    for (int j = 0; j <= i; j++) {
      dL_delevated[j] += grad;
    }
  }
#pragma unroll
  for (int i = 0; i < pos_dim; i++) {
    dL_delevated[i + 1] -= grad_p_cur[i] * scale_factor[level][i] * (i + 1);
    // dL_delevated[i+1]-=grad_p_cur[i] * scale_factor_constant[level*pos_dim +
    // i] * (i+1);
  }
  // dE/dB
  float dL_dbarycentric[pos_dim + 2]{0.0f};
// in the forward pass we did:
// dL_dbarycentric[pos_dim + 1] += dL_dbarycentric[0]; //order here is important
// btw, we first add B0 to B5 and only afterwards we double B0
// dL_dbarycentric[0]=dL_dbarycentric[0]*2;
// float dL_delevated[pos_dim + 1]{0};
// #pragma unroll
// for (int i = 0; i <= pos_dim; i++) {
//     dL_delevated[i]+=  dL_dbarycentric[pos_dim - rank[i]] * (1.0 / (pos_dim +
//     1)); dL_delevated[i]-=  dL_dbarycentric[pos_dim + 1 - rank[i]] * (1.0 /
//     (pos_dim + 1));
// }
// So now we do this
#pragma unroll
  for (int i = 0; i <= pos_dim; i++) {
    dL_dbarycentric[pos_dim - rank[i]] +=
        dL_delevated[i] * (1.0f / (pos_dim + 1));
    dL_dbarycentric[pos_dim + 1 - rank[i]] -=
        dL_delevated[i] * (1.0f / (pos_dim + 1));
  }
  // dL_dbarycentric[0]=dL_dbarycentric[0]*2;
  dL_dbarycentric[0] += dL_dbarycentric[pos_dim + 1];
  // push gradient into values_lattice and grad_sliced
  // float grad_grad_sliced_val_cur[val_dim]{0};
  for (int remainder = 0; remainder <= pos_dim; remainder++) {
// Compute the location of the lattice point explicitly (all but
// the last coordinate - it's redundant because they sum to zero)
#pragma unroll
    for (int i = 0; i < pos_dim; i++) {
      key[i] = rem0[i] + remainder;
      if (rank[i] > pos_dim - remainder) key[i] -= (pos_dim + 1);
    }
    // Retrieve pointer to the value at this vertex.
    int idx_val = idx_hash_with_collision<pos_dim>(key, lattice_capacity);

    // Load the value for this vertex
    // const float* fv=&lattice_values_monolithic[level][idx_val][0];
    // const float2 val_lattice_vertex=reinterpret_cast<const float2*>( fv )[0];
    // add to the dL_d_barycentric
    // dL_dbarycentric[remainder]+=val_lattice_vertex.x*w_lvl   *
    // grad_sliced_val_cur[0];
    // dL_dbarycentric[remainder]+=val_lattice_vertex.y*w_lvl   *
    // grad_sliced_val_cur[1]; lattice_values_monolithic_grad[level][idx_val][0]
    // += dL_dbarycentric[remainder]* w_lvl * grad_sliced_val_cur[0];
    // lattice_values_monolithic_grad[level][idx_val][1] +=
    // dL_dbarycentric[remainder]* w_lvl * grad_sliced_val_cur[1];

    // atomicAdd(&lattice_values_monolithic_grad[level][idx_val][0],
    // dL_dbarycentric[remainder]* w_lvl * grad_sliced_val_cur[0]  );
    // atomicAdd(&lattice_values_monolithic_grad[level][idx_val][1],
    // dL_dbarycentric[remainder]* w_lvl * grad_sliced_val_cur[1]  );
    atomicAdd(&lattice_values_monolithic_grad[level][0][idx_val],
              dL_dbarycentric[remainder] * w_lvl * grad_sliced_val_cur[0]);
    atomicAdd(&lattice_values_monolithic_grad[level][1][idx_val],
              dL_dbarycentric[remainder] * w_lvl * grad_sliced_val_cur[1]);

    // push gradient into grad_sliced_val_cur
    // grad_sliced_val_cur add towards all the barycentric coord, so in the
    // backward pass the gradient from b0 to all the grad_sliced_val_cur
    // grad_grad_sliced_val_cur[0]+=dL_dbarycentric[remainder]* w_lvl *
    // val_lattice_vertex.x;
    // grad_grad_sliced_val_cur[1]+=dL_dbarycentric[remainder]* w_lvl *
    // val_lattice_vertex.y;
  }
  // finish the accumulation of grad_grad_sliced
  // atomicAdd(&grad_grad_sliced_values_monolithic[level][0][idx],
  // grad_grad_sliced_val_cur[0]  );
  // atomicAdd(&grad_grad_sliced_values_monolithic[level][1][idx],
  // grad_grad_sliced_val_cur[1]  );
}

// double back
template <int pos_dim, int val_dim>
__global__ void __launch_bounds__(
    BLOCK_SIZE_DOUBLE_BACK)  // since the block size is known at compile time we
                             // can specify it to the kernel and therefore cuda
                             // doesnt need to use heuristics based on code
                             // complexity to minimize registry usage
    double_backward_from_positions_gpu_2(
        const int nr_positions, const int lattice_capacity,
        const int nr_resolutions,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits>
            double_positions_grad,
        const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits>
            lattice_values_monolithic,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits>
            positions,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits>
            scale_factor,
        const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits>
            random_shift_monolithic,
        const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits>
            anneal_window,
        const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits>
            grad_sliced_values_monolithic,
        const bool concat_points,
        // output
        torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits>
            grad_grad_sliced_values_monolithic,
        torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits>
            lattice_values_monolithic_grad) {
  // values_vertices refers to the values that the lattice had in the forward
  // pass. it has size m_hash_table_capcity x (val_dim+1) grad_sliced_values is
  // the gradient of the loss with respect to the sliced out values which has
  // size nr_positions x val_dim
  const int idx = blockIdx.x * blockDim.x +
                  threadIdx.x;  // each thread will deal with one position
  if (idx >= nr_positions) {
    return;
  }

  const uint32_t level =
      blockIdx.y;  // <- the level is the same for all threads

  if (level >= nr_resolutions) {
    // we are in one of the extra resolutions so we just write zero in the grad
    // sliced grad
    grad_grad_sliced_values_monolithic[level][0][idx] = 0;
    grad_grad_sliced_values_monolithic[level][1][idx] = 0;
    return;
  }

  float pos[pos_dim];
  for (int i = 0; i < pos_dim; i++) {
    pos[i] = positions[idx][i];
  }

  float elevated[pos_dim + 1];
  float sm = 0;
#pragma unroll
  for (int i = pos_dim; i > 0; i--) {
    // float cf = (pos[i-1] +random_shift_constant[level*pos_dim + i-1]  ) *
    // scale_factor_constant[level*pos_dim + i-1];
    float cf = (pos[i - 1] + random_shift_monolithic[level][i - 1]) *
               scale_factor[level][i - 1];
    elevated[i] = sm - i * cf;
    sm += cf;
  }
  elevated[0] = sm;

  int rem0[pos_dim + 1];
  int rank[pos_dim + 1]{};

  // Find the closest 0-colored simplex through rounding
  // greedily search for the closest zero-colored lattice point
  int sum = 0;
#pragma unroll
  for (int i = 0; i <= pos_dim; i++) {
    float v = elevated[i] * (1.0 / (pos_dim + 1));
    float up = ceil(v) * (pos_dim + 1);
    float down = floor(v) * (pos_dim + 1);
    if (up - elevated[i] < elevated[i] - down) {
      rem0[i] = (int)up;
    } else {
      rem0[i] = (int)down;
    }
    sum += rem0[i];
  }
  sum /= pos_dim + 1;

// Find the simplex we are in and store it in rank (where rank describes what
// position coordinate i has in the sorted order of the features values)
#pragma unroll
  for (int i = 0; i < pos_dim; i++) {
    double di = elevated[i] - rem0[i];
    for (int j = i + 1; j <= pos_dim; j++)
      if (di < elevated[j] - rem0[j])
        rank[i]++;
      else
        rank[j]++;
  }

// If the point doesn't lie on the plane (sum != 0) bring it back
#pragma unroll
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

  float barycentric[pos_dim + 2]{0.0f};
// Compute the barycentric coordinates (p.10 in [Adams etal 2010])
#pragma unroll
  for (int i = 0; i <= pos_dim; i++) {
    float delta = (elevated[i] - rem0[i]) * (1.0f / (pos_dim + 1));
    barycentric[pos_dim - rank[i]] += delta;
    barycentric[pos_dim + 1 - rank[i]] -= delta;
  }
  // Wrap around
  barycentric[0] += 1.0 + barycentric[pos_dim + 1];

  float w_lvl = anneal_window[level];

  // get eh gradient at the curent position
  float grad_p_cur[pos_dim];
#pragma unroll
  for (int j = 0; j < pos_dim; j++) {
    grad_p_cur[j] = double_positions_grad[idx][j];
  }

  int key[pos_dim];

  // We have upstream gradient dL/dPos which is double_positions_grad
  // we want dL/dV and dL/dS, so we want to push the gradient into
  // lattice_values_monolithic_grad    grad_grad_sliced_values_monolithic
  // dL/dS = dL/dP * dP/dE * dE/dB * dB/dS
  // dL/dV = dL/dP * dP/dE * dE/dB * dB/dV
  // STARTING
  // dP/dE
  float dL_delevated[pos_dim + 1]{0.0f};
  //-------hardocded for 3 positions----------
  // dL_delevated[0] =   grad_p_cur[0] * scale_factor[level][0] +
  //                     grad_p_cur[1] * scale_factor[level][1] +
  //                     grad_p_cur[2] * scale_factor[level][2];
  // dL_delevated[1] =   grad_p_cur[0] * (-scale_factor[level][0]) +
  //                     grad_p_cur[1] * scale_factor[level][1] +
  //                     grad_p_cur[2] * scale_factor[level][2];
  // dL_delevated[2] =   grad_p_cur[1] * (-2*scale_factor[level][1]) +
  //                     grad_p_cur[2] * scale_factor[level][2];
  // dL_delevated[3] =   grad_p_cur[2] * (-3*scale_factor[level][2]);
  //------doing it so that it support all pos_dims--------
  // in the forward pass we do:
  // for(int i=0; i<pos_dim; i++){
  //     for(int j=0; j<=i; j++){
  //         dL_dPos[i]+=dL_delevated[j]*scale_factor[level][i];
  //     }
  // }
  // for(int i=0; i<pos_dim; i++){
  //     dL_dPos[i]-=dL_delevated[i+1] * scale_factor[level][i] * (i+1);
  // }
  // so the gradient from grad_p_cur[i] will go into each elevated <= i.
  // Afterwards we have another loop which passes the gradient from
  // grad_p_cur[i] into elevated[i+1]
  for (int i = 0; i < pos_dim; i++) {
    float grad = grad_p_cur[i] * scale_factor[level][i];
// float grad=grad_p_cur[i]*scale_factor_constant[level*pos_dim + i];
#pragma unroll
    for (int j = 0; j <= i; j++) {
      dL_delevated[j] += grad;
    }
  }
#pragma unroll
  for (int i = 0; i < pos_dim; i++) {
    dL_delevated[i + 1] -= grad_p_cur[i] * scale_factor[level][i] * (i + 1);
    // dL_delevated[i+1]-=grad_p_cur[i] * scale_factor_constant[level*pos_dim +
    // i] * (i+1);
  }
  // dE/dB
  float dL_dbarycentric[pos_dim + 2]{0.0f};
// in the forward pass we did:
// dL_dbarycentric[pos_dim + 1] += dL_dbarycentric[0]; //order here is important
// btw, we first add B0 to B5 and only afterwards we double B0
// dL_dbarycentric[0]=dL_dbarycentric[0]*2;
// float dL_delevated[pos_dim + 1]{0};
// #pragma unroll
// for (int i = 0; i <= pos_dim; i++) {
//     dL_delevated[i]+=  dL_dbarycentric[pos_dim - rank[i]] * (1.0 / (pos_dim +
//     1)); dL_delevated[i]-=  dL_dbarycentric[pos_dim + 1 - rank[i]] * (1.0 /
//     (pos_dim + 1));
// }
// So now we do this
#pragma unroll
  for (int i = 0; i <= pos_dim; i++) {
    dL_dbarycentric[pos_dim - rank[i]] +=
        dL_delevated[i] * (1.0f / (pos_dim + 1));
    dL_dbarycentric[pos_dim + 1 - rank[i]] -=
        dL_delevated[i] * (1.0f / (pos_dim + 1));
  }
  // dL_dbarycentric[0]=dL_dbarycentric[0]*2;
  dL_dbarycentric[0] += dL_dbarycentric[pos_dim + 1];
  // push gradient into values_lattice and grad_sliced
  float grad_grad_sliced_val_cur[val_dim]{0.0f};
  for (int remainder = 0; remainder <= pos_dim; remainder++) {
// Compute the location of the lattice point explicitly (all but
// the last coordinate - it's redundant because they sum to zero)
#pragma unroll
    for (int i = 0; i < pos_dim; i++) {
      key[i] = rem0[i] + remainder;
      if (rank[i] > pos_dim - remainder) key[i] -= (pos_dim + 1);
    }
    // Retrieve pointer to the value at this vertex.
    int idx_val = idx_hash_with_collision<pos_dim>(key, lattice_capacity);

    // Load the value for this vertex
    const float* fv = &lattice_values_monolithic[level][idx_val][0];
    const float2 val_lattice_vertex = reinterpret_cast<const float2*>(fv)[0];
    // add to the dL_d_barycentric
    // dL_dbarycentric[remainder]+=val_lattice_vertex.x*w_lvl   *
    // grad_sliced_val_cur[0];
    // dL_dbarycentric[remainder]+=val_lattice_vertex.y*w_lvl   *
    // grad_sliced_val_cur[1]; lattice_values_monolithic_grad[level][idx_val][0]
    // += dL_dbarycentric[remainder]* w_lvl * grad_sliced_val_cur[0];
    // lattice_values_monolithic_grad[level][idx_val][1] +=
    // dL_dbarycentric[remainder]* w_lvl * grad_sliced_val_cur[1];
    // atomicAdd(&lattice_values_monolithic_grad[level][idx_val][0],
    // dL_dbarycentric[remainder]* w_lvl * grad_sliced_val_cur[0]  );
    // atomicAdd(&lattice_values_monolithic_grad[level][idx_val][1],
    // dL_dbarycentric[remainder]* w_lvl * grad_sliced_val_cur[1]  );

    // push gradient into grad_sliced_val_cur
    // grad_sliced_val_cur add towards all the barycentric coord, so in the
    // backward pass the gradient from b0 to all the grad_sliced_val_cur
    grad_grad_sliced_val_cur[0] +=
        dL_dbarycentric[remainder] * w_lvl * val_lattice_vertex.x;
    grad_grad_sliced_val_cur[1] +=
        dL_dbarycentric[remainder] * w_lvl * val_lattice_vertex.y;
  }
  // finish the accumulation of grad_grad_sliced
  // atomicAdd(&grad_grad_sliced_values_monolithic[level][0][idx],
  // grad_grad_sliced_val_cur[0]  );
  // atomicAdd(&grad_grad_sliced_values_monolithic[level][1][idx],
  // grad_grad_sliced_val_cur[1]  );
  grad_grad_sliced_values_monolithic[level][0][idx] =
      grad_grad_sliced_val_cur[0];
  grad_grad_sliced_values_monolithic[level][1][idx] =
      grad_grad_sliced_val_cur[1];
}
