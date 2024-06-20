#include <chrono>
#include <string>

#include "Encoding.cuh"
#include "EncodingGPU.cuh"

using torch::Tensor;

template <typename T>
T div_round_up(T val, T divisor) {
  return (val + divisor - 1) / divisor;
}

template <uint32_t POS_DIM, uint32_t NR_FEAT_PER_LEVEL>
Encoding<POS_DIM, NR_FEAT_PER_LEVEL>::Encoding(const EncodingFixedParams& fixed_params)
    : m_fixed_params(fixed_params) {}

template <uint32_t POS_DIM, uint32_t NR_FEAT_PER_LEVEL>
Encoding<POS_DIM, NR_FEAT_PER_LEVEL>::~Encoding() {}

template <uint32_t POS_DIM, uint32_t NR_FEAT_PER_LEVEL>
void Encoding<POS_DIM, NR_FEAT_PER_LEVEL>::check_positions(const torch::Tensor& positions_raw) {
  CHECK(positions_raw.is_cuda()) << "positions should be in GPU memory. Please call .cuda() on the tensor";
  CHECK(positions_raw.dim() == 3) << "positions should have shape (num_batch, num_points, "
                                     "dim_points), however it has shape"
                                  << positions_raw.sizes();
  CHECK(positions_raw.size(0) != 0) << "Why do we have batch size 0";
  CHECK(positions_raw.size(1) != 0) << "Why do we have 0 points";
  CHECK(positions_raw.size(2) != 0) << "Why do we have dimension 0 for the points";
  CHECK(positions_raw.size(2) == m_fixed_params.m_pos_dim)
      << "Pos dim for the lattice doesn't correspond with the position of the "
         "points. Lattice was initialized with "
      << m_fixed_params.m_pos_dim << " and points have pos dim " << positions_raw.size(2);
}

template <uint32_t POS_DIM, uint32_t NR_FEAT_PER_LEVEL>
torch::Tensor Encoding<POS_DIM, NR_FEAT_PER_LEVEL>::forward(const EncodingInput& input) {
  check_positions(input.m_positions_raw);
  const int batch_size_pos = input.m_positions_raw.size(0);
  const int nr_positions = input.m_positions_raw.size(1);
  const int pos_dim = input.m_positions_raw.size(2);

  const int batch_size_features = input.m_features.size(0);
  int nr_resolutions = input.m_features.size(1);
  const int lattice_capacity = input.m_features.size(2);
  const int val_dim = input.m_features.size(3);

  CHECK(batch_size_pos == batch_size_features) << "Batch size of positions and lattice must be the same.";
  CHECK(m_fixed_params.m_random_shift_per_level.size(0) == nr_resolutions)
      << "Random shift should have the first dimension the same as the nr of resolutions";
  CHECK(m_fixed_params.m_random_shift_per_level.size(1) == pos_dim)
      << "Random shift should have the second dimension the same as the pos dim";
  // check the anneal window
  CHECK(input.m_anneal_window.size(0) == nr_resolutions)
      << "anneal_window should have the first dimension the same as the nr of resolutions";

  // if we concat also the points, we add a series of extra resolutions to contain those points
  int nr_resolutions_extra = 0;
  if (m_fixed_params.m_concat_points) {
    nr_resolutions_extra = std::ceil(float(pos_dim) / val_dim);
  }

  // initialize the output values
  Tensor sliced_values_hom_tensor = torch::empty(
      {batch_size_pos, nr_resolutions + nr_resolutions_extra, val_dim, nr_positions},
      torch::dtype(input.m_features.scalar_type()).device(torch::kCUDA, 0)
  );

  const dim3 blocks = {
      (unsigned int)div_round_up(nr_positions, BLOCK_SIZE),
      (unsigned int)(nr_resolutions + nr_resolutions_extra), (unsigned int)batch_size_pos};

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.m_features.scalar_type(), "forward_gpu", ([&] {
        forward_gpu<POS_DIM, NR_FEAT_PER_LEVEL, scalar_t>
            <<<blocks, BLOCK_SIZE, 0, at::cuda::getCurrentCUDAStream()>>>(
                nr_positions, lattice_capacity, nr_resolutions, nr_resolutions_extra,
                input.m_positions_raw.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                input.m_features.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                m_fixed_params.m_scale_factor.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                m_fixed_params.m_random_shift_per_level
                    .packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                input.m_anneal_window.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                sliced_values_hom_tensor.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                m_fixed_params.m_concat_points, m_fixed_params.m_points_scaling,
                input.m_require_features_grad, input.m_require_positions_grad
            );
      })
  );

  return sliced_values_hom_tensor;
}

template <uint32_t POS_DIM, uint32_t NR_FEAT_PER_LEVEL>
std::tuple<torch::Tensor, torch::Tensor> Encoding<POS_DIM, NR_FEAT_PER_LEVEL>::backward(
    const EncodingInput& input, torch::Tensor& grad_outs
) {
  check_positions(input.m_positions_raw);
  const int batch_size_pos = input.m_positions_raw.size(0);
  const int nr_positions = input.m_positions_raw.size(1);
  const int pos_dim = input.m_positions_raw.size(2);

  const int lattice_capacity = input.m_features.size(2);
  CHECK(
      grad_outs.dim() == 4
  ) << "grad_outs should be batch_size x nr_resolutions x val_dim x nr_positions, so it "
       "should have 4 dimensions. However it has "
    << grad_outs.dim();
  CHECK(grad_outs.is_contiguous()
  ) << "Grad sliced values needs to be contiguous. Please call .contiguous() on it";

  int nr_resolutions = grad_outs.size(1);
  const int val_dim = grad_outs.size(2);
  CHECK(nr_positions == grad_outs.size(3))
      << "The nr of positions should match between the input positions and the sliced values";
  CHECK(
      input.m_features.dim() == 4
  ) << "grad_outs should be batch_size x nr_resolutions x val_dim x nr_positions, so it "
       "should have 3 dimensions. However it has "
    << input.m_features.dim();
  CHECK(input.m_features.is_contiguous())
      << "We assume that the features are contiguous because in the cuda code we make a load of 2 float "
         "values at a time and that assumes that they are contiguous";

  // if we concat also the points, we add a series of extra resolutions to
  // contain those points
  int nr_resolutions_extra = 0;
  if (m_fixed_params.m_concat_points) {
    nr_resolutions_extra = std::ceil(float(pos_dim) / val_dim);
    nr_resolutions = nr_resolutions - nr_resolutions_extra;
  }

  auto tensor_options = torch::dtype(input.m_features.scalar_type()).device(torch::kCUDA, 0);

  Tensor grad_features;  // dL/dLatticeValues
  grad_features = torch::zeros({batch_size_pos, nr_resolutions, lattice_capacity, val_dim}, tensor_options);

  Tensor grad_positions;  // dL/dPos
  grad_positions = torch::zeros({batch_size_pos, nr_positions, pos_dim}, tensor_options);

  const dim3 blocks = {
      (unsigned int)div_round_up(nr_positions, BLOCK_SIZE_BACK), (unsigned int)nr_resolutions,
      (unsigned int)batch_size_pos};

  if (input.m_require_features_grad) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.m_features.scalar_type(), "backward_gpu", ([&] {
          backward_gpu<POS_DIM, NR_FEAT_PER_LEVEL, scalar_t>
              <<<blocks, BLOCK_SIZE_BACK, 0, at::cuda::getCurrentCUDAStream()>>>(
                  nr_positions, lattice_capacity,
                  input.m_features.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                  input.m_positions_raw.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                  m_fixed_params.m_scale_factor.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                  m_fixed_params.m_random_shift_per_level
                      .packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                  input.m_anneal_window.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                  grad_outs.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                  grad_features.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                  m_fixed_params.m_concat_points
              );
        })
    );
  }

  if (input.m_require_positions_grad) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.m_features.scalar_type(), "backward_gpu_only_pos", ([&] {
          backward_gpu_only_pos<POS_DIM, NR_FEAT_PER_LEVEL>
              <<<blocks, BLOCK_SIZE_BACK, 0, at::cuda::getCurrentCUDAStream()>>>(
                  nr_positions, lattice_capacity,
                  input.m_features.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                  input.m_positions_raw.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                  m_fixed_params.m_scale_factor.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                  m_fixed_params.m_random_shift_per_level
                      .packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                  input.m_anneal_window.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                  grad_outs.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                  grad_positions.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                  m_fixed_params.m_concat_points, input.m_require_features_grad,
                  input.m_require_positions_grad
              );
        })
    );
  }

  return std::make_tuple(grad_features, grad_positions);
}

template <uint32_t POS_DIM, uint32_t NR_FEAT_PER_LEVEL>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Encoding<POS_DIM, NR_FEAT_PER_LEVEL>::double_backward(
    const EncodingInput& input, const torch::Tensor& grad_grad_positions,
    const torch::Tensor& grad_grad_features, torch::Tensor& grad_outs
) {
  check_positions(input.m_positions_raw);
  const int batch_size_pos = input.m_positions_raw.size(0);
  const int nr_positions = input.m_positions_raw.size(1);
  const int pos_dim = input.m_positions_raw.size(2);

  const int batch_size_features = input.m_features.size(0);
  int nr_resolutions = input.m_features.size(1);
  const int lattice_capacity = input.m_features.size(2);
  const int val_dim = input.m_features.size(3);

  CHECK(
      grad_outs.dim() == 4
  ) << "grad_outs should be batch_size x nr_resolutions x val_dim x nr_positions, so it should "
       "have 4 dimensions. However it has "
    << grad_outs.dim();
  CHECK(grad_outs.is_contiguous()
  ) << "Grad sliced values needs to be contiguous. Please call .contiguous() on it";
  CHECK(nr_positions == grad_outs.size(3))
      << "The nr of positions should match between the number of encoded positions";
  CHECK(input.m_features.dim() == 4) << "grad_outs should be batch_size x nr_resolutions x val_dim x "
                                        "nr_positions, so it should have 4 dimensions. However it has "
                                     << input.m_features.dim();
  CHECK(input.m_features.is_contiguous())
      << "We assume that the features are contiguous because in the cuda code we make a load of 2 float "
         "values at a time and that assumes that they are contiguous";

  // if we concat also the points, we add a series of extra resolutions to contain those points
  int nr_resolutions_extra = 0;
  if (m_fixed_params.m_concat_points) {
    nr_resolutions_extra = std::ceil(float(pos_dim) / val_dim);
    nr_resolutions = nr_resolutions - nr_resolutions_extra;
  }

  auto tensor_options = torch::dtype(input.m_features.scalar_type()).device(torch::kCUDA, 0);

  Tensor grad_features =
      torch::zeros({batch_size_features, nr_resolutions, lattice_capacity, val_dim}, tensor_options);
  Tensor grad_positions = torch::zeros({batch_size_pos, nr_positions, pos_dim}, tensor_options);
  Tensor grad_grad_outs = torch::zeros(
      {batch_size_pos, nr_resolutions + nr_resolutions_extra, val_dim, nr_positions}, tensor_options
  );

  const dim3 blocks = {
      (unsigned int)div_round_up(nr_positions, BLOCK_SIZE_DOUBLE_BACK), (unsigned int)nr_resolutions,
      (unsigned int)batch_size_pos};

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.m_features.scalar_type(), "double_backward_gpu", ([&] {
        double_backward_gpu<POS_DIM, NR_FEAT_PER_LEVEL, scalar_t>
            <<<blocks, BLOCK_SIZE_DOUBLE_BACK, 0, at::cuda::getCurrentCUDAStream()>>>(
                nr_positions, lattice_capacity, nr_resolutions,
                grad_grad_positions.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                grad_grad_features.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                input.m_features.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                input.m_positions_raw.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                m_fixed_params.m_scale_factor.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                m_fixed_params.m_random_shift_per_level
                    .packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                input.m_anneal_window.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                grad_outs.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                m_fixed_params.m_concat_points,
                // output
                grad_grad_outs.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                grad_features.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                grad_positions.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>()
            );
      })
  );

  return std::make_tuple(grad_features, grad_positions, grad_grad_outs);
}

// explicit instantiation
// https://stackoverflow.com/a/495056
// https://isocpp.org/wiki/faq/templates#separate-template-class-defn-from-decl
// for val 2
template class Encoding<2, 2>;
template class Encoding<3, 2>;
template class Encoding<4, 2>;
template class Encoding<5, 2>;
template class Encoding<6, 2>;
template class Encoding<7, 2>;
// for val 4
// TODO not implemented other values other than 2 because we assume we load only
// 2 floats in the CUDA kernels
