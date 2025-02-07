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
Encoding<POS_DIM, NR_FEAT_PER_LEVEL>::Encoding(
    const EncodingFixedParams& fixed_params)
    : m_fixed_params(fixed_params) {
  // this->copy_to_constant_mem(fixed_params);
}

template <uint32_t POS_DIM, uint32_t NR_FEAT_PER_LEVEL>
Encoding<POS_DIM, NR_FEAT_PER_LEVEL>::~Encoding() {}

template <uint32_t POS_DIM, uint32_t NR_FEAT_PER_LEVEL>
void Encoding<POS_DIM, NR_FEAT_PER_LEVEL>::check_positions(
    const torch::Tensor& positions_raw) {
  CHECK(positions_raw.is_cuda())
      << "positions should be in GPU memory. Please call .cuda() on the tensor";
  CHECK(positions_raw.dim() == 3)
      << "positions should have shape (num_batch, num_points, "
         "dim_points), however it has shape"
      << positions_raw.sizes();
  CHECK(positions_raw.size(0) != 0) << "Why do we have batch size 0";
  CHECK(positions_raw.size(1) != 0) << "Why do we have 0 points";
  CHECK(positions_raw.size(2) != 0)
      << "Why do we have dimension 0 for the points";
  CHECK(positions_raw.size(2) == m_fixed_params.m_pos_dim)
      << "Pos dim for the lattice doesn't correspond with the position of the "
         "points. Lattice was initialized with "
      << m_fixed_params.m_pos_dim << " and points have pos dim "
      << positions_raw.size(2);
}

template <uint32_t POS_DIM, uint32_t NR_FEAT_PER_LEVEL>
torch::Tensor Encoding<POS_DIM, NR_FEAT_PER_LEVEL>::forward(
    const EncodingInput& input) {
  check_positions(input.m_positions_raw);
  const int batch_size_pos = input.m_positions_raw.size(0);
  const int nr_positions = input.m_positions_raw.size(1);
  const int pos_dim = input.m_positions_raw.size(2);

  const int batch_size_lattice = input.m_lattice_values.size(0);
  int nr_resolutions = input.m_lattice_values.size(1);
  const int lattice_capacity = input.m_lattice_values.size(2);
  const int val_dim = input.m_lattice_values.size(3);

  CHECK(batch_size_pos == batch_size_lattice)
      << "Batch size of positions and lattice must be the same.";
  CHECK(m_fixed_params.m_random_shift_per_level.size(0) == nr_resolutions)
      << "Random shift should have the first dimension the same as the nr of "
         "resolutions";
  CHECK(m_fixed_params.m_random_shift_per_level.size(1) == pos_dim)
      << "Random shift should have the second dimension the same as the pos "
         "dim";
  // check the anneal window
  CHECK(input.m_anneal_window.size(0) == nr_resolutions)
      << "anneal_window should have the first dimension the same as the nr of "
         "resolutions";

  // if we concat also the points, we add a series of extra resolutions to
  // contain those points
  int nr_resolutions_extra = 0;
  if (m_fixed_params.m_concat_points) {
    nr_resolutions_extra = std::ceil(float(pos_dim) / val_dim);
  }

  // initialize the output values
  Tensor sliced_values_hom_tensor =
      torch::empty({batch_size_pos, nr_resolutions + nr_resolutions_extra,
                    val_dim, nr_positions},
                   torch::dtype(input.m_lattice_values.scalar_type())
                       .device(torch::kCUDA, 0));

  // try again with a monolithic kernel
  const dim3 blocks = {
      (unsigned int)div_round_up(nr_positions, BLOCK_SIZE),
      (unsigned int)(nr_resolutions + nr_resolutions_extra),
      (unsigned int)batch_size_pos};  // the blocks are executed in order, first
                                      // the blocks for the first resolution,
                                      // then the second and so on

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.m_lattice_values.scalar_type(), "forward_gpu", ([&] {
        forward_gpu<POS_DIM, NR_FEAT_PER_LEVEL, scalar_t>
            <<<blocks, BLOCK_SIZE, 0, at::cuda::getCurrentCUDAStream()>>>(
                nr_positions, lattice_capacity, nr_resolutions,
                nr_resolutions_extra,
                input.m_positions_raw
                    .packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                input.m_lattice_values
                    .packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                m_fixed_params.m_scale_factor
                    .packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                m_fixed_params.m_random_shift_per_level
                    .packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                input.m_anneal_window
                    .packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                sliced_values_hom_tensor
                    .packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                m_fixed_params.m_concat_points, m_fixed_params.m_points_scaling,
                input.m_require_lattice_values_grad,
                input.m_require_positions_grad);
      }));

  return sliced_values_hom_tensor;
}

template <uint32_t POS_DIM, uint32_t NR_FEAT_PER_LEVEL>
std::tuple<torch::Tensor, torch::Tensor>
Encoding<POS_DIM, NR_FEAT_PER_LEVEL>::backward(
    const EncodingInput& input, torch::Tensor& grad_sliced_values_monolithic) {
  check_positions(input.m_positions_raw);
  const int batch_size_pos = input.m_positions_raw.size(0);
  const int nr_positions = input.m_positions_raw.size(1);
  const int pos_dim = input.m_positions_raw.size(2);

  const int lattice_capacity = input.m_lattice_values.size(2);
  CHECK(grad_sliced_values_monolithic.dim() == 4)
      << "grad_sliced_values_monolithic should be batch_size x nr_resolutions "
         "x val_dim x nr_positions, so it should have 3 dimensions. However it "
         "has "
      << grad_sliced_values_monolithic.dim();
  CHECK(grad_sliced_values_monolithic.is_contiguous())
      << "Grad sliced values needs to be contiguous. Please call .contiguous() "
         "on it";

  int nr_resolutions = grad_sliced_values_monolithic.size(1);
  const int val_dim = grad_sliced_values_monolithic.size(2);
  CHECK(nr_positions == grad_sliced_values_monolithic.size(3))
      << "The nr of positions should match between the input positions and the "
         "sliced values";
  CHECK(input.m_lattice_values.dim() == 4)
      << "grad_sliced_values_monolithic should be batch_size x nr_resolutions "
         "x val_dim x nr_positions, so it should have 3 dimensions. However it "
         "has "
      << input.m_lattice_values.dim();
  CHECK(input.m_lattice_values.is_contiguous())
      << "We assume that the lattice_values_monolithic are contiguous because "
         "in the cuda code we make a load of 2 float values at a time and that "
         "assumes that they are contiguous";

  // if we concat also the points, we add a series of extra resolutions to
  // contain those points
  int nr_resolutions_extra = 0;
  if (m_fixed_params.m_concat_points) {
    nr_resolutions_extra = std::ceil(float(pos_dim) / val_dim);
    nr_resolutions = nr_resolutions - nr_resolutions_extra;
  }

  Tensor lattice_values_monolithic_grad;  // dL/dLatticeValues
  if (input.m_require_lattice_values_grad) {
    lattice_values_monolithic_grad =
        torch::zeros({batch_size_pos, nr_resolutions, val_dim, lattice_capacity},
                     torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
  } else {
    lattice_values_monolithic_grad = torch::empty(
        {1, 1, 1, 1}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
  }

  Tensor positions_grad;  // dL/dPos
  if (input.m_require_positions_grad) {
    positions_grad =
        torch::zeros({batch_size_pos, pos_dim, nr_positions},
                     torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
  } else {
    positions_grad = torch::empty(
        {1, 1, 1}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));
  }

  const dim3 blocks = {
      (unsigned int)div_round_up(nr_positions, BLOCK_SIZE_BACK),
      (unsigned int)nr_resolutions,
      (unsigned int)batch_size_pos};  // the blocks are executed in order, first the blocks for the first
           // resolution, then the second and so on

  if (input.m_require_lattice_values_grad) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.m_lattice_values.scalar_type(), "backward_gpu", ([&] {
          backward_gpu<POS_DIM, NR_FEAT_PER_LEVEL, scalar_t><<<
              blocks, BLOCK_SIZE_BACK, 0, at::cuda::getCurrentCUDAStream()>>>(
              nr_positions, lattice_capacity,
              input.m_lattice_values
                  .packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
              input.m_positions_raw
                  .packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
              m_fixed_params.m_scale_factor
                  .packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
              m_fixed_params.m_random_shift_per_level
                  .packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
              input.m_anneal_window
                  .packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
              grad_sliced_values_monolithic
                  .packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
              lattice_values_monolithic_grad
                  .packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
              m_fixed_params.m_concat_points);
        }));
  }

  if (input.m_require_positions_grad) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.m_lattice_values.scalar_type(), "backward_gpu_only_pos", ([&] {
          backward_gpu_only_pos<POS_DIM, NR_FEAT_PER_LEVEL><<<
              blocks, BLOCK_SIZE_BACK, 0, at::cuda::getCurrentCUDAStream()>>>(
              nr_positions, lattice_capacity,
              input.m_lattice_values
                  .packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
              input.m_positions_raw
                  .packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
              m_fixed_params.m_scale_factor
                  .packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
              m_fixed_params.m_random_shift_per_level
                  .packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
              input.m_anneal_window
                  .packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
              grad_sliced_values_monolithic
                  .packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
              positions_grad
                  .packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
              m_fixed_params.m_concat_points,
              input.m_require_lattice_values_grad,
              input.m_require_positions_grad);
        }));
  }

  lattice_values_monolithic_grad =
      lattice_values_monolithic_grad.permute({0, 1, 3, 2});
  positions_grad = positions_grad.permute({0, 2, 1});

  return std::make_tuple(lattice_values_monolithic_grad, positions_grad);
}

template <uint32_t POS_DIM, uint32_t NR_FEAT_PER_LEVEL>
std::tuple<torch::Tensor, torch::Tensor>
Encoding<POS_DIM, NR_FEAT_PER_LEVEL>::double_backward_from_positions(
    const EncodingInput& input, const torch::Tensor& double_positions_grad,
    torch::Tensor& grad_sliced_values_monolithic) {
  check_positions(input.m_positions_raw);
  int nr_positions = input.m_positions_raw.size(0);
  int pos_dim = input.m_positions_raw.size(1);
  int capacity = input.m_lattice_values.size(1);
  CHECK(grad_sliced_values_monolithic.dim() == 3)
      << "grad_sliced_values_monolithic should be nr_resolutions x val_dim x "
         "nr_positions, so it should have 3 dimensions. However it has "
      << grad_sliced_values_monolithic.dim();
  CHECK(grad_sliced_values_monolithic.is_contiguous())
      << "Grad sliced values needs to be contiguous. Please call .contiguous() "
         "on it";
  int nr_resolutions = grad_sliced_values_monolithic.size(0);
  int val_dim = grad_sliced_values_monolithic.size(1);
  CHECK(nr_positions == grad_sliced_values_monolithic.size(2))
      << "The nr of positions should match between the input positions and the "
         "sliced values";
  CHECK(input.m_lattice_values.dim() == 3)
      << "grad_sliced_values_monolithic should be nr_resolutions x val_dim x "
         "nr_positions, so it should have 3 dimensions. However it has "
      << input.m_lattice_values.dim();
  CHECK(input.m_lattice_values.is_contiguous())
      << "We assume that the lattice_values_monolithic are contiguous because "
         "in the cuda code we make a load of 2 float values at a time and that "
         "assumes that they are contiguous";

  // if we concat also the points, we add a series of extra resolutions to
  // contain those points
  int nr_resolutions_extra = 0;
  if (m_fixed_params.m_concat_points) {
    nr_resolutions_extra = std::ceil(float(pos_dim) / val_dim);
    nr_resolutions = nr_resolutions - nr_resolutions_extra;
  }

  // nr_resolutions x nr_lattice_vertices x nr_lattice_featues
  // dL/dLattiveValues
  // Tensor lattice_values_monolithic_grad=torch::zeros({ nr_resolutions,
  // capacity, val_dim },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)
  // );
  Tensor lattice_values_monolithic_grad =
      torch::zeros({nr_resolutions, val_dim, capacity},
                   torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));

  Tensor grad_grad_sliced_values_monolithic = torch::empty(
      {nr_resolutions + nr_resolutions_extra, val_dim, nr_positions},
      torch::dtype(torch::kFloat32).device(torch::kCUDA, 0));

  // writes gradient to lattice_values_monolithic_grad, assumes the
  // lattice_values_monolithic_grad is tranposed so we have to transpose back
  // afterwards
  dim3 blocks = {
      (unsigned int)div_round_up(nr_positions, BLOCK_SIZE_DOUBLE_BACK),
      (unsigned int)nr_resolutions,
      1};  // the blocks are executed in order, first the blocks for the first
           // resolution, then the second and so on
  double_backward_from_positions_gpu_1<POS_DIM, NR_FEAT_PER_LEVEL>
      <<<blocks, BLOCK_SIZE_DOUBLE_BACK, 0, at::cuda::getCurrentCUDAStream()>>>(
          nr_positions, capacity, nr_resolutions,
          double_positions_grad
              .packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
          input.m_lattice_values
              .packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
          input.m_positions_raw
              .packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
          m_fixed_params.m_scale_factor
              .packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
          m_fixed_params.m_random_shift_per_level
              .packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
          input.m_anneal_window
              .packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
          grad_sliced_values_monolithic
              .packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
          m_fixed_params.m_concat_points,
          // output
          grad_grad_sliced_values_monolithic
              .packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
          lattice_values_monolithic_grad
              .packed_accessor32<float, 3, torch::RestrictPtrTraits>());

  // writes gradient to grad_grad_sliced_values_monolithic
  // the last few resolutions might be extra resolutions so we just write zero
  // grad there
  blocks = {(unsigned int)div_round_up(nr_positions, BLOCK_SIZE_DOUBLE_BACK),
            (unsigned int)(nr_resolutions + nr_resolutions_extra),
            1};  // the blocks are executed in order, first the blocks for the
                 // first resolution, then the second and so on
  double_backward_from_positions_gpu_2<POS_DIM, NR_FEAT_PER_LEVEL>
      <<<blocks, BLOCK_SIZE_DOUBLE_BACK, 0, at::cuda::getCurrentCUDAStream()>>>(
          nr_positions, capacity, nr_resolutions,
          double_positions_grad
              .packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
          input.m_lattice_values
              .packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
          input.m_positions_raw
              .packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
          m_fixed_params.m_scale_factor
              .packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
          m_fixed_params.m_random_shift_per_level
              .packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
          input.m_anneal_window
              .packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
          grad_sliced_values_monolithic
              .packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
          m_fixed_params.m_concat_points,
          // output
          grad_grad_sliced_values_monolithic
              .packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
          lattice_values_monolithic_grad
              .packed_accessor32<float, 3, torch::RestrictPtrTraits>());

  lattice_values_monolithic_grad =
      lattice_values_monolithic_grad.permute({0, 2, 1});

  return std::make_tuple(lattice_values_monolithic_grad,
                         grad_grad_sliced_values_monolithic);
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
