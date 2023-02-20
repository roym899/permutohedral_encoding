#include "permutohedral_encoding/Encoding.cuh"

//c++
#include <string>


//my stuff
#include "permutohedral_encoding/EncodingGPU.cuh"


using torch::Tensor;


template <typename T>
T div_round_up(T val, T divisor) {
	return (val + divisor - 1) / divisor;
}




// Encoding::Encoding(const int pos_dim, const int capacity, const int nr_levels, const int nr_feat_per_level):
//     m_expected_pos_dim(pos_dim),
//     m_capacity(capacity),
//     m_nr_levels(nr_levels),
//     m_nr_feat_per_level(nr_feat_per_level)
//     {

//     // init_params(config_file);
//     // VLOG(3) << "Creating lattice";

// }

template<uint32_t POS_DIM, uint32_t NR_FEAT_PER_LEVEL>
Encoding<POS_DIM, NR_FEAT_PER_LEVEL>::Encoding(const EncodingFixedParams& fixed_params):
    m_fixed_params(fixed_params)
    {


}


template<uint32_t POS_DIM, uint32_t NR_FEAT_PER_LEVEL>
Encoding<POS_DIM, NR_FEAT_PER_LEVEL>::~Encoding(){
    // LOG(WARNING) << "Deleting lattice: " << m_name;
}

// template<uint32_t POS_DIM, uint32_t NR_FEAT_PER_LEVEL>
// void Encoding<POS_DIM, NR_FEAT_PER_LEVEL>::test(const torch::Tensor& tensor){

// }

template<uint32_t POS_DIM, uint32_t NR_FEAT_PER_LEVEL>
void Encoding<POS_DIM, NR_FEAT_PER_LEVEL>::check_positions(const torch::Tensor& positions_raw){
    CHECK(positions_raw.is_cuda()) << "positions should be in GPU memory. Please call .cuda() on the tensor";
    CHECK(positions_raw.scalar_type()==at::kFloat) << "positions should be of type float";
    CHECK(positions_raw.dim()==2) << "positions should have dim 2 correspondin to HW. However it has sizes" << positions_raw.sizes();
    int pos_dim=positions_raw.size(1);
    // CHECK(m_sigmas.size()==pos_dim) <<"One must set sigmas for each dimension of the positions. Use set_sigmas. m_sigmas is " << m_sigmas.size() << " m_pos dim is " <<pos_dim;
    // CHECK(m_expected_pos_dim==positions_raw.size(1)) << "The expected pos dim is " << m_expected_pos_dim << " whole the input points have pos_dim " << positions_raw.size(1);
    CHECK(positions_raw.size(0)!=0) << "Why do we have 0 points";
    CHECK(positions_raw.size(1)!=0) << "Why do we have dimension 0 for the points";
    // CHECK(positions_raw.is_contiguous()) << "Positions raw is not contiguous. Please call .contiguous() on it";
    // CHECK(pos_dim==m_expected_position_dimensions) << "The pos dim should be the same as the expected positions dimensions given by the sigmas. Pos dim is " << pos_dim << " m_expected_position_dimensions " << m_expected_position_dimensions;
}
template<uint32_t POS_DIM, uint32_t NR_FEAT_PER_LEVEL>
void Encoding<POS_DIM, NR_FEAT_PER_LEVEL>::check_values(const torch::Tensor& values){
    CHECK(values.is_cuda()) << "lattice values should be in GPU memory. Please call .cuda() on the tensor";
    CHECK(values.scalar_type()==at::kFloat) << "values should be of type float";
    CHECK(values.dim()==2) << "values should have dim 2 correspondin to HW. However it has sizes" << values.sizes();
    CHECK(values.is_contiguous()) << "Values is not contiguous. Please call .contiguous() on it";
}
template<uint32_t POS_DIM, uint32_t NR_FEAT_PER_LEVEL>
void Encoding<POS_DIM, NR_FEAT_PER_LEVEL>::check_positions_and_values(const torch::Tensor& positions_raw, const torch::Tensor& values){
    //check input
    CHECK(positions_raw.size(0)==values.size(0)) << "Sizes of positions and values should match. Meaning that that there should be a value for each position. Positions_raw has sizes "<<positions_raw.sizes() << " and the values has size " << values.sizes();
    check_positions(positions_raw);
    check_values(positions_raw);
}







template<uint32_t POS_DIM, uint32_t NR_FEAT_PER_LEVEL>
// std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Encoding<POS_DIM, NR_FEAT_PER_LEVEL>::forward(const torch::Tensor& lattice_values, torch::Tensor& positions_raw, torch::Tensor& anneal_window,const bool require_lattice_values_grad, const bool require_positions_grad){
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Encoding<POS_DIM, NR_FEAT_PER_LEVEL>::forward(const EncodingInput& input){

    check_positions(input.m_positions_raw); 
    int nr_positions=input.m_positions_raw.size(0);
    int pos_dim=input.m_positions_raw.size(1);
    int nr_resolutions=input.m_lattice_values.size(0);
    int lattice_capacity=input.m_lattice_values.size(1);
    int val_dim=input.m_lattice_values.size(2);
    CHECK(m_fixed_params.m_random_shift_per_level.size(0)==nr_resolutions ) <<"Random shift should have the first dimension the same as the nr of resolutions";
    CHECK(m_fixed_params.m_random_shift_per_level.size(1)==pos_dim ) <<"Random shift should have the second dimension the same as the pos dim";
    //check the anneal window
    CHECK(input.m_anneal_window.size(0)==nr_resolutions ) <<"anneal_window should have the first dimension the same as the nr of resolutions";

     //to cuda
    // positions_raw=positions_raw.to("cuda");
    // anneal_window=anneal_window.to("cuda");

    Tensor positions=input.m_positions_raw; //the sigma scaling is done inside the kernel
    

    //if we concat also the points, we add a series of extra resolutions to contain those points
    int nr_resolutions_extra=0;
    if (m_fixed_params.m_concat_points){
        nr_resolutions_extra=std::ceil(float(pos_dim)/val_dim);
    }
   



    //initialize the output values 
    Tensor sliced_values_hom_tensor=torch::empty({nr_resolutions+nr_resolutions_extra, val_dim, nr_positions }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );

    
    Tensor splatting_indices_tensor;
    Tensor splatting_weights_tensor;
    if (input.m_require_lattice_values_grad || input.m_require_positions_grad){
        splatting_indices_tensor = torch::empty({ nr_resolutions, nr_positions, (pos_dim+1) }, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) );
        #if LATTICE_HALF_PRECISION
            splatting_weights_tensor = torch::empty({ nr_resolutions, nr_positions, (pos_dim+1) }, torch::dtype(torch::kFloat16).device(torch::kCUDA, 0) );
        #else 
            splatting_weights_tensor = torch::empty({ nr_resolutions, nr_positions, (pos_dim+1) }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
        #endif
        splatting_indices_tensor.fill_(-1);
        // splatting_weights_tensor.fill_(0);
    }else{
        splatting_indices_tensor = torch::empty({ 1,1,1 }, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) );
        #if LATTICE_HALF_PRECISION
            splatting_weights_tensor = torch::empty({ 1,1,1 }, torch::dtype(torch::kFloat16).device(torch::kCUDA, 0) );
        #else
            splatting_weights_tensor = torch::empty({ 1,1,1 }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
        #endif
    }



    //make scalefactor for each lvl
    // Tensor scale_factor_tensor=m_scale_factor;



    //try again with a monolithic kernel
    const dim3 blocks = { (unsigned int)div_round_up(nr_positions, BLOCK_SIZE), (unsigned int)(nr_resolutions+nr_resolutions_extra), 1 }; //the blocks are executed in order, first the blocks for the first resolution, then the second and so on
    

    //debug by wirting also the elevated ones
    // if (pos_dim==3){
    //     if(val_dim==2){
            slice_with_collisions_no_precomputation_fast_mr_monolithic<POS_DIM, NR_FEAT_PER_LEVEL><<<blocks, BLOCK_SIZE>>>(
                nr_positions, 
                lattice_capacity,
                nr_resolutions,
                nr_resolutions_extra,
                positions.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                #if LATTICE_HALF_PRECISION
                    input.m_lattice_values.packed_accessor32<at::Half,3,torch::RestrictPtrTraits>(),
                #else 
                    input.m_lattice_values.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                #endif
                m_fixed_params.m_scale_factor.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                m_fixed_params.m_random_shift_per_level.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                input.m_anneal_window.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
                // #if LATTICE_HALF_PRECISION
                    // sliced_values_hom_tensor.packed_accessor32<at::Half,3,torch::RestrictPtrTraits>(),
                // #else
                    sliced_values_hom_tensor.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                // #endif
                // elevated.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                splatting_indices_tensor.packed_accessor32<int,3,torch::RestrictPtrTraits>(),   
                #if LATTICE_HALF_PRECISION
                    splatting_weights_tensor.packed_accessor32<at::Half,3,torch::RestrictPtrTraits>(),   
                #else 
                    splatting_weights_tensor.packed_accessor32<float,3,torch::RestrictPtrTraits>(),   
                #endif
                m_fixed_params.m_concat_points,
                m_fixed_params.m_points_scaling,
                input.m_require_lattice_values_grad,
                input.m_require_positions_grad
            );
    //     }else{
    //         LOG(FATAL) <<"Instantiation for this val_dim doesnt exist. Please add it to the source code here.";
    //     }
    // }else{
    //     LOG(FATAL) <<"Instantiation for this PosDIM doesnt exist. Please add it to the source code here.";
    // }


    auto ret = std::make_tuple (sliced_values_hom_tensor, splatting_indices_tensor, splatting_weights_tensor ); 
    return ret;

}




// template<uint32_t POS_DIM, uint32_t NR_FEAT_PER_LEVEL>
// std::tuple<torch::Tensor, torch::Tensor> Encoding<POS_DIM, NR_FEAT_PER_LEVEL>::slice_backwards_standalone_no_precomputation_no_homogeneous_mr_monolithic(torch::Tensor& positions_raw,  torch::Tensor& lattice_values_monolithic, torch::Tensor& grad_sliced_values_monolithic, const torch::Tensor& scale_factor, torch::Tensor& random_shift_monolithic, torch::Tensor& anneal_window, const bool concat_points, const bool require_lattice_values_grad, const bool require_positions_grad){

//     check_positions(positions_raw); 
//     // CHECK(val_dim()>0) << "m_val_dim is 0 or lwoer. We have to splat something first so that we have values from where to slice. Val dim is " << val_dim();
//     int nr_positions=positions_raw.size(0);
//     int pos_dim=positions_raw.size(1);
//     // CHECK(pos_dim==this->pos_dim()) << " The position dimension do not coreespond with the ones we used for creating the lattice";
//     CHECK(grad_sliced_values_monolithic.dim()==3) <<"grad_sliced_values_monolithic should be nr_resolutions x val_dim x nr_positions, so it should have 3 dimensions. However it has "<< grad_sliced_values_monolithic.dim();
//     CHECK(grad_sliced_values_monolithic.is_contiguous()) << "Grad sliced values needs to be contiguous. Please call .contiguous() on it";
//     // splatting_indices_tensor=splatting_indices_tensor.contiguous();
//     // splatting_weights_tensor=splatting_weights_tensor.contiguous();
//     int nr_resolutions=grad_sliced_values_monolithic.size(0);
//     int val_dim=grad_sliced_values_monolithic.size(1);
//     CHECK(nr_positions==grad_sliced_values_monolithic.size(2)) << "The nr of positions should match between the input positions and the sliced values";
//     CHECK(lattice_values_monolithic.dim()==3) <<"grad_sliced_values_monolithic should be nr_resolutions x val_dim x nr_positions, so it should have 3 dimensions. However it has "<< lattice_values_monolithic.dim();
//     CHECK(lattice_values_monolithic.is_contiguous()) <<"We assume that the lattice_values_monolithic are contiguous because in the cuda code we make a load of 2 float values at a time and that assumes that they are contiguous";
    
    

//     //if we concat also the points, we add a series of extra resolutions to contain those points
//     int nr_resolutions_extra=0;
//     if (concat_points){
//         nr_resolutions_extra=std::ceil(float(pos_dim)/val_dim);
//         // VLOG(1) << "grad_sliced_values_monolithic" << grad_sliced_values_monolithic.sizes();
//         // VLOG(1) << "grad_sliced_values_monolithic" << grad_sliced_values_monolithic;
//         // grad_sliced_values_monolithic=grad_sliced_values_monolithic.slice(0, 0, nr_resolutions-nr_resolutions_extra); //dim, start, end
//         // VLOG(1) << "grad_sliced_values_monolithic after slicing" << grad_sliced_values_monolithic.sizes();
//         // VLOG(1) << "grad_sliced_values_monolithic after slicing" << grad_sliced_values_monolithic;
//         nr_resolutions=nr_resolutions-nr_resolutions_extra;
//     }

//     int capacity=lattice_values_monolithic.size(1);
//     // std::cout << "capacity is " << capacity << std::endl;

    

//     // nr_resolutions x nr_lattice_vertices x nr_lattice_featues
//     // Tensor lattice_values_monolithic=torch::zeros({ nr_resolutions, nr_lattice_vertices(), val_dim  },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
//     // Tensor lattice_values_monolithic=torch::zeros({ nr_resolutions, val_dim, m_capacity },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
//     // TIME_START("slice_b_create_output");
//     Tensor lattice_values_monolithic_grad; //dL/dLattiveValues
//     if (require_lattice_values_grad){
//         #if LATTICE_HALF_PRECISION
//             // Tensor lattice_values_monolithic=torch::zeros({ nr_resolutions, val_dim, m_capacity },  torch::dtype(torch::kFloat16).device(torch::kCUDA, 0)  );
//             lattice_values_monolithic_grad=torch::zeros({ nr_resolutions, capacity, val_dim },  torch::dtype(torch::kFloat16).device(torch::kCUDA, 0)  );
//         #else
//             // Tensor lattice_values_monolithic=torch::zeros({ nr_resolutions, val_dim, m_capacity },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
//             lattice_values_monolithic_grad=torch::zeros({ nr_resolutions, capacity, val_dim },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
//         #endif
//     }else{
//         lattice_values_monolithic_grad=torch::empty({ 1,1,1 },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
//     }
    


//     Tensor positions_grad; //dL/dPos
//     if (require_positions_grad){
//         positions_grad=torch::zeros({ nr_positions, pos_dim },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
//     }else{
//         positions_grad=torch::empty({ 1,1 },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
//     }

//     // TIME_END("slice_b_create_output");

//     //make scalefactor for each lvl
//     // Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> scale_factor_eigen;
//     // scale_factor_eigen.resize(nr_resolutions, pos_dim);
//     // double invStdDev = 1.0;
//     // for(int res_idx=0; res_idx<nr_resolutions; res_idx++){
//     //     for (int i = 0; i < pos_dim; i++) {
//     //         scale_factor_eigen(res_idx,i) =  1.0 / (std::sqrt((double) (i + 1) * (i + 2))) * invStdDev;
//     //         scale_factor_eigen(res_idx,i)=scale_factor_eigen(res_idx,i)/ sigmas_list[res_idx];
//     //         // VLOG(1) << "scalinbg by " << sigmas_list[res_idx];
//     //     }
//     // }
//     // // VLOG(1) << "scale_factor_eigen" << scale_factor_eigen;
//     // Tensor scale_factor_tensor=eigen2tensor(scale_factor_eigen.cast<float>()).cuda();
//     // scale_factor_tensor=scale_factor_tensor.view({nr_resolutions, pos_dim}); //nr_resolutuons x pos_dim

//     // Tensor scale_factor_tensor=Lattice::compute_scale_factor_tensor(sigmas_list, pos_dim);
//     Tensor scale_factor_tensor=scale_factor;


//     const dim3 blocks = { (unsigned int)div_round_up(nr_positions, BLOCK_SIZE_BACK), (unsigned int)nr_resolutions, 1 }; //the blocks are executed in order, first the blocks for the first resolution, then the second and so on

//     // TIME_START("slice_back_cuda");
//     if (pos_dim==2){
//         if(val_dim==2){
//             slice_backwards_no_precomputation_no_homogeneous_mr_monolithic<2,2><<<blocks, BLOCK_SIZE_BACK>>>(
//                 nr_positions,
//                 capacity, 
//                 lattice_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
//                 positions_raw.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
//                 scale_factor_tensor.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
//                 random_shift_monolithic.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
//                 anneal_window.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
//                 // grad_sliced_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
//                 // #if LATTICE_HALF_PRECISION
//                     // grad_sliced_values_monolithic.packed_accessor32<at::Half,3,torch::RestrictPtrTraits>(),
//                 // #else
//                     grad_sliced_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
//                 // #endif
//                 // lattice_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>()
//                 #if LATTICE_HALF_PRECISION
//                     lattice_values_monolithic_grad.packed_accessor32<at::Half,3,torch::RestrictPtrTraits>(),
//                 #else   
//                     lattice_values_monolithic_grad.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
//                 #endif
//                 positions_grad.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
//                 concat_points,
//                 require_lattice_values_grad,
//                 require_positions_grad
//             );
//         }else{
//             LOG(FATAL) <<"Instantiation for this val_dim doesnt exist. Please add it to the source code here.";
//         }
//     }else if(pos_dim==3){
//         // LOG(FATAL) <<"Instantiation for this pos_dim doesnt exist. Please add it to the source code here.";
//         if(val_dim==2){
//             slice_backwards_no_precomputation_no_homogeneous_mr_monolithic<3,2><<<blocks, BLOCK_SIZE_BACK>>>(
//                 nr_positions,
//                 capacity, 
//                 lattice_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
//                 positions_raw.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
//                 scale_factor_tensor.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
//                 random_shift_monolithic.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
//                 anneal_window.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
//                 // grad_sliced_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
//                 // #if LATTICE_HALF_PRECISION
//                     // grad_sliced_values_monolithic.packed_accessor32<at::Half,3,torch::RestrictPtrTraits>(),
//                 // #else
//                     grad_sliced_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
//                 // #endif
//                 // lattice_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>()
//                 #if LATTICE_HALF_PRECISION
//                     lattice_values_monolithic_grad.packed_accessor32<at::Half,3,torch::RestrictPtrTraits>(),
//                 #else   
//                     lattice_values_monolithic_grad.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
//                 #endif
//                 positions_grad.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
//                 concat_points,
//                 require_lattice_values_grad,
//                 require_positions_grad
                
//             );
//         }else{
//             LOG(FATAL) << "I'll implement it later. For now ther are a lot of stuff in the kernel hard coded for valdim=2 and pos either 3 or 4";
//         }
//     }else if(pos_dim==4){
//         // LOG(FATAL) <<"Instantiation for this pos_dim doesnt exist. Please add it to the source code here.";
//         if(val_dim==2){
//             slice_backwards_no_precomputation_no_homogeneous_mr_monolithic<4,2><<<blocks, BLOCK_SIZE_BACK>>>(
//                 nr_positions,
//                 capacity, 
//                 lattice_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
//                 positions_raw.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
//                 scale_factor_tensor.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
//                 random_shift_monolithic.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
//                 anneal_window.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
//                 // grad_sliced_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
//                 // #if LATTICE_HALF_PRECISION
//                     // grad_sliced_values_monolithic.packed_accessor32<at::Half,3,torch::RestrictPtrTraits>(),
//                 // #else
//                     grad_sliced_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
//                 // #endif
//                 // lattice_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>()
//                 #if LATTICE_HALF_PRECISION
//                     lattice_values_monolithic_grad.packed_accessor32<at::Half,3,torch::RestrictPtrTraits>(),
//                 #else   
//                     lattice_values_monolithic_grad.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
//                 #endif
//                 positions_grad.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
//                 concat_points,
//                 require_lattice_values_grad,
//                 require_positions_grad
                
//             );
//         }else{
//             LOG(FATAL) << "I'll implement it later. For now ther are a lot of stuff in the kernel hard coded for valdim=2 and pos either 3 or 4";
//         }
//     }else if(pos_dim==5){
//         // LOG(FATAL) <<"Instantiation for this pos_dim doesnt exist. Please add it to the source code here.";
//         if(val_dim==2){
//             slice_backwards_no_precomputation_no_homogeneous_mr_monolithic<5,2><<<blocks, BLOCK_SIZE_BACK>>>(
//                 nr_positions,
//                 capacity, 
//                 lattice_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
//                 positions_raw.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
//                 scale_factor_tensor.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
//                 random_shift_monolithic.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
//                 anneal_window.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
//                 // grad_sliced_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
//                 // #if LATTICE_HALF_PRECISION
//                     // grad_sliced_values_monolithic.packed_accessor32<at::Half,3,torch::RestrictPtrTraits>(),
//                 // #else
//                     grad_sliced_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
//                 // #endif
//                 // lattice_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>()
//                 #if LATTICE_HALF_PRECISION
//                     lattice_values_monolithic_grad.packed_accessor32<at::Half,3,torch::RestrictPtrTraits>(),
//                 #else   
//                     lattice_values_monolithic_grad.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
//                 #endif
//                 positions_grad.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
//                 concat_points,
//                 require_lattice_values_grad,
//                 require_positions_grad
                
//             );
//         }else{
//             LOG(FATAL) << "I'll implement it later. For now ther are a lot of stuff in the kernel hard coded for valdim=2 and pos either 3 or 4";
//         }
//     }else if(pos_dim==6){
//         // LOG(FATAL) <<"Instantiation for this pos_dim doesnt exist. Please add it to the source code here.";
//         if(val_dim==2){
//             slice_backwards_no_precomputation_no_homogeneous_mr_monolithic<6,2><<<blocks, BLOCK_SIZE_BACK>>>(
//                 nr_positions,
//                 capacity, 
//                 lattice_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
//                 positions_raw.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
//                 scale_factor_tensor.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
//                 random_shift_monolithic.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
//                 anneal_window.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
//                 // grad_sliced_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
//                 // #if LATTICE_HALF_PRECISION
//                     // grad_sliced_values_monolithic.packed_accessor32<at::Half,3,torch::RestrictPtrTraits>(),
//                 // #else
//                     grad_sliced_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
//                 // #endif
//                 // lattice_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>()
//                 #if LATTICE_HALF_PRECISION
//                     lattice_values_monolithic_grad.packed_accessor32<at::Half,3,torch::RestrictPtrTraits>(),
//                 #else   
//                     lattice_values_monolithic_grad.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
//                 #endif
//                 positions_grad.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
//                 concat_points,
//                 require_lattice_values_grad,
//                 require_positions_grad
                
//             );
//         }else{
//             LOG(FATAL) << "I'll implement it later. For now ther are a lot of stuff in the kernel hard coded for valdim=2 and pos either 3 or 4";
//         }
//     }
//     // TIME_END("slice_back_cuda");
    
   

//     return std::make_tuple(lattice_values_monolithic_grad,positions_grad);

// }
template<uint32_t POS_DIM, uint32_t NR_FEAT_PER_LEVEL>
std::tuple<torch::Tensor, torch::Tensor> Encoding<POS_DIM, NR_FEAT_PER_LEVEL>::backward(const EncodingInput& input, torch::Tensor& grad_sliced_values_monolithic){

    std::cout << " inside cpp calling back  " << std::endl;

    check_positions(input.m_positions_raw); 
    // CHECK(val_dim()>0) << "m_val_dim is 0 or lwoer. We have to splat something first so that we have values from where to slice. Val dim is " << val_dim();
    int nr_positions=input.m_positions_raw.size(0);
    int pos_dim=input.m_positions_raw.size(1);
    // CHECK(pos_dim==this->pos_dim()) << " The position dimension do not coreespond with the ones we used for creating the lattice";
    CHECK(grad_sliced_values_monolithic.dim()==3) <<"grad_sliced_values_monolithic should be nr_resolutions x val_dim x nr_positions, so it should have 3 dimensions. However it has "<< grad_sliced_values_monolithic.dim();
    CHECK(grad_sliced_values_monolithic.is_contiguous()) << "Grad sliced values needs to be contiguous. Please call .contiguous() on it";
    // splatting_indices_tensor=splatting_indices_tensor.contiguous();
    // splatting_weights_tensor=splatting_weights_tensor.contiguous();
    int nr_resolutions=grad_sliced_values_monolithic.size(0);
    int val_dim=grad_sliced_values_monolithic.size(1);
    CHECK(nr_positions==grad_sliced_values_monolithic.size(2)) << "The nr of positions should match between the input positions and the sliced values";
    CHECK(input.m_lattice_values.dim()==3) <<"grad_sliced_values_monolithic should be nr_resolutions x val_dim x nr_positions, so it should have 3 dimensions. However it has "<< input.m_lattice_values.dim();
    CHECK(input.m_lattice_values.is_contiguous()) <<"We assume that the lattice_values_monolithic are contiguous because in the cuda code we make a load of 2 float values at a time and that assumes that they are contiguous";
    
    

    //if we concat also the points, we add a series of extra resolutions to contain those points
    int nr_resolutions_extra=0;
    if (m_fixed_params.m_concat_points){
        nr_resolutions_extra=std::ceil(float(pos_dim)/val_dim);
        // VLOG(1) << "grad_sliced_values_monolithic" << grad_sliced_values_monolithic.sizes();
        // VLOG(1) << "grad_sliced_values_monolithic" << grad_sliced_values_monolithic;
        // grad_sliced_values_monolithic=grad_sliced_values_monolithic.slice(0, 0, nr_resolutions-nr_resolutions_extra); //dim, start, end
        // VLOG(1) << "grad_sliced_values_monolithic after slicing" << grad_sliced_values_monolithic.sizes();
        // VLOG(1) << "grad_sliced_values_monolithic after slicing" << grad_sliced_values_monolithic;
        nr_resolutions=nr_resolutions-nr_resolutions_extra;
    }

    int capacity=input.m_lattice_values.size(1);
    // std::cout << "capacity is " << capacity << std::endl;

    

    // nr_resolutions x nr_lattice_vertices x nr_lattice_featues
    // Tensor lattice_values_monolithic=torch::zeros({ nr_resolutions, nr_lattice_vertices(), val_dim  },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
    // Tensor lattice_values_monolithic=torch::zeros({ nr_resolutions, val_dim, m_capacity },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
    // TIME_START("slice_b_create_output");
    Tensor lattice_values_monolithic_grad; //dL/dLattiveValues
    if (input.m_require_lattice_values_grad){
        #if LATTICE_HALF_PRECISION
            // Tensor lattice_values_monolithic=torch::zeros({ nr_resolutions, val_dim, m_capacity },  torch::dtype(torch::kFloat16).device(torch::kCUDA, 0)  );
            lattice_values_monolithic_grad=torch::zeros({ nr_resolutions, capacity, val_dim },  torch::dtype(torch::kFloat16).device(torch::kCUDA, 0)  );
        #else
            // Tensor lattice_values_monolithic=torch::zeros({ nr_resolutions, val_dim, m_capacity },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
            lattice_values_monolithic_grad=torch::zeros({ nr_resolutions, capacity, val_dim },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
        #endif
    }else{
        lattice_values_monolithic_grad=torch::empty({ 1,1,1 },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
    }
    


    Tensor positions_grad; //dL/dPos
    if (input.m_require_positions_grad){
        positions_grad=torch::zeros({ nr_positions, pos_dim },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
    }else{
        positions_grad=torch::empty({ 1,1 },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
    }

    // TIME_END("slice_b_create_output");

    //make scalefactor for each lvl
    // Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> scale_factor_eigen;
    // scale_factor_eigen.resize(nr_resolutions, pos_dim);
    // double invStdDev = 1.0;
    // for(int res_idx=0; res_idx<nr_resolutions; res_idx++){
    //     for (int i = 0; i < pos_dim; i++) {
    //         scale_factor_eigen(res_idx,i) =  1.0 / (std::sqrt((double) (i + 1) * (i + 2))) * invStdDev;
    //         scale_factor_eigen(res_idx,i)=scale_factor_eigen(res_idx,i)/ sigmas_list[res_idx];
    //         // VLOG(1) << "scalinbg by " << sigmas_list[res_idx];
    //     }
    // }
    // // VLOG(1) << "scale_factor_eigen" << scale_factor_eigen;
    // Tensor scale_factor_tensor=eigen2tensor(scale_factor_eigen.cast<float>()).cuda();
    // scale_factor_tensor=scale_factor_tensor.view({nr_resolutions, pos_dim}); //nr_resolutuons x pos_dim

    // Tensor scale_factor_tensor=Lattice::compute_scale_factor_tensor(sigmas_list, pos_dim);
    // Tensor scale_factor_tensor=scale_factor;


    const dim3 blocks = { (unsigned int)div_round_up(nr_positions, BLOCK_SIZE_BACK), (unsigned int)nr_resolutions, 1 }; //the blocks are executed in order, first the blocks for the first resolution, then the second and so on

    
    slice_backwards_no_precomputation_no_homogeneous_mr_monolithic<POS_DIM,NR_FEAT_PER_LEVEL><<<blocks, BLOCK_SIZE_BACK>>>(
        nr_positions,
        capacity, 
        input.m_lattice_values.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        input.m_positions_raw.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        m_fixed_params.m_scale_factor.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        m_fixed_params.m_random_shift_per_level.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        input.m_anneal_window.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
        // grad_sliced_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        // #if LATTICE_HALF_PRECISION
            // grad_sliced_values_monolithic.packed_accessor32<at::Half,3,torch::RestrictPtrTraits>(),
        // #else
            grad_sliced_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        // #endif
        // lattice_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>()
        #if LATTICE_HALF_PRECISION
            input.m_lattice_values.packed_accessor32<at::Half,3,torch::RestrictPtrTraits>(),
        #else   
            input.m_lattice_values.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        #endif
        positions_grad.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        m_fixed_params.m_concat_points,
        input.m_require_lattice_values_grad,
        input.m_require_positions_grad
    );

    
   

    return std::make_tuple(lattice_values_monolithic_grad,positions_grad);

}



// //double back
// template<uint32_t POS_DIM, uint32_t NR_FEAT_PER_LEVEL>
// std::tuple<torch::Tensor, torch::Tensor> Encoding<POS_DIM, NR_FEAT_PER_LEVEL>::slice_double_back_from_positions_grad(const torch::Tensor& double_positions_grad, torch::Tensor& positions_raw, torch::Tensor& lattice_values_monolithic, torch::Tensor& grad_sliced_values_monolithic, const torch::Tensor& scale_factor, torch::Tensor& random_shift_monolithic, torch::Tensor& anneal_window, const bool concat_points){

//     check_positions(positions_raw); 
//     // CHECK(val_dim()>0) << "m_val_dim is 0 or lwoer. We have to splat something first so that we have values from where to slice. Val dim is " << val_dim();
//     int nr_positions=positions_raw.size(0);
//     int pos_dim=positions_raw.size(1);
//     // CHECK(pos_dim==this->pos_dim()) << " The position dimension do not coreespond with the ones we used for creating the lattice";
//     CHECK(grad_sliced_values_monolithic.dim()==3) <<"grad_sliced_values_monolithic should be nr_resolutions x val_dim x nr_positions, so it should have 3 dimensions. However it has "<< grad_sliced_values_monolithic.dim();
//     CHECK(grad_sliced_values_monolithic.is_contiguous()) << "Grad sliced values needs to be contiguous. Please call .contiguous() on it";
//     // splatting_indices_tensor=splatting_indices_tensor.contiguous();
//     // splatting_weights_tensor=splatting_weights_tensor.contiguous();
//     int nr_resolutions=grad_sliced_values_monolithic.size(0);
//     int val_dim=grad_sliced_values_monolithic.size(1);
//     CHECK(nr_positions==grad_sliced_values_monolithic.size(2)) << "The nr of positions should match between the input positions and the sliced values";
//     CHECK(lattice_values_monolithic.dim()==3) <<"grad_sliced_values_monolithic should be nr_resolutions x val_dim x nr_positions, so it should have 3 dimensions. However it has "<< lattice_values_monolithic.dim();
//     CHECK(lattice_values_monolithic.is_contiguous()) <<"We assume that the lattice_values_monolithic are contiguous because in the cuda code we make a load of 2 float values at a time and that assumes that they are contiguous";


//     //if we concat also the points, we add a series of extra resolutions to contain those points
//     int nr_resolutions_extra=0;
//     if (concat_points){
//         nr_resolutions_extra=std::ceil(float(pos_dim)/val_dim);
//         // VLOG(1) << "grad_sliced_values_monolithic" << grad_sliced_values_monolithic.sizes();
//         // VLOG(1) << "grad_sliced_values_monolithic" << grad_sliced_values_monolithic;
//         // grad_sliced_values_monolithic=grad_sliced_values_monolithic.slice(0, 0, nr_resolutions-nr_resolutions_extra); //dim, start, end
//         // VLOG(1) << "grad_sliced_values_monolithic after slicing" << grad_sliced_values_monolithic.sizes();
//         // VLOG(1) << "grad_sliced_values_monolithic after slicing" << grad_sliced_values_monolithic;
//         nr_resolutions=nr_resolutions-nr_resolutions_extra;
//     }

//     int capacity=lattice_values_monolithic.size(1);
    

//     // nr_resolutions x nr_lattice_vertices x nr_lattice_featues
//     // Tensor lattice_values_monolithic=torch::zeros({ nr_resolutions, nr_lattice_vertices(), val_dim  },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
//     // Tensor lattice_values_monolithic=torch::zeros({ nr_resolutions, val_dim, m_capacity },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
//     // TIME_START("slice_b_create_output");
//     Tensor lattice_values_monolithic_grad; //dL/dLattiveValues
//         #if LATTICE_HALF_PRECISION
//             // Tensor lattice_values_monolithic=torch::zeros({ nr_resolutions, val_dim, m_capacity },  torch::dtype(torch::kFloat16).device(torch::kCUDA, 0)  );
//             lattice_values_monolithic_grad=torch::zeros({ nr_resolutions, capacity, val_dim },  torch::dtype(torch::kFloat16).device(torch::kCUDA, 0)  );
//         #else
//             // Tensor lattice_values_monolithic=torch::zeros({ nr_resolutions, val_dim, m_capacity },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
//             lattice_values_monolithic_grad=torch::zeros({ nr_resolutions, capacity, val_dim },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
//         #endif


//     Tensor grad_grad_sliced_values_monolithic = torch::zeros({ nr_resolutions+nr_resolutions_extra, val_dim, nr_positions },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
    


    
//     Tensor scale_factor_tensor=scale_factor;


//     const dim3 blocks = { (unsigned int)div_round_up(nr_positions, BLOCK_SIZE_BACK), (unsigned int)nr_resolutions, 1 }; //the blocks are executed in order, first the blocks for the first resolution, then the second and so on

//     // TIME_START("slice_double_back");
//     if (pos_dim==3){
//         if(val_dim==1){
//             // slice_backwards_no_precomputation_no_homogeneous_mr_monolithic<3,1><<<blocks, BLOCK_SIZE_BACK>>>(
//             //     nr_positions, 
//             //     m_capacity,
//             //     positions_raw.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
//             //     scale_factor_tensor.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
//             //     grad_sliced_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
//             //     lattice_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>()
//             // );
//             LOG(FATAL) << "I'll implement it later. For now ther are a lot of stuff in the kernel hard coded for valdim=2 and pos either 3 or 4";
//         }else if(val_dim==2){
//             // slice_backwards_no_precomputation_no_homogeneous_mr_monolithic<3,2><<<blocks, BLOCK_SIZE_BACK>>>(
//             //     nr_positions, 
//             //     m_capacity,
//             //     positions_raw.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
//             //     scale_factor_tensor.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
//             //     grad_sliced_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
//             //     lattice_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>()
//             // );
//             slice_double_back_from_positions_grad_gpu<3,2><<<blocks, BLOCK_SIZE_BACK>>>(
//                 nr_positions,
//                 capacity, 
//                 double_positions_grad.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
//                 lattice_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
//                 positions_raw.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
//                 scale_factor_tensor.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
//                 random_shift_monolithic.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
//                 anneal_window.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
//                 grad_sliced_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
//                 concat_points,
//                 //output
//                 grad_grad_sliced_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
//                 #if LATTICE_HALF_PRECISION
//                     lattice_values_monolithic_grad.packed_accessor32<at::Half,3,torch::RestrictPtrTraits>()
//                 #else   
//                     lattice_values_monolithic_grad.packed_accessor32<float,3,torch::RestrictPtrTraits>()
//                 #endif
//             );
//         }else if(val_dim==4){
//             // slice_backwards_no_precomputation_no_homogeneous_mr_monolithic<3,4><<<blocks, BLOCK_SIZE_BACK>>>(
//             //     nr_positions, 
//             //     m_capacity,
//             //     positions_raw.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
//             //     scale_factor_tensor.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
//             //     grad_sliced_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
//             //     lattice_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>()
//             // );
//             LOG(FATAL) << "I'll implement it later. For now ther are a lot of stuff in the kernel hard coded for valdim=2 and pos either 3 or 4";
//         }else{
//             LOG(FATAL) <<"Instantiation for this val_dim doesnt exist. Please add it to the source code here.";
//         }
//     }else{
//         LOG(FATAL) <<"Instantiation for this pos_dim doesnt exist. Please add it to the source code here.";
//     }
//     // TIME_END("slice_double_back");
    
   

//     return std::make_tuple(lattice_values_monolithic_grad,  grad_grad_sliced_values_monolithic);

// }
template<uint32_t POS_DIM, uint32_t NR_FEAT_PER_LEVEL>
std::tuple<torch::Tensor, torch::Tensor> Encoding<POS_DIM, NR_FEAT_PER_LEVEL>::double_backward(const EncodingInput& input, const torch::Tensor& double_positions_grad, torch::Tensor& grad_sliced_values_monolithic){

    check_positions(input.m_positions_raw); 
    // CHECK(val_dim()>0) << "m_val_dim is 0 or lwoer. We have to splat something first so that we have values from where to slice. Val dim is " << val_dim();
    int nr_positions=input.m_positions_raw.size(0);
    int pos_dim=input.m_positions_raw.size(1);
    // CHECK(pos_dim==this->pos_dim()) << " The position dimension do not coreespond with the ones we used for creating the lattice";
    CHECK(grad_sliced_values_monolithic.dim()==3) <<"grad_sliced_values_monolithic should be nr_resolutions x val_dim x nr_positions, so it should have 3 dimensions. However it has "<< grad_sliced_values_monolithic.dim();
    CHECK(grad_sliced_values_monolithic.is_contiguous()) << "Grad sliced values needs to be contiguous. Please call .contiguous() on it";
    // splatting_indices_tensor=splatting_indices_tensor.contiguous();
    // splatting_weights_tensor=splatting_weights_tensor.contiguous();
    int nr_resolutions=grad_sliced_values_monolithic.size(0);
    int val_dim=grad_sliced_values_monolithic.size(1);
    CHECK(nr_positions==grad_sliced_values_monolithic.size(2)) << "The nr of positions should match between the input positions and the sliced values";
    CHECK(input.m_lattice_values.dim()==3) <<"grad_sliced_values_monolithic should be nr_resolutions x val_dim x nr_positions, so it should have 3 dimensions. However it has "<< input.m_lattice_values.dim();
    CHECK(input.m_lattice_values.is_contiguous()) <<"We assume that the lattice_values_monolithic are contiguous because in the cuda code we make a load of 2 float values at a time and that assumes that they are contiguous";


    //if we concat also the points, we add a series of extra resolutions to contain those points
    int nr_resolutions_extra=0;
    if (m_fixed_params.m_concat_points){
        nr_resolutions_extra=std::ceil(float(pos_dim)/val_dim);
        // VLOG(1) << "grad_sliced_values_monolithic" << grad_sliced_values_monolithic.sizes();
        // VLOG(1) << "grad_sliced_values_monolithic" << grad_sliced_values_monolithic;
        // grad_sliced_values_monolithic=grad_sliced_values_monolithic.slice(0, 0, nr_resolutions-nr_resolutions_extra); //dim, start, end
        // VLOG(1) << "grad_sliced_values_monolithic after slicing" << grad_sliced_values_monolithic.sizes();
        // VLOG(1) << "grad_sliced_values_monolithic after slicing" << grad_sliced_values_monolithic;
        nr_resolutions=nr_resolutions-nr_resolutions_extra;
    }

    int capacity=input.m_lattice_values.size(1);
    

    // nr_resolutions x nr_lattice_vertices x nr_lattice_featues
    // Tensor lattice_values_monolithic=torch::zeros({ nr_resolutions, nr_lattice_vertices(), val_dim  },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
    // Tensor lattice_values_monolithic=torch::zeros({ nr_resolutions, val_dim, m_capacity },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
    // TIME_START("slice_b_create_output");
    Tensor lattice_values_monolithic_grad; //dL/dLattiveValues
        #if LATTICE_HALF_PRECISION
            // Tensor lattice_values_monolithic=torch::zeros({ nr_resolutions, val_dim, m_capacity },  torch::dtype(torch::kFloat16).device(torch::kCUDA, 0)  );
            lattice_values_monolithic_grad=torch::zeros({ nr_resolutions, capacity, val_dim },  torch::dtype(torch::kFloat16).device(torch::kCUDA, 0)  );
        #else
            // Tensor lattice_values_monolithic=torch::zeros({ nr_resolutions, val_dim, m_capacity },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
            lattice_values_monolithic_grad=torch::zeros({ nr_resolutions, capacity, val_dim },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
        #endif


    Tensor grad_grad_sliced_values_monolithic = torch::zeros({ nr_resolutions+nr_resolutions_extra, val_dim, nr_positions },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
    


    
    // Tensor scale_factor_tensor=scale_factor;


    const dim3 blocks = { (unsigned int)div_round_up(nr_positions, BLOCK_SIZE_BACK), (unsigned int)nr_resolutions, 1 }; //the blocks are executed in order, first the blocks for the first resolution, then the second and so on

   
    slice_double_back_from_positions_grad_gpu<POS_DIM, NR_FEAT_PER_LEVEL><<<blocks, BLOCK_SIZE_BACK>>>(
        nr_positions,
        capacity, 
        double_positions_grad.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        input.m_lattice_values.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        input.m_positions_raw.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        m_fixed_params.m_scale_factor.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        m_fixed_params.m_random_shift_per_level.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        input.m_anneal_window.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
        grad_sliced_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        m_fixed_params.m_concat_points,
        //output
        grad_grad_sliced_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        #if LATTICE_HALF_PRECISION
            lattice_values_monolithic_grad.packed_accessor32<at::Half,3,torch::RestrictPtrTraits>()
        #else   
            lattice_values_monolithic_grad.packed_accessor32<float,3,torch::RestrictPtrTraits>()
        #endif
    );

    
   

    return std::make_tuple(lattice_values_monolithic_grad,  grad_grad_sliced_values_monolithic);

}

// template<uint32_t POS_DIM, uint32_t NR_FEAT_PER_LEVEL>
// torch::Tensor Encoding<POS_DIM, NR_FEAT_PER_LEVEL>::compute_scale_factor_tensor(const std::vector<float> sigmas_list, const int pos_dim){

//     int nr_resolutions=sigmas_list.size();

//     // Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> scale_factor_eigen;
//     // scale_factor_eigen.resize(nr_resolutions, pos_dim);
//     // double invStdDev = 1.0;
//     // for(int res_idx=0; res_idx<nr_resolutions; res_idx++){
//     //     for (int i = 0; i < pos_dim; i++) {
//     //         scale_factor_eigen(res_idx,i) =  1.0 / (std::sqrt((double) (i + 1) * (i + 2))) * invStdDev;
//     //         scale_factor_eigen(res_idx,i)=scale_factor_eigen(res_idx,i)/ sigmas_list[res_idx];
//     //         // VLOG(1) << "scalinbg by " << sigmas_list[res_idx];
//     //     }
//     // }
//     // // VLOG(1) << "scale_factor_eigen" << scale_factor_eigen;
//     // Tensor scale_factor_tensor=eigen2tensor(scale_factor_eigen.cast<float>()).cuda();
//     // scale_factor_tensor=scale_factor_tensor.view({nr_resolutions, pos_dim}); //nr_resolutuons x pos_dim



//     Tensor scale_factor_tensor=torch::zeros({ nr_resolutions, pos_dim },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
//     double invStdDev = 1.0;
//     for(int res_idx=0; res_idx<nr_resolutions; res_idx++){
//         for (int i = 0; i < pos_dim; i++) {
//             scale_factor_tensor[res_idx][i] =  1.0 / (std::sqrt((double) (i + 1) * (i + 2))) * invStdDev;
//             scale_factor_tensor[res_idx][i]=scale_factor_tensor[res_idx][i]/ sigmas_list[res_idx];
//         }
//     }



//     return scale_factor_tensor;

// }











//explicit instantiation 
// https://stackoverflow.com/a/495056
// https://isocpp.org/wiki/faq/templates#separate-template-class-defn-from-decl
template class Encoding<3,2>;



