#include "isomer_batch.hh"
#include <tuple>

using std::get;

template<typename T, typename K>
IsomerBatch<T,K>::IsomerBatch(size_t n_atoms, size_t n_isomers, cl::sycl::queue& Q){
    TEMPLATE_TYPEDEFS(T,K);
    this->n_atoms          = n_atoms;
    this->isomer_capacity  = n_isomers;
    this->n_faces          = n_atoms/2 + 2;
    this->ctx              = Q.get_context();
    pointers =   {{"cubic_neighbours",(void**)&cubic_neighbours, sizeof(node_t)*n_atoms*3, true}, {"dual_neighbours", (void**)&dual_neighbours, sizeof(node_t) * (n_atoms/2 +2) * 6, true}, {"face_degrees", (void**)&face_degrees, sizeof(node_t)*(n_atoms/2 +2), true},{"X", (void**)&X, sizeof(real_t)*n_atoms*3, true}, {"xys", (void**)&xys, sizeof(real_t)*n_atoms*2, true}, {"statuses", (void**)&statuses, sizeof(IsomerStatus), false}, {"IDs", (void**)&IDs, sizeof(size_t), false}, {"iterations", (void**)&iterations, sizeof(size_t), false}};
    if (!Q.is_host()){
        for (size_t i = 0; i < pointers.size(); i++) {
    //        std::cout << "Allocating " << isomer_capacity * get<2>(pointers[i]) << " bytes of device memory for " << get<0>(pointers[i]) << std::endl;
            *get<1>(pointers[i]) = cl::sycl::malloc_device(n_isomers * get<2>(pointers[i]), Q); 
            Q.memset(*get<1>(pointers[i]),0,n_isomers*get<2>(pointers[i]));
        }
    } else if(Q.is_host()){
        for (size_t i = 0; i < pointers.size(); i++) {
            //For asynchronous memory transfers host memory must be pinned.
            *get<1>(pointers[i]) = cl::sycl::malloc_host(n_isomers * get<2>(pointers[i]), Q); 
            Q.memset(*get<1>(pointers[i]),0, n_isomers*get<2>(pointers[i]));
        }
    }
    Q.wait_and_throw();
    allocated = true;
}

template<typename T, typename K>
void IsomerBatch<T,K>::operator=(const IsomerBatch<T,K>& input){

    pointers =   {{"cubic_neighbours",(void**)&cubic_neighbours, sizeof(node_t)*n_atoms*3, true}, {"dual_neighbours", (void**)&dual_neighbours, sizeof(node_t) * (n_atoms/2 +2) * 6, true}, {"face_degrees", (void**)&face_degrees, sizeof(node_t)*(n_atoms/2 +2), true},{"X", (void**)&X, sizeof(real_t)*n_atoms*3, true}, {"xys", (void**)&xys, sizeof(real_t)*n_atoms*2, true}, {"statuses", (void**)&statuses, sizeof(IsomerStatus), false}, {"IDs", (void**)&IDs, sizeof(size_t), false}, {"iterations", (void**)&iterations, sizeof(size_t), false}};
    if (allocated == true){
        for (size_t i = 0; i < pointers.size(); i++) {
            cl::sycl::free(*get<1>(pointers[i]), *Q_ptr);
        }
        allocated = false;
    }
    //Construct a tempory batch: allocates the needed amount of memory.
    this->Q_ptr           = input.Q_ptr;
    this->isomer_capacity = input.isomer_capacity;
    this->n_atoms = input.n_atoms;
    this->n_faces = input.n_faces;
    
    
    //Copy contents of old batch into newly allocated memory.
    if (!Q_ptr->is_host()){
        for (size_t i = 0; i < pointers.size(); i++) {
            *get<1>(pointers[i]) = cl::sycl::malloc_device(isomer_capacity * get<2>(pointers[i]), *Q_ptr);
            //cudaMalloc(get<1>(pointers[i]), isomer_capacity * get<2>(pointers[i]));
            Q_ptr -> memcpy(*get<1>(pointers[i]), *get<1>(input.pointers[i]), isomer_capacity * get<2>(pointers[i]));
            //cudaMemcpy(*get<1>(pointers[i]), *get<1>(input.pointers[i]), isomer_capacity * get<2>(pointers[i]), cudaMemcpyDeviceToDevice);
        }
    } else if(Q_ptr->is_host()){
        for (size_t i = 0; i < pointers.size(); i++) {
            *get<1>(pointers[i]) = cl::sycl::malloc_host(isomer_capacity * get<2>(pointers[i]), *Q_ptr);
            //cudaMallocHost(get<1>(pointers[i]), isomer_capacity * get<2>(pointers[i]));
            Q_ptr -> memcpy(*get<1>(pointers[i]), *get<1>(input.pointers[i]), isomer_capacity * get<2>(pointers[i]));
	    //TODO: Isn't this a bug? Nothing is being copied!
        }
    }
    printLastCudaError("Failed to copy IsomerBatch");
}

template<typename T, typename K>
IsomerBatch<T,K>::~IsomerBatch(){
    //if (allocated == true);
    //{
    //    for (size_t i = 0; i < pointers.size(); i++) {
    //        cl::sycl::free(*get<1>(pointers[i]), ctx);
    //    }
    //}
    //allocated = false;
}