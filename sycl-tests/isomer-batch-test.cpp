#include <CL/sycl.hpp>
#include <optional>
#include <vector>
#include <tuple>
#include <string>
#include <iostream>
#define FLOAT_TYPEDEFS(T) static_assert(std::is_floating_point<T>::value, "T must be float"); typedef std::array<T,3> coord3d; typedef std::array<T,2> coord2d; typedef T real_t;
#define INT_TYPEDEFS(K) static_assert(std::is_integral<K>::value, "K must be integral type"); typedef std::array<K,3> node3; typedef std::array<K,2> node2; typedef K node_t; typedef std::array<K,6> node6;
#define TEMPLATE_TYPEDEFS(T,K) FLOAT_TYPEDEFS(T) INT_TYPEDEFS(K)

using namespace cl::sycl;

enum class IsomerStatus {EMPTY, CONVERGED, PLZ_CHECK, FAILED, NOT_CONVERGED};
enum BatchMember {COORDS3D, COORDS2D, CUBIC_NEIGHBOURS, DUAL_NEIGHBOURS, FACE_DEGREES, IDS, ITERATIONS, STATUSES};
enum SortOrder {ASCENDING, DESCENDING};
enum class LaunchPolicy {SYNC, ASYNC};
enum Device   {CPU, GPU};

template<typename T, typename K>
struct IsomerBatch
{ 
    //template typename T::LaunchCtx LaunchCtx;
    int isomer_capacity = 0;
    bool allocated = false;
    size_t n_atoms = 0;
    size_t n_faces = 0;
    size_t n_isomers = 0;
    T* X;
    T* xys;

    K* cubic_neighbours;
    K* dual_neighbours;
    K* face_degrees;

    size_t* IDs;
    size_t* iterations;
    IsomerStatus* statuses;
    std::vector<std::tuple<std::string,void**,size_t,bool>> pointers;

    IsomerBatch(){
      pointers =   {{"cubic_neighbours",(void**)&cubic_neighbours, sizeof(K)*3, true}, {"dual_neighbours", (void**)&dual_neighbours, sizeof(K)*4, true}, {"face_degrees", (void**)&face_degrees, sizeof(K)*1, true}, {"X", (void**)&X, sizeof(T)*3, true}, {"xys", (void**)&xys, sizeof(T)*2, true}, {"statuses", (void**)&statuses, sizeof(IsomerStatus), false}, {"IDs", (void**)&IDs, sizeof(size_t), false}, {"iterations", (void**)&iterations, sizeof(size_t), false}};
    }

    void operator=(const IsomerBatch &);

    ~IsomerBatch();
    IsomerBatch(size_t n_atoms, size_t n_isomers, cl::sycl::queue& Q);
    //void set_print_simple() {verbose = false;} 
    //void set_print_verbose() {verbose = true;} 
    //bool get_print_mode() const {return verbose;}
    ////Prints a specific parameter from the batch
    //void print(const BatchMember param, const std::pair<int,int>& range = {-1,-1}); 
    //int size() const {return m_size;}
    //int capacity() const {return isomer_capacity;}
//
    //std::vector<size_t> find_ids(const IsomerStatus status); //Returns a vector of IDs with a given status
    //void shrink_to_fit();        
//
    //void clear(sycl::queue& Q, const LaunchPolicy = LaunchPolicy::SYNC);                 //Clears the batch and resets the size to 0
    //bool operator==(const IsomerBatch& b); //Returns true if the two batches are equal
    //bool operator!=(const IsomerBatch& b) {return !(*this == b);}
    ////friend std::ostream& operator<<(std::ostream& os, const IsomerBatch& a); //Prints the batch to the given stream

  private:
    int m_size = 0;
    cl::sycl::context ctx;
    bool verbose = false;

};

using std::get;

template<typename T, typename K>
IsomerBatch<T,K>::IsomerBatch(size_t n_atoms, size_t n_isomers, cl::sycl::queue& Q){
    TEMPLATE_TYPEDEFS(T,K);
    this->n_atoms          = n_atoms;
    this->isomer_capacity  = n_isomers;
    this->n_faces          = n_atoms/2 + 2;
    this->ctx            =  Q.get_context();
    pointers =   {{"cubic_neighbours",(void**)&cubic_neighbours, sizeof(node_t)*n_atoms*3, true}, {"dual_neighbours", (void**)&dual_neighbours, sizeof(node_t) * (n_atoms/2 +2) * 6, true}, {"face_degrees", (void**)&face_degrees, sizeof(node_t)*(n_atoms/2 +2), true},{"X", (void**)&X, sizeof(real_t)*n_atoms*3, true}, {"xys", (void**)&xys, sizeof(real_t)*n_atoms*2, true}, {"statuses", (void**)&statuses, sizeof(IsomerStatus), false}, {"IDs", (void**)&IDs, sizeof(size_t), false}, {"iterations", (void**)&iterations, sizeof(size_t), false}};
    if (!Q.is_host()){
        for (size_t i = 0; i < pointers.size(); i++) {
            std::cout << "Allocating " << isomer_capacity * get<2>(pointers[i]) << " bytes of device memory for " << get<0>(pointers[i]) << std::endl;
            *get<1>(pointers[i]) = cl::sycl::malloc_device(n_isomers * get<2>(pointers[i]), Q); 
            Q.memset(*get<1>(pointers[i]),0,n_isomers*get<2>(pointers[i]));
            auto device = get_pointer_device(*get<1>(pointers[i]), Q.get_context());
            std::cout << "Allocated on device: " << device.get_info<info::device::name>() << std::endl;
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
IsomerBatch<T,K>::~IsomerBatch(){
    if (allocated == true);
    {
        for (size_t i = 0; i < pointers.size(); i++) {
            auto device = get_pointer_device(*get<1>(pointers[i]), ctx);
            std::cout << "Attempting to free ptr: "<< i << ": "<< get<0>(pointers[i]) << " " << device.get_info<info::device::name>() << std::endl;
            //cl::sycl::free(*get<1>(pointers[i]), ctx);

        }
    }
    allocated = false;
}


int main(int argc, char const *argv[])
{

    queue Q(gpu_selector{}, property::queue::in_order());
    IsomerBatch<float, int> batch(20, 1, Q);
    /* code */
    //Q.wait_and_throw();
//
    Q.submit([&](handler &h) {
        // Create a command group to issue GPU work.
        h.parallel_for<class hello_world>(nd_range(range{1}, range{1}), [=](nd_item<1> idx) {
            printf("Hello World from GPU thread %e!\n", batch.X[0]);

        });
    });
//
    Q.wait_and_throw();

    return 0;
}
