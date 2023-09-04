#pragma once
#include <optional>
#include <vector>
#include <tuple>
#include <string>

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
    IsomerBatch(size_t n_atoms, size_t n_isomers, sycl::queue& Q);
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
    sycl::context ctx;
    bool verbose = false;

};
