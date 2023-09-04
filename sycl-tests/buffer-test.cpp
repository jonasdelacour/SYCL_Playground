#include <CL/sycl.hpp>
#include <iostream>
#include "numeric"
#include <vector>
#include <tuple>
using namespace cl::sycl;

#define UINT_TYPE uint16_t
enum class IsomerStatus
{
    EMPTY,
    CONVERGED,
    PLZ_CHECK,
    FAILED,
    NOT_CONVERGED
};

template <typename T, typename K>
struct IsomerBatch
{
    size_t m_capacity = 0;
    size_t m_size = 0;
    size_t n_atoms = 0;
    size_t n_faces = 0;
    buffer<T, 1> X{};
    buffer<T, 1> xys;//(sycl::range<1>{1});
    buffer<K, 1> cubic_neighbours;//(sycl::range<1>{1});
    buffer<K, 1> dual_neighbours;//(sycl::range<1>{1});
    buffer<K, 1> face_degrees;//(sycl::range<1>{1});
    buffer<size_t, 1> IDs;//(sycl::range<1>{1});
    buffer<size_t, 1> iterations;//(sycl::range<1>{1});
    buffer<IsomerStatus, 1> statuses;//(sycl::range<1>{1});

    bool allocated = false;

    // std::vector<std::tuple<void**,size_t,bool>> pointers;

    IsomerBatch(size_t n_atoms, size_t n_isomers) : n_atoms(n_atoms), m_capacity(n_isomers),
                                                    X{range<1>(n_isomers * n_atoms * 3)}, 
                                                    xys{range<1>(n_isomers * n_atoms * 2)}, 
                                                    cubic_neighbours{range<1>(n_isomers * n_atoms * 3)}, 
                                                    dual_neighbours{range<1>(6 * n_isomers * (n_atoms / 2 + 2))}, 
                                                    face_degrees{range<1>((n_atoms / 2 + 2) * 1)}, 
                                                    IDs{range<1>(n_isomers)}, 
                                                    iterations{range<1>(n_isomers)}, 
                                                    statuses{range<1>(n_isomers)}
    {   
        //X = buffer(range<1>{3});
        host_accessor X_acc(X, no_init);
        host_accessor xys_acc(xys, no_init);
        host_accessor cubic_neighbours_acc(cubic_neighbours, no_init);
        host_accessor dual_neighbours_acc(dual_neighbours, no_init);
        host_accessor face_degrees_acc(face_degrees, no_init);
        host_accessor IDs_acc(IDs, no_init);
        host_accessor iterations_acc(iterations, no_init);
        host_accessor statuses_acc(statuses, no_init);


        for (size_t i = 0; i < n_isomers; i++)
        {
            for (size_t j = 0; j < n_atoms; j++)
            {
                X_acc[i * n_atoms * 3 + j * 3 + 0] = T(0.0);
                X_acc[i * n_atoms * 3 + j * 3 + 1] = T(0.0);
                X_acc[i * n_atoms * 3 + j * 3 + 2] = T(0.0);
                xys_acc[i * n_atoms * 2 + j * 2 + 0] = T(0.0);
                xys_acc[i * n_atoms * 2 + j * 2 + 1] = T(0.0);
                cubic_neighbours_acc[i * n_atoms * 3 + j * 3 + 0] = std::numeric_limits<K>::max();
                cubic_neighbours_acc[i * n_atoms * 3 + j * 3 + 1] = std::numeric_limits<K>::max();
                cubic_neighbours_acc[i * n_atoms * 3 + j * 3 + 2] = std::numeric_limits<K>::max();
            }
            for (size_t j = 0; j < 6 * (n_atoms / 2 + 2); j++)
            {
                dual_neighbours_acc[i * 6 * (n_atoms / 2 + 2) + j] = std::numeric_limits<K>::max();
            }
            for (size_t j = 0; j < (n_atoms / 2 + 2); j++)
            {
                face_degrees_acc[i * (n_atoms / 2 + 2) + j] = std::numeric_limits<K>::max();
            }
            IDs_acc[i] = std::numeric_limits<size_t>::max();
            iterations_acc[i] = 0;
            statuses_acc[i] = IsomerStatus::EMPTY;
        }
    }
};

int main()
{
    IsomerBatch<float, UINT_TYPE> batch(20, 10);
//    std::cout << "batch allocated: " << batch.allocated << std::endl;
//    std::cout << "batch size: " << batch.m_size << std::endl;
//    std::cout << "batch capacity: " << batch.m_capacity << std::endl;
//    std::cout << "batch n_atoms: " << batch.n_atoms << std::endl;
//    std::cout << "batch n_faces: " << batch.n_faces << std::endl;
//    std::cout << "batch X size: " << batch.X.get_count() << std::endl;
//    std::cout << "batch xys size: " << batch.xys.get_count() << std::endl;
//    std::cout << "batch cubic_neighbours size: " << batch.cubic_neighbours.get_count() << std::endl;
//    std::cout << "batch dual_neighbours size: " << batch.dual_neighbours.get_count() << std::endl;
//    std::cout << "batch face_degrees size: " << batch.face_degrees.get_count() << std::endl;
//    std::cout << "batch IDs size: " << batch.IDs.get_count() << std::endl;
//    std::cout << "batch iterations size: " << batch.iterations.get_count() << std::endl;
//    std::cout << "batch statuses size: " << batch.statuses.get_count() << std::endl;

    //sycl::host_accessor X_acc(batch.IDs, sycl::read_only);
    //std::cout << "IDs_acc: " << X_acc[0] << std::endl;

    return 0;
}
