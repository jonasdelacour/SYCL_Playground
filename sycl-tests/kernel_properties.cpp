#include <CL/sycl.hpp>
#include <iostream>
using namespace cl::sycl;

int main() {
    constexpr size_t size = 200;
    queue Q(gpu_selector{}); // Select a SYCL device to use.
    
    Q.submit([&](handler &h) {
        // Create a command group to issue GPU work.
        h.parallel_for<class hello_world>(nd_range(range{size*68}, range{size}), [=](nd_item<1> i) {
            if(i.get_local_linear_id() == 0) printf("Hello, World from work item %d!\n", i.get_group_linear_id());
        });
    });

    program prog(Q.get_context());
    prog.build_with_kernel_type<class hello_world>("--maxrregcount 40");

    auto work_group_size = prog.get_kernel<class hello_world>().get_work_group_info<info::kernel_work_group::compile_work_group_size>(Q.get_device());
    std::cout << "Kernel compiled with work-group-size: " << work_group_size[0] << ", " << work_group_size[1] << ", " << work_group_size[2] << "\n";

    return 0;
}