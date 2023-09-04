#include <CL/sycl.hpp>
#include <iostream>

using namespace cl::sycl;

std::string output_memtype(info::local_mem_type memtype) {
    switch(memtype) {
        case info::local_mem_type::none:
            return "none";
        case info::local_mem_type::local:
            return "local";
        case info::local_mem_type::global:
            return "global";
        default:
            return "unknown";
    }
}

std::string output_global_memtype(info::global_mem_cache_type memtype) {
    switch(memtype) {
        case info::global_mem_cache_type::none:
            return "none";
        case info::global_mem_cache_type::read_only:
            return "read_only";
        case info::global_mem_cache_type::read_write:
            return "read_write";
        default:
            return "unknown";
    }
}

std::string output_partition_property(info::partition_property partition_type) {
    switch(partition_type) {
        case info::partition_property::no_partition:
            return "no_partition";
        case info::partition_property::partition_equally:
            return "partition_equally";
        case info::partition_property::partition_by_counts:
            return "partition_by_counts";
        case info::partition_property::partition_by_affinity_domain:
            return "partition_by_affinity_domain";
        default:
            return "unknown";
    }
}

std::string output_partition_affinity_domain(info::partition_affinity_domain partition_domain) {
    switch(partition_domain) {
        case info::partition_affinity_domain::not_applicable:
            return "not_applicable";
        case info::partition_affinity_domain::numa:
            return "numa";
        case info::partition_affinity_domain::L4_cache:
            return "L4_cache";
        case info::partition_affinity_domain::L3_cache:
            return "L3_cache";
        case info::partition_affinity_domain::L2_cache:
            return "L2_cache";
        case info::partition_affinity_domain::L1_cache:
            return "L1_cache";
        case info::partition_affinity_domain::next_partitionable:
            return "next_partitionable";
        default:
            return "unknown";
    }
}

int main() {
    // Loop through the available platforms
    for (auto const& this_platform : platform::get_platforms() ) {
    std::cout << "Found Platform:\n";
    std::cout << this_platform.get_info<info::platform::name>() << "\n"; ; 
    std::cout << "  Vendor: "
                << this_platform.get_info<info::platform::vendor>() << "\n";
        std::cout << "  Version: "
                << this_platform.get_info<info::platform::version>() << "\n";
        std::cout << "  Profile: "
                    << this_platform.get_info<info::platform::profile>() << "\n";
        // Query the platform for devices of type GPU
        auto devices = this_platform.get_devices(info::device_type::gpu);
        std::cout << "  Devices:\n";
        // Loop through the devices available in this plaform
        for (auto &dev : devices ) {
        std::cout << "    Device: "
                    << dev.get_info<info::device::name>() << "\n";
        std::cout << "      is_host(): "
                    << (dev.is_host() ? "Yes" : "No") << "\n";
        std::cout << "      is_cpu(): "
                    << (dev.is_cpu() ? "Yes" : "No") << "\n";
        std::cout << "      is_gpu(): "
                    << (dev.is_gpu() ? "Yes" : "No") << "\n";
        std::cout << "      is_accelerator(): "
                    << (dev.is_accelerator() ? "Yes" : "No") << "\n";
        std::cout << "      Vendor: "
                    << dev.get_info<info::device::vendor>() << "\n";
        std::cout << "      Driver Version: "
                    << dev.get_info<info::device::driver_version>() << "\n";
        std::cout << "      Max Work Item Dimensions: "
                    << dev.get_info<info::device::max_work_item_dimensions>() << "\n";
        std::cout << "      Max Work Group Size: "
                    << dev.get_info<info::device::max_work_group_size>() << "\n";
        std::cout << "      Mem Base Addr Align: "
                    << dev.get_info<info::device::mem_base_addr_align>() << "\n";
        std::cout << "      Sub Group Size: "
                    << dev.get_info<info::device::sub_group_sizes>()[0] << "\n";
        std::cout << "      Max Local Mem Per Work Group: "
                    << dev.get_info<info::device::local_mem_size>() << "\n";
        std::cout << "      Number of Compute Units: "
                    << dev.get_info<info::device::max_compute_units>() << "\n";
        std::cout << "      Max Clock Frequency: "
                    << dev.get_info<info::device::max_clock_frequency>() << "\n";   
        std::cout << "      Max Mem Alloc Size: "
                    << dev.get_info<info::device::max_mem_alloc_size>() << "\n";
        std::cout << "      Global Mem Size: "
                    << dev.get_info<info::device::global_mem_size>() << "\n";
        std::cout << "      Max Constant Buffer Size: "
                    << dev.get_info<info::device::max_constant_buffer_size>() << "\n";
        std::cout << "      Max Constant Args: "
                    << dev.get_info<info::device::max_constant_args>() << "\n";
        std::cout << "      Local Mem Type: "
                    << output_memtype(dev.get_info<info::device::local_mem_type>()) << "\n";
        std::cout << "      Global Mem Cache Type: "
                    << output_global_memtype(dev.get_info<info::device::global_mem_cache_type>()) << "\n";
        std::cout << "      Global Mem Cache Size: "
                    << dev.get_info<info::device::global_mem_cache_size>() << "\n";
        std::cout << "      Global Mem Cache Line Size: "
                    << dev.get_info<info::device::global_mem_cache_line_size>() << "\n";
        std::cout << "      Max Block Dimensions: "
                    << dev.get_info<info::device::max_work_item_sizes<3>>()[0] << ", " << dev.get_info<info::device::max_work_item_sizes<3>>()[1] << ", " << dev.get_info<info::device::max_work_item_sizes<3>>()[2] << "\n";
        std::cout << "      Max Block Dimensions: "
                    << dev.get_info<info::device::max_work_item_dimensions>() << "\n";
        std::cout << "      Preffered half vector width: "
                    << dev.get_info<info::device::preferred_vector_width_half>() << "\n";
        std::cout << "      Preffered float vector width: "
                    << dev.get_info<info::device::preferred_vector_width_float>() << "\n"; 
        std::cout << "      Preffered double vector width: "
                    << dev.get_info<info::device::preferred_vector_width_double>() << "\n";
        std::cout << "      Partition Max Sub Devices: "
                    << dev.get_info<info::device::partition_max_sub_devices>() << "\n";
        /* std::cout << "      Partition Properties: "
                    << output_partition_property(dev.get_info<info::device::partition_properties>()[0]) << "\n";
        std::cout << "      Partition Affinity Domains: "
                    << output_partition_affinity_domain(dev.get_info<info::device::partition_affinity_domains>()[0]) << "\n"; */
        } std::cout << "\n";
        
    }

    return 0;
}

