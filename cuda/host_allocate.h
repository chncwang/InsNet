#ifndef N3LDG_PLUS_HOST_ALLOCATE_H
#define N3LDG_PLUS_HOST_ALLOCATE_H

#include <cstddef>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <Memory_cuda.h>

namespace n3ldg_cuda {

class HostAlloc {
public:
    cudaError_t malloc(void **p, size_t size);
};

template <class T>
class HostAllocator
{
public:
    using value_type    = T;

    HostAllocator() noexcept {}  // not required, unless used
    template <class U> HostAllocator(HostAllocator<U> const&) noexcept {}

    value_type*  // Use pointer if pointer is not a value_type*
    allocate(std::size_t n)
    {
        T *host_ptr = nullptr;
        cudaError_t error = MemoryPool<HostAlloc>::Ins().Malloc((void**)&host_ptr,
                n * sizeof(value_type));
        if (error != cudaSuccess) {
            std::cerr << error << " host allocation error!" << std::endl;
            abort();
        }
        return static_cast<value_type*>(host_ptr);
    }

    void
    deallocate(value_type* p, std::size_t) noexcept  // Use pointer if pointer is not a value_type*
    {
        cudaFreeHost(p);
    }

//     value_type*
//     allocate(std::size_t n, const_void_pointer)
//     {
//         return allocate(n);
//     }

//     template <class U, class ...Args>
//     void
//     construct(U* p, Args&& ...args)
//     {
//         ::new(p) U(std::forward<Args>(args)...);
//     }

//     template <class U>
//     void
//     destroy(U* p) noexcept
//     {
//         p->~U();
//     }

//     std::size_t
//     max_size() const noexcept
//     {
//         return std::numeric_limits<size_type>::max();
//     }

//     HostAllocator
//     select_on_container_copy_construction() const
//     {
//         return *this;
//     }

//     using propagate_on_container_copy_assignment = std::false_type;
//     using propagate_on_container_move_assignment = std::false_type;
//     using propagate_on_container_swap            = std::false_type;
//     using is_always_equal                        = std::is_empty<HostAllocator>;
};

template <class T, class U>
bool
operator==(HostAllocator<T> const&, HostAllocator<U> const&) noexcept
{
    return true;
}

template <class T, class U>
bool
operator!=(HostAllocator<T> const& x, HostAllocator<U> const& y) noexcept
{
    return !(x == y);
}

template<typename T>
//using PageLockedVector = std::vector<T, HostAllocator<T>>;
using PageLockedVector = std::vector<T>;

}

#endif
