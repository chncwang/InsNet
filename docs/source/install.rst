Installing N3LDG++
========================

How to build N3LDG++ and link it with your C++ programs, assuming you are using Ubuntu 16.04 or higher.

Prerequisites
-------------

There are several prerequisites for building N3LDG++ as below:

- g++ 9.3.0 (lower versions should also work but are not guaranteed)
- CMake 3.20 (3.8 should also work)

For the CPU version, eigen is already included in N3LDG++ (keeping the original MPL 2.0 license), so you need not specify external libraries. For the GPU version, N3LDG++ only relies on CUDA 11.1 (lower versions should also work).

Building & linking
-------------

To get and build N3LDG++, clone the repository as below:

::

    git clone http://github.com/chncwang/n3ldg-plus

Then the recommended way to link N3LDG++ to your project is to place the source codes under your project and then add the following lines to your project's CMakeLists.txt:

::

    add_definitions(-DUSE_GPU) # the GPU version only
    add_subdirectory(path_to_n3ldg-plus/n3ldg-plus)
    target_link_libraries(your_executable n3ldg_plus)

If you use the GPU version, you need to specify your GPU architecture when building your project:

::

    cmake .. -DCARD_NUMBER=75 # "75" means Turing architecture.


