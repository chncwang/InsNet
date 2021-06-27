Installing InsNet
==================

In this topic, we will demonstrate how to build InsNet and link it with your C++ programs, assuming you are using Ubuntu 16.04 or higher.

Prerequisites
-------------

There are several prerequisites for building InsNet as below:

- g++ 9.3.0 (lower versions supporting C++ 14 should also work but are not guaranteed)
- CMake 3.20 (lower versions such as 3.10 would probably fail to find cublas)

For the CPU version, eigen is already included in InsNet (keeping the original MPL 2.0 license), so you need not specify external libraries. For the GPU version, InsNet only relies on CUDA 11.1 (lower versions or the latest should also work).

Building & Linking
------------------

To get and build InsNet, clone the repository as below:

.. code-block:: console

    git clone http://github.com/chncwang/insnet

Then the recommended way to link InsNet is to place the source codes under your project and then add the following lines to your project's CMakeLists.txt:

.. code-block:: cmake

    if (USE_CUDA)
        add_definitions(-DUSE_GPU)
    endif()
    add_subdirectory(path_to_insnet)
    target_link_libraries(your_executable insnet)

If you use the GPU version, you need to set USE_CUDA to 1 and specify your GPU architecture when building your project:

.. code-block:: console

    cmake .. -DUSE_CUDA=1 -DCARD_NUMBER=75 # "75" means Turing architecture.
