Benchmarks
==============================================

This topic will conduct the benchmarks (`Source Code <https://github.com/chncwang/insnet-benchmark>`_) to measure InsNet's training speed and memory usage on Transformer-based seq2seq models, with various model size settings.

Dataset
---------

We use a neural machine translation dataset IWSLT'15 English-Vietnamese with sentence length statistics as follows:

.. list-table::
    :widths: 5 10
    :header-rows: 1

    * -
      - Length (mean and standard deviation)
    * - source sentence
      - :math:`20.3\pm15.0`
    * - target sentence
      - :math:`24.8\pm18.8`

The source and target vocabulary size is 17191 and 7709, respectively.

Model
-------

We use two Transformer-based seq2seq models, namely *BASE* and *LARGE*, with their settings as follows:

.. list-table::
    :widths: 5 5 5 5 5 5 5
    :header-rows: 1

    * -
      - enc layer
      - dec layer
      - hidden size
      - inner size
      - head size
      - dropout
    * - BASE
      - 3
      - 3
      - 512
      - 2048
      - 8
      - 0.1
    * - LARGE
      - 6
      - 6
      - 768
      - 3072
      - 12
      - 0.1

For Both models, we use the ADAM optimizer.

Environment
-------------

All benchmarks are conducted on our laptop with configuration as follows:

.. list-table::
    :widths: 5 5
    :header-rows: 0

    * - CPU Frequency
      - 2.6G Hz
    * - GPU
      - RTX 2060 *Laptop*
    * - GPU Memory
      - 6GB
    * - OS
      - Ubuntu 16.04

The completed GPU configuration is as follows:

::

    Device 0: "GeForce RTX 2060"
      CUDA Driver Version / Runtime Version          11.1 / 10.0
      CUDA Capability Major/Minor version number:    7.5
      Total amount of global memory:                 5935 MBytes (6222839808 bytes)
      (30) Multiprocessors, ( 64) CUDA Cores/MP:     1920 CUDA Cores
      GPU Max Clock rate:                            1200 MHz (1.20 GHz)
      Memory Clock rate:                             7001 Mhz
      Memory Bus Width:                              192-bit
      L2 Cache Size:                                 3145728 bytes
      Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
      Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
      Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
      Total amount of constant memory:               65536 bytes
      Total amount of shared memory per block:       49152 bytes
      Total number of registers available per block: 65536
      Warp size:                                     32
      Maximum number of threads per multiprocessor:  1024
      Maximum number of threads per block:           1024
      Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
      Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
      Maximum memory pitch:                          2147483647 bytes
      Texture alignment:                             512 bytes
      Concurrent copy and kernel execution:          Yes with 3 copy engine(s)
      Run time limit on kernels:                     No
      Integrated GPU sharing Host Memory:            No
      Support host page-locked memory mapping:       Yes
      Alignment requirement for Surfaces:            Yes
      Device has ECC support:                        Disabled
      Device supports Unified Addressing (UVA):      Yes
      Device supports Compute Preemption:            Yes
      Supports Cooperative Kernel Launch:            Yes
      Supports MultiDevice Co-op Kernel Launch:      Yes
      Device PCI Domain ID / Bus ID / location ID:   0 / 1 / 0
      Compute Mode:
         < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

PyTorch Implementation
----------------------------

We use `OpenNMT <https://github.com/OpenNMT/OpenNMT-py>`_ as the PyTorch implementation for comparison.

Results
--------

Training Throughput on GPU
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Transformer (BASE) Training Throughput on GPU (tokens per second)
    :widths: 3 3 3
    :header-rows: 1

    * - batch_size
      - insnet
      - PyTorch
    * - 500
      - 14885
      - 12443
    * - 1000
      - 16782
      - 17822
    * - 1500
      - 17401
      - 19846
    * - 2000
      - 17767
      - 20953
    * - 2500
      - 17952
      - 21555
    * - 3000
      - 17994
      - 21912
    * - 4500
      - 17885
      - 22674
    * - 6000
      - 17348
      - 22474

.. list-table:: Transformer (LARGE) Training Throughput on GPU (tokens per second)
    :widths: 3 3 3
    :header-rows: 1

    * - batch_size
      - insnet
      - PyTorch
    * - 500
      - 4788
      - 5901
    * - 1000
      - 5235
      - 5968
    * - 1500
      - OOM
      - 6970
    * - 2000
      - OOM
      - OOM
    * - 2500
      - OOM
      - OOM
    * - 3000
      - OOM
      - OOM
    * - 4500
      - OOM
      - OOM
    * - 6000
      - OOM
      - OOM

Memory Usage on GPU
^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Transformer (BASE) Memory Usage on GPU (MB)
    :widths: 3 3 3
    :header-rows: 1

    * - batch_size
      - insnet
      - PyTorch
    * - 500
      - 1667
      - 2397
    * - 1000
      - 1907
      - 2353
    * - 1500
      - 2243
      - 2551
    * - 2000
      - 2165
      - 3345
    * - 2500
      - 2163
      - 3754
    * - 3000
      - 2850
      - 3727
    * - 4500
      - 3158
      - 5284
    * - 6000
      - 3255
      - 5447

.. list-table:: Transformer (LARGE) Memory Usage on GPU (MB)
    :widths: 3 3 3
    :header-rows: 1

    * - batch_size
      - insnet
      - PyTorch
    * - 500
      - 5169
      - 5271
    * - 1000
      - 5793
      - 5277
    * - 1500
      - OOM
      - 5813
    * - 2000
      - OOM
      - OOM
    * - 2500
      - OOM
      - OOM
    * - 3000
      - OOM
      - OOM
    * - 4500
      - OOM
      - OOM
    * - 6000
      - OOM
      - OOM

The low memory usage shall be partly attributed to InsNet's Padding-free Dynamic Batching feature.

We will illustrate InsNet's efficient memory management in detail in the future.
