Benchmarks
==============================================

This topic will conduct the benchmarks (`Source Code <https://github.com/chncwang/n3ldg-plus-benchmark>`_) to measure N3LDG++'s training speed and memory usage on Transformer-based seq2seq models, respectively, with various model size settings.

Dataset
---------

We use an open-domain conversation dataset (`paper <https://arxiv.org/pdf/1503.02364.pdf>`_) preprocessed with BPE, with sentence length statistics as follows:

.. list-table::
    :widths: 5 10
    :header-rows: 1

    * -
      - Length (mean and standard deviation)
    * - source sentence
      - :math:`14.8\pm4.8`
    * - target sentence
      - :math:`11.2\pm4.3`

In addition to BPE, We also set words appearing less than 1000 times as *UNK*, resulting in a vocabulary of size 8435.

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
      - 1024
      - 4096
      - 16
      - 0.1

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

Results
--------

Training Throughput on GPU
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Transformer Training Throughput on GPU (tokens per second)
    :widths: 3 3 2
    :header-rows: 1

    * - batch_size
      - BASE
      - LARGE
    * - 500
      - 14886
      - 3178
    * - 1000
      - 17804
      - 3517
    * - 1500
      - 18450
      - 3657
    * - 2000
      - 18745
      - 3711
    * - 2500
      - 18965
      - 3753
    * - 3000
      - 19256
      - 3807
    * - 4500
      - 19508
      - OOM
    * - 6000
      - 19502
      - OOM

Note that the training throughput is tested on our laptop, and it should be much higher on average machines for deep learning training.

Memory Usage on GPU
^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Transformer Memory Usage on GPU (MB)
    :widths: 3 3 2
    :header-rows: 1

    * - batch_size
      - BASE
      - LARGE
    * - 500
      - 843
      - 3475
    * - 1000
      - 997
      - 3751
    * - 1500
      - 967
      - 3667
    * - 2000
      - 1329
      - 4259
    * - 2500
      - 1325
      - 4507
    * - 3000
      - 1333
      - 4523
    * - 4500
      - 2177
      - OOM
    * - 6000
      - 2077
      - OOM

The low memory usage is due to N3LDG++'s JIT Dynamic Batching mechanism which does not require paddings.

We will illustrate N3LDG++'s efficient memory management in detail in the future.
