.. N3LDG++ documentation master file, created by
   sphinx-quickstart on Sun Mar 21 22:16:57 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

N3LDG++ documentation
===================================

`N3LDG++ <http://github.com/chncwang/n3ldg-plus>`_  is a powerful neural network library aiming at building flexible NLP models, especially those with different computation graphs for different instances. It is designed to run dynamic models on-the-fly and allow users to focus on building the graph for a single instance, leaving batching (both mini-batch and instance level) to the library's lazy execution. This design has at least three advantages as follows:

1. It makes it super easy to build NLP models with instance-dependent computation graphs, such as tree-LSTM or hierarchical transformers.
2. It reduces users' intellectual burden of manual batching, as N3LDG++ can efficiently take over all batching works. As such, users even need not know the concept of tensors, but only matrices (such as K, V, and Q in transformers) and vectors (which are one-column matrices), neither the concept of paddings.
3. It reduces memory allocation since no padding is needed.

Besides, N3LDG++ has the following features:

1. It is written in C++ 11 and is header-only for CPU computation.
2. For GPU computation, we write almost all CUDA kernels by hand, allowing efficient parallel computation for matrices and vectors of different shapes.
3. Both lazy and eager execution is supported, with the latter facilitating users' debugging.
4. It currently provides more than twenty operations with GPU and CPU implementations, supporting building modern NLP models for sentence classification, sequence tagging, and language generation. It furthermore provides NLP modules such as attention blocks, RNNs, and transformers, built with the aforementioned operations.

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
