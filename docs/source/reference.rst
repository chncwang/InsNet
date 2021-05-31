APIs
=================

Operators
----------

.. doxygenfunction:: add
.. doxygenfunction:: affine
.. doxygenfunction:: avgPool
.. doxygenfunction:: bias
.. doxygenfunction:: cat(const std::vector<Node*> &, int col = 1)
.. doxygenfunction:: div
.. doxygenfunction:: dropout
.. doxygenfunction:: embedding(Graph &, const std::vector<std::string> &, EmbeddingAbs &, bool freeze = false)
.. doxygenfunction:: embedding(Graph &, const std::string &, EmbeddingAbs &, bool freeze = false)
.. doxygenfunction:: embedding(Graph &, const std::vector<int> &, BaseParam &, bool freeze = false)
.. doxygenfunction:: embedding(Graph &, int, BaseParam &, bool freeze = false)
.. doxygenfunction:: exp
.. doxygenfunction:: expandColumnwisely
.. doxygenfunction:: expandRowwisely
.. doxygenfunction:: layerNorm(Node &, int)
.. doxygenfunction:: layerNorm(Node &, LayerNormParams &)
.. doxygenfunction:: linear(Node &, LinearParams &)
.. doxygenfunction:: linear(Node &, Param &)
.. doxygenfunction:: matmul
.. doxygenfunction:: max
.. doxygenfunction:: maxPool
.. doxygenfunction:: minPool
.. doxygenfunction:: mul(Node &, dtype)
.. doxygenfunction:: mul(Node &, Node &)
.. doxygenfunction:: relu
.. doxygenfunction:: sigmoid
.. doxygenfunction:: softmax(Node &, int)
.. doxygenfunction:: softmax(Node &)
.. doxygenfunction:: split(Node &, int, int, int input_col = 1)
.. doxygenfunction:: sqrt
.. doxygenfunction:: sum
.. doxygenfunction:: sumPool
.. doxygenfunction:: tanh
.. doxygenfunction:: tensor(Graph &, const std::vector<dtype> &)
.. doxygenfunction:: tensor(Graph &, int, dtype)
