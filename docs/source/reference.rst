Manual Reference
=================

Operators
----------

.. doxygenfunction:: add
.. doxygenfunction:: affine
.. doxygenfunction:: bias
.. doxygenfunction:: cat(const std::vector<Node*> &, int col = 1)
.. doxygenfunction:: div
.. doxygenfunction:: dropout
.. doxygenfunction:: embedding(Graph &, const std::vector<std::string> &, EmbeddingAbs &, bool freeze = false)
.. doxygenfunction:: embedding(Graph &, const std::string &, EmbeddingAbs &, bool freeze = false)
.. doxygenfunction:: embedding(Graph &, const std::vector<int> &, BaseParam &, bool freeze = false)
.. doxygenfunction:: embedding(Graph &, int, BaseParam &param, bool freeze = false)
.. doxygenfunction:: exp
.. doxygenfunction:: expandColumnwisely
.. doxygenfunction:: expandRowwisely
.. doxygenfunction:: layerNorm(Node &, int)
.. doxygenfunction:: layerNorm(Node &, LayerNormParams &)
.. doxygenfunction:: linear(Node &, LinearParams &)
.. doxygenfunction:: linear(Node &, Param &)
.. doxygenfunction:: matmul
.. doxygenfunction:: max
.. doxygenfunction:: mul(Node &, dtype)
.. doxygenfunction:: mul(Node &, Node &)
.. doxygenfunction:: relu
.. doxygenfunction:: sigmoid
.. doxygenfunction:: sqrt
.. doxygenfunction:: sum
.. doxygenfunction:: tanh
.. doxygenfunction:: tensor(Graph &, const std::vector<dtype> &)
.. doxygenfunction:: tensor(Graph &, int, dtype)
