APIs
=================

There are several principles for API design as follows:

1. Tensor dimensions are no greater than 2, i.e., they are all column-major matrices.
2. The shape information is not kept in and broadcasted among Node objects but only coupled with operators, while Node objects only keep their sizes. For example, shapes of *matmul*'s input matrix A and B are inferred from A.size(), B.size() and b_row, while methods such as *add* do not care shapes, but only sizes. So we do not need shape-related methods such as reshape and view.
3. Batch rules are designed to be ignored by users.
4. Operators and modules are all functions.
5. We do not invent but borrow function names from PyTorch, except for those not included in PyTorch but implemented in InsNet for strong reasons.

Operators
----------

.. doxygenfunction:: add
.. doxygenfunction:: affine
.. doxygenfunction:: argmax
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
.. doxygenfunction:: sub
.. doxygenfunction:: sum
.. doxygenfunction:: sumPool
.. doxygenfunction:: tanh
.. doxygenfunction:: tensor(Graph &, const std::vector<dtype> &)
.. doxygenfunction:: tensor(Graph &, int, dtype)

Modules
---------------------

.. doxygenfunction:: multiheadAttention
.. doxygenfunction:: gru(Node &, Node &, GRUParams &, dtype)
.. doxygenfunction:: gru(Node &, const std::vector<Node *> &, GRUParams &, dtype)
.. doxygenfunction:: lstm(LSTMState &, Node &, LSTMParams &, dtype)
.. doxygenfunction:: lstm(LSTMState &initial_state, const std::vector<Node *> &, LSTMParams &, dtype)
.. doxygenfunction:: transformerDecoder(Node &, Node &, TransformerDecoderParams &, dtype)
.. doxygenfunction:: transformerDecoder(TransformerDecoderState &, const std::vector<Node*> &, const std::vector<Node*> &, Node &, TransformerDecoderParams &, dtype);
.. doxygenfunction:: transformerEncoder

Loss Functions
---------
.. doxygenfunction:: NLLLoss
