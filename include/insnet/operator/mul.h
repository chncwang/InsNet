#ifndef INSNET_MUL_H
#define INSNET_MUL_H

#include "insnet/computation-graph/graph.h"

namespace insnet {

/// \ingroup operator
/// Pointwise multiplication. \f$[{a_0}{b_0}, {a_1}{b_1}, ..., {a_n}{b_n}]\f$
///
/// **The operators with equal a.size()(also b.size()) will be executed in batch.**
/// For example, in RNN networks, the pointwise multiplication of the forget gate and the last hidden state in the same mini-batch or beam search will be executed in batch.
/// \param a The first input tensor.
/// \param b The second input tensor. Of course they are exchangable.
/// \return The result tensor. Its size is equal to both *a.size()* and *b.size()*.
Node *mul(Node &a, Node &b);

}

#endif
