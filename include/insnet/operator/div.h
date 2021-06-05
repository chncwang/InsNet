#ifndef INSNET_DIV_H
#define INSNET_DIV_H

#include "insnet/computation-graph/graph.h"

namespace insnet {

/// The pointwise div operator.
///
/// For example, div([0.1, 0.2], [0.1, 0.2]) will return [1, 1].
///
/// **All div operators will be executed in batch**.
/// \param dividend The dividend number. Its size should be equal to *divisor.size()*.
/// \param divisor The divisor number. Its size should be equal to *dividend.size()*.
/// \return The result tensor. Its size is equal to *dividend.size()* and *divisor.size()*.
Node *div(Node &dividend, Node &divisor);

}

#endif
