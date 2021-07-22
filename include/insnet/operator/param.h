#ifndef INSNET_OP_PARAM_H
#define INSNET_OP_PARAM_H

#include "insnet/computation-graph/graph.h"
#include "insnet/param/base-param.h"

namespace insnet {

/// \ingroup operator
/// Copy the parameters to the *Node* object.
///
/// **The param operator should be used only once in a computation graph.**
/// \param graph The computation graph.
/// \param param The parameters.
/// \Return The result tensor. Its size is equal to param.size().
Node* param(Graph &graph, BaseParam &param);

}

#endif
