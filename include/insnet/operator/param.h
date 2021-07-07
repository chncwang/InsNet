#ifndef INSNET_OP_PARAM_H
#define INSNET_OP_PARAM_H

#include "insnet/computation-graph/graph.h"
#include "insnet/param/base-param.h"

namespace insnet {

Node* param(Graph &graph, BaseParam &param);

}

#endif
