#ifndef N3LDG_PLUS_SUB_H
#define N3LDG_PLUS_SUB_H

#include "n3ldg-plus/computation-graph/graph.h"

namespace n3ldg_plus {

Node *sub(Graph &graph, Node &minuend, Node &subtrahend);

BatchedNode *sub(Graph &graph, BatchedNode &minuend, BatchedNode &subtrahend);

}

#endif
