#ifndef N3LDG_PLUS_MUL_H
#define N3LDG_PLUS_MUL_H

#include "n3ldg-plus/computation-graph/graph.h"

namespace n3ldg_plus {

Node *pointwiseMultiply(Graph &graph, Node &a, Node &b);

BatchedNode *pointwiseMultiply(Graph &graph, BatchedNode &a, BatchedNode &b);

}

#endif
