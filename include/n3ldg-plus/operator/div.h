#ifndef N3LDG_PLUS_DIV_H
#define N3LDG_PLUS_DIV_H

#include "n3ldg-plus/computation-graph/graph.h"

namespace n3ldg_plus {

Node *fullDiv(Graph &graph, Node &numerator, Node &denominator);

BatchedNode *fullDiv(Graph &graph, BatchedNode &numerator, BatchedNode &denominator);

}

#endif
