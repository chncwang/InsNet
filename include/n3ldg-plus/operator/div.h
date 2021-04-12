#ifndef N3LDG_PLUS_DIV_H
#define N3LDG_PLUS_DIV_H

#include "n3ldg-plus/computation-graph/graph.h"

namespace n3ldg_plus {

Node *fullDiv(Node &numerator, Node &denominator);

BatchedNode *fullDiv(BatchedNode &numerator, BatchedNode &denominator);

}

#endif
