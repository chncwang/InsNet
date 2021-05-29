#ifndef N3LDG_PLUS_POOLING_H
#define N3LDG_PLUS_POOLING_H

#include "n3ldg-plus/computation-graph/graph.h"

namespace n3ldg_plus {

Node *maxPool(Node &input, int row);

Node *minPool(Node &input, int row);

Node *sumPool(Node &input, int row);

Node *avgPool(Node &input, int row);

}

#endif
