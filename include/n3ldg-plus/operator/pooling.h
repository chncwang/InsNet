#ifndef N3LDG_PLUS_POOLING_H
#define N3LDG_PLUS_POOLING_H

#include "n3ldg-plus/computation-graph/graph.h"

namespace n3ldg_plus {

Node *maxPool(std::vector<Node *> &inputs);

Node *minPool(std::vector<Node *> &inputs);

Node *sumPool(std::vector<Node *> &inputs);

Node *avgPool(std::vector<Node *> &inputs);

Node *avgPool(Node *input, int row);

}

#endif
