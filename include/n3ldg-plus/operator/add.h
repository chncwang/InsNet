#ifndef N3LDG_PLUS_ADD_H
#define N3LDG_PLUS_ADD_H

#include "n3ldg-plus/computation-graph/graph.h"

namespace n3ldg_plus {

Node *add(Graph &graph, const ::std::vector<Node*> &inputs);

BatchedNode *addInBatch(Graph &graph, const ::std::vector<BatchedNode *> &inputs);

}

#endif
