#ifndef N3LDG_PLUS_BUCKET_H
#define N3LDG_PLUS_BUCKET_H

#include "n3ldg-plus/computation-graph/graph.h"

namespace n3ldg_plus {

Node *tensor(Graph &graph, int dim, float v);

Node *tensor(Graph &graph, const std::vector<dtype> &v);

}

#endif
