#ifndef N3LDG_PLUS_BUCKET_H
#define N3LDG_PLUS_BUCKET_H

#include "n3ldg-plus/computation-graph/graph.h"

namespace n3ldg_plus {

Node *bucket(Graph &graph, int dim, float v);

Node *bucket(Graph &graph, const ::std::vector<dtype> &v);

BatchedNode *bucket(Graph &graph, int batch_size, const ::std::vector<dtype> &v);

BatchedNode *bucket(Graph &graph, int dim, int batch_size, dtype v);

}

#endif
