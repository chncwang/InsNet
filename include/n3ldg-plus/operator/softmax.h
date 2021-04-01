#ifndef N3LDG_PLUS_SOFTMAX_NODE
#define N3LDG_PLUS_SOFTMAX_NODE

#include <n3ldg-plus/computation-graph/graph.h>

namespace n3ldg_plus {

Node* softmax(Graph &graph, Node &input, int input_col);

BatchedNode* softmax(Graph &graph, BatchedNode &input, int col = 1);

}

#endif
