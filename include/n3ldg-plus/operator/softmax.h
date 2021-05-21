#ifndef N3LDG_PLUS_SOFTMAX_NODE
#define N3LDG_PLUS_SOFTMAX_NODE

#include <n3ldg-plus/computation-graph/graph.h>

namespace n3ldg_plus {

Node* softmax(Node &input, int row);

inline Node* softmax(Node &input) {
    return softmax(input, input.size());
}

BatchedNode* softmax(BatchedNode &input, int row);

inline BatchedNode* softmax(BatchedNode &input) {
    return softmax(input, input.size());
}

}

#endif
