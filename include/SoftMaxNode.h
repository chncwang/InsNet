#ifndef N3LDG_PLUS_SOFTMAX_NODE
#define N3LDG_PLUS_SOFTMAX_NODE

#include "AtomicOP.h"
#include "Sub.h"
#include "AtomicOP.h"
#include "Div.h"
#include "Split.h"

#include <boost/format.hpp>

namespace n3ldg_plus {

Node *minusMaxScalar(Graph &graph, Node &input, int input_col) {
    using namespace n3ldg_plus;

    Node *max_scalar = maxScalar(graph, input, input_col);
    int input_row = input.getDim() / input_col;
    Node *scalar_to_vector = scalarToVector(graph, *max_scalar, input_row);
    Node *subtracted = sub(graph, input, *scalar_to_vector);
    return subtracted;
}

BatchedNode *minusMaxScalar(Graph &graph, BatchedNode &input, int input_col) {
    using namespace n3ldg_plus;

    BatchedNode *max_scalar = maxScalar(graph, input, input_col);
    vector<int> input_rows;
    for (int dim : input.getDims()) {
        input_rows.push_back(dim / input_col);
    }
    BatchedNode *scalar_to_vector = scalarToVector(graph, *max_scalar, input_rows);
    BatchedNode *subtracted = sub(graph, input, *scalar_to_vector);
    return subtracted;
}

Node* softmax(Graph &graph, Node &input, int input_col) {
    using namespace n3ldg_plus;
    Node *subtracted = minusMaxScalar(graph, input, input_col);
    Node *exp = n3ldg_plus::exp(graph, *subtracted);
    Node *sum = vectorSum(graph, *exp, input_col);
    int input_row = input.getDim() / input_col;
    sum = scalarToVector(graph, *sum, input_row);
    Node *div = n3ldg_plus::fullDiv(graph, *exp, *sum);
    return div;
}

BatchedNode* softmax(Graph &graph, BatchedNode &input, int input_col) {
    using namespace n3ldg_plus;
    BatchedNode *subtracted = minusMaxScalar(graph, input, input_col);
    BatchedNode *exp = n3ldg_plus::exp(graph, *subtracted);
    BatchedNode *sum = vectorSum(graph, *exp, input_col);
    vector<int> input_rows;
    for (int dim : input.getDims()) {
        input_rows.push_back(dim / input_col);
    }
    sum = scalarToVector(graph, *sum, input_rows);
    BatchedNode *div = n3ldg_plus::fullDiv(graph, *exp, *sum);
    return div;
}

};

#endif
