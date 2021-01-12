#ifndef N3LDG_PLUS_SOFTMAX_NODE
#define N3LDG_PLUS_SOFTMAX_NODE

#include "Layer.h"
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
    Node *scalar_to_vector = scalarToVector(graph, input_row, *max_scalar);
    Node *subtracted = sub(graph, input, *scalar_to_vector);
    return subtracted;
}

Node* softmax(Graph &graph, Node &input, int input_col) {
    using namespace n3ldg_plus;
    Node *subtracted = minusMaxScalar(graph, input, input_col);
    Node *exp = n3ldg_plus::exp(graph, *subtracted);
    Node *sum = vectorSum(graph, *exp, input_col);
    int input_row = input.getDim() / input_col;
    sum = scalarToVector(graph, input_row, *sum);
    Node *div = n3ldg_plus::fullDiv(graph, *exp, *sum);
    return div;
}

};

#endif
