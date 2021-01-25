#ifndef N3LDG_PLUS_SOFTMAX_NODE
#define N3LDG_PLUS_SOFTMAX_NODE

#include "AtomicOP.h"
#include "Sub.h"
#include "AtomicOP.h"
#include "Div.h"
#include "Split.h"

#include <boost/format.hpp>

namespace n3ldg_plus {

AtomicNode *minusMaxScalar(Graph &graph, AtomicNode &input, int input_col) {
    using namespace n3ldg_plus;

    AtomicNode *max_scalar = maxScalar(graph, input, input_col);
    int input_row = input.getDim() / input_col;
    AtomicNode *scalar_to_vector = scalarToVector(graph, input_row, *max_scalar);
    AtomicNode *subtracted = sub(graph, input, *scalar_to_vector);
    return subtracted;
}

AtomicNode* softmax(Graph &graph, AtomicNode &input, int input_col) {
    using namespace n3ldg_plus;
    AtomicNode *subtracted = minusMaxScalar(graph, input, input_col);
    AtomicNode *exp = n3ldg_plus::exp(graph, *subtracted);
    AtomicNode *sum = vectorSum(graph, *exp, input_col);
    int input_row = input.getDim() / input_col;
    sum = scalarToVector(graph, input_row, *sum);
    AtomicNode *div = n3ldg_plus::fullDiv(graph, *exp, *sum);
    return div;
}

};

#endif
