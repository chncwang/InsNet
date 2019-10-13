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

Node* softmax(Graph &graph, Node &input) {
    int dim = input.getDim();

    MaxScalarNode *max_scalar = new MaxScalarNode();
    max_scalar->initAsScalar();
    max_scalar->forward(graph, input);

    ScalarToVectorNode *scalar_to_vector = new ScalarToVectorNode;
    scalar_to_vector->init(dim);
    scalar_to_vector->forward(graph, *max_scalar);

    SubNode *subtracted = new SubNode;
    subtracted->init(dim);
    subtracted->forward(graph, input, *scalar_to_vector);

    ExpNode *exp = new ExpNode;
    exp->init(dim);
    exp->forward(graph, *subtracted);

    SumNode *sum = new SumNode;
    sum->initAsScalar();
    sum->forward(graph, *exp);

    DivNode *div = new DivNode;
    div->init(dim);
    div->forward(graph, *exp, *sum);

    return div;
}

};

#endif
