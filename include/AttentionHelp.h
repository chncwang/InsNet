#ifndef ATTENTION_HELP
#define ATTENTION_HELP

/*
*  AttentionHelp.h:
*  attention softmax help nodes
*
*  Created on: Apr 22, 2017
*      Author: mszhang
*/

#include "MyLib.h"
#include "Node.h"
#include "Graph.h"
#include "SoftMaxNode.h"
#include "Concat.h"
#include "Split.h"
#include "PMultiOP.h"
#include "PAddOP.h"
#include "Pooling.h"
#include "MatrixNode.h"
#include <memory>

namespace n3ldg_plus {

AtomicNode* attention(Graph &graph, vector<AtomicNode *>& inputs, vector<AtomicNode *>& scores) {
    using namespace n3ldg_plus;

    if (inputs.empty() || inputs.size() != scores.size()) {
        cerr << boost::format("inputs size:%1% scores:%2%") % inputs.size() % scores.size() <<
            endl;
        abort();
    }

    int input_dim = inputs.front()->getDim();
    for (int i = 1; i < inputs.size(); ++i) {
        if (input_dim != inputs.at(i)->getDim()) {
            cerr << boost::format("dim1:%1% dim2:%2%") % input_dim % inputs.at(i)->getDim() <<
                endl;
            abort();
        }
    }

    for (AtomicNode *score : scores) {
        if (score->getDim() != 1) {
            cerr << "score dim:" << score->getDim() << endl;
            abort();
        }
    }

    AtomicNode *concated = scalarConcat(graph, scores);

    AtomicNode *softmax = n3ldg_plus::softmax(graph, *concated, 1);

    vector<AtomicNode*> splitted_vector;
    for (int i = 0; i < scores.size(); ++i) {
        AtomicNode * split = n3ldg_plus::split(graph, 1, *softmax, i);
        splitted_vector.push_back(scalarToVector(graph, input_dim, *split));
    }

    if (splitted_vector.size() != inputs.size()) {
        cerr << "splitted_vector and inputs size not equal" << endl;
        abort();
    }

    vector<AtomicNode*> multiplied_nodes;
    for (int i = 0; i < inputs.size(); ++i) {
        AtomicNode *node = pointwiseMultiply(graph, *inputs.at(i), *splitted_vector.at(i));
        multiplied_nodes.push_back(node);
    }

    AtomicNode *result = n3ldg_plus::sumPool(graph, multiplied_nodes);

    return result;
}

}

#endif
