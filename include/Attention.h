#ifndef ATTENTION_BUILDER
#define ATTENTION_BUILDER

/*
*  Attention.h:
*  a set of attention builders
*
*  Created on: Apr 22, 2017
*      Author: mszhang
*/

#include "MyLib.h"
#include "Node.h"
#include "UniOP.h"
#include "Graph.h"
#include "AttentionHelp.h"
#include "AtomicOP.h"
#include <memory>
#include <boost/format.hpp>

class DotAttentionBuilder {
public:
    vector<Node *> _weights;
    Node* _hidden;

    void forward(Graph &cg, vector<Node*> &keys, vector<Node *>& values, Node& guide) {
        using namespace n3ldg_plus;
        if (values.empty()) {
            std::cerr << "empty inputs for attention operation" << std::endl;
            abort();
        }

        for (int idx = 0; idx < values.size(); idx++) {
            Node *pro = n3ldg_plus::pointwiseMultiply(cg, *keys.at(idx), guide);
            Node *sum = n3ldg_plus::vectorSum(cg, *pro);
            _weights.push_back(sum);
        }

        _hidden = attention(cg, values, _weights);
    }
};

#endif
