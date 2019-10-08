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

    void forward(Graph &cg, vector<Node *>& x, Node& guide) {
        using namespace n3ldg_plus;
        if (x.empty()) {
            std::cerr << "empty inputs for attention operation" << std::endl;
            abort();
        }

        for (int idx = 0; idx < x.size(); idx++) {
            Node *pro = n3ldg_plus::pointwiseMultiply(cg, *x.at(idx), guide);
            Node *sum = n3ldg_plus::vectorSum(cg, *pro);
            _weights.push_back(sum);
        }

        _hidden = attention(cg, x, _weights);
    }
};

#endif
