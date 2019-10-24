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

struct AdditiveAttentionParams : public N3LDGSerializable, TunableCombination<BaseParam>
#if USE_GPU
, public TransferableComponents
#endif
{
    UniParams k, q, w3t;

    AdditiveAttentionParams(const string &name) : k(name + "-k"), q(name + "-q"),
    w3t(name + "-w3t") {}

    Json::Value toJson() const override {
        Json::Value json;
        json["k"] = k.toJson();
        json["q"] = q.toJson();
        json["w3t"] = w3t.toJson();
        return json;
    }

    void fromJson(const Json::Value &json) override {
        k.fromJson(json["k"]);
        q.fromJson(json["q"]);
        w3t.fromJson(json["w3t"]);
    }

    void init(int k_size, int q_size) {
        int out = std::max(k_size, q_size);
        k.init(out, k_size, false);
        q.init(out, q_size, false);
        w3t.init(1, out, false);
    }

#if USE_GPU
    std::vector<Transferable *> transferablePtrs() override {
        return {&k, &q, &w3t};
    }

    virtual std::string name() const {
        return "AdditiveAttention";
    }
#endif

protected:
    std::vector<Tunable<BaseParam> *> tunableComponents() override {
        return {&k, &q, &w3t};
    }
};

class AdditiveAttentionBuilder {
public:
    vector<Node *> _weights;
    Node* _hidden;

    void forward(Graph &graph, AdditiveAttentionParams &params, vector<Node *>& values,
            Node& guide) {
        using namespace n3ldg_plus;
        if (values.empty()) {
            std::cerr << "empty inputs for attention operation" << std::endl;
            abort();
        }

        Node *q = linear(graph, params.q, guide);

        for (int idx = 0; idx < values.size(); idx++) {
            Node *k = linear(graph, params.k, *values.at(idx));
            Node *sum = add(graph, {k, q});
            Node *nonlinear = tanh(graph, *sum);
            Node *w = linear(graph, params.w3t, *nonlinear);
            _weights.push_back(w);
        }

        _hidden = attention(graph, values, _weights);
    }
};

#endif
