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
#include <memory>
#include <boost/format.hpp>

struct DotAttentionParams : public N3LDGSerializable, public TunableCombination<BaseParam>
#if USE_GPU
, public TransferableComponents
#endif
{
    UniParams uni1;
    UniParams uni2;
    int hidden_dim;
    int guide_dim;

    DotAttentionParams(const string &name) : uni1(name + "-hidden"), uni2(name + "-guide") {}

    void init(int nHidden, int nGuide) {
        uni1.init(1, nHidden, false);
        uni2.init(1, nGuide, false);
        hidden_dim = nHidden;
        guide_dim = nGuide;
    }

    Json::Value toJson() const override {
        Json::Value json;
        json["uni1"] = uni1.toJson();
        json["uni2"] = uni2.toJson();
        json["hidden_dim"] = hidden_dim;
        json["guide_dim"] = guide_dim;
        return json;
    }

    void fromJson(const Json::Value &json) override {
        uni1.fromJson(json["uni1"]);
        uni2.fromJson(json["uni2"]);
        hidden_dim = json["hidden_dim"].asInt();
        guide_dim = json["guide_dim"].asInt();
    }

#if USE_GPU
    std::vector<Transferable *> transferablePtrs() override {
        return {&uni1, &uni2};
    }

    virtual std::string name() const {
        return "AttentionParams";
    }
#endif

protected:
    std::vector<Tunable<BaseParam>*> tunableComponents() override {
        return {&uni1, &uni2};
    }
};

class DotAttentionBuilder {
public:
    vector<Node *> _weights;
    Node* _hidden;

    DotAttentionParams* _param = nullptr;

    void init(DotAttentionParams &paramInit) {
        _param = &paramInit;
    }

    void forward(Graph &cg, vector<Node *>& x, Node& guide) {
        using namespace n3ldg_plus;
        if (x.size() == 0) {
            std::cerr << "empty inputs for lstm operation" << std::endl;
            abort();
        }

        if (x.at(0)->getDim() != _param->hidden_dim || guide.getDim() != _param->guide_dim) {
            std::cerr << "input dim does not match for attention  operation" << std::endl;
            cerr << boost::format("x.at(0)->dim:%1%, _param->hidden_dim:%2% guide.dim:%3% _param->guide_dim:%4%") % x.at(0)->getDim() % _param->hidden_dim % guide.getDim() % _param->guide_dim << endl;
            abort();
        }

        for (int idx = 0; idx < x.size(); idx++) {
            Node *uni1 = n3ldg_plus::uni(cg, 1, _param->uni1, *x.at(idx));
            Node *uni2 = n3ldg_plus::uni(cg, 1, _param->uni2, guide);

            _weights.push_back(n3ldg_plus::add(cg, {uni1, uni2}));
        }
        _hidden = attention(cg, x, _weights);
    }
};

#endif
