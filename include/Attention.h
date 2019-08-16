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
#include "BiOP.h"
#include "UniOP.h"
#include "Graph.h"
#include "AttentionHelp.h"
#include <memory>
#include <boost/format.hpp>

struct DotAttentionParams : public N3LDGSerializable
#if USE_GPU
, public TransferableComponents
#endif
{
    BiParams bi_atten;
    int hidden_dim;
    int guide_dim;

    DotAttentionParams() = default;

    void exportAdaParams(ModelUpdate& ada) {
        bi_atten.exportAdaParams(ada);
    }

    void init(int nHidden, int nGuide) {
        bi_atten.init(1, nHidden, nGuide, false);
        hidden_dim = nHidden;
        guide_dim = nGuide;
    }

    Json::Value toJson() const override {
        Json::Value json;
        json["bi_atten"] = bi_atten.toJson();
        json["hidden_dim"] = hidden_dim;
        json["guide_dim"] = guide_dim;
        return json;
    }

    void fromJson(const Json::Value &json) override {
        bi_atten.fromJson(json["bi_atten"]);
        hidden_dim = json["hidden_dim"].asInt();
        guide_dim = json["guide_dim"].asInt();
    }

#if USE_GPU
    std::vector<Transferable *> transferablePtrs() override {
        return {&bi_atten};
    }

    virtual std::string name() const {
        return "AttentionParams";
    }
#endif
};

class DotAttentionBuilder {
public:
    vector<BiNode *> _weights;
    AttentionSoftMaxNode* _hidden = new AttentionSoftMaxNode;

    DotAttentionParams* _param = nullptr;

    void init(DotAttentionParams &paramInit) {
        _param = &paramInit;
        _hidden->init(paramInit.hidden_dim);
    }

    void forward(Graph &cg, vector<Node *>& x, Node& guide) {
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
            BiNode* intermediate_node(new BiNode);
            intermediate_node->init(1);
            intermediate_node->setParam(_param->bi_atten);
            intermediate_node->forward(cg, *x.at(idx), guide);
            _weights.push_back(intermediate_node);
        }
        vector<Node *> weights = toNodePointers<BiNode>(_weights);
        _hidden->forward(cg, x, weights);
    }
};

struct AttentionParams : public N3LDGSerializable
#if USE_GPU
, public TransferableComponents
#endif
{
    BiParams bi_atten;
    UniParams to_scalar_params;
    int hidden_dim;
    int guide_dim;

    AttentionParams() = default;

    void exportAdaParams(ModelUpdate& ada) {
        bi_atten.exportAdaParams(ada);
    }

    void init(int nHidden, int nGuide) {
        bi_atten.init(nHidden + nGuide, nHidden, nGuide, true);
        to_scalar_params.init(1, nHidden + nGuide, false);
        hidden_dim = nHidden;
        guide_dim = nGuide;
    }

    Json::Value toJson() const override {
        Json::Value json;
        json["bi_atten"] = bi_atten.toJson();
        json["to_scalar_params"] = to_scalar_params.toJson();
        json["hidden_dim"] = hidden_dim;
        json["guide_dim"] = guide_dim;
        return json;
    }

    void fromJson(const Json::Value &json) override {
        bi_atten.fromJson(json["bi_atten"]);
        to_scalar_params.fromJson(json["to_scalar_params"]);
        hidden_dim = json["hidden_dim"].asInt();
        guide_dim = json["guide_dim"].asInt();
    }

#if USE_GPU
    std::vector<Transferable *> transferablePtrs() override {
        return {&bi_atten, &to_scalar_params};
    }

    virtual std::string name() const {
        return "AttentionParams";
    }
#endif
};

class AttentionBuilder {
public:
    vector<LinearNode *> _weights;
    vector<BiNode *> _intermediate_nodes;
    AttentionSoftMaxNode* _hidden = new AttentionSoftMaxNode;

    AttentionParams* _param = nullptr;

    void init(AttentionParams &paramInit) {
        _param = &paramInit;
        _hidden->init(paramInit.hidden_dim);
    }

    void forward(Graph &cg, vector<Node *>& x, Node& guide) {
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
            BiNode* intermediate_node(new BiNode);
            intermediate_node->setParam(_param->bi_atten);
            intermediate_node->init(_param->guide_dim + _param->hidden_dim);
            intermediate_node->forward(cg, *x.at(idx), guide);
            _intermediate_nodes.push_back(intermediate_node);

            LinearNode* uni_node(new LinearNode);
            uni_node->setParam(_param->to_scalar_params);
            uni_node->init(1);
            uni_node->forward(cg, *intermediate_node);
            _weights.push_back(uni_node);
        }
        vector<Node *> weights = toNodePointers<LinearNode>(_weights);
        _hidden->forward(cg, x, weights);
    }
};



struct AttentionVParams : public N3LDGSerializable
#if USE_GPU
, public TransferableComponents
#endif
{
    BiParams bi_atten;
    int hidden_dim;
    int guide_dim;

    AttentionVParams() = default;

    void exportAdaParams(ModelUpdate& ada) {
        bi_atten.exportAdaParams(ada);
    }

    void init(int nHidden, int nGuide) {
        bi_atten.init(nHidden, nHidden, nGuide, false);
        hidden_dim = nHidden;
        guide_dim = nGuide;
    }

    Json::Value toJson() const override {
        Json::Value json;
        json["bi_atten"] = bi_atten.toJson();
        json["hidden_dim"] = hidden_dim;
        json["guide_dim"] = guide_dim;
        return json;
    }

    void fromJson(const Json::Value &json) override {
        bi_atten.fromJson(json["bi_atten"]);
        hidden_dim = json["hidden_dim"].asInt();
        guide_dim = json["guide_dim"].asInt();
    }

#if USE_GPU
    std::vector<Transferable *> transferablePtrs() override {
        return {&bi_atten};
    }

    virtual std::string name() const {
        return "AttentionVParams";
    }
#endif
};

class AttentionVBuilder {
public:
    vector<BiNode *> _weights;
    AttentionSoftMaxVNode* _hidden = new AttentionSoftMaxVNode;
    AttentionVParams* _param;

    void init(AttentionVParams& paramInit) {
        _param = &paramInit;
        _hidden->init(paramInit.hidden_dim);
    }

    void forward(Graph &cg, const vector<Node *>& x, Node &guide) {
        if (x.size() == 0) {
            std::cerr << "empty inputs for lstm operation" << std::endl;
            abort();
        }
        if (x.at(0)->getDim() != _param->hidden_dim || guide.getDim() != _param->guide_dim) {
            std::cerr << "input dim does not match for attention  operation" << std::endl;
            std::cerr << boost::format("input dim:%1% hidden dim:%2% guide.getDim():%3% param->guide_dim:%4%") % x.at(0)->getDim() % _param->hidden_dim % guide.getDim() % _param->guide_dim << std::endl;
            abort();
        }

        for (int i = 0; i < x.size(); ++i) {
            BiNode *weight_node = new BiNode;
            weight_node->setParam(_param->bi_atten);
            weight_node->init(_param->hidden_dim);
            weight_node->forward(cg, *x.at(i), guide);
            _weights.push_back(weight_node);
        }

        vector<Node*> weights = toNodePointers<BiNode>(_weights);
        _hidden->forward(cg, x, weights);
    }
};


struct SelfAttentionParams {
    UniParams uni_atten;
    int hidden_dim;

    SelfAttentionParams() {
    }

    void exportAdaParams(ModelUpdate& ada) {
        uni_atten.exportAdaParams(ada);
    }

    void init(int nHidden) {
        uni_atten.init(1, nHidden, false);
        hidden_dim = nHidden;
    }
};

class SelfAttentionBuilder {
public:
    int _nSize;
    int _nHiddenDim;

    vector<UniNode> _weights;
    AttentionSoftMaxNode _hidden;

    SelfAttentionParams* _param;

    void init(SelfAttentionParams* paramInit) {
        _param = paramInit;
        _nHiddenDim = _param->hidden_dim;

        int maxsize = _weights.size();
        for (int idx = 0; idx < maxsize; idx++) {
            _weights.at(idx).setParam(_param->uni_atten);
            _weights.at(idx).init(1);
        }
        _hidden.init(_nHiddenDim);
    }

    void forward(Graph &cg, vector<Node *>& x) {
        if (x.size() == 0) {
            std::cout << "empty inputs for lstm operation" << std::endl;
            return;
        }
        _nSize = x.size();
        if (x.at(0)->getDim() != _nHiddenDim) {
            std::cout << "input dim does not match for attention  operation" << std::endl;
            return;
        }

        vector<Node *> aligns;
        for (int idx = 0; idx < _nSize; idx++) {
            _weights.at(idx).forward(cg, *x.at(idx));
            aligns.push_back(&_weights.at(idx));
        }

        _hidden.forward(cg, x, aligns);
    }
};



struct SelfAttentionVParams {
    UniParams uni_atten;
    int hidden_dim;

    SelfAttentionVParams() {
    }

    void exportAdaParams(ModelUpdate& ada) {
        uni_atten.exportAdaParams(ada);
    }

    void init(int nHidden) {
        uni_atten.init(nHidden, nHidden, false);
        hidden_dim = nHidden;
    }
};

class SelfAttentionVBuilder {
public:
    int _nSize;
    int _nHiddenDim;

    vector<UniNode> _weights;
    AttentionSoftMaxVNode _hidden;
    SelfAttentionVParams* _param;

    void init(SelfAttentionVParams* paramInit) {
        _param = paramInit;
        _nHiddenDim = _param->hidden_dim;

        int maxsize = _weights.size();
        for (int idx = 0; idx < maxsize; idx++) {
            _weights.at(idx).setParam(_param->uni_atten);
            _weights.at(idx).init(_nHiddenDim);
        }
        _hidden.init(_nHiddenDim);
    }

    void forward(Graph &cg, const vector<Node *>& x) {
        if (x.size() == 0) {
            std::cout << "empty inputs for lstm operation" << std::endl;
            return;
        }
        _nSize = x.size();
        if (x.at(0)->getDim() != _nHiddenDim) {
            std::cout << "input dim does not match for attention  operation" << std::endl;
            return;
        }

        vector<Node *> aligns;
        for (int idx = 0; idx < _nSize; idx++) {
            _weights.at(idx).forward(cg, *x.at(idx));
            aligns.push_back(&_weights.at(idx));
        }
        _hidden.forward(cg, x, aligns);
    }
};

#endif
