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

struct AttentionParams {
    BiParams bi_atten;
    int hidden_dim;
    int guide_dim;

    AttentionParams() {
    }

    inline void exportAdaParams(ModelUpdate& ada) {
        bi_atten.exportAdaParams(ada);
    }

    inline void initial(int nHidden, int nGuide) {
        bi_atten.initial(1, nHidden, nGuide, false);
        hidden_dim = nHidden;
        guide_dim = nGuide;
    }


    inline void save(std::ofstream &os) const {
        bi_atten.save(os);
    }

    inline void load(std::ifstream &is) {
        bi_atten.load(is);
    }

};

class AttentionBuilder {
  public:
    int _nSize;
    int _nHiddenDim;
    int _nGuideDim;

    vector<BiNode> _weights;
    AttentionSoftMaxNode _hidden;

    AttentionParams* _param;

  public:
    AttentionBuilder() {
        clear();
    }

    ~AttentionBuilder() {
        clear();
    }

  public:
    inline void resize(int maxsize) {
        _weights.resize(maxsize);
        _hidden.setParam(maxsize);
    }

    inline void clear() {
        _weights.clear();
    }

  public:
    inline void init(AttentionParams* paramInit) {
        _param = paramInit;
        _nHiddenDim = _param->hidden_dim;
        _nGuideDim = _param->guide_dim;

        int maxsize = _weights.size();
        for (int idx = 0; idx < maxsize; idx++) {
            _weights[idx].setParam(&_param->bi_atten);
            _weights[idx].init(1, -1);
        }
        _hidden.init(_nHiddenDim, -1);
    }

  public:
    inline void forward(Graph *cg, const vector<PNode>& x, PNode guide) {
        if (x.size() == 0) {
            std::cout << "empty inputs for lstm operation" << std::endl;
            return;
        }
        _nSize = x.size();
        if (x[0]->dim != _nHiddenDim || guide->dim != _nGuideDim) {
            std::cout << "input dim does not match for attention  operation" << std::endl;
            return;
        }

        vector<PNode> aligns;
        for (int idx = 0; idx < _nSize; idx++) {
            _weights[idx].forward(cg, x[idx], guide);
            aligns.push_back(&_weights[idx]);
        }

        _hidden.forward(cg, x, aligns);
    }

};



struct AttentionVParams {
    BiParams bi_atten;
    int hidden_dim;
    int guide_dim;

    AttentionVParams() {
    }

    inline void exportAdaParams(ModelUpdate& ada) {
        bi_atten.exportAdaParams(ada);
    }

    inline void initial(int nHidden, int nGuide) {
        bi_atten.initial(nHidden, nHidden, nGuide, false);
        hidden_dim = nHidden;
        guide_dim = nGuide;
    }


    inline void save(std::ofstream &os) const {
        bi_atten.save(os);
    }

    inline void load(std::ifstream &is) {
        bi_atten.load(is);
    }

};

class AttentionVBuilder {
  public:
    int _nSize;
    int _nHiddenDim;
    int _nGuideDim;

    vector<BiNode> _weights;
    AttentionSoftMaxVNode _hidden;
    AttentionVParams* _param;

  public:
    AttentionVBuilder() {
        clear();
    }

    ~AttentionVBuilder() {
        clear();
    }

  public:
    inline void resize(int maxsize) {
        _weights.resize(maxsize);
        _hidden.setParam(maxsize);
    }

    inline void clear() {
        _weights.clear();
    }

  public:
    inline void init(AttentionVParams* paramInit) {
        _param = paramInit;
        _nHiddenDim = _param->hidden_dim;
        _nGuideDim = _param->guide_dim;

        int maxsize = _weights.size();
        for (int idx = 0; idx < maxsize; idx++) {
            _weights[idx].setParam(&_param->bi_atten);
            _weights[idx].init(_nHiddenDim, -1);
        }
        _hidden.init(_nHiddenDim, -1);
    }

  public:
    inline void forward(Graph *cg, const vector<PNode>& x, PNode guide) {
        if (x.size() == 0) {
            std::cout << "empty inputs for lstm operation" << std::endl;
            return;
        }
        _nSize = x.size();
        if (x[0]->dim != _nHiddenDim || guide->dim != _nGuideDim) {
            std::cout << "input dim does not match for attention  operation" << std::endl;
            return;
        }

        vector<PNode> aligns;
        for (int idx = 0; idx < _nSize; idx++) {
            _weights[idx].forward(cg, x[idx], guide);
            aligns.push_back(&_weights[idx]);
        }
        _hidden.forward(cg, x, aligns);
    }
};


struct SelfAttentionParams {
    UniParams uni_atten;
    int hidden_dim;

    SelfAttentionParams() {
    }

    inline void exportAdaParams(ModelUpdate& ada) {
        uni_atten.exportAdaParams(ada);
    }

    inline void initial(int nHidden) {
        uni_atten.initial(1, nHidden, false);
        hidden_dim = nHidden;
    }


    inline void save(std::ofstream &os) const {
        uni_atten.save(os);
    }

    inline void load(std::ifstream &is) {
        uni_atten.load(is);
    }

};

class SelfAttentionBuilder {
  public:
    int _nSize;
    int _nHiddenDim;

    vector<UniNode> _weights;
    AttentionSoftMaxNode _hidden;

    SelfAttentionParams* _param;

  public:
    SelfAttentionBuilder() {
        clear();
    }

    ~SelfAttentionBuilder() {
        clear();
    }

  public:
    inline void resize(int maxsize) {
        _weights.resize(maxsize);
        _hidden.setParam(maxsize);
    }

    inline void clear() {
        _weights.clear();
    }

  public:
    inline void init(SelfAttentionParams* paramInit) {
        _param = paramInit;
        _nHiddenDim = _param->hidden_dim;

        int maxsize = _weights.size();
        for (int idx = 0; idx < maxsize; idx++) {
            _weights[idx].setParam(&_param->uni_atten);
            _weights[idx].init(1, -1);
        }
        _hidden.init(_nHiddenDim, -1);
    }

  public:
    inline void forward(Graph *cg, const vector<PNode>& x) {
        if (x.size() == 0) {
            std::cout << "empty inputs for lstm operation" << std::endl;
            return;
        }
        _nSize = x.size();
        if (x[0]->dim != _nHiddenDim) {
            std::cout << "input dim does not match for attention  operation" << std::endl;
            return;
        }

        vector<PNode> aligns;
        for (int idx = 0; idx < _nSize; idx++) {
            _weights[idx].forward(cg, x[idx]);
            aligns.push_back(&_weights[idx]);
        }

        _hidden.forward(cg, x, aligns);
    }

};



struct SelfAttentionVParams {
    UniParams uni_atten;
    int hidden_dim;

    SelfAttentionVParams() {
    }

    inline void exportAdaParams(ModelUpdate& ada) {
        uni_atten.exportAdaParams(ada);
    }

    inline void initial(int nHidden) {
        uni_atten.initial(nHidden, nHidden, false);
        hidden_dim = nHidden;
    }


    inline void save(std::ofstream &os) const {
        uni_atten.save(os);
    }

    inline void load(std::ifstream &is) {
        uni_atten.load(is);
    }

};

class SelfAttentionVBuilder {
  public:
    int _nSize;
    int _nHiddenDim;

    vector<UniNode> _weights;
    AttentionSoftMaxVNode _hidden;
    SelfAttentionVParams* _param;

  public:
    SelfAttentionVBuilder() {
        clear();
    }

    ~SelfAttentionVBuilder() {
        clear();
    }

  public:
    inline void resize(int maxsize) {
        _weights.resize(maxsize);
        _hidden.setParam(maxsize);
    }

    inline void clear() {
        _weights.clear();
    }

  public:
    inline void init(SelfAttentionVParams* paramInit) {
        _param = paramInit;
        _nHiddenDim = _param->hidden_dim;

        int maxsize = _weights.size();
        for (int idx = 0; idx < maxsize; idx++) {
            _weights[idx].setParam(&_param->uni_atten);
            _weights[idx].init(_nHiddenDim, -1);
        }
        _hidden.init(_nHiddenDim, -1);
    }

  public:
    inline void forward(Graph *cg, const vector<PNode>& x) {
        if (x.size() == 0) {
            std::cout << "empty inputs for lstm operation" << std::endl;
            return;
        }
        _nSize = x.size();
        if (x[0]->dim != _nHiddenDim) {
            std::cout << "input dim does not match for attention  operation" << std::endl;
            return;
        }

        vector<PNode> aligns;
        for (int idx = 0; idx < _nSize; idx++) {
            _weights[idx].forward(cg, x[idx]);
            aligns.push_back(&_weights[idx]);
        }
        _hidden.forward(cg, x, aligns);
    }
};

#endif
