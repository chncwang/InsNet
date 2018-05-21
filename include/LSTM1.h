#ifndef LSTM1
#define LSTM1

/*
*  LSTM1.h:
*  LSTM variation 1
*
*  Created on: June 13, 2017
*      Author: mszhang
*/

#include <functional>
#include <array>
#include "MyLib.h"
#include "Node.h"
#include "BiOP.h"
#include "AtomicOP.h"
#include "Graph.h"
#include "PMultiOP.h"
#include "PAddOP.h"
#include "BucketOP.h"

class LSTMActivatedNode : public Node {
  public:
    std::array<Node *, 2> ins;
    int quaterDim = 0;

    void init(int ndim, dtype dropout) override {
        if ((ndim & 3) != 0) {
            abort();
        }
        quaterDim = ndim >> 2;
        Node::init(ndim, dropout);
    }

    void forward(Graph *graph, Node *in1, Node *in2) {
        if (dim != in1->dim || dim != in2->dim) {
            abort();
        }
        degree = 0;

        ins.at(0) = in1;
        ins.at(1) = in2;
        for (Node *in : ins) {
            in->addParent(this);
        }
        graph->addNode(in1);
        graph->addNode(in2);
    }

    void compute() override {
        sigmoidValue() = (sigmoidInValue(0) + sigmoidInValue(1)).sigmoid();
        tanhValue() = (tanhInValue(0) + tanhInValue(1)).tanh();
    }

    void backward() override {
        for (int i = 0; i < 2; ++i) {
            sigmoidInLoss(i) += sigmoidLoss() *
                sigmoidInValue(i).binaryExpr(sigmoidValue(), ptr_fun(dsigmoid));
            tanhInLoss(i) += tanhLoss() *
                tanhInValue(i).binaryExpr(tanhValue(), ptr_fun(dtanh));
        }
    }

    size_t typeHashCode() const override {
        return Node::typeHashCode();
    }

    Vec sigmoidValue() {
        return Vec(val.v, 3 * quaterDim);
    }

    Vec sigmoidLoss() {
        return Vec(loss.v, 3 * quaterDim);
    }

    Vec sigmoidInValue(int index) {
        return Vec(ins.at(index)->val.v, 3 * quaterDim);
    }

    Vec sigmoidInLoss(int index) {
        return Vec(ins.at(index)->loss.v, 3 * quaterDim);
    }

    Vec tanhValue() {
        return Vec(val.v + 3 * quaterDim, quaterDim);
    }

    Vec tanhLoss() {
        return Vec(loss.v + 3 * quaterDim, quaterDim);
    }

    Vec tanhInValue(int index) {
        return Vec(ins.at(index)->val.v + 3 * quaterDim, quaterDim);
    }

    Vec tanhInLoss(int index) {
        return Vec(ins.at(index)->loss.v + 3 * quaterDim, quaterDim);
    }

    Execute* generate(bool bTrain, dtype cur_drop_factor);
};

#if USE_GPU
class LSTMActivatedExecute : public Execute {
};
#else
class LSTMActivatedExecute : public Execute {};
#endif

Execute *LSTMActivatedNode::generate(bool bTrain, dtype cur_drop_factor) {
    LSTMActivatedExecute *execute = new LSTMActivatedExecute;
    execute->bTrain = bTrain;
    execute->drop_factor = cur_drop_factor;
    return execute;
}

class LSTMCellNode : public Node {
  public:
    Node *previousCell = NULL;
    Node *forgetAndInput = NULL;

    void forward(Graph *graph, Node *previouscell, Node *forgetandinput) {
        if (dim != previousCell->dim || dim != (forgetAndInput->dim >> 2)) {
            abort();
        }
        degree = 0;
        previousCell = previouscell;
        forgetAndInput = forgetandinput;
        previousCell->addParent(this);
        forgetAndInput->addParent(this);
        graph->addNode(previouscell);
        graph->addNode(forgetandinput);
    }

    void compute() override {
        val.vec() = forgetValue() * previousCell->val.vec() +
            inputValue() * newCellValue();
    }

    void backward() override {
        forgetLoss() += loss.vec() * previousCell->val.vec();
        previousCell->loss.vec() += loss.vec() * forgetValue();
        inputLoss() += loss.vec() * newCellValue();
        newCellLoss() += loss.vec() * inputValue();
    }

    size_t typeHashCode() const override {
        return Node::typeHashCode();
    }

    Vec forgetValue() {
        return Vec(forgetAndInput->val.v, dim);
    }

    Vec forgetLoss() {
        return Vec(forgetAndInput->loss.v, dim);
    }

    Vec inputValue() {
        return Vec(forgetAndInput->val.v + dim, dim);
    }

    Vec inputLoss() {
        return Vec(forgetAndInput->loss.v + dim, dim);
    }

    Vec newCellValue() {
        return Vec(forgetAndInput->val.v + 3 * dim, dim);
    }

    Vec newCellLoss() {
        return Vec(forgetAndInput->loss.v + 3 * dim, dim);
    }

    Execute* generate(bool bTrain, dtype cur_drop_factor);
};

#if USE_GPU
class LSTMCellExecute : public Execute {};
#else
class LSTMCellExecute : public Execute {};
#endif

Execute *LSTMCellNode::generate(bool bTrain, dtype cur_drop_factor) {
    LSTMCellExecute *execute = new LSTMCellExecute;
    execute->bTrain = bTrain;
    execute->drop_factor = cur_drop_factor;
    return execute;
}

class LSTMHiddenNode : public Node {
  public:
    Node *output = NULL;
    Node *cell = NULL;
    Tensor1D tanhCell;

    void init(int ndim, dtype dropout) override {
        Node::init(ndim, dropout);
        tanhCell.init(ndim);
    }

    void forward(Graph *graph, Node *outputte, Node *sell) {
        if (dim != outputte->dim) {
            abort();
        }
        degree = 0;
        output = outputte;
        cell = sell;
        output->addParent(this);
        cell->addParent(this);
        graph->addNode(outputte);
        graph->addNode(sell);
    }

    void compute() override {
        tanhCell.vec() = cell->val.vec().tanh();
        val.vec() = outputValue() * tanhCell.vec();
    }

    void backward() override {
        outputLoss() += loss.vec() * tanhCell.vec();
        cell->loss.vec() += loss.vec() * outputValue() *
            cell->val.vec().binaryExpr(tanhCell.vec(), ptr_fun(dtanh));
    }

    size_t typeHashCode() const override {
        return Node::typeHashCode();
    }

    Vec outputValue() {
        return Vec(output->val.v + 2 * dim, dim);
    }

    Vec outputLoss() {
        return Vec(output->loss.v + 2 * dim, dim);
    }

    Execute* generate(bool bTrain, dtype cur_drop_factor);
};

#if USE_GPU
class LSTMHiddenExecute : public Execute {};
#else
class LSTMHiddenExecute : public Execute {};
#endif

Execute *LSTMHiddenNode::generate(bool bTrain, dtype cur_drop_factor) {
    LSTMHiddenExecute *execute = new LSTMHiddenExecute;
    execute->bTrain = bTrain;
    execute->drop_factor = cur_drop_factor;
    return execute;
}

struct LSTM1Params {
    UniParams Wx;
    UniParams Wh;

    void exportAdaParams(ModelUpdate& ada) {
        Wx.exportAdaParams(ada);
        Wh.exportAdaParams(ada);
    }

    void initial(int outDim, int inDim) {
        Wx.initial(4 * outDim, inDim, true);
        Wh.initial(4 * outDim, outDim, false);
    }

    int inDim() {
        return Wx.W.inDim();
    }

    int outDim() {
        return Wh.W.inDim();
    }

    void save(std::ofstream &os) const {
        Wx.save(os);
        Wh.save(os);
    }

    void load(std::ifstream &is) {
        Wx.load(is);
        Wh.load(is);
    }
};

// standard LSTM1 using tanh as activation function
// other conditions are not implemented unless they are clear
class LSTM1Builder {
  public:
    int _size;;
    int _inDim;
    int _outDim;

    std::vector<LinearNode> _linearTransformedXs;
    std::vector<LinearNode> _linearTransformedHiddens;
    std::vector<LSTMActivatedNode> _activatedNodes;
    std::vector<LSTMCellNode> _cellNodes;
    std::vector<LSTMHiddenNode> _hiddenNodes;

    BucketNode _long_bucket;
    BucketNode _short_bucket;

    LSTM1Params* _param;

    bool _left2right;

    void init(LSTM1Params* paramInit, dtype dropout, bool left2right = true) {
        _param = paramInit;
        _inDim = paramInit->inDim();
        _outDim = paramInit->outDim();
        int size = _hiddenNodes.size();

        for (int i = 0; i < size; ++i) {
            _linearTransformedXs.at(i).setParam(&paramInit->Wx);
            _linearTransformedHiddens.at(i).setParam(&paramInit->Wh);
        }

        _left2right = left2right;

        int fourOutDim = _outDim << 2;
        for (int i = 0; i < size; ++i) {
            _linearTransformedXs.at(i).init(fourOutDim, -1);
            _linearTransformedHiddens.at(i).init(fourOutDim, -1);
            _activatedNodes.at(i).init(fourOutDim, -1);
            _cellNodes.at(i).init(_outDim, -1);
            _hiddenNodes.at(i).init(_outDim, dropout);
        }

        _long_bucket.init(fourOutDim, -1);
        _short_bucket.init(_outDim, -1);
    }


    void resize(int maxsize) {
        _linearTransformedXs.resize(maxsize);
        _linearTransformedHiddens.resize(maxsize);
        _activatedNodes.resize(maxsize);
        _cellNodes.resize(maxsize);
        _hiddenNodes.resize(maxsize);
    }

    //whether vectors have been allocated
    bool empty() {
        return _hiddenNodes.empty();
    }

    void forward(Graph *graph, const vector<PNode>& xs) {
        if (xs.size() == 0) {
            std::cout << "empty inputs for lstm operation" << std::endl;
            abort();
        }
        if (xs[0]->val.dim != _inDim) {
            std::cout << "input dim does not match for lstm operation" << std::endl;
            abort();
        }

        int i_begin, i_step;

        if (_left2right) {
            i_begin = 0;
            i_step = 1;
        } else {
            i_begin = _size - 1;
            i_step = -1;
        }

        for (int i = i_begin; _left2right ? i < xs.size() : i >= 0;
                i += i_step) {
            _linearTransformedXs.at(i).forward(graph, xs.at(i));
            _linearTransformedHiddens.at(i).forward(graph, i == i_begin ? 
                    static_cast<Node *>(&_long_bucket) : 
                    static_cast<Node *>(&_hiddenNodes.at(i - i_step)));
            _activatedNodes.at(i).forward(graph, &_linearTransformedXs.at(i),
                    &_linearTransformedHiddens.at(i));
            _cellNodes.at(i).forward(graph, i == i_begin ?
                    static_cast<Node *>(&_short_bucket) :
                    static_cast<Node *>(&_cellNodes.at(i - i_step)),
                    &_activatedNodes.at(i));
            _hiddenNodes.at(i).forward(graph, &_activatedNodes.at(i),
                    &_cellNodes.at(i));
        }
    }
};

#endif
