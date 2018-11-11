#ifndef LSTM1
#define LSTM1

/*
*  LSTM1.h:
*  LSTM variation 1
*
*  Created on: June 13, 2017
*      Author: mszhang, chncwang
*/

#include "MyLib.h"
#include "Node.h"
#include "BiOP.h"
#include "AtomicOP.h"
#include "Graph.h"
#include "PMultiOP.h"
#include "PAddOP.h"
#include "BucketOP.h"

#include <memory>

struct LSTM1Params
#if USE_GPU
: public TransferableComponents
#endif
{
    UniParams input_hidden;
    UniParams input_input;
    UniParams output_hidden;
    UniParams output_input;
    UniParams forget_hidden;
    UniParams forget_input;
    UniParams cell_hidden;
    UniParams cell_input;

    LSTM1Params() = default;

    void exportAdaParams(ModelUpdate& ada) {
        input_hidden.exportAdaParams(ada);
        input_input.exportAdaParams(ada);
        output_hidden.exportAdaParams(ada);
        output_input.exportAdaParams(ada);
        forget_hidden.exportAdaParams(ada);
        forget_input.exportAdaParams(ada);
        cell_hidden.exportAdaParams(ada);
        cell_input.exportAdaParams(ada);
    }

    void initial(int nOSize, int nISize) {
        input_hidden.initial(nOSize, nOSize, false);
        input_input.initial(nOSize, nISize, true);
        output_hidden.initial(nOSize, nOSize, false);
        output_input.initial(nOSize, nISize, true);
        forget_hidden.initial(nOSize, nOSize, false);
        forget_input.initial(nOSize, nISize, true);
        cell_hidden.initial(nOSize, nOSize, false);
        cell_input.initial(nOSize, nISize, true);
    }

    int inDim() {
        return input_input.W.inDim();
    }

    int outDim() {
        return input_input.W.outDim();
    }

    void save(std::ofstream &os) const {
        input_hidden.save(os);
        input_input.save(os);
        output_hidden.save(os);
        output_input.save(os);
        forget_hidden.save(os);
        forget_input.save(os);
        cell_hidden.save(os);
        cell_input.save(os);

    }

    void load(std::ifstream &is) {
        input_hidden.load(is);
        input_input.load(is);
        output_hidden.load(is);
        output_input.load(is);
        forget_hidden.load(is);
        forget_input.load(is);
        cell_hidden.load(is);
        cell_input.load(is);
    }

#if USE_GPU
    std::vector<Transferable *> transferablePtrs() override {
        return {&input_hidden, &input_input, &output_hidden, &output_input, &forget_input,
            &forget_hidden, &cell_hidden, &cell_input};
    }

    virtual std::string name() const {
        return "LSTM1Params";
    }
#endif
};

// standard LSTM1 using tanh as activation function
// other conditions are not implemented unless they are clear
class LSTM1Builder {
public:
    int _nSize;
    int _inDim;
    int _outDim;

    vector<LinearNode> _inputgates_hidden;
    vector<LinearNode> _inputgates_input;
    vector<PAddNode> _inputgates_add;
    vector<SigmoidNode> _inputgates;

    vector<LinearNode> _forgetgates_hidden;
    vector<LinearNode> _forgetgates_input;
    vector<PAddNode> _forgetgates_add;
    vector<SigmoidNode> _forgetgates;

    vector<LinearNode> _halfcells_hidden;
    vector<LinearNode> _halfcells_input;
    vector<PAddNode> _halfcells_add;
    vector<TanhNode> _halfcells;

    vector<LinearNode> _outputgates_hidden;
    vector<LinearNode> _outputgates_input;
    vector<PAddNode> _outputgates_add;
    vector<SigmoidNode> _outputgates;

    vector<PMultiNode> _inputfilters;
    vector<PMultiNode> _forgetfilters;

    vector<PAddNode> _cells;

    vector<TanhNode> _halfhiddens;
    vector<PMultiNode> _hiddens;  // intermediate result without dropout

    BucketNode _bucket;

    LSTM1Params* _param;

    bool _left2right;

public:
    LSTM1Builder() {
        clear();
    }

    ~LSTM1Builder() {
        clear();
    }

public:
    void init(LSTM1Params* paramInit, dtype dropout, bool left2right = true) {
        _param = paramInit;
        _inDim = _param->input_input.W.inDim();
        _outDim = _param->input_input.W.outDim();
        int maxsize = _inputgates_hidden.size();

        for (int idx = 0; idx < maxsize; idx++) {
            _inputgates_input.at(idx).setParam(&_param->input_input);
            _outputgates_input.at(idx).setParam(&_param->output_input);
            _forgetgates_input.at(idx).setParam(&_param->forget_input);
            _halfcells_input.at(idx).setParam(&_param->cell_input);

            _inputgates_hidden.at(idx).setParam(&_param->input_hidden);
            _outputgates_hidden.at(idx).setParam(&_param->output_hidden);
            _forgetgates_hidden.at(idx).setParam(&_param->forget_hidden);
            _halfcells_hidden.at(idx).setParam(&_param->cell_hidden);
        }
        _left2right = left2right;

        for (int idx = 0; idx < maxsize; idx++) {
            _inputgates_hidden.at(idx).init(_outDim, -1);
            _inputgates_input.at(idx).init(_outDim, -1);
            _forgetgates_hidden.at(idx).init(_outDim, -1);
            _forgetgates_input.at(idx).init(_outDim, -1);
            _halfcells_hidden.at(idx).init(_outDim, -1);
            _halfcells_input.at(idx).init(_outDim, -1);
            _outputgates_hidden.at(idx).init(_outDim, -1);
            _outputgates_input.at(idx).init(_outDim, -1);

            _inputfilters.at(idx).init(_outDim, -1);
            _forgetfilters.at(idx).init(_outDim, -1);
            _cells.at(idx).init(_outDim, -1);

            _halfhiddens.at(idx).init(_outDim, -1);
            _hiddens.at(idx).init(_outDim, dropout);

            _inputgates_add.at(idx).init(_outDim, -1);
            _inputgates.at(idx).init(_outDim, -1);
            _forgetgates_add.at(idx).init(_outDim, -1);
            _forgetgates.at(idx).init(_outDim, -1);
            _halfcells_add.at(idx).init(_outDim, -1);
            _halfcells.at(idx).init(_outDim, -1);
            _outputgates_add.at(idx).init(_outDim, -1);
            _outputgates.at(idx).init(_outDim, -1);
        }
        _bucket.init(_outDim, -1);

    }

    void resize(int maxsize) {
        _inputgates_hidden.resize(maxsize);
        _inputgates_input.resize(maxsize);
        _inputgates_add.resize(maxsize);
        _inputgates.resize(maxsize);
        _forgetgates_hidden.resize(maxsize);
        _forgetgates_input.resize(maxsize);
        _forgetgates_add.resize(maxsize);
        _forgetgates.resize(maxsize);
        _halfcells_hidden.resize(maxsize);
        _halfcells_input.resize(maxsize);
        _halfcells_add.resize(maxsize);
        _halfcells.resize(maxsize);
        _outputgates_hidden.resize(maxsize);
        _outputgates_input.resize(maxsize);
        _outputgates_add.resize(maxsize);
        _outputgates.resize(maxsize);

        _inputfilters.resize(maxsize);
        _forgetfilters.resize(maxsize);
        _cells.resize(maxsize);
        _halfhiddens.resize(maxsize);
        _hiddens.resize(maxsize);
    }

    //whether vectors have been allocated
    bool empty() {
        return _hiddens.empty();
    }

    void clear() {
        _inputgates_hidden.clear();
        _inputgates_input.clear();
        _inputgates_add.clear();
        _inputgates.clear();
        _forgetgates_hidden.clear();
        _forgetgates_input.clear();
        _forgetgates_add.clear();
        _forgetgates.clear();
        _halfcells_hidden.clear();
        _halfcells_input.clear();
        _halfcells_add.clear();
        _halfcells.clear();
        _outputgates_hidden.clear();
        _outputgates_input.clear();
        _outputgates_add.clear();
        _outputgates.clear();

        _inputfilters.clear();
        _forgetfilters.clear();
        _cells.clear();
        _halfhiddens.clear();
        _hiddens.clear();

        _left2right = true;
        _param = NULL;
        _nSize = 0;
        _inDim = 0;
        _outDim = 0;
    }

public:
    void forward(Graph *cg, const vector<PNode>& x) {
        if (x.size() == 0) {
            std::cout << "empty inputs for lstm operation" << std::endl;
            return;
        }
        _nSize = x.size();
        if (x[0]->val.dim != _inDim) {
            std::cout << "input dim does not match for lstm operation" << std::endl;
            return;
        }

        int idx_begin = _left2right ? 0 : x.size() - 1;
        int idx_step = _left2right ? 1 : -1;

        for (int idx = idx_begin; _left2right ? idx < _nSize : idx >= 0; idx += idx_step) {
            if (idx == idx_begin) {
                _bucket.forward(cg, 0);
                _inputgates_hidden[idx].forward(cg, &_bucket);
                _inputgates_input[idx].forward(cg, x[idx]);
                _inputgates_add[idx].forward(cg, &_inputgates_hidden[idx], &_inputgates_input[idx]);
                _inputgates[idx].forward(cg, &_inputgates_add[idx]);

                _outputgates_hidden[idx].forward(cg, &_bucket);
                _outputgates_input[idx].forward(cg, x[idx]);
                _outputgates_add[idx].forward(cg, &_outputgates_hidden[idx], &_outputgates_input[idx]);
                _outputgates[idx].forward(cg, &_outputgates_add[idx]);

                _halfcells_hidden[idx].forward(cg, &_bucket);
                _halfcells_input[idx].forward(cg, x[idx]);
                _halfcells_add[idx].forward(cg, &_halfcells_hidden[idx], &_halfcells_input[idx]);
                _halfcells[idx].forward(cg, &_halfcells_add[idx]);

                _inputfilters[idx].forward(cg, &_halfcells[idx], &_inputgates[idx]);
                _cells[idx].forward(cg, &_inputfilters[idx], &_bucket);
                _halfhiddens[idx].forward(cg, &_cells[idx]);
                _hiddens[idx].forward(cg, &_halfhiddens[idx], &_outputgates[idx]);
            } else {
                _inputgates_hidden[idx].forward(cg, &_hiddens[idx - idx_step]);
                _inputgates_input[idx].forward(cg, x[idx]);
                _inputgates_add[idx].forward(cg, &_inputgates_hidden[idx], &_inputgates_input[idx]);
                _inputgates[idx].forward(cg, &_inputgates_add[idx]);

                _outputgates_hidden[idx].forward(cg, &_hiddens[idx - idx_step]);
                _outputgates_input[idx].forward(cg, x[idx]);
                _outputgates_add[idx].forward(cg, &_outputgates_hidden[idx], &_outputgates_input[idx]);
                _outputgates[idx].forward(cg, &_outputgates_add[idx]);

                _halfcells_hidden[idx].forward(cg, &_hiddens[idx - idx_step]);
                _halfcells_input[idx].forward(cg, x[idx]);
                _halfcells_add[idx].forward(cg, &_halfcells_hidden[idx], &_halfcells_input[idx]);
                _halfcells[idx].forward(cg, &_halfcells_add[idx]);

                _forgetgates_hidden[idx].forward(cg, &_hiddens[idx - idx_step]);
                _forgetgates_input[idx].forward(cg, x[idx]);
                _forgetgates_add[idx].forward(cg, &_forgetgates_hidden[idx], &_forgetgates_input[idx]);
                _forgetgates[idx].forward(cg, &_forgetgates_add[idx]);

                _inputfilters[idx].forward(cg, &_halfcells[idx], &_inputgates[idx]);
                _forgetfilters[idx].forward(cg, &_cells[idx - idx_step], &_forgetgates[idx]);
                _cells[idx].forward(cg, &_inputfilters[idx], &_forgetfilters[idx]);
                _halfhiddens[idx].forward(cg, &_cells[idx]);
                _hiddens[idx].forward(cg, &_halfhiddens[idx], &_outputgates[idx]);
            }
        }
    }
};

class IncLSTM1Builder {
public:
    int _nSize;
    int _inDim;
    int _outDim;

    IncLSTM1Builder *pPrev;

    LinearNode _inputgate_hidden;
    LinearNode _inputgate_input;
    PAddNode _inputgate_add;
    SigmoidNode _inputgate;

    LinearNode _forgetgate_hidden;
    LinearNode _forgetgate_input;
    PAddNode _forgetgate_add;
    SigmoidNode _forgetgate;

    LinearNode _halfcell_hidden;
    LinearNode _halfcell_input;
    PAddNode _halfcell_add;
    TanhNode _halfcell;

    LinearNode _outputgate_hidden;
    LinearNode _outputgate_input;
    PAddNode _outputgate_add;
    SigmoidNode _outputgate;

    PMultiNode _inputfilter;
    PMultiNode _forgetfilter;

    PAddNode _cells;

    TanhNode _halfhidden;
    PMultiNode _hidden;  // intermediate result without dropout

    BucketNode _bucket;

    LSTM1Params* _param;

    void init(LSTM1Params* paramInit, dtype dropout, bool left2right = true) {
        _param = paramInit;
        _inDim = _param->input_input.W.inDim();
        _outDim = _param->input_input.W.outDim();

        _inputgate_input.setParam(&_param->input_input);
        _outputgate_input.setParam(&_param->output_input);
        _forgetgate_input.setParam(&_param->forget_input);
        _halfcell_input.setParam(&_param->cell_input);

        _inputgate_hidden.setParam(&_param->input_hidden);
        _outputgate_hidden.setParam(&_param->output_hidden);
        _forgetgate_hidden.setParam(&_param->forget_hidden);
        _halfcell_hidden.setParam(&_param->cell_hidden);

        _inputgate_hidden.init(_outDim, -1);
        _inputgate_input.init(_outDim, -1);
        _forgetgate_hidden.init(_outDim, -1);
        _forgetgate_input.init(_outDim, -1);
        _halfcell_hidden.init(_outDim, -1);
        _halfcell_input.init(_outDim, -1);
        _outputgate_hidden.init(_outDim, -1);
        _outputgate_input.init(_outDim, -1);

        _inputfilter.init(_outDim, -1);
        _forgetfilter.init(_outDim, -1);
        _cells.init(_outDim, -1);

        _halfhidden.init(_outDim, -1);
        _hidden.init(_outDim, dropout);

        _inputgate_add.init(_outDim, -1);
        _inputgate.init(_outDim, -1);
        _forgetgate_add.init(_outDim, -1);
        _forgetgate.init(_outDim, -1);
        _halfcell_add.init(_outDim, -1);
        _halfcell.init(_outDim, -1);
        _outputgate_add.init(_outDim, -1);
        _outputgate.init(_outDim, -1);
        _bucket.init(_outDim, -1);

    }

    void forward(Graph *cg, PNode x, IncLSTM1Builder *prev = NULL) {
        if (prev == NULL) {
            _bucket.forward(cg, 0);
            _inputgate_hidden.forward(cg, &_bucket);
            _inputgate_input.forward(cg, x);
            _inputgate_add.forward(cg, &_inputgate_hidden, &_inputgate_input);
            _inputgate.forward(cg, &_inputgate_add);

            _outputgate_hidden.forward(cg, &_bucket);
            _outputgate_input.forward(cg, x);
            _outputgate_add.forward(cg, &_outputgate_hidden, &_outputgate_input);
            _outputgate.forward(cg, &_outputgate_add);

            _halfcell_hidden.forward(cg, &_bucket);
            _halfcell_input.forward(cg, x);
            _halfcell_add.forward(cg, &_halfcell_hidden, &_halfcell_input);
            _halfcell.forward(cg, &_halfcell_add);

            _inputfilter.forward(cg, &_halfcell, &_inputgate);
            _cells.forward(cg, &_inputfilter, &_bucket);
            _halfhidden.forward(cg, &_cells);
            _hidden.forward(cg, &_halfhidden, &_outputgate);
        } else {
            _inputgate_hidden.forward(cg, &prev->_hidden);
            _inputgate_input.forward(cg, x);
            _inputgate_add.forward(cg, &prev->_inputgate_hidden, &prev->_inputgate_input);
            _inputgate.forward(cg, &prev->_inputgate_add);

            _outputgate_hidden.forward(cg, &prev->_hidden);
            _outputgate_input.forward(cg, x);
            _outputgate_add.forward(cg, &_outputgate_hidden, &_outputgate_input);
            _outputgate.forward(cg, &_outputgate_add);

            _halfcell_hidden.forward(cg, &prev->_hidden);
            _halfcell_input.forward(cg, x);
            _halfcell_add.forward(cg, &_halfcell_hidden, &_halfcell_input);
            _halfcell.forward(cg, &_halfcell_add);

            _forgetgate_hidden.forward(cg, &prev->_hidden);
            _forgetgate_input.forward(cg, x);
            _forgetgate_add.forward(cg, &_forgetgate_hidden, &_forgetgate_input);
            _forgetgate.forward(cg, &_forgetgate_add);

            _inputfilter.forward(cg, &_halfcell, &_inputgate);
            _forgetfilter.forward(cg, &prev->_cells, &_forgetgate);
            _cells.forward(cg, &_inputfilter, &_forgetfilter);
            _halfhidden.forward(cg, &_cells);
            _hidden.forward(cg, &_halfhidden, &_outputgate);
        }
    }
};

struct DynamicLSTMBuilder {
    std::vector<std::shared_ptr<LinearNode>> _inputgates_hidden;
    std::vector<std::shared_ptr<LinearNode>> _inputgates_input;
    std::vector<std::shared_ptr<PAddNode>> _inputgates_add;
    std::vector<std::shared_ptr<SigmoidNode>> _inputgates;

    std::vector<std::shared_ptr<LinearNode>> _forgetgates_hidden;
    std::vector<std::shared_ptr<LinearNode>> _forgetgates_input;
    std::vector<std::shared_ptr<PAddNode>> _forgetgates_add;
    std::vector<std::shared_ptr<SigmoidNode>> _forgetgates;

    std::vector<std::shared_ptr<LinearNode>> _halfcells_hidden;
    std::vector<std::shared_ptr<LinearNode>> _halfcells_input;
    std::vector<std::shared_ptr<PAddNode>> _halfcells_add;
    std::vector<std::shared_ptr<TanhNode>> _halfcells;

    std::vector<std::shared_ptr<LinearNode>> _outputgates_hidden;
    std::vector<std::shared_ptr<LinearNode>> _outputgates_input;
    std::vector<std::shared_ptr<PAddNode>> _outputgates_add;
    std::vector<std::shared_ptr<SigmoidNode>> _outputgates;

    std::vector<std::shared_ptr<PMultiNode>> _inputfilters;
    std::vector<std::shared_ptr<PMultiNode>> _forgetfilters;

    std::vector<std::shared_ptr<PAddNode>> _cells;

    std::vector<std::shared_ptr<TanhNode>> _halfhiddens;
    std::vector<std::shared_ptr<PMultiNode>> _hiddens;

    void forward(Graph &graph, LSTM1Params &lstm_params, Node &input, Node &h0, Node &c0) {
        Node *last_hidden, *last_cell;
        int len = _hiddens.size();
        if (len == 0) {
            last_hidden = &h0;
            last_cell = &c0;
        } else {
            last_hidden = _hiddens.at(len - 1).get();
            last_cell = _cells.at(len - 1).get();
        }
        int out_dim = lstm_params.input_hidden.W.outDim();

        shared_ptr<LinearNode> inputgate_hidden(new LinearNode);
        inputgate_hidden->init(out_dim, -1);
        inputgate_hidden->setParam(lstm_params.input_hidden);
        inputgate_hidden->node_name = "inputgate_hidden";
        _inputgates_hidden.push_back(inputgate_hidden);

        shared_ptr<LinearNode> inputgate_input(new LinearNode);
        inputgate_input->init(out_dim, -1);
        inputgate_input->setParam(lstm_params.input_input);
        inputgate_input->node_name = "inputgate_input";
        _inputgates_input.push_back(inputgate_input);

        shared_ptr<LinearNode> forgetgate_hidden(new LinearNode);
        forgetgate_hidden->init(out_dim, -1);
        forgetgate_hidden->setParam(lstm_params.forget_hidden);
        forgetgate_hidden->node_name = "forgetgate_hidden";
        _forgetgates_hidden.push_back(forgetgate_hidden);

        shared_ptr<LinearNode> forgetgate_input(new LinearNode);
        forgetgate_input->init(out_dim, -1);
        forgetgate_input->setParam(lstm_params.forget_input);
        forgetgate_input->node_name = "forgetgate_input";
        _forgetgates_input.push_back(forgetgate_input);

        shared_ptr<LinearNode> halfcell_hidden(new LinearNode);
        halfcell_hidden->init(out_dim, -1);
        halfcell_hidden->setParam(lstm_params.cell_hidden);
        halfcell_hidden->node_name = "halfcell_hidden";
        _halfcells_hidden.push_back(halfcell_hidden);

        shared_ptr<LinearNode> halfcell_input(new LinearNode);
        halfcell_input->init(out_dim, -1);
        halfcell_input->setParam(lstm_params.cell_input);
        halfcell_input->node_name = "halfcell_input";
        _halfcells_input.push_back(halfcell_input);

        shared_ptr<LinearNode> outputgate_hidden(new LinearNode);
        outputgate_hidden->init(out_dim, -1);
        outputgate_hidden->setParam(lstm_params.output_hidden);
        outputgate_hidden->node_name = "outputgate_hidden";
        _outputgates_hidden.push_back(outputgate_hidden);

        shared_ptr<LinearNode> outputgate_input(new LinearNode);
        outputgate_input->init(out_dim, -1);
        outputgate_input->setParam(lstm_params.output_input);
        outputgate_input->node_name = "outputgate_input";
        _outputgates_input.push_back(outputgate_input);

        shared_ptr<PMultiNode> inputfilter(new PMultiNode);
        inputfilter->init(out_dim, -1);
        inputfilter->node_name = "inputfilter";
        _inputfilters.push_back(inputfilter);

        shared_ptr<PMultiNode> forgetfilter(new PMultiNode);
        forgetfilter->init(out_dim, -1);
        forgetfilter->node_name = "forgetfilter";
        _forgetfilters.push_back(forgetfilter);

        shared_ptr<PAddNode> cell(new PAddNode);
        cell->init(out_dim, -1);
        cell->node_name = "cell";
        _cells.push_back(cell);

        shared_ptr<TanhNode> halfhidden(new TanhNode);
        halfhidden->init(out_dim, -1);
        halfhidden->node_name = "halfhidden";
        _halfhiddens.push_back(halfhidden);

        shared_ptr<PMultiNode> hidden(new PMultiNode);
        hidden->init(out_dim, -1);
        hidden->node_name = "hidden";
        _hiddens.push_back(hidden);

        shared_ptr<PAddNode> inputgate_add(new PAddNode);
        inputgate_add->init(out_dim, -1);
        inputgate_add->node_name = "inputgate_add";
        _inputgates_add.push_back(inputgate_add);

        shared_ptr<SigmoidNode> inputgate(new SigmoidNode);
        inputgate->init(out_dim, -1);
        inputgate->node_name = "inputgate";
        _inputgates.push_back(inputgate);

        shared_ptr<PAddNode> forgetgate_add(new PAddNode);
        forgetgate_add->init(out_dim, -1);
        forgetgate_add->node_name = "forgetgate_add";
        _forgetgates_add.push_back(forgetgate_add);

        shared_ptr<SigmoidNode> forgetgate(new SigmoidNode);
        forgetgate->init(out_dim, -1);
        forgetgate->node_name = "forgetgate";
        _forgetgates.push_back(forgetgate);

        shared_ptr<PAddNode> halfcell_add(new PAddNode);
        halfcell_add->init(out_dim, -1);
        halfcell_add->node_name = "halfcell_add";
        _halfcells_add.push_back(halfcell_add);

        shared_ptr<TanhNode> halfcell(new TanhNode);
        halfcell->init(out_dim, -1);
        halfcell->node_name = "halfcell";
        _halfcells.push_back(halfcell);

        shared_ptr<PAddNode> outputgate_add(new PAddNode);
        outputgate_add->init(out_dim, -1);
        outputgate_add->node_name = "outputgate_add";
        _outputgates_add.push_back(outputgate_add);

        shared_ptr<SigmoidNode> outputgate(new SigmoidNode);
        outputgate->init(out_dim, -1);
        outputgate->node_name = "outputgate";
        _outputgates.push_back(outputgate);

        _inputgates_hidden.at(len)->forward(graph, *last_hidden);
        _inputgates_input.at(len)->forward(graph, input);
        _inputgates_add.at(len)->forward(graph, *_inputgates_hidden.at(len),
                *_inputgates_input.at(len));
        _inputgates.at(len)->forward(graph, *_inputgates_add.at(len));

        _outputgates_hidden.at(len)->forward(graph, *last_hidden);
        _outputgates_input.at(len)->forward(graph, input);
        _outputgates_add.at(len)->forward(graph, *_outputgates_hidden.at(len),
                *_outputgates_input.at(len));
        _outputgates.at(len)->forward(graph, *_outputgates_add.at(len));

        _halfcells_hidden.at(len)->forward(graph, *last_hidden);
        _halfcells_input.at(len)->forward(graph, input);
        _halfcells_add.at(len)->forward(graph, *_halfcells_hidden.at(len),
                *_halfcells_input.at(len));
        _halfcells.at(len)->forward(graph, *_halfcells_add.at(len));

        _forgetgates_hidden.at(len)->forward(graph, *last_hidden);
        _forgetgates_input.at(len)->forward(graph, input);
        _forgetgates_add.at(len)->forward(graph, *_forgetgates_hidden.at(len),
                *_forgetgates_input.at(len));
        _forgetgates.at(len)->forward(graph, *_forgetgates_add.at(len));

        _inputfilters.at(len)->forward(graph, *_halfcells.at(len), *_inputgates.at(len));
        _forgetfilters.at(len)->forward(graph, *last_cell, *_forgetgates.at(len));
        _cells.at(len)->forward(graph, *_inputfilters.at(len), *_forgetfilters.at(len));
        _halfhiddens.at(len)->forward(graph, *_cells.at(len));
        _hiddens.at(len)->forward(graph, *_halfhiddens.at(len), *_outputgates.at(len));
    }
};

#endif
