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
#include "AtomicOP.h"
#include "Graph.h"
#include "PMultiOP.h"
#include "PAddOP.h"
#include "BucketOP.h"
#include "UniOP.h"

#include <memory>

struct LSTM1Params : public N3LDGSerializable, TunableCombination<BaseParam>
#if USE_GPU
, public TransferableComponents
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

    LSTM1Params(const string &name) : input_hidden(name + "-input_hidden"),
    input_input(name + "-input_input"), output_hidden(name + "-output_hidden"),
    output_input(name + "-output_input"), forget_hidden(name + "-forget_hidden"),
    forget_input(name + "-forget_input"), cell_hidden(name + "-cell_hidden"),
    cell_input(name + "-cell_input") {}

    Json::Value toJson() const override {
        Json::Value json;
        json["input_hidden"] = input_hidden.toJson();
        json["input_input"] = input_input.toJson();
        json["output_hidden"] = output_hidden.toJson();
        json["output_input"] = output_input.toJson();
        json["forget_hidden"] = forget_hidden.toJson();
        json["forget_input"] = forget_input.toJson();
        json["cell_hidden"] = cell_hidden.toJson();
        json["cell_input"] = cell_input.toJson();
        return json;
    }

    void fromJson(const Json::Value &json) override {
        input_hidden.fromJson(json["input_hidden"]);
        input_input.fromJson(json["input_input"]);
        output_hidden.fromJson(json["output_hidden"]);
        output_input.fromJson(json["output_input"]);
        forget_hidden.fromJson(json["forget_hidden"]);
        forget_input.fromJson(json["forget_input"]);
        cell_hidden.fromJson(json["cell_hidden"]);
        cell_input.fromJson(json["cell_input"]);
    }

    void init(int nOSize, int nISize) {
        input_hidden.init(nOSize, nOSize, false);
        input_input.init(nOSize, nISize, true);
        output_hidden.init(nOSize, nOSize, false);
        output_input.init(nOSize, nISize, true);
        forget_hidden.init(nOSize, nOSize, false);
        forget_input.init(nOSize, nISize, true);
        cell_hidden.init(nOSize, nOSize, false);
        cell_input.init(nOSize, nISize, true);

        forget_input.b.val.assignAll(1.0f);
    }

    int inDim() {
        return input_input.W.inDim();
    }

    int outDim() {
        return input_input.W.outDim();
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

protected:
    std::vector<Tunable<BaseParam> *> tunableComponents() override {
        return {&input_hidden, &input_input, &output_hidden, &output_input, &forget_hidden,
            &forget_input, &cell_hidden, &cell_input};
    }
};

struct DynamicLSTMBuilder {
    std::vector<LinearNode*> _inputgates_hidden;
    std::vector<LinearNode*> _inputgates_input;
    std::vector<PAddNode*> _inputgates_add;
    std::vector<SigmoidNode*> _inputgates;

    std::vector<LinearNode*> _forgetgates_hidden;
    std::vector<LinearNode*> _forgetgates_input;
    std::vector<PAddNode*> _forgetgates_add;
    std::vector<SigmoidNode*> _forgetgates;

    std::vector<LinearNode*> _halfcells_hidden;
    std::vector<LinearNode*> _halfcells_input;
    std::vector<PAddNode*> _halfcells_add;
    std::vector<TanhNode*> _halfcells;

    std::vector<LinearNode*> _outputgates_hidden;
    std::vector<LinearNode*> _outputgates_input;
    std::vector<PAddNode*> _outputgates_add;
    std::vector<SigmoidNode*> _outputgates;

    std::vector<PMultiNode*> _inputfilters;
    std::vector<PMultiNode*> _forgetfilters;

    std::vector<PAddNode*> _cells;

    std::vector<TanhNode*> _halfhiddens;
    std::vector<PMultiNode*> _hiddens_before_dropout;
    std::vector<Node*> _hiddens;

    int size() {
        return _hiddens.size();
    }

    void forward(Graph &graph, LSTM1Params &lstm_params, Node &input, Node &h0, Node &c0,
            dtype dropout, bool is_training) {
        Node *last_hidden, *last_cell;
        int len = _hiddens.size();
        if (len == 0) {
            last_hidden = &h0;
            last_cell = &c0;
        } else {
            last_hidden = _hiddens.at(len - 1);
            last_cell = _cells.at(len - 1);
        }
        int out_dim = lstm_params.input_hidden.W.outDim();

        LinearNode *inputgate_hidden = new LinearNode;
        inputgate_hidden->init(out_dim);
        inputgate_hidden->setParam(lstm_params.input_hidden);
        inputgate_hidden->setNodeName("lstm inputgate_hidden");
        _inputgates_hidden.push_back(inputgate_hidden);

        LinearNode *inputgate_input = new LinearNode;
        inputgate_input->init(out_dim);
        inputgate_input->setParam(lstm_params.input_input);
        inputgate_input->setNodeName("lstm inputgate_input");
        _inputgates_input.push_back(inputgate_input);

        LinearNode *forgetgate_hidden = new LinearNode;
        forgetgate_hidden->init(out_dim);
        forgetgate_hidden->setParam(lstm_params.forget_hidden);
        forgetgate_hidden->setNodeName("lstm forgetgate_hidden");
        _forgetgates_hidden.push_back(forgetgate_hidden);

        LinearNode *forgetgate_input = new LinearNode;
        forgetgate_input->init(out_dim);
        forgetgate_input->setParam(lstm_params.forget_input);
        forgetgate_input->setNodeName("lstm forgetgate_input");
        _forgetgates_input.push_back(forgetgate_input);

        LinearNode *halfcell_hidden = new LinearNode;
        halfcell_hidden->init(out_dim);
        halfcell_hidden->setParam(lstm_params.cell_hidden);
        halfcell_hidden->setNodeName("lstm halfcell_hidden");
        _halfcells_hidden.push_back(halfcell_hidden);

        LinearNode *halfcell_input = new LinearNode;
        halfcell_input->init(out_dim);
        halfcell_input->setParam(lstm_params.cell_input);
        halfcell_input->setNodeName("lstm halfcell_input");
        _halfcells_input.push_back(halfcell_input);

        LinearNode *outputgate_hidden = new LinearNode;
        outputgate_hidden->init(out_dim);
        outputgate_hidden->setParam(lstm_params.output_hidden);
        outputgate_hidden->setNodeName("lstm outputgate_hidden");
        _outputgates_hidden.push_back(outputgate_hidden);

        LinearNode *outputgate_input = new LinearNode;
        outputgate_input->init(out_dim);
        outputgate_input->setParam(lstm_params.output_input);
        outputgate_input->setNodeName("lstm outputgate_input");
        _outputgates_input.push_back(outputgate_input);

        PMultiNode *inputfilter = new PMultiNode;
        inputfilter->init(out_dim);
        _inputfilters.push_back(inputfilter);

        PMultiNode *forgetfilter = new PMultiNode;
        forgetfilter->init(out_dim);
        _forgetfilters.push_back(forgetfilter);

        PAddNode *cell = new PAddNode;
        cell->init(out_dim);
        _cells.push_back(cell);

        TanhNode *halfhidden = new TanhNode;
        halfhidden->init(out_dim);
        _halfhiddens.push_back(halfhidden);

        PMultiNode *hidden_before_dropout = new PMultiNode;
        hidden_before_dropout->init(out_dim);
        _hiddens_before_dropout.push_back(hidden_before_dropout);

        DropoutNode *hidden = new DropoutNode(dropout, is_training);
        hidden->init(out_dim);
        _hiddens.push_back(hidden);

        PAddNode * inputgate_add = new PAddNode;
        inputgate_add->init(out_dim);
        _inputgates_add.push_back(inputgate_add);

        SigmoidNode *inputgate = new SigmoidNode;
        inputgate->init(out_dim);
        _inputgates.push_back(inputgate);

        PAddNode *forgetgate_add = new PAddNode;
        forgetgate_add->init(out_dim);
        _forgetgates_add.push_back(forgetgate_add);

        SigmoidNode *forgetgate = new SigmoidNode;
        forgetgate->init(out_dim);
        _forgetgates.push_back(forgetgate);

        PAddNode *halfcell_add = new PAddNode;
        halfcell_add->init(out_dim);
        _halfcells_add.push_back(halfcell_add);

        TanhNode *halfcell = new TanhNode;
        halfcell->init(out_dim);
        _halfcells.push_back(halfcell);

        PAddNode *outputgate_add = new PAddNode;
        outputgate_add->init(out_dim);
        _outputgates_add.push_back(outputgate_add);

        SigmoidNode *outputgate = new SigmoidNode;
        outputgate->init(out_dim);
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
        _hiddens_before_dropout.at(len)->forward(graph, *_halfhiddens.at(len),
                *_outputgates.at(len));
        ((DropoutNode*)_hiddens.at(len))->forward(graph, *_hiddens_before_dropout.at(len));
    }
};

#endif
