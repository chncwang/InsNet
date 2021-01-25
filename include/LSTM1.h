#ifndef LSTM1
#define LSTM1

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
    std::vector<AtomicNode*> _cells;
    std::vector<AtomicNode*> _hiddens;

    int size() {
        return _hiddens.size();
    }

    void forward(Graph &graph, LSTM1Params &lstm_params, AtomicNode &input, AtomicNode &h0,
            AtomicNode &c0,
            dtype dropout,
            bool is_training) {
        AtomicNode *last_hidden, *last_cell;
        int len = _hiddens.size();
        if (len == 0) {
            last_hidden = &h0;
            last_cell = &c0;
        } else {
            last_hidden = _hiddens.at(len - 1);
            last_cell = _cells.at(len - 1);
        }

        using namespace n3ldg_plus;
        AtomicNode *inputgate_hidden = linear(graph, lstm_params.input_hidden, *last_hidden);
        AtomicNode *inputgate_input = linear(graph, lstm_params.input_input, input);
        AtomicNode *inputgate_add = add(graph, {inputgate_hidden, inputgate_input});
        AtomicNode *inputgate = sigmoid(graph, *inputgate_add);

        AtomicNode *forgetgate_hidden = linear(graph, lstm_params.forget_hidden, *last_hidden);
        AtomicNode *forgetgate_input = linear(graph, lstm_params.forget_input, input);
        AtomicNode *forgetgate_add = add(graph, {forgetgate_hidden, forgetgate_input});
        AtomicNode *forgetgate = sigmoid(graph, *forgetgate_add);

        AtomicNode *outputgate_hidden = linear(graph, lstm_params.output_hidden, *last_hidden);
        AtomicNode *outputgate_input = linear(graph, lstm_params.output_input, input);
        AtomicNode *outputgate_add = add(graph, {outputgate_hidden, outputgate_input});
        AtomicNode *outputgate = sigmoid(graph, *outputgate_add);

        AtomicNode *halfcell_hidden = linear(graph, lstm_params.cell_hidden, *last_hidden);
        AtomicNode *halfcell_input = linear(graph, lstm_params.cell_input, input);
        AtomicNode *halfcell_add = add(graph, {halfcell_hidden, halfcell_input});
        AtomicNode *halfcell = tanh(graph, *halfcell_add);
        AtomicNode *inputfilter = pointwiseMultiply(graph, *inputgate, *halfcell);
        AtomicNode *forgetfilter = pointwiseMultiply(graph, *last_cell, *forgetgate);
        AtomicNode *cell = add(graph, {inputfilter, forgetfilter});
        AtomicNode *halfhidden = tanh(graph, *cell);
        AtomicNode *hidden = pointwiseMultiply(graph, *halfhidden, *outputgate);
        hidden = n3ldg_plus::dropout(graph, *hidden, dropout, is_training);
        _hiddens.push_back(hidden);
        _cells.push_back(cell);
    }
};

#endif
