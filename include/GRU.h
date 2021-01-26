#ifndef N3LDG_PLUS_GRU
#define N3LDG_PLUS_GRU

#include "MyLib.h"
#include "Node.h"
#include "AtomicOP.h"
#include "Graph.h"
#include "PMultiOP.h"
#include "PAddOP.h"
#include "BucketOP.h"
#include "UniOP.h"
#include "Sub.h"

#include <memory>

struct GRUParams : public N3LDGSerializable, TunableCombination<BaseParam>
#if USE_GPU
, public TransferableComponents
#endif
{
    UniParams update_input;
    UniParams update_hidden;
    UniParams reset_input;
    UniParams reset_hidden;
    UniParams candidate_input;
    UniParams candidate_hidden;

    GRUParams(const string &name) : update_input(name + "-update_input"),
    update_hidden(name + "-update_hidden"), reset_input(name + "reset_input"),
    reset_hidden(name + "reset_hidden"), candidate_input(name + "candidate_input"),
    candidate_hidden(name + "candidate_hidden") {}

    Json::Value toJson() const override {
        Json::Value json;
        json["update_input"] = update_input.toJson();
        json["update_hidden"] = update_hidden.toJson();
        json["reset_input"] = reset_input.toJson();
        json["reset_hidden"] = reset_hidden.toJson();
        json["candidate_input"] = candidate_input.toJson();
        json["candidate_hidden"] = candidate_hidden.toJson();
        return json;
    }

    void fromJson(const Json::Value &json) override {
        update_input.fromJson(json["update_input"]);
        update_hidden.fromJson(json["update_hidden"]);
        reset_input.fromJson(json["reset_input"]);
        reset_hidden.fromJson(json["reset_hidden"]);
        candidate_input.fromJson(json["candidate_input"]);
        candidate_hidden.fromJson(json["candidate_hidden"]);
    }

    void init(int out_size, int in_size) {
        update_input.init(out_size, in_size);
        update_hidden.init(out_size, out_size);
        reset_input.init(out_size, in_size);
        reset_hidden.init(out_size, out_size);
        candidate_input.init(out_size, in_size);
        candidate_hidden.init(out_size, out_size);
    }

    int inDim() {
        return update_input.W.inDim();
    }

    int outDim() {
        return update_input.W.outDim();
    }

#if USE_GPU
    std::vector<Transferable *> transferablePtrs() override {
        return {&update_input, &update_hidden, &reset_input, &reset_hidden, &candidate_input,
            &candidate_hidden};
    }

    virtual std::string name() const {
        return "GRUParams";
    }
#endif

protected:
    std::vector<Tunable<BaseParam> *> tunableComponents() override {
        return {&update_input, &update_hidden, &reset_input, &reset_hidden, &candidate_input,
            &candidate_hidden};
    }
};

struct DynamicGRUBuilder {
    vector<Node*> hiddens;

    int size() {
        return hiddens.size();
    }

    void forward(Graph &graph, GRUParams &gru_params, Node &input, Node &h0,
            dtype dropout,
            bool is_training) {
        int len = hiddens.size();
        Node *last_hidden = len == 0 ? &h0 : hiddens.at(len - 1);
        using namespace n3ldg_plus;

        Node *update_input = linear(graph, gru_params.update_input, input);
        Node *update_hidden = linear(graph, gru_params.update_hidden, *last_hidden);
        Node *update_gate = add(graph, {update_input, update_hidden});
        update_gate = sigmoid(graph, *update_gate);

        Node *reset_input = linear(graph, gru_params.reset_input, input);
        Node *reset_hidden = linear(graph, gru_params.reset_hidden, *last_hidden);
        Node *reset_gate = add(graph, {reset_input, reset_hidden});
        reset_gate = sigmoid(graph, *reset_gate);

        Node *candidate_input = linear(graph, gru_params.candidate_input, input);
        Node *updated_hidden = pointwiseMultiply(graph, *reset_gate, *last_hidden);
        Node *candidate_hidden = linear(graph, gru_params.candidate_hidden, *updated_hidden);
        Node *candidate = add(graph, {candidate_input, candidate_hidden});
        candidate = tanh(graph, *candidate);

        int hidden_dim = h0.getDim();
        Node *one = bucket(graph, hidden_dim, 1);
        Node *reversal_update = sub(graph, *one, *update_gate);
        Node *passed_last_hidden = pointwiseMultiply(graph, *reversal_update, *last_hidden);
        Node *updated_candidate = pointwiseMultiply(graph, *update_gate, *candidate);
        Node *h = add(graph, {passed_last_hidden, updated_candidate});
        hiddens.push_back(h);
    }
};

#endif
