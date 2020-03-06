#ifndef N3LDG_PLUS_TRANSFORMER_H
#define N3LDG_PLUS_TRANSFORMER_H

#include "MyLib.h"
#include "Node.h"
#include "AtomicOP.h"
#include "Graph.h"
#include "UniOP.h"
#include "PAddOP.h"
#include "UniOP.h"
#include "LookupTable.h"
#include "Attention.h"
#include "layer_normalization.h"

class AttentionHeadParams : public N3LDGSerializable, public TunableCombination<BaseParam>
#if USE_GPU
, public TransferableComponents
#endif
{
public:
    AttentionHeadParams(const string &name) : q_(name + "-q"), k_(name + "-k"), v_(name + "-v") {}

    void init(int out_dim, int in_dim) {
        cout << boost::format("AttentionHeadParams init - out_dim:%1% in_dim:%2%") % out_dim %
            in_dim << endl;
        q_.init(out_dim, in_dim, false);
        k_.init(out_dim, in_dim, false);
        v_.init(out_dim, in_dim, false);
    }

    UniParams &q() {
        return q_;
    }

    UniParams &k() {
        return k_;
    }

    UniParams &v() {
        return v_;
    }

    Json::Value toJson() const override {
        Json::Value json;
        json["q"] = q_.toJson();
        json["k"] = k_.toJson();
        json["v"] = v_.toJson();
        return json;
    }

    void fromJson(const Json::Value &json) override {
        q_.fromJson(json["q"]);
        k_.fromJson(json["k"]);
        v_.fromJson(json["v"]);
    }

#if USE_GPU
    std::vector<n3ldg_cuda::Transferable *> transferablePtrs() override {
        return {&q_, &k_, &v_};
    }
#endif

protected:
    virtual std::vector<Tunable<BaseParam>*> tunableComponents() override {
        return {&q_, &k_, &v_};
    }

private:
    UniParams q_;
    UniParams k_;
    UniParams v_;
};

class TransformerEncoderLayerParams : public N3LDGSerializable,
    public TunableCombination<BaseParam>
#if USE_GPU
, public TransferableComponents
#endif
{
public:
    TransformerEncoderLayerParams(const string &name) :
        multi_head_attention_params_(name + "-multi_head_attention_params"),
        ffn_inner_params_(name + "-ffn_inner_params"),
        ffn_outter_params_(name + "-ffn_outter_params"),
        layer_norm_a_(name + "-layer_norm_a"), layer_norm_b_(name + "-layer_norm_b") {}

    void init(int dim, int head_count) {
        if (dim % head_count != 0) {
            cerr << "out_dim:" << dim << " head_count:" << head_count << endl;
            abort();
        }
        int section_out_dim = dim / head_count;
        cout << boost::format("section_out_dim:%1% dim:%2% head_count:%3%") % section_out_dim %
            dim % head_count << endl;
        function<void(AttentionHeadParams&, int)> param_init =
            [&](AttentionHeadParams &params, int len) {
            params.init(section_out_dim, dim);
        };
        multi_head_attention_params_.init(head_count, param_init);

        function<dtype(int, int)> init_relu = [](int out, int in) ->dtype {
            return sqrt(2.0 / (out + in));
        };
        ffn_inner_params_.init(4 * dim, dim, true, &init_relu);
        ffn_outter_params_.init(dim, 4 * dim, true, &init_relu);
        layer_norm_a_.init(dim);
        layer_norm_b_.init(dim);
    }

    ParamArray<AttentionHeadParams> &multiHeadAttentionParams() {
        return multi_head_attention_params_;
    }

    UniParams &ffnInnerParams() {
        return ffn_inner_params_;
    }

    UniParams &ffnOutterParams() {
        return ffn_outter_params_;
    }

    LayerNormalizationParams &layerNormA() {
        return layer_norm_a_;
    }

    LayerNormalizationParams &layerNormB() {
        return layer_norm_b_;
    }

    Json::Value toJson() const override {
        Json::Value json;
        json["multi_head_attention_params"] = multi_head_attention_params_.toJson();
        json["ffn_inner_params"] = ffn_inner_params_.toJson();
        json["ffn_outter_params"] = ffn_outter_params_.toJson();
        json["layer_norm_a"] = layer_norm_a_.toJson();
        json["layer_norm_b"] = layer_norm_b_.toJson();
        return json;
    }

    void fromJson(const Json::Value &json) override {
        multi_head_attention_params_.fromJson(json["multi_head_attention_params"]);
        ffn_inner_params_.fromJson(json["ffn_inner_params"]);
        ffn_outter_params_.fromJson(json["ffn_outter_params"]);
        layer_norm_a_.fromJson(json["layer_norm_a"]);
        layer_norm_b_.fromJson(json["layer_norm_b"]);
    }

#if USE_GPU
    std::vector<n3ldg_cuda::Transferable *> transferablePtrs() override {
        return {&multi_head_attention_params_, &ffn_inner_params_, &ffn_outter_params_,
            &layer_norm_a_, &layer_norm_b_};
    }
#endif

protected:
    virtual std::vector<Tunable<BaseParam>*> tunableComponents() override {
        return {&multi_head_attention_params_, &ffn_inner_params_, &ffn_outter_params_,
            &layer_norm_a_, &layer_norm_b_};
    }

private:
    ParamArray<AttentionHeadParams> multi_head_attention_params_;
    UniParams ffn_inner_params_;
    UniParams ffn_outter_params_;
    LayerNormalizationParams layer_norm_a_;
    LayerNormalizationParams layer_norm_b_;
};

class TransformerEncoderParams : public N3LDGSerializable, public TunableCombination<BaseParam>
#if USE_GPU
, public TransferableComponents
#endif
{
public:
    TransformerEncoderParams(const string &name) :
        positional_encoding_param_(name + "-positional_encoding_param"),
        input_linear_(name + "-input_linear"), layer_params_(name + "-layer_params") {}

    void init(int layer, int hidden_dim, int input_dim, int head_count, int max_sentence_len) {
        positional_encoding_param_.init(hidden_dim, max_sentence_len);
        for (int pos_i = 0; pos_i < max_sentence_len; ++pos_i) {
            for (int dim_i = 0; dim_i < hidden_dim; ++dim_i) {
                dtype v;
                if (dim_i % 2 == 0) {
                    int half = dim_i / 2;
                    v = sin(pos_i / pow(1e4, 2.0 * half / hidden_dim));
                } else {
                    int half = (dim_i - 1) / 2;
                    v = cos(pos_i / pow(1e4, 2.0 * half / hidden_dim));
                }
                positional_encoding_param_.val[pos_i][dim_i] = v;
            }
        }
#if USE_GPU
        positional_encoding_param_.val.copyFromHostToDevice();
#endif

        input_linear_.init(hidden_dim, input_dim, false);

        function<void(TransformerEncoderLayerParams &, int)> init_param =
            [&](TransformerEncoderLayerParams &params, int layer) {
                params.init(hidden_dim, head_count);
            };
        layer_params_.init(layer, init_param);
        head_count_ = head_count;
        hidden_dim_ = hidden_dim;
    }

    Param &positionalEncodingParam() {
        return positional_encoding_param_;
    }

    ParamArray<TransformerEncoderLayerParams> &layerParams() {
        return layer_params_;
    }

    UniParams &inputLinear() {
        return input_linear_;
    }

    int headCount() const {
        return head_count_;
    }

    int layerCount() const {
        return layer_params_.size();
    }

    int hiddenDim() const {
        return hidden_dim_;
    }

    Json::Value toJson() const override {
        Json::Value json;
        json["positional_encoding_param"] = positional_encoding_param_.toJson();
        json["input_linear"] = input_linear_.toJson();
        json["layer_params"] = layer_params_.toJson();
        return json;
    }

    void fromJson(const Json::Value &json) override {
        positional_encoding_param_.fromJson(json["positional_encoding_param"]);
        input_linear_.fromJson(json["input_linear"]);
        layer_params_.fromJson(json["layer_params"]);
    }

#if USE_GPU
    std::vector<n3ldg_cuda::Transferable *> transferablePtrs() override {
        return {&positional_encoding_param_, &input_linear_, &layer_params_};
    }
#endif

protected:
    virtual std::vector<Tunable<BaseParam>*> tunableComponents() override {
        return {&input_linear_, &layer_params_};
    }

private:
    Param positional_encoding_param_;
    UniParams input_linear_;
    ParamArray<TransformerEncoderLayerParams> layer_params_;
    int head_count_;
    int hidden_dim_;
};

namespace n3ldg_plus {

vector<Node *> transformerEncoder(Graph &graph, TransformerEncoderParams &params,
        vector<Node *> inputs,
        dtype dropout,
        bool is_training) {
    using namespace n3ldg_plus;
    vector<Node *> pos_encoded_layer;
    int sentence_len = inputs.size();
    pos_encoded_layer.reserve(sentence_len);
    for (int i = 0; i < sentence_len; ++i) {
        Node *embedding = n3ldg_plus::embedding(graph, params.positionalEncodingParam(), i, false);
        Node *input = linear(graph, params.inputLinear(), *inputs.at(i));
        Node *pos_encoded = add(graph, {input, embedding});
        pos_encoded = n3ldg_plus::dropout(graph, *pos_encoded, dropout, is_training);
        pos_encoded_layer.push_back(pos_encoded);
    }

    int layer_count = params.layerCount();

    vector<Node *> last_layer = pos_encoded_layer;
    for (int i = 0; i < layer_count; ++i) {
        if (last_layer.size() != sentence_len) {
            cerr << "transformer - last_layer.size():" << last_layer.size() << " sentence_len:"
                << sentence_len << endl;
            abort();
        }
        auto &layer_params = *params.layerParams().ptrs().at(i);

        int head_count = params.headCount();
        vector<vector<Node *>> key_heads, value_heads;
        key_heads.reserve(head_count);
        value_heads.reserve(head_count);
        for (int k = 0; k < head_count; ++k) {
            vector<Node *> keys, values;
            keys.reserve(sentence_len);
            values.reserve(sentence_len);
            auto &attention_head_params = *layer_params.multiHeadAttentionParams().ptrs().at(k);
            for (int m = 0; m < sentence_len; ++m) {
                Node *kv_input = last_layer.at(m);
                Node *k = linear(graph, attention_head_params.k(), *kv_input);
                keys.push_back(k);
                Node *v = linear(graph, attention_head_params.v(), *kv_input);
                values.push_back(v);
            }
            key_heads.push_back(move(keys));
            value_heads.push_back(move(values));
        }

        vector<Node *> sub_layer;
        for (int j = 0; j < sentence_len; ++j) {
            vector<Node *> attended_segments;
            attended_segments.reserve(head_count);
            for (int k = 0; k < head_count; ++k) {
                Node *q_input = last_layer.at(j);
                auto &attention_head_params =
                    *layer_params.multiHeadAttentionParams().ptrs().at(k);
                Node *q = linear(graph, attention_head_params.q(), *q_input);

                DotAttentionBuilder attention_builder;
                attention_builder.forward(graph, key_heads.at(k), value_heads.at(k), *q);
                if (attention_builder._hidden->getDim() * head_count != params.hiddenDim()) {
                    cerr << boost::format("attended_seg dim:%1% head_count:%2% hiddendim:%3%") %
                        attention_builder._hidden->getDim() % head_count % params.hiddenDim()
                        << endl;
                    abort();
                }
                attended_segments.push_back(attention_builder._hidden);
            }

            Node *concated = concat(graph, attended_segments);
            concated = n3ldg_plus::dropout(graph, *concated, dropout, is_training);
            Node *added = add(graph, {concated, last_layer.at(j)});
            Node *normed = layerNormalization(graph, layer_params.layerNormA(), *added);
            Node *t = linear(graph, layer_params.ffnInnerParams(), *normed);
            t = relu(graph, *t);
            t = linear(graph, layer_params.ffnOutterParams(), *t);
            t = n3ldg_plus::dropout(graph, *t, dropout, is_training);
            t = add(graph, {normed, t});
            Node *normed2 = layerNormalization(graph, layer_params.layerNormB(), *t);
            sub_layer.push_back(normed2);
        }
        last_layer = sub_layer;
    }

    return last_layer;
}

}

#endif
