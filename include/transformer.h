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

void initPositionalEncodingParam(Param &param, int dim, int max_sentence_len) {
    param.init(dim, max_sentence_len);
    for (int pos_i = 0; pos_i < max_sentence_len; ++pos_i) {
        for (int dim_i = 0; dim_i < dim; ++dim_i) {
            dtype v;
            if (dim_i % 2 == 0) {
                int half = dim_i / 2;
                v = sin(pos_i / pow(1e4, 2.0 * half / dim));
            } else {
                int half = (dim_i - 1) / 2;
                v = cos(pos_i / pow(1e4, 2.0 * half / dim));
            }
            param.val[pos_i][dim_i] = v;
        }
    }
#if USE_GPU
    param.val.copyFromHostToDevice();
#endif
}

template <typename T>
class TransformerParams : public N3LDGSerializable, public TunableCombination<BaseParam>
#if USE_GPU
, public TransferableComponents
#endif
{
public:
    TransformerParams(const string &name) :
        positional_encoding_param_(name + "-positional_encoding_param"),
        input_linear_(name + "-input_linear"), layer_params_(name + "-layer_params") {}

    TransformerParams(const TransformerParams<T> &p) = delete;

    void init(int layer, int hidden_dim, int input_dim, int head_count, int max_sentence_len) {
        initPositionalEncodingParam(positional_encoding_param_, hidden_dim, max_sentence_len);
        input_linear_.init(hidden_dim, input_dim, false);

        function<void(T &, int)> init_param = [&](T &params, int layer) {
                params.init(hidden_dim, head_count);
            };
        layer_params_.init(layer, init_param);
        head_count_ = head_count;
        hidden_dim_ = hidden_dim;
    }

    Param &positionalEncodingParam() {
        return positional_encoding_param_;
    }

    ParamArray<T> &layerParams() {
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
    ParamArray<T> layer_params_;
    int head_count_;
    int hidden_dim_;
};

typedef TransformerParams<TransformerEncoderLayerParams> TransformerEncoderParams;

class TransformerDecoderLayerParams : public N3LDGSerializable,
    public TunableCombination<BaseParam>
#if USE_GPU
, public TransferableComponents
#endif
{
public:
    TransformerDecoderLayerParams(const string &name) :
        masked_multi_head_attention_params_(name + "-masked_multi_head_attention_params"),
        multi_head_attention_params_(name + "-multi_head_attention_params"),
        ffn_inner_params_(name + "-ffn_inner_params"),
        ffn_outter_params_(name + "-ffn_outter_params"),
        layer_norm_a_(name + "-layer_norm_a"), layer_norm_b_(name + "-layer_norm_b"),
        layer_norm_c_(name + "-layer_norm_c") {}

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
        masked_multi_head_attention_params_.init(head_count, param_init);
        multi_head_attention_params_.init(head_count, param_init);

        function<dtype(int, int)> init_relu = [](int out, int in) ->dtype {
            return sqrt(2.0 / (out + in));
        };
        ffn_inner_params_.init(4 * dim, dim, true, &init_relu);
        ffn_outter_params_.init(dim, 4 * dim, true, &init_relu);
        layer_norm_a_.init(dim);
        layer_norm_b_.init(dim);
        layer_norm_c_.init(dim);
    }

    ParamArray<AttentionHeadParams> &maskedMultiHeadAttentionParams() {
        return masked_multi_head_attention_params_;
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

    LayerNormalizationParams &layerNormC() {
        return layer_norm_c_;
    }

    Json::Value toJson() const override {
        Json::Value json;
        json["masked_multi_head_attention_params"] = masked_multi_head_attention_params_.toJson();
        json["multi_head_attention_params"] = multi_head_attention_params_.toJson();
        json["ffn_inner_params"] = ffn_inner_params_.toJson();
        json["ffn_outter_params"] = ffn_outter_params_.toJson();
        json["layer_norm_a"] = layer_norm_a_.toJson();
        json["layer_norm_b"] = layer_norm_b_.toJson();
        json["layer_norm_c"] = layer_norm_c_.toJson();
        return json;
    }

    void fromJson(const Json::Value &json) override {
        masked_multi_head_attention_params_.fromJson(json["masked_multi_head_attention_params"]);
        multi_head_attention_params_.fromJson(json["multi_head_attention_params"]);
        ffn_inner_params_.fromJson(json["ffn_inner_params"]);
        ffn_outter_params_.fromJson(json["ffn_outter_params"]);
        layer_norm_a_.fromJson(json["layer_norm_a"]);
        layer_norm_b_.fromJson(json["layer_norm_b"]);
        layer_norm_c_.fromJson(json["layer_norm_c"]);
    }

#if USE_GPU
    std::vector<n3ldg_cuda::Transferable *> transferablePtrs() override {
        return {&masked_multi_head_attention_params_, &multi_head_attention_params_,
            &ffn_inner_params_, &ffn_outter_params_, &layer_norm_a_, &layer_norm_b_,
            &layer_norm_c_};
    }
#endif

protected:
    virtual std::vector<Tunable<BaseParam>*> tunableComponents() override {
        return {&masked_multi_head_attention_params_, &multi_head_attention_params_,
            &ffn_inner_params_, &ffn_outter_params_, &layer_norm_a_, &layer_norm_b_,
            &layer_norm_c_};
    }

private:
    ParamArray<AttentionHeadParams> masked_multi_head_attention_params_;
    ParamArray<AttentionHeadParams> multi_head_attention_params_;
    UniParams ffn_inner_params_;
    UniParams ffn_outter_params_;
    LayerNormalizationParams layer_norm_a_;
    LayerNormalizationParams layer_norm_b_;
    LayerNormalizationParams layer_norm_c_;
};

typedef TransformerParams<TransformerDecoderLayerParams> TransformerDecoderParams;

namespace n3ldg_plus {

vector<Node *> transformerEncoder(Graph &graph, TransformerEncoderParams &params,
        vector<Node *> &inputs,
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
        vector<Node *> key_heads, value_heads;
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
            Node *key_matrix = concatToMatrix(graph, keys);
            key_heads.push_back(key_matrix);
            Node *value_matrix = concatToMatrix(graph, values);
            value_heads.push_back(value_matrix);
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

                Node *attended = n3ldg_plus::dotAttention(graph, *key_heads.at(k),
                        *value_heads.at(k), *q).first;
                if (attended->getDim() * head_count != params.hiddenDim()) {
                    cerr << boost::format("attended_seg dim:%1% head_count:%2% hiddendim:%3%") %
                        attended->getDim() % head_count % params.hiddenDim()
                        << endl;
                    abort();
                }
                attended_segments.push_back(attended);
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

class TransformerDecoderBuilder {
public:
    TransformerDecoderBuilder(Graph &graph, TransformerDecoderParams &params,
            const vector<Node *> &encoder_hiddens,
            dtype dropout,
            bool is_training) :
        graph_(&graph), params_(&params), encoder_hiddens_(encoder_hiddens), dropout_(dropout),
    is_training_(is_training) {
        int layer_count = params.layerCount();
        for (int i = 0; i < layer_count; ++i) {
            vector<Node *> keys, values;
            for (int j = 0; j < params.headCount(); ++j) {
                keys.push_back(nullptr);
                values.push_back(nullptr);
            }
            key_heads_layers_.push_back(keys);
            value_heads_layers_.push_back(values);
        }
    }

    void forward(Node &decoder_input) {
        using namespace n3ldg_plus;
        int decoded_len = pos_encoded_layer_.size();
        Node *embedding = n3ldg_plus::embedding(*graph_, params_->positionalEncodingParam(),
                decoded_len, false);
        Node *input = linear(*graph_, params_->inputLinear(), decoder_input);
        Node *pos_encoded = add(*graph_, {input, embedding});
        pos_encoded = n3ldg_plus::dropout(*graph_, *pos_encoded, dropout_, is_training_);
        pos_encoded_layer_.push_back(pos_encoded);

        int layer_count = params_->layerCount();

        vector<Node *> &last_layer = pos_encoded_layer_;
        for (int i = 0; i < layer_count; ++i) {
            auto &layer_params = *params_->layerParams().ptrs().at(i);

            int head_count = params_->headCount();
            for (int j = 0; j < head_count; ++j) {
                auto &attention_head_params =
                    *layer_params.multiHeadAttentionParams().ptrs().at(j);
                Node *kv_input = last_layer.at(decoded_len);
                Node *k = linear(*graph_, attention_head_params.k(), *kv_input);
                Node *&key_matrix = key_heads_layers_.at(i).at(j);
                key_matrix = key_matrix == nullptr ? k : concat(*graph_, {key_matrix, k});
                key_matrix->setColumn(key_matrix->getDim() / k->getDim());
                Node *v = linear(*graph_, attention_head_params.v(), *kv_input);
                Node *&value_matrix = value_heads_layers_.at(i).at(j);
                value_matrix = value_matrix == nullptr ? v : concat(*graph_, {value_matrix, v});
                value_matrix->setColumn(value_matrix->getDim() / v->getDim());
            }

            vector<Node *> sub_layer;
            for (int j = 0; j < decoded_len + 1; ++j) {
                vector<Node *> attended_segments;
                attended_segments.reserve(head_count);
                for (int k = 0; k < head_count; ++k) {
                    Node *q_input = last_layer.at(j);
                    auto &attention_head_params =
                        *layer_params.multiHeadAttentionParams().ptrs().at(k);
                    Node *q = linear(*graph_, attention_head_params.q(), *q_input);

                    Node *attended = n3ldg_plus::dotAttention(*graph_,
                            *key_heads_layers_.at(i).at(k), *value_heads_layers_.at(i).at(k),
                            *q).first;
                    if (attended->getDim() * head_count != params_->hiddenDim()) {
                        cerr << boost::format("attended_seg dim:%1% head_count:%2% hiddendim:%3%")
                            % attended->getDim() % head_count % params_->hiddenDim() << endl;
                        abort();
                    }
                    attended_segments.push_back(attended);
                }

                Node *concated = concat(*graph_, attended_segments);
                concated = n3ldg_plus::dropout(*graph_, *concated, dropout_, is_training_);
                Node *added = add(*graph_, {concated, last_layer.at(j)});
                Node *normed = layerNormalization(*graph_, layer_params.layerNormA(), *added);
                Node *t = linear(*graph_, layer_params.ffnInnerParams(), *normed);
                t = relu(*graph_, *t);
                t = linear(*graph_, layer_params.ffnOutterParams(), *t);
                t = n3ldg_plus::dropout(*graph_, *t, dropout_, is_training_);
                t = add(*graph_, {normed, t});
                Node *normed2 = layerNormalization(*graph_, layer_params.layerNormB(), *t);
                sub_layer.push_back(normed2);
            }
            last_layer = sub_layer;
            hidden_layers_.push_back(last_layer);
        }
    }

    const vector<vector<Node *>> &hiddenLayers() {
        return hidden_layers_;
    }

private:
    Graph *graph_ = nullptr;
    TransformerDecoderParams *params_ = nullptr;
    vector<Node *> encoder_hiddens_;
    dtype dropout_;
    bool is_training_;
    vector<Node *> pos_encoded_layer_;
    vector<vector<Node *>> key_heads_layers_, value_heads_layers_;
    vector<vector<Node *>> hidden_layers_;
};

}

#endif
