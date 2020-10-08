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
        multi_head_attention_params_.init(dim, dim);

        function<dtype(int, int)> init_relu = [](int out, int in) ->dtype {
            return sqrt(2.0 / (out + in));
        };
        ffn_inner_params_.init(4 * dim, dim, true, &init_relu);
        ffn_outter_params_.init(dim, 4 * dim, true, &init_relu);
        layer_norm_a_.init(dim);
        layer_norm_b_.init(dim);
    }

    AttentionHeadParams &multiHeadAttentionParams() {
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
    AttentionHeadParams multi_head_attention_params_;
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
        masked_multi_head_attention_params_.init(dim, dim);
        multi_head_attention_params_.init(dim, dim);

        function<dtype(int, int)> init_relu = [](int out, int in) ->dtype {
            return sqrt(2.0 / (out + in));
        };
        ffn_inner_params_.init(4 * dim, dim, true, &init_relu);
        ffn_outter_params_.init(dim, 4 * dim, true, &init_relu);
        layer_norm_a_.init(dim);
        layer_norm_b_.init(dim);
        layer_norm_c_.init(dim);
    }

    AttentionHeadParams &maskedMultiHeadAttentionParams() {
        return masked_multi_head_attention_params_;
    }

    AttentionHeadParams &multiHeadAttentionParams() {
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
    AttentionHeadParams masked_multi_head_attention_params_;
    AttentionHeadParams multi_head_attention_params_;
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
    n3ldg_cuda::Profiler &profiler = n3ldg_cuda::Profiler::Ins();
    vector<Node *> pos_encoded_layer;
    int sentence_len = inputs.size();
    pos_encoded_layer.reserve(sentence_len);
    for (int i = 0; i < sentence_len; ++i) {
        Node *embedding = n3ldg_plus::embedding(graph, params.positionalEncodingParam(), i, false);
        Node *input = linear(graph, params.inputLinear(), *inputs.at(i));
        int hidden_dim = params.hiddenDim();
        input = scaled(graph, *input, ::sqrt(hidden_dim));
        Node *pos_encoded = add(graph, {input, embedding});
        pos_encoded = n3ldg_plus::dropout(graph, *pos_encoded, dropout, is_training);
        pos_encoded_layer.push_back(pos_encoded);
    }

    int layer_count = params.layerCount();

    vector<Node *> last_layer = pos_encoded_layer;
    for (int i = 0; i < layer_count; ++i) {
        profiler.BeginEvent("multi head");
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
        auto &attention_head_params = layer_params.multiHeadAttentionParams();
        vector<Node *> keys, values;
        keys.reserve(sentence_len);
        values.reserve(sentence_len);
        for (int m = 0; m < sentence_len; ++m) {
            Node *kv_input = last_layer.at(m);
            Node *k = linear(graph, attention_head_params.k(), *kv_input);
            keys.push_back(k);
            Node *v = linear(graph, attention_head_params.v(), *kv_input);
            values.push_back(v);
        }
        int section_dim = keys.front()->getDim() / head_count;
        for (int j = 0; j < head_count; ++j) {
            vector<Node *> split_keys, split_values;
            split_keys.reserve(sentence_len);
            split_values.reserve(sentence_len);
            for (int m = 0; m < sentence_len; ++m) {
                split_keys.push_back(split(graph, section_dim, *keys.at(m), section_dim * j));
            }
            Node *key_matrix = concatToMatrix(graph, split_keys);
            key_heads.push_back(key_matrix);
            for (int m = 0; m < sentence_len; ++m) {
                split_values.push_back(split(graph, section_dim, *values.at(m), section_dim * j));
            }
            Node *value_matrix = concatToMatrix(graph, split_values);
            value_heads.push_back(value_matrix);
        }
        profiler.EndEvent();

        vector<Node *> sub_layer;
        for (int j = 0; j < sentence_len; ++j) {
            profiler.BeginEvent("multi head");
            vector<Node *> attended_segments;
            attended_segments.reserve(head_count);
            auto &attention_head_params = layer_params.multiHeadAttentionParams();
            Node *q_input = last_layer.at(j);
            Node *q = linear(graph, attention_head_params.q(), *q_input);

            for (int k = 0; k < head_count; ++k) {
                Node *split_q = split(graph, section_dim, *q, section_dim * k);
                Node *attended = n3ldg_plus::dotAttention(graph, *key_heads.at(k),
                        *value_heads.at(k), *split_q).first;
                if (attended->getDim() * head_count != params.hiddenDim()) {
                    cerr << boost::format("attended_seg dim:%1% head_count:%2% hiddendim:%3%") %
                        attended->getDim() % head_count % params.hiddenDim()
                        << endl;
                    abort();
                }
                attended_segments.push_back(attended);
            }

            Node *concated = concat(graph, attended_segments);
            profiler.EndEvent();
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

class TransformerDecoderBuilderAbs {
public:
    TransformerDecoderBuilderAbs(Graph &graph, TransformerDecoderParams &params,
            const vector<Node *> &encoder_hiddens,
            dtype dropout,
            bool is_training) :
        graph_(&graph), params_(&params), encoder_hiddens_(encoder_hiddens), dropout_(dropout),
    is_training_(is_training) {
        int layer_count = params.layerCount();
        hidden_layers_.resize(layer_count);
    }

    virtual ~TransformerDecoderBuilderAbs() = default;

    void prepare() {
        int layer_count = params_->layerCount();
        for (int i = 0; i < layer_count; ++i) {
            auto &layer_params = *params_->layerParams().ptrs().at(i);
            auto &attention_head_params = layer_params.multiHeadAttentionParams();
            vector<Node *> keys, values;
            for (Node *hidden : encoder_hiddens_) {
                Node *k = linear(*graph_, attention_head_params.k(), *hidden);
                keys.push_back(k);
                Node *v = linear(*graph_, attention_head_params.v(), *hidden);
                values.push_back(v);
            }
            int head_count = params_->headCount();
            int section_dim = keys.front()->getDim() / head_count;
            int sentence_len = encoder_hiddens_.size();
            vector<Node *> key_heads, value_heads;
            for (int j = 0; j < head_count; ++j) {
                vector<Node *> split_keys, split_values;
                for (int m = 0; m < sentence_len; ++m) {
                    split_keys.push_back(split(*graph_, section_dim, *keys.at(m),
                                section_dim * j));
                }
                Node *key_matrix = concatToMatrix(*graph_, split_keys);
                key_heads.push_back(key_matrix);
                for (int m = 0; m < sentence_len; ++m) {
                    split_values.push_back(split(*graph_, section_dim, *values.at(m),
                                section_dim * j));
                }
                Node *value_matrix = concatToMatrix(*graph_, split_values);
                value_heads.push_back(value_matrix);
            }
            encoder_keys_.push_back(key_heads);
            encoder_values_.push_back(value_heads);
        }
        prepared_ = true;
    }

    const vector<vector<Node *>> &hiddenLayers() {
        return hidden_layers_;
    }

protected:
    Graph *graph_ = nullptr;
    TransformerDecoderParams *params_ = nullptr;

    vector<Node *> encoder_hiddens_;

    // The outter vector represents layers, and the inner represents heads.
    vector<vector<Node *>> encoder_keys_, encoder_values_;

    dtype dropout_;
    bool is_training_;

    // The outter vector represents layers, and the inner represents sentences.
    vector<vector<Node *>> hidden_layers_;

    bool prepared_ = false;
};

class TransformerDecoderCellBuilder : public TransformerDecoderBuilderAbs {
public:
    TransformerDecoderCellBuilder(Graph &graph, TransformerDecoderParams &params,
            const vector<Node *> &encoder_hiddens,
            dtype dropout,
            bool is_training) :
        TransformerDecoderBuilderAbs(graph, params, encoder_hiddens, dropout, is_training) {
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
        if (!prepared_) {
            cerr << "TransformerDecoderBuilder forward - not prepared" << endl;
            abort();
        }
        using namespace n3ldg_plus;
        Node *embedding = n3ldg_plus::embedding(*graph_, params_->positionalEncodingParam(),
                decoded_len_, false);
        Node *input = linear(*graph_, params_->inputLinear(), decoder_input);
        Node *pos_encoded = add(*graph_, {input, embedding});
        pos_encoded = n3ldg_plus::dropout(*graph_, *pos_encoded, dropout_, is_training_);

        int layer_count = params_->layerCount();

        Node *last_layer_node = pos_encoded;
        for (int i = 0; i < layer_count; ++i) {
            auto &layer_params = *params_->layerParams().ptrs().at(i);
            auto &attention_head_params = layer_params.maskedMultiHeadAttentionParams();
            Node *k = linear(*graph_, attention_head_params.k(), *last_layer_node);
            Node *v = linear(*graph_, attention_head_params.v(), *last_layer_node);

            int head_count = params_->headCount();
            int section_dim = k->getDim() / head_count;
            for (int j = 0; j < head_count; ++j) {
                Node *&key_matrix = key_heads_layers_.at(i).at(j);
                Node *split_k = split(*graph_, section_dim, *k, section_dim * j);
                key_matrix = key_matrix == nullptr ? split_k :
                    concat(*graph_, {key_matrix, split_k});
                key_matrix->setColumn(key_matrix->getDim() / split_k->getDim());
                Node *&value_matrix = value_heads_layers_.at(i).at(j);
                Node *split_v = split(*graph_, section_dim, *v, section_dim * j);
                value_matrix = value_matrix == nullptr ? split_v :
                    concat(*graph_, {value_matrix, split_v});
                value_matrix->setColumn(value_matrix->getDim() / split_v->getDim());
            }

            vector<Node *> attended_segments;
            attended_segments.reserve(head_count);
            Node *q = linear(*graph_, attention_head_params.q(), *last_layer_node);
            for (int k = 0; k < head_count; ++k) {
                Node *split_q = split(*graph_, section_dim, *q, section_dim * k);
                Node *attended = n3ldg_plus::dotAttention(*graph_,
                        *key_heads_layers_.at(i).at(k), *value_heads_layers_.at(i).at(k),
                        *split_q).first;
                if (attended->getDim() * head_count != params_->hiddenDim()) {
                    cerr << boost::format("attended_seg dim:%1% head_count:%2% hiddendim:%3%")
                        % attended->getDim() % head_count % params_->hiddenDim() << endl;
                    abort();
                }
                attended_segments.push_back(attended);
            }
            Node *concated = concat(*graph_, attended_segments);
            concated = n3ldg_plus::dropout(*graph_, *concated, dropout_, is_training_);
            Node *added = add(*graph_, {concated, last_layer_node});
            Node *normed = layerNormalization(*graph_, layer_params.layerNormA(), *added);

            auto &attention_head_params_for_encoder = layer_params.multiHeadAttentionParams();
            Node *q_for_encoder = linear(*graph_, attention_head_params_for_encoder.q(), *normed);
            vector<Node *> encoder_attended_segments;
            for (int k = 0; k < head_count; ++k) {
                Node *split_q = split(*graph_, section_dim, *q_for_encoder, section_dim * k);
                Node *attended = n3ldg_plus::dotAttention(*graph_,
                        *encoder_keys_.at(i).at(k), *encoder_values_.at(i).at(k),
                        *split_q).first;
                if (attended->getDim() * head_count != params_->hiddenDim()) {
                    cerr << boost::format("attended_seg dim:%1% head_count:%2% hiddendim:%3%")
                        % attended->getDim() % head_count % params_->hiddenDim() << endl;
                    abort();
                }
                encoder_attended_segments.push_back(attended);
            }
            concated = concat(*graph_, encoder_attended_segments);
            concated = n3ldg_plus::dropout(*graph_, *concated, dropout_, is_training_);
            added = add(*graph_, {concated, normed});
            normed = layerNormalization(*graph_, layer_params.layerNormB(), *added);

            Node *t = linear(*graph_, layer_params.ffnInnerParams(), *normed);
            t = relu(*graph_, *t);
            t = linear(*graph_, layer_params.ffnOutterParams(), *t);
            t = n3ldg_plus::dropout(*graph_, *t, dropout_, is_training_);
            t = add(*graph_, {normed, t});
            Node *normed3 = layerNormalization(*graph_, layer_params.layerNormC(), *t);
            last_layer_node = normed3;
            hidden_layers_.at(i).push_back(last_layer_node);
        }
        decoded_len_++;
    }

private:
//     The outter vector represents layers, and the inner represents heads.
    vector<vector<Node *>> key_heads_layers_, value_heads_layers_;

    int decoded_len_ = 0;
};

class TransformerDecoderBuilder : public TransformerDecoderBuilderAbs {
public:
    TransformerDecoderBuilder(Graph &graph, TransformerDecoderParams &params,
            const vector<Node *> &encoder_hiddens,
            dtype dropout,
            bool is_training) :
        TransformerDecoderBuilderAbs(graph, params, encoder_hiddens, dropout, is_training) {}

    void forward(vector<Node *> &decoder_inputs) {
        if (!prepared_) {
            cerr << "TransformerDecoderBuilder forward - not prepared" << endl;
            abort();
        }
        using namespace n3ldg_plus;

        vector<Node *> pos_encoded_layer;
        int i = 0;
        for (Node *decoder_input : decoder_inputs) {
            Node *embedding = n3ldg_plus::embedding(*graph_, params_->positionalEncodingParam(),
                    i++, false);
            Node *input = linear(*graph_, params_->inputLinear(), *decoder_input);
            int hidden_dim = params_->hiddenDim();
            input = scaled(*graph_, *input, ::sqrt(hidden_dim));
            Node *pos_encoded = add(*graph_, {input, embedding});
            pos_encoded = n3ldg_plus::dropout(*graph_, *pos_encoded, dropout_, is_training_);
            pos_encoded_layer.push_back(pos_encoded);
        }

        int layer_count = params_->layerCount();
        int head_count = params_->headCount();
        int hidden_dim = pos_encoded_layer.front()->getDim();
        int section_dim = hidden_dim / head_count;

        vector<Node *> *last_layer = &pos_encoded_layer;
        for (int i = 0; i < layer_count; ++i) {
            auto &layer_params = *params_->layerParams().ptrs().at(i);
            auto &attention_head_params = layer_params.maskedMultiHeadAttentionParams();

            // The outter vector represents heads, and the inner represents tokens;
            vector<vector<Node *>> ks, vs;
            ks.resize(head_count);
            vs.resize(head_count);

            for (Node *last_layer_node : *last_layer) {
                Node *k = linear(*graph_, attention_head_params.k(), *last_layer_node);
                Node *v = linear(*graph_, attention_head_params.v(), *last_layer_node);
                for (int j = 0; j < head_count; ++j) {
                    ks.at(j).push_back(split(*graph_, section_dim, *k, j * section_dim));
                    vs.at(j).push_back(split(*graph_, section_dim, *v, j * section_dim));
                }
            }
            vector<Node *> key_matrices, value_matrices;
            for (int j = 0; j < head_count; ++j) {
                Node *key_matrix = concat(*graph_, ks.at(j));
                key_matrices.push_back(key_matrix);
                Node *value_matrix = concat(*graph_, vs.at(j));
                value_matrices.push_back(value_matrix);
            }
            int token_i = 0;
            for (Node *last_layer_node : *last_layer) {
                vector<Node *> key_matrix_heads, value_matrix_heads;
                for (int j = 0; j < head_count; ++j) {
                    Node *key_matrix = split(*graph_, (token_i + 1) * section_dim,
                            *key_matrices.at(j), 0);
                    key_matrix->setColumn(token_i + 1);
                    key_matrix_heads.push_back(key_matrix);
                    Node *value_matrix = split(*graph_, (token_i + 1) * section_dim,
                            *value_matrices.at(j), 0);
                    value_matrix->setColumn(token_i + 1);
                    value_matrix_heads.push_back(value_matrix);
                }

                vector<Node *> attended_segments;
                attended_segments.reserve(head_count);
                Node *q = linear(*graph_, attention_head_params.q(), *last_layer_node);
                for (int k = 0; k < head_count; ++k) {
                    Node *split_q = split(*graph_, section_dim, *q, section_dim * k);
                    Node *attended = n3ldg_plus::dotAttention(*graph_, *key_matrix_heads.at(k),
                            *value_matrix_heads.at(k), *split_q).first;
                    if (attended->getDim() * head_count != params_->hiddenDim()) {
                        cerr << boost::format("attended_seg dim:%1% head_count:%2% hiddendim:%3%")
                            % attended->getDim() % head_count % params_->hiddenDim() << endl;
                        abort();
                    }
                    attended_segments.push_back(attended);
                }
                Node *concated = concat(*graph_, attended_segments);
                concated = n3ldg_plus::dropout(*graph_, *concated, dropout_, is_training_);
                Node *added = add(*graph_, {concated, last_layer_node});
                Node *normed = layerNormalization(*graph_, layer_params.layerNormA(), *added);

                auto &attention_head_params_for_encoder = layer_params.multiHeadAttentionParams();
                Node *q_for_encoder = linear(*graph_, attention_head_params_for_encoder.q(),
                        *normed);
                vector<Node *> encoder_attended_segments;
                for (int k = 0; k < head_count; ++k) {
                    Node *split_q = split(*graph_, section_dim, *q_for_encoder, section_dim * k);
                    Node *attended = n3ldg_plus::dotAttention(*graph_,
                            *encoder_keys_.at(i).at(k), *encoder_values_.at(i).at(k),
                            *split_q).first;
                    if (attended->getDim() * head_count != params_->hiddenDim()) {
                        cerr << boost::format("attended_seg dim:%1% head_count:%2% hiddendim:%3%")
                            % attended->getDim() % head_count % params_->hiddenDim() << endl;
                        abort();
                    }
                    encoder_attended_segments.push_back(attended);
                }
                concated = concat(*graph_, encoder_attended_segments);
                concated = n3ldg_plus::dropout(*graph_, *concated, dropout_, is_training_);
                added = add(*graph_, {concated, normed});
                normed = layerNormalization(*graph_, layer_params.layerNormB(), *added);

                Node *t = linear(*graph_, layer_params.ffnInnerParams(), *normed);
                t = relu(*graph_, *t);
                t = linear(*graph_, layer_params.ffnOutterParams(), *t);
                t = n3ldg_plus::dropout(*graph_, *t, dropout_, is_training_);
                t = add(*graph_, {normed, t});
                Node *normed3 = layerNormalization(*graph_, layer_params.layerNormC(), *t);
                last_layer_node = normed3;
                hidden_layers_.at(i).push_back(last_layer_node);
                ++token_i;
            }
        }
    }
};

}

#endif
