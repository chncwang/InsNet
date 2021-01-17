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
        function<dtype(int, int)> init_att = [](int out, int in) ->dtype {
            return sqrt(2.0 / (out * 5));
        };
        q_.init(out_dim, in_dim, false, &init_att, InitDistribution::NORM);
        k_.init(out_dim, in_dim, false, &init_att, InitDistribution::NORM);
        v_.init(out_dim, in_dim, false, &init_att, InitDistribution::NORM);
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
        heads_fusion_params_(name + "-heads_fusion_params"),
        ffn_inner_params_(name + "-ffn_inner_params"),
        ffn_outter_params_(name + "-ffn_outter_params"),
        layer_norm_a_(name + "-layer_norm_a"), layer_norm_b_(name + "-layer_norm_b") {}

    void init(int dim, int head_count) {
        if (dim % head_count != 0) {
            cerr << "out_dim:" << dim << " head_count:" << head_count << endl;
            abort();
        }
        multi_head_attention_params_.init(dim, dim);

        function<dtype(int, int)> init_relu = [](int out, int in) ->dtype {
            return sqrt(2.0 / (out + in));
        };
        heads_fusion_params_.init(dim, dim, false, &init_relu, InitDistribution::NORM);
        ffn_inner_params_.init(4 * dim, dim, true, &init_relu, InitDistribution::NORM);
        ffn_outter_params_.init(dim, 4 * dim, true, &init_relu, InitDistribution::NORM);
        layer_norm_a_.init(dim);
        layer_norm_b_.init(dim);
    }

    AttentionHeadParams &multiHeadAttentionParams() {
        return multi_head_attention_params_;
    }

    UniParams &headsFusionParams() {
        return heads_fusion_params_;
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
        json["heads_fusion_params"] = heads_fusion_params_.toJson();
        json["ffn_inner_params"] = ffn_inner_params_.toJson();
        json["ffn_outter_params"] = ffn_outter_params_.toJson();
        json["layer_norm_a"] = layer_norm_a_.toJson();
        json["layer_norm_b"] = layer_norm_b_.toJson();
        return json;
    }

    void fromJson(const Json::Value &json) override {
        multi_head_attention_params_.fromJson(json["multi_head_attention_params"]);
        heads_fusion_params_.fromJson(json["heads_fusion_params"]);
        ffn_inner_params_.fromJson(json["ffn_inner_params"]);
        ffn_outter_params_.fromJson(json["ffn_outter_params"]);
        layer_norm_a_.fromJson(json["layer_norm_a"]);
        layer_norm_b_.fromJson(json["layer_norm_b"]);
    }

#if USE_GPU
    std::vector<n3ldg_cuda::Transferable *> transferablePtrs() override {
        return {&multi_head_attention_params_, &heads_fusion_params_,
            &ffn_inner_params_, &ffn_outter_params_, &layer_norm_a_, &layer_norm_b_};
    }
#endif

protected:
    virtual std::vector<Tunable<BaseParam>*> tunableComponents() override {
        return {&multi_head_attention_params_, &heads_fusion_params_, &ffn_inner_params_,
            &ffn_outter_params_, &layer_norm_a_, &layer_norm_b_};
    }

private:
    AttentionHeadParams multi_head_attention_params_;
    UniParams heads_fusion_params_;
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
        layer_params_(name + "-layer_params") {}

    TransformerParams(const TransformerParams<T> &p) = delete;

    void init(int layer, int hidden_dim, int head_count, int max_sentence_len) {
        initPositionalEncodingParam(positional_encoding_param_, hidden_dim, max_sentence_len);

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
        json["layer_params"] = layer_params_.toJson();
        return json;
    }

    void fromJson(const Json::Value &json) override {
        positional_encoding_param_.fromJson(json["positional_encoding_param"]);
        layer_params_.fromJson(json["layer_params"]);
    }

#if USE_GPU
    std::vector<n3ldg_cuda::Transferable *> transferablePtrs() override {
        return {&positional_encoding_param_, &layer_params_};
    }
#endif

protected:
    virtual std::vector<Tunable<BaseParam>*> tunableComponents() override {
        return {&layer_params_};
    }

private:
    Param positional_encoding_param_;
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
    vector<Node *> pos_encoded_layer;
    int sentence_len = inputs.size();
    pos_encoded_layer.reserve(sentence_len);
    for (int i = 0; i < sentence_len; ++i) {
        Node *embedding = n3ldg_plus::embedding(graph, params.positionalEncodingParam(), i, false);
        Node *input = inputs.at(i);
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

        vector<Node *> normed;
        for (int m = 0; m < sentence_len; ++m) {
            Node *input = last_layer.at(m);
            input = layerNormalization(graph, layer_params.layerNormA(), *input);
            normed.push_back(input);
        }

        auto &attention_head_params = layer_params.multiHeadAttentionParams();
        vector<Node *> keys, values;
        keys.reserve(sentence_len);
        values.reserve(sentence_len);
        for (int m = 0; m < sentence_len; ++m) {
            Node *kv_input = normed.at(m);
            Node *k = linear(graph, attention_head_params.k(), *kv_input);
            keys.push_back(k);
            Node *v = linear(graph, attention_head_params.v(), *kv_input);
            values.push_back(v);
        }
        Node *key_matrix = concatToMatrix(graph, keys);
        Node *value_matrix = concatToMatrix(graph, values);

        vector<Node *> sub_layer;
        for (int j = 0; j < sentence_len; ++j) {
            auto &attention_head_params = layer_params.multiHeadAttentionParams();
            Node *q_input = normed.at(j);
            Node *q = linear(graph, attention_head_params.q(), *q_input);

            Node *attended = n3ldg_plus::dotAttention(graph, *key_matrix,
                    *value_matrix, *q, sentence_len, params.headCount()).first;
            attended = n3ldg_plus::linear(graph, layer_params.headsFusionParams(), *attended);
            attended = n3ldg_plus::dropout(graph, *attended, dropout, is_training);
            Node *added = add(graph, {attended, last_layer.at(j)});
            Node *normed = layerNormalization(graph, layer_params.layerNormB(), *added);
            Node *t = linear(graph, layer_params.ffnInnerParams(), *normed);
            t = relu(graph, *t);
            t = linear(graph, layer_params.ffnOutterParams(), *t);
            t = n3ldg_plus::dropout(graph, *t, dropout, is_training);
            t = add(graph, {added, t});
            sub_layer.push_back(t);
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
        vector<Node *> encoder_keys, encoder_values;
        for (int i = 0; i < layer_count; ++i) {
            auto &layer_params = *params_->layerParams().ptrs().at(i);
            auto &attention_head_params = layer_params.multiHeadAttentionParams();
            for (Node *hidden : encoder_hiddens_) {
                Node *k = linear(*graph_, attention_head_params.k(), *hidden);
                encoder_keys.push_back(k);
                Node *v = linear(*graph_, attention_head_params.v(), *hidden);
                encoder_values.push_back(v);
            }
        }

        encoder_key_matrix_ = concat(*graph_, encoder_keys);
        encoder_value_matrix_ = concat(*graph_, encoder_values);

        prepared_ = true;
    }

    const vector<vector<Node *>> &hiddenLayers() {
        return hidden_layers_;
    }

protected:
    Graph *graph_ = nullptr;
    TransformerDecoderParams *params_ = nullptr;

    vector<Node *> encoder_hiddens_;

    Node *encoder_key_matrix_;
    Node *encoder_value_matrix_;

    dtype dropout_;
    bool is_training_;

    vector<vector<Node *>> hidden_layers_;

    bool prepared_ = false;
};

class TransformerDecoderCellBuilder : public TransformerDecoderBuilderAbs {
public:
    TransformerDecoderCellBuilder(Graph &graph, TransformerDecoderParams &params,
            const vector<Node *> &encoder_hiddens,
            dtype dropout,
            bool is_training) : TransformerDecoderBuilderAbs(graph, params, encoder_hiddens,
            dropout, is_training) {
        for (int i = 0; i < params.layerCount(); ++i) {
            key_matrix_layers_.push_back(nullptr);
            value_matrix_layers_.push_back(nullptr);
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
        Node *pos_encoded = add(*graph_, {&decoder_input, embedding});
        pos_encoded = n3ldg_plus::dropout(*graph_, *pos_encoded, dropout_, is_training_);

        int layer_count = params_->layerCount();

        Node *last_layer_node = pos_encoded;
        for (int i = 0; i < layer_count; ++i) {
            auto &layer_params = *params_->layerParams().ptrs().at(i);
            auto &attention_head_params = layer_params.maskedMultiHeadAttentionParams();
            Node *normed = layerNormalization(*graph_, layer_params.layerNormA(),
                    *last_layer_node);
            Node *k = linear(*graph_, attention_head_params.k(), *normed);
            Node *v = linear(*graph_, attention_head_params.v(), *normed);

            Node *&key_matrix = key_matrix_layers_.at(i);
            key_matrix = key_matrix == nullptr ? k : concat(*graph_, {key_matrix, k});
            Node *&value_matrix = value_matrix_layers_.at(i);
            value_matrix = value_matrix == nullptr ? v : concat(*graph_, {value_matrix, v});

            Node *q = linear(*graph_, attention_head_params.q(), *last_layer_node);
            Node *attended = n3ldg_plus::dotAttention(*graph_, *key_matrix, *value_matrix, *q,
                    decoded_len_ + 1, params_->headCount()).first;
            attended = n3ldg_plus::dropout(*graph_, *attended, dropout_, is_training_);
            Node *added = add(*graph_, {attended, last_layer_node});
            normed = layerNormalization(*graph_, layer_params.layerNormB(), *added);

            auto &attention_head_params_for_encoder = layer_params.multiHeadAttentionParams();
            q = linear(*graph_, attention_head_params_for_encoder.q(), *normed);
            attended = dotAttention(*graph_, *encoder_key_matrix_, *encoder_value_matrix_, *q,
                    encoder_hiddens_.size(), params_->headCount()).first;
            attended = n3ldg_plus::dropout(*graph_, *attended, dropout_, is_training_);
            added = add(*graph_, {added, attended});
            normed = layerNormalization(*graph_, layer_params.layerNormC(), *added);

            Node *t = linear(*graph_, layer_params.ffnInnerParams(), *normed);
            t = relu(*graph_, *t);
            t = linear(*graph_, layer_params.ffnOutterParams(), *t);
            t = n3ldg_plus::dropout(*graph_, *t, dropout_, is_training_);
            added = add(*graph_, {added, t});
            last_layer_node = added;
            hidden_layers_.at(i).push_back(last_layer_node);
        }
        decoded_len_++;
    }

private:
    vector<Node *> key_matrix_layers_, value_matrix_layers_;

    int decoded_len_ = 0;
};

//class TransformerDecoderBuilder : public TransformerDecoderBuilderAbs {
//public:
//    TransformerDecoderBuilder(Graph &graph, TransformerDecoderParams &params,
//            const vector<Node *> &encoder_hiddens,
//            dtype dropout,
//            bool is_training) :
//        TransformerDecoderBuilderAbs(graph, params, encoder_hiddens, dropout, is_training) {}

//    void forward(vector<Node *> &decoder_inputs) {
//        if (!prepared_) {
//            cerr << "TransformerDecoderBuilder forward - not prepared" << endl;
//            abort();
//        }
//        using namespace n3ldg_plus;

//        vector<Node *> pos_encoded_layer;
//        int i = 0;
//        for (Node *decoder_input : decoder_inputs) {
//            Node *embedding = n3ldg_plus::embedding(*graph_, params_->positionalEncodingParam(),
//                    i++, false);
//            Node *input = linear(*graph_, params_->inputLinear(), *decoder_input);
//            Node *pos_encoded = add(*graph_, {input, embedding});
//            pos_encoded = n3ldg_plus::dropout(*graph_, *pos_encoded, dropout_, is_training_);
//            pos_encoded_layer.push_back(pos_encoded);
//        }

//        int layer_count = params_->layerCount();
//        int head_count = params_->headCount();
//        int hidden_dim = pos_encoded_layer.front()->getDim();
//        int section_dim = hidden_dim / head_count;

//        vector<Node *> *last_layer = &pos_encoded_layer;
//        for (int i = 0; i < layer_count; ++i) {
//            auto &layer_params = *params_->layerParams().ptrs().at(i);

//            vector<Node *> normed_nodes;
//            for (Node *last_layer_node : *last_layer) {
//                Node *normed = layerNormalization(*graph_, layer_params.layerNormA(),
//                        *last_layer_node);
//                normed_nodes.push_back(normed);
//            }

//            auto &attention_head_params = layer_params.maskedMultiHeadAttentionParams();

//             The outter vector represents heads, and the inner represents tokens;
//            vector<vector<Node *>> ks, vs;
//            ks.resize(head_count);
//            vs.resize(head_count);

//            for (Node *normed : normed_nodes) {
//                Node *k = linear(*graph_, attention_head_params.k(), *normed);
//                Node *v = linear(*graph_, attention_head_params.v(), *normed);
//                for (int j = 0; j < head_count; ++j) {
//                    ks.at(j).push_back(split(*graph_, section_dim, *k, j * section_dim));
//                    vs.at(j).push_back(split(*graph_, section_dim, *v, j * section_dim));
//                }
//            }
//            vector<Node *> key_matrices, value_matrices;
//            for (int j = 0; j < head_count; ++j) {
//                Node *key_matrix = concat(*graph_, ks.at(j));
//                key_matrices.push_back(key_matrix);
//                Node *value_matrix = concat(*graph_, vs.at(j));
//                value_matrices.push_back(value_matrix);
//            }
//            int token_i = 0;
//            for (Node *normed : normed_nodes) {
//                vector<Node *> key_matrix_heads, value_matrix_heads;
//                for (int j = 0; j < head_count; ++j) {
//                    Node *key_matrix = split(*graph_, (token_i + 1) * section_dim,
//                            *key_matrices.at(j), 0);
//                    key_matrix->setColumn(token_i + 1);
//                    key_matrix_heads.push_back(key_matrix);
//                    Node *value_matrix = split(*graph_, (token_i + 1) * section_dim,
//                            *value_matrices.at(j), 0);
//                    value_matrix->setColumn(token_i + 1);
//                    value_matrix_heads.push_back(value_matrix);
//                }

//                vector<Node *> attended_segments;
//                attended_segments.reserve(head_count);
//                Node *q = linear(*graph_, attention_head_params.q(), *normed);
//                for (int k = 0; k < head_count; ++k) {
//                    Node *split_q = split(*graph_, section_dim, *q, section_dim * k);
//                    Node *attended = n3ldg_plus::dotAttention(*graph_, *key_matrix_heads.at(k),
//                            *value_matrix_heads.at(k), *split_q).first;
//                    if (attended->getDim() * head_count != params_->hiddenDim()) {
//                        cerr << boost::format("attended_seg dim:%1% head_count:%2% hiddendim:%3%")
//                            % attended->getDim() % head_count % params_->hiddenDim() << endl;
//                        abort();
//                    }
//                    attended_segments.push_back(attended);
//                }
//                Node *concated = concat(*graph_, attended_segments);
//                concated = n3ldg_plus::dropout(*graph_, *concated, dropout_, is_training_);
//                Node *added = add(*graph_, {concated, last_layer->at(token_i)});

//                normed = layerNormalization(*graph_, layer_params.layerNormB(), *added);
//                auto &attention_head_params_for_encoder = layer_params.multiHeadAttentionParams();
//                Node *q_for_encoder = linear(*graph_, attention_head_params_for_encoder.q(),
//                        *normed);
//                vector<Node *> encoder_attended_segments;
//                for (int k = 0; k < head_count; ++k) {
//                    Node *split_q = split(*graph_, section_dim, *q_for_encoder, section_dim * k);
//                    Node *attended = n3ldg_plus::dotAttention(*graph_,
//                            *encoder_keys_.at(i).at(k), *encoder_values_.at(i).at(k),
//                            *split_q).first;
//                    if (attended->getDim() * head_count != params_->hiddenDim()) {
//                        cerr << boost::format("attended_seg dim:%1% head_count:%2% hiddendim:%3%")
//                            % attended->getDim() % head_count % params_->hiddenDim() << endl;
//                        abort();
//                    }
//                    encoder_attended_segments.push_back(attended);
//                }
//                concated = concat(*graph_, encoder_attended_segments);
//                concated = n3ldg_plus::dropout(*graph_, *concated, dropout_, is_training_);
//                added = add(*graph_, {concated, added});

//                normed = layerNormalization(*graph_, layer_params.layerNormC(), *added);
//                Node *t = linear(*graph_, layer_params.ffnInnerParams(), *normed);
//                t = relu(*graph_, *t);
//                t = linear(*graph_, layer_params.ffnOutterParams(), *t);
//                t = n3ldg_plus::dropout(*graph_, *t, dropout_, is_training_);
//                t = add(*graph_, {added, t});
//                hidden_layers_.at(i).push_back(t);
//                ++token_i;
//            }
//            last_layer = &hidden_layers_.at(i);
//        }
//    }
//};

}

#endif
