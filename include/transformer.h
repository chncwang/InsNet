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
        self_attention_(name + "-self_attention"),
        encoder_attention_(name + "-encoder_attention"),
        self_fusion_(name + "-self_fusion"),
        encoder_fusion_(name+"encoder_fusion"),
        ffn_inner_params_(name + "-ffn_inner_params"),
        ffn_outter_params_(name + "-ffn_outter_params"),
        layer_norm_a_(name + "-layer_norm_a"), layer_norm_b_(name + "-layer_norm_b"),
        layer_norm_c_(name + "-layer_norm_c") {}

    void init(int dim, int head_count) {
        if (dim % head_count != 0) {
            cerr << "out_dim:" << dim << " head_count:" << head_count << endl;
            abort();
        }
        self_attention_.init(dim, dim);
        encoder_attention_.init(dim, dim);
        self_fusion_.init(dim, dim);
        encoder_fusion_.init(dim, dim);

        function<dtype(int, int)> init_relu = [](int out, int in) ->dtype {
            return sqrt(2.0 / (out + in));
        };
        ffn_inner_params_.init(4 * dim, dim, true, &init_relu);
        ffn_outter_params_.init(dim, 4 * dim, true, &init_relu);
        layer_norm_a_.init(dim);
        layer_norm_b_.init(dim);
        layer_norm_c_.init(dim);
    }

    AttentionHeadParams &selfAttention() {
        return self_attention_;
    }

    AttentionHeadParams &encoderAttention() {
        return encoder_attention_;
    }

    UniParams &selfFusion() {
        return self_fusion_;
    }

    UniParams &encoderFusion() {
        return encoder_fusion_;
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
        json["self_attention"] = self_attention_.toJson();
        json["encoder_attention"] = encoder_attention_.toJson();
        json["self_fusion"] = self_fusion_.toJson();
        json["encoder_fusion"] = encoder_fusion_.toJson();
        json["ffn_inner_params"] = ffn_inner_params_.toJson();
        json["ffn_outter_params"] = ffn_outter_params_.toJson();
        json["layer_norm_a"] = layer_norm_a_.toJson();
        json["layer_norm_b"] = layer_norm_b_.toJson();
        json["layer_norm_c"] = layer_norm_c_.toJson();
        return json;
    }

    void fromJson(const Json::Value &json) override {
        self_attention_.fromJson(json["self_attention"]);
        encoder_attention_.fromJson(json["encoder_attention"]);
        self_fusion_.fromJson(json["self_fusion"]);
        encoder_fusion_.fromJson(json["encoder_fusion"]);
        ffn_inner_params_.fromJson(json["ffn_inner_params"]);
        ffn_outter_params_.fromJson(json["ffn_outter_params"]);
        layer_norm_a_.fromJson(json["layer_norm_a"]);
        layer_norm_b_.fromJson(json["layer_norm_b"]);
        layer_norm_c_.fromJson(json["layer_norm_c"]);
    }

#if USE_GPU
    std::vector<n3ldg_cuda::Transferable *> transferablePtrs() override {
        return {&self_attention_, &encoder_attention_, &ffn_inner_params_, &ffn_outter_params_,
            &layer_norm_a_, &layer_norm_b_, &layer_norm_c_};
    }
#endif

protected:
    virtual std::vector<Tunable<BaseParam>*> tunableComponents() override {
        return {&self_attention_, &encoder_attention_, &ffn_inner_params_, &ffn_outter_params_,
            &layer_norm_a_, &layer_norm_b_, &layer_norm_c_};
    }

private:
    AttentionHeadParams self_attention_;
    AttentionHeadParams encoder_attention_;
    UniParams self_fusion_;
    UniParams encoder_fusion_;
    UniParams ffn_inner_params_;
    UniParams ffn_outter_params_;
    LayerNormalizationParams layer_norm_a_;
    LayerNormalizationParams layer_norm_b_;
    LayerNormalizationParams layer_norm_c_;
};

typedef TransformerParams<TransformerDecoderLayerParams> TransformerDecoderParams;

class BatchedConcatNodeForDotAtt : public BatchedConcatNode {
public:
    void init(Graph &graph, BatchedNode &input, int head) {
        if (input.batch().size() % head != 0) {
            cerr << boost::format("BatchedConcatNode init input batch size:%1% head:%2%") %
                input.batch().size() % head << endl;
            abort();
        }
        int sentence_len = input.batch().size() / head;
        int dim = head * input.getDim();
        allocateBatch(dim, sentence_len);
        for (int i = 0; i < sentence_len; ++i) {
            Node *node = batch().at(i);
            vector<Node *> inputs(head);
            for (int j = 0; j < head; ++j) {
                inputs.at(j) = input.batch().at(j * sentence_len + i);
            }
            node->setInputs(inputs);
        }
        afterInit(graph, {&input});
    }
};

BatchedNode *concat(Graph &graph, BatchedNode &input, int head) {
    BatchedConcatNodeForDotAtt *node = new BatchedConcatNodeForDotAtt;
    node->init(graph, input, head);
    return node;
}

namespace n3ldg_plus {

BatchedNode *dotAttention(Graph &graph, BatchedNode& k, BatchedNode& v, BatchedNode& q,
        bool is_decoder_self_att,
        int head_count,
        UniParams &fusion_param,
        dtype dropout,
        bool is_training) {
    vector<int> dims(k.batch().size());
    for (int i = 1; i <= k.batch().size(); ++i) {
        dims.at(i - 1) = i;
    }

    int head_dim = q.getDim() / head_count;
    vector<int> offsets(head_count);
    for (int i = 0; i < head_count; ++i) {
        offsets.at(i) = i * head_dim;
    }

    BatchedNode *split_q = split(graph, q, head_dim, offsets);
    BatchedNode *split_k = split(graph, k, head_dim, offsets);
    BatchedNode *key_matrix = concatToMatrix(graph, *split_k, head_count);
    BatchedNode *split_v = split(graph, v, head_dim, offsets);
    BatchedNode *value_matrix = concatToMatrix(graph, *split_v, head_count);
    BatchedNode *split_attended = n3ldg_plus::dotAttention(graph, *key_matrix,
            *value_matrix, *split_q, is_decoder_self_att ? &dims : nullptr).first;

    BatchedNode *attended = concat(graph, *split_attended, head_count);
    attended = n3ldg_plus::linear(graph, *attended, fusion_param);
    attended = n3ldg_plus::dropout(graph, *attended, dropout, is_training);
    return attended;
}

BatchedNode *dotAttentionEncoder(Graph &graph, BatchedNode& k, BatchedNode& v, BatchedNode& q,
        bool is_decoder_self_att,
        int head_count,
        UniParams &fusion_param,
        dtype dropout,
        bool is_training) {
    int dim = k.batch().size();

    int head_dim = q.getDim() / head_count;
    vector<int> offsets(head_count);
    for (int i = 0; i < head_count; ++i) {
        offsets.at(i) = i * head_dim;
    }

    BatchedNode *split_q = split(graph, q, head_dim, offsets);
    BatchedNode *query_matrix = concatToMatrix(graph, *split_q, head_count);
    BatchedNode *split_k = split(graph, k, head_dim, offsets);
    BatchedNode *key_matrix = concatToMatrix(graph, *split_k, head_count);
    BatchedNode *split_v = split(graph, v, head_dim, offsets);
    BatchedNode *value_matrix = concatToMatrix(graph, *split_v, head_count);
    BatchedNode *split_attended = n3ldg_plus::dotAttention(graph, *key_matrix,
            *value_matrix, *query_matrix, dim).first;

    BatchedNode *attended = concat(graph, *split_attended, head_count);
    attended = n3ldg_plus::linear(graph, *attended, fusion_param);
    attended = n3ldg_plus::dropout(graph, *attended, dropout, is_training);
    return attended;
}

BatchedNode *transformerEncoder(Graph &graph, TransformerEncoderParams &params,
        BatchedNode &inputs,
        dtype dropout,
        bool is_training) {
    using namespace n3ldg_plus;
    int sentence_len = inputs.batch().size();
    vector<int> pos_ids;
    pos_ids.reserve(sentence_len);
    for (int i = 0; i < sentence_len; ++i) {
        pos_ids.push_back(i);
    }

    BatchedNode *pos_emb = embedding(graph, params.positionalEncodingParam(), pos_ids, false);
    BatchedNode *scaled_input = scaled(graph, inputs, ::sqrt(inputs.getDim()));
    BatchedNode *pos_encoded = addInBatch(graph, {&pos_emb, &scaled_input});
    pos_encoded = n3ldg_plus::dropout(graph, *pos_encoded, dropout, is_training);

    int layer_count = params.layerCount();

    BatchedNode *last_layer = pos_encoded;
    for (int i = 0; i < layer_count; ++i) {
        if (last_layer->batch().size() != sentence_len) {
            cerr << "transformer - last_layer.size():" << last_layer->batch().size() <<
                " sentence_len:" << sentence_len << endl;
            abort();
        }
        auto &layer_params = *params.layerParams().ptrs().at(i);

        BatchedNode *normed = layerNormalization(graph, layer_params.layerNormA(), *last_layer);
        auto &attention_head_params = layer_params.multiHeadAttentionParams();
        BatchedNode *key = linear(graph, *normed, attention_head_params.k());
        BatchedNode *value = linear(graph, *normed, attention_head_params.v());
        BatchedNode *q = linear(graph, *normed, attention_head_params.q());
        BatchedNode *attended = dotAttentionEncoder(graph, *key, *value, *q, false,
                params.headCount(), layer_params.headsFusionParams(), dropout, is_training);
        BatchedNode *added = addInBatch(graph, {attended, last_layer});
        normed = layerNormalization(graph, layer_params.layerNormB(), *added);
        BatchedNode *t = linear(graph, *normed, layer_params.ffnInnerParams());
        t = relu(graph, *t);
        t = linear(graph, *t, layer_params.ffnOutterParams());
        t = n3ldg_plus::dropout(graph, *t, dropout, is_training);
        t = addInBatch(graph, {added, t});
        last_layer = t;
    }

    return last_layer;
//    return &inputs;
}

class TransformerDecoderBuilderAbs {
public:
    TransformerDecoderBuilderAbs(Graph &graph, TransformerDecoderParams &params,
            BatchedNode &encoder_hiddens,
            dtype dropout,
            bool is_training) :
        graph_(&graph), params_(&params), encoder_hiddens_(&encoder_hiddens), dropout_(dropout),
    is_training_(is_training) {}

    virtual ~TransformerDecoderBuilderAbs() = default;

    void prepare() {
        int layer_count = params_->layerCount();
        for (int i = 0; i < layer_count; ++i) {
            auto &layer_params = *params_->layerParams().ptrs().at(i);
            auto &attention_head_params = layer_params.encoderAttention();
            BatchedNode *k = linear(*graph_, *encoder_hiddens_, attention_head_params.k());
            BatchedNode *v = linear(*graph_, *encoder_hiddens_, attention_head_params.v());
            encoder_key_matrices_.push_back(k);
            encoder_value_matrices_.push_back(v);
        }

        prepared_ = true;
    }

protected:
    Graph *graph_ = nullptr;
    TransformerDecoderParams *params_ = nullptr;

    BatchedNode *encoder_hiddens_;

    vector<BatchedNode *> encoder_key_matrices_;
    vector<BatchedNode *> encoder_value_matrices_;

    dtype dropout_;
    bool is_training_;

    bool prepared_ = false;
};

class TransformerDecoderCellBuilder : public TransformerDecoderBuilderAbs {
public:
    TransformerDecoderCellBuilder(Graph &graph, TransformerDecoderParams &params,
            BatchedNode &encoder_hiddens,
            dtype dropout,
            bool is_training) : TransformerDecoderBuilderAbs(graph, params, encoder_hiddens,
            dropout, is_training) {
        for (int i = 0; i < params.layerCount(); ++i) {
            key_matrix_layers_.push_back(nullptr);
            value_matrix_layers_.push_back(nullptr);
        }
    }

    const vector<vector<Node *>> &hiddenLayers() {
        return hidden_layers_;
    }

    void forward(Node &decoder_input) {
//        if (!prepared_) {
//            cerr << "TransformerDecoderBuilder forward - not prepared" << endl;
//            abort();
//        }
//        using namespace n3ldg_plus;
//        Node *scaled_input = n3ldg_plus::scaled(*graph_, decoder_input,
//                ::sqrt(decoder_input.getDim()));
//        Node *embedding = n3ldg_plus::embedding(*graph_, params_->positionalEncodingParam(),
//                decoded_len_, false);
//        Node *pos_encoded = add(*graph_, {scaled_input, embedding});
//        pos_encoded = n3ldg_plus::dropout(*graph_, *pos_encoded, dropout_, is_training_);

//        int layer_count = params_->layerCount();

//        Node *last_layer_node = pos_encoded;
//        for (int i = 0; i < layer_count; ++i) {
//            auto &layer_params = *params_->layerParams().ptrs().at(i);
//            auto &attention_head_params = layer_params.selfAttention();
//            Node *normed = layerNormalization(*graph_, layer_params.layerNormA(),
//                    *last_layer_node);
//            Node *k = linear(*graph_, *normed, attention_head_params.k());
//            Node *v = linear(*graph_, *normed, attention_head_params.v());

//            Node *&key_matrix = key_matrix_layers_.at(i);
//            key_matrix = key_matrix == nullptr ? k : concat(*graph_, {key_matrix, k});
//            Node *&value_matrix = value_matrix_layers_.at(i);
//            value_matrix = value_matrix == nullptr ? v : concat(*graph_, {value_matrix, v});

//            Node *q = linear(*graph_, *normed, attention_head_params.q());
//            Node *attended = n3ldg_plus::dotAttention(*graph_, *key_matrix, *value_matrix, *q,
//                    decoded_len_ + 1, params_->headCount()).first;
//            attended = n3ldg_plus::linear(*graph_, *attended, layer_params.selfFusion());
//            attended = n3ldg_plus::dropout(*graph_, *attended, dropout_, is_training_);
//            Node *added = add(*graph_, {attended, last_layer_node});
//            normed = layerNormalization(*graph_, layer_params.layerNormB(), *added);

//            auto &attention_head_params_for_encoder = layer_params.encoderAttention();
//            q = linear(*graph_, *normed, attention_head_params_for_encoder.q());
//            attended = dotAttention(*graph_, *encoder_key_matrices_.at(i),
//                    *encoder_value_matrices_.at(i), *q, encoder_hiddens_->batch().size(),
//                    params_->headCount()).first;
//            attended = n3ldg_plus::linear(*graph_, *attended, layer_params.encoderFusion());
//            attended = n3ldg_plus::dropout(*graph_, *attended, dropout_, is_training_);
//            added = add(*graph_, {added, attended});
//            normed = layerNormalization(*graph_, layer_params.layerNormC(), *added);

//            Node *t = linear(*graph_, *normed, layer_params.ffnInnerParams());
//            t = relu(*graph_, *t);
//            t = linear(*graph_, *t, layer_params.ffnOutterParams());
//            t = n3ldg_plus::dropout(*graph_, *t, dropout_, is_training_);
//            added = add(*graph_, {added, t});
//            last_layer_node = added;
//            hidden_layers_.at(i).push_back(last_layer_node);
//        }
//        decoded_len_++;
    }

private:
    vector<Node *> key_matrix_layers_, value_matrix_layers_;
    vector<vector<Node *>> hidden_layers_;
//    int decoded_len_ = 0;

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
//            Node *input = linear(*graph_, *decoder_input, params_->inputLinear());
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
//                Node *k = linear(*graph_, *normed, attention_head_params.k());
//                Node *v = linear(*graph_, *normed, attention_head_params.v());
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
//                Node *q = linear(*graph_, *normed, attention_head_params.q());
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
//                Node *t = linear(*graph_, *normed, layer_params.ffnInnerParams());
//                t = relu(*graph_, *t);
//                t = linear(*graph_, *t, layer_params.ffnOutterParams());
//                t = n3ldg_plus::dropout(*graph_, *t, dropout_, is_training_);
//                t = add(*graph_, {added, t});
//                hidden_layers_.at(i).push_back(t);
//                ++token_i;
//            }
//            last_layer = &hidden_layers_.at(i);
//        }
//    }
//};

class TransformerDecoderBuilder : public TransformerDecoderBuilderAbs {
public:
    TransformerDecoderBuilder(Graph &graph, TransformerDecoderParams &params,
            BatchedNode &encoder_hiddens,
            dtype dropout,
            bool is_training) :
        TransformerDecoderBuilderAbs(graph, params, encoder_hiddens, dropout, is_training) {}

    void forward(BatchedNode &inputs) {
        using namespace n3ldg_plus;
        if (!prepared_) {
            cerr << "TransformerDecoderBuilder forward - not prepared" << endl;
            abort();
        }

        int sentence_len = inputs.batch().size();
        vector<int> pos_ids;
        for (int i = 0; i < sentence_len; ++i) {
            pos_ids.push_back(i);
        }

        BatchedNode *pos_emb = embedding(*graph_, params_->positionalEncodingParam(), pos_ids,
                false);
        BatchedNode *scaled_input = scaled(*graph_, inputs, ::sqrt(inputs.getDim()));
        BatchedNode *pos_encoded = addInBatch(*graph_, {&pos_emb, &scaled_input});
        pos_encoded = n3ldg_plus::dropout(*graph_, *pos_encoded, dropout_, is_training_);

        int layer_count = params_->layerCount();
        BatchedNode *last_layer = pos_encoded;
        for (int i = 0; i < layer_count; ++i) {
            auto &layer_params = *params_->layerParams().ptrs().at(i);
            auto &attention_head_params = layer_params.selfAttention();
            BatchedNode *normed = layerNormalization(*graph_, layer_params.layerNormA(),
                    *last_layer);
            BatchedNode *k = linear(*graph_, *normed, attention_head_params.k());
            BatchedNode *v = linear(*graph_, *normed, attention_head_params.v());
            BatchedNode *q = linear(*graph_, *normed, attention_head_params.q());
            BatchedNode *attended = dotAttention(*graph_, *k, *v, *q, true, params_->headCount(),
                    layer_params.selfFusion(), dropout_, is_training_);
            BatchedNode *added = addInBatch(*graph_, {attended, last_layer});
            normed = layerNormalization(*graph_, layer_params.layerNormB(), *added);

            auto &attention_head_params_for_encoder = layer_params.encoderAttention();
            q = linear(*graph_, *normed, attention_head_params_for_encoder.q());
            attended = dotAttention(*graph_, *encoder_key_matrices_.at(i),
                    *encoder_value_matrices_.at(i), *q, false, params_->headCount(),
                    layer_params.encoderFusion(), dropout_, is_training_);
            added = addInBatch(*graph_, {added, attended});
            normed = layerNormalization(*graph_, layer_params.layerNormC(), *added);

            BatchedNode *t = linear(*graph_, *normed, layer_params.ffnInnerParams());
            t = relu(*graph_, *t);
            t = linear(*graph_, *t, layer_params.ffnOutterParams());
            t = n3ldg_plus::dropout(*graph_, *t, dropout_, is_training_);
            added = addInBatch(*graph_, {added, t});
            last_layer = added;

            hidden_layers_.push_back(last_layer);
        }
    }

    vector<BatchedNode *> &hiddenLayers() {
        return hidden_layers_;
    }

private:
    vector<BatchedNode *> hidden_layers_;
};

}

#endif
