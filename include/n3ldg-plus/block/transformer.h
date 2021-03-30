#ifndef N3LDG_PLUS_TRANSFORMER_H
#define N3LDG_PLUS_TRANSFORMER_H

#include "n3ldg-plus/operator/atomic.h"
#include "n3ldg-plus/operator/linear.h"
#include "n3ldg-plus/operator/concat.h"
#include "n3ldg-plus/operator/split.h"
#include "n3ldg-plus/operator/add.h"
#include "n3ldg-plus/operator/embedding.h"
#include "n3ldg-plus/operator/concat.h"
#include "n3ldg-plus/operator/layer_normalization.h"
#include "n3ldg-plus/block/attention.h"

namespace n3ldg_plus {

class AttentionHeadParams : public TunableCombination<BaseParam>
#if USE_GPU
, public TransferableComponents
#endif
{
public:
    AttentionHeadParams(const std::string &name) : q_(name + "-q"), k_(name + "-k"),
    v_(name + "-v") {}

    void init(int out_dim, int in_dim) {
        std::function<dtype(int, int)> init_att = [](int out, int in) ->dtype {
            return std::sqrt(2.0 / (out * 5));
        };
        q_.init(out_dim, in_dim, false, &init_att, InitDistribution::NORM);
        k_.init(out_dim, in_dim, false, &init_att, InitDistribution::NORM);
        v_.init(out_dim, in_dim, false, &init_att, InitDistribution::NORM);
    }

    LinearParam &q() {
        return q_;
    }

    LinearParam &k() {
        return k_;
    }

    LinearParam &v() {
        return v_;
    }

    template<typename Archive>
    void serialize(Archive &ar) {
        ar(q_, k_, v_);
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
    LinearParam q_;
    LinearParam k_;
    LinearParam v_;
};

class TransformerEncoderLayerParams :
    public TunableCombination<BaseParam>
#if USE_GPU
, public TransferableComponents
#endif
{
public:
    TransformerEncoderLayerParams(const std::string &name) :
        multi_head_attention_params_(name + "-multi_head_attention_params"),
        heads_fusion_params_(name + "-heads_fusion_params"),
        ffn_inner_params_(name + "-ffn_inner_params"),
        ffn_outter_params_(name + "-ffn_outter_params"),
        layer_norm_a_(name + "-layer_norm_a"), layer_norm_b_(name + "-layer_norm_b") {}

    void init(int dim, int head_count) {
        if (dim % head_count != 0) {
            std::cerr << "out_dim:" << dim << " head_count:" << head_count << std::endl;
            abort();
        }
        multi_head_attention_params_.init(dim, dim);

        std::function<dtype(int, int)> init_relu = [](int out, int in) ->dtype {
            return std::sqrt(2.0 / (out + in));
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

    LinearParam &headsFusionParams() {
        return heads_fusion_params_;
    }

    LinearParam &ffnInnerParams() {
        return ffn_inner_params_;
    }

    LinearParam &ffnOutterParams() {
        return ffn_outter_params_;
    }

    LayerNormalizationParams &layerNormA() {
        return layer_norm_a_;
    }

    LayerNormalizationParams &layerNormB() {
        return layer_norm_b_;
    }

    template<typename Archive>
    void serialize(Archive &ar) {
        ar(multi_head_attention_params_, heads_fusion_params_, ffn_inner_params_,
                ffn_outter_params_, layer_norm_a_, layer_norm_b_);
    }

#if USE_GPU
    std::vector<n3ldg_cuda::Transferable *> transferablePtrs() override {
        return {&multi_head_attention_params_, &heads_fusion_params_, &ffn_inner_params_,
            &ffn_outter_params_, &layer_norm_a_, &layer_norm_b_};
    }
#endif

protected:
    virtual std::vector<Tunable<BaseParam>*> tunableComponents() override {
        return {&multi_head_attention_params_, &heads_fusion_params_, &ffn_inner_params_,
            &ffn_outter_params_, &layer_norm_a_, &layer_norm_b_};
    }

private:
    AttentionHeadParams multi_head_attention_params_;
    LinearParam heads_fusion_params_;
    LinearParam ffn_inner_params_;
    LinearParam ffn_outter_params_;
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
class TransformerParams : public TunableCombination<BaseParam>
#if USE_GPU
, public TransferableComponents
#endif
{
public:
    TransformerParams(const std::string &name) :
        positional_encoding_param_(name + "-positional_encoding_param"),
        layer_params_(name + "-layer_params") {}

    TransformerParams(const TransformerParams<T> &p) = delete;

    void init(int layer, int hidden_dim, int head_count, int max_sentence_len) {
        initPositionalEncodingParam(positional_encoding_param_, hidden_dim, max_sentence_len);

        std::function<void(T &, int)> init_param = [&](T &params, int layer) {
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

    template<typename Archive>
    void serialize(Archive &ar) {
        ar(positional_encoding_param_, layer_params_);
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

class TransformerDecoderLayerParams : public TunableCombination<BaseParam>
#if USE_GPU
, public TransferableComponents
#endif
{
public:
    TransformerDecoderLayerParams(const std::string &name) :
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
            std::cerr << "out_dim:" << dim << " head_count:" << head_count << std::endl;
            abort();
        }
        self_attention_.init(dim, dim);
        encoder_attention_.init(dim, dim);
        self_fusion_.init(dim, dim);
        encoder_fusion_.init(dim, dim);

        std::function<dtype(int, int)> init_relu = [](int out, int in) ->dtype {
            return std::sqrt(2.0 / (out + in));
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

    LinearParam &selfFusion() {
        return self_fusion_;
    }

    LinearParam &encoderFusion() {
        return encoder_fusion_;
    }

    LinearParam &ffnInnerParams() {
        return ffn_inner_params_;
    }

    LinearParam &ffnOutterParams() {
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

    template<typename Archive>
    void serialize(Archive &ar) {
        ar(self_attention_, encoder_attention_, self_fusion_, encoder_fusion_, ffn_inner_params_,
                ffn_outter_params_, layer_norm_a_, layer_norm_b_, layer_norm_c_);
    }

#if USE_GPU
    std::vector<n3ldg_cuda::Transferable *> transferablePtrs() override {
        return {&self_attention_, &encoder_attention_, &self_fusion_, &encoder_fusion_,
            &ffn_inner_params_, &ffn_outter_params_, &layer_norm_a_, &layer_norm_b_,
            &layer_norm_c_};
    }
#endif

protected:
    virtual std::vector<Tunable<BaseParam>*> tunableComponents() override {
        return {&self_attention_, &encoder_attention_, &self_fusion_, &encoder_fusion_,
            &ffn_inner_params_, &ffn_outter_params_, &layer_norm_a_, &layer_norm_b_,
            &layer_norm_c_};
    }

private:
    AttentionHeadParams self_attention_;
    AttentionHeadParams encoder_attention_;
    LinearParam self_fusion_;
    LinearParam encoder_fusion_;
    LinearParam ffn_inner_params_;
    LinearParam ffn_outter_params_;
    LayerNormalizationParams layer_norm_a_;
    LayerNormalizationParams layer_norm_b_;
    LayerNormalizationParams layer_norm_c_;
};

typedef TransformerParams<TransformerDecoderLayerParams> TransformerDecoderParams;

class BatchedConcatNodeForDotAtt : public BatchedConcatNode {
public:
    void init(Graph &graph, BatchedNode &input, int head) {
        if (input.batch().size() % head != 0) {
            std::cerr << fmt::format("BatchedConcatNode init input batch size:{} head:{}",
                input.batch().size(), head) << std::endl;
            abort();
        }
        int sentence_len = input.batch().size() / head;
        int dim = head * input.getDim();
        allocateBatch(dim, sentence_len);
        for (int i = 0; i < sentence_len; ++i) {
            Node *node = batch().at(i);
            std::vector<Node *> inputs(head);
            for (int j = 0; j < head; ++j) {
                inputs.at(j) = input.batch().at(j * sentence_len + i);
            }
            node->setInputs(inputs);
        }
        afterInit(graph, {&input});
    }
};

BatchedNode *concatHeads(Graph &graph, BatchedNode &input, int head) {
    BatchedConcatNodeForDotAtt *node = new BatchedConcatNodeForDotAtt;
    node->init(graph, input, head);
    return node;
}

Node *dotAttention(Graph &graph, Node& k, Node& v, int v_col, Node& q, int q_col,
        int head_count,
        LinearParam &fusion_param,
        dtype dropout_value,
        bool use_mask,
        bool is_training) {
    int row = q.getDim() / q_col;
    int head_dim = row / head_count;
    std::vector<int> offsets(head_count);
    for (int i = 0; i < head_count; ++i) {
        offsets.at(i) = i * head_dim;
    }

    BatchedNode *split_q = split(graph, q, head_dim, offsets, q_col);
    BatchedNode *split_k = split(graph, k, head_dim, offsets, v_col);
    BatchedNode *split_v = split(graph, v, head_dim, offsets, v_col);
    BatchedNode *split_attended = dotAttention(graph, *split_k, *split_v, *split_q,
            q_col, use_mask).first;
    Node *attended_matrix = concat(graph, *split_attended, q_col);
    attended_matrix = linear(graph, *attended_matrix, fusion_param);
    attended_matrix = dropout(graph, *attended_matrix, dropout_value, is_training);
    return attended_matrix;
}

Node *transformerEncoder(Graph &graph, TransformerEncoderParams &params, Node &inputs,
        int sentence_len,
        dtype dropout_value,
        bool is_training) {
    std::vector<int> pos_ids;
    pos_ids.reserve(sentence_len);
    for (int i = 0; i < sentence_len; ++i) {
        pos_ids.push_back(i);
    }

    Node *pos_emb = embedding(graph, params.positionalEncodingParam(), pos_ids, false);
    Node *scaled_input = scaled(graph, inputs, ::sqrt(inputs.getDim() / inputs.getColumn()));
    Node *pos_encoded = add(graph, {pos_emb, scaled_input});
    pos_encoded = dropout(graph, *pos_encoded, dropout_value, is_training);

    int layer_count = params.layerCount();

    Node *last_layer = pos_encoded;
    for (int i = 0; i < layer_count; ++i) {
        auto &layer_params = *params.layerParams().ptrs().at(i);

        Node *normed = layerNormalization(graph, layer_params.layerNormA(), *last_layer,
                sentence_len);
        auto &attention_head_params = layer_params.multiHeadAttentionParams();
        Node *key = linear(graph, *normed, attention_head_params.k());
        Node *value = linear(graph, *normed, attention_head_params.v());
        Node *q = linear(graph, *normed, attention_head_params.q());
        Node *attended = dotAttention(graph, *key, *value, sentence_len, *q, sentence_len,
                params.headCount(), layer_params.headsFusionParams(), dropout_value, false,
                is_training);
        Node *added = add(graph, {attended, last_layer});
        normed = layerNormalization(graph, layer_params.layerNormB(), *added, sentence_len);
        Node *t = linear(graph, *normed, layer_params.ffnInnerParams());
        t = relu(graph, *t);
        t = linear(graph, *t, layer_params.ffnOutterParams());
        t = dropout(graph, *t, dropout_value, is_training);
        t = add(graph, {added, t});
        last_layer = t;
    }

    return last_layer;
}

class TransformerDecoderBuilderAbs {
public:
    TransformerDecoderBuilderAbs(Graph &graph, TransformerDecoderParams &params,
            Node &encoder_hiddens,
            int encoder_sentence_len,
            dtype dropout,
            bool is_training) : graph_(&graph), params_(&params),
    encoder_hiddens_(&encoder_hiddens), encoder_sentence_len_(encoder_sentence_len),
    dropout_(dropout), is_training_(is_training) {}

    virtual ~TransformerDecoderBuilderAbs() = default;

    virtual void prepare() {
        if (prepared_) {
            return;
        }
        int layer_count = params_->layerCount();
        for (int i = 0; i < layer_count; ++i) {
            auto &layer_params = *params_->layerParams().ptrs().at(i);
            auto &attention_head_params = layer_params.encoderAttention();
            Node *k = linear(*graph_, *encoder_hiddens_, attention_head_params.k());
            Node *v = linear(*graph_, *encoder_hiddens_, attention_head_params.v());
            encoder_key_matrices_.push_back(k);
            encoder_value_matrices_.push_back(v);
        }

        prepared_ = true;
    }

protected:
    Graph *graph_ = nullptr;
    TransformerDecoderParams *params_ = nullptr;

    Node *encoder_hiddens_;

    std::vector<Node *> encoder_key_matrices_;
    std::vector<Node *> encoder_value_matrices_;

    int encoder_sentence_len_;
    dtype dropout_;
    bool is_training_;

    bool prepared_ = false;
};

class TransformerDecoderCellBuilder : public TransformerDecoderBuilderAbs {
public:
    TransformerDecoderCellBuilder(Graph &graph, TransformerDecoderParams &params,
            Node &encoder_hiddens,
            int encoder_sentence_len,
            dtype dropout,
            bool is_training) : TransformerDecoderBuilderAbs(graph, params, encoder_hiddens,
                encoder_sentence_len, dropout, is_training) {
        for (int i = 0; i < params.layerCount(); ++i) {
            key_matrix_layers_.push_back(nullptr);
            value_matrix_layers_.push_back(nullptr);
        }
    }

    const std::vector<std::vector<Node *>> &hiddenLayers() {
        return hidden_layers_;
    }

    void prepare() override {
        if (prepared_) {
            return;
        }
        TransformerDecoderBuilderAbs::prepare();

        hidden_layers_.reserve(params_->layerCount());
        for (int i = 0; i < params_->layerCount(); ++i) {
            std::vector<Node *> nodes;
            hidden_layers_.push_back(nodes);
        }
    }

    void connect(Node &decoder_input) {
        if (!prepared_) {
            std::cerr << "TransformerDecoderBuilder forward - not prepared" << std::endl;
            abort();
        }
        Node *scaled_input = scaled(*graph_, decoder_input, std::sqrt(decoder_input.getDim()));
        Node *emb = embedding(*graph_, params_->positionalEncodingParam(),
                decoded_len_, false);
        Node *pos_encoded = add(*graph_, {scaled_input, emb});
        pos_encoded = dropout(*graph_, *pos_encoded, dropout_, is_training_);

        int layer_count = params_->layerCount();

        Node *last_layer_node = pos_encoded;
        for (int i = 0; i < layer_count; ++i) {
            auto &layer_params = *params_->layerParams().ptrs().at(i);
            auto &attention_head_params = layer_params.selfAttention();
            Node *normed = layerNormalization(*graph_, layer_params.layerNormA(),
                    *last_layer_node);
            Node *k = linear(*graph_, *normed, attention_head_params.k());
            Node *v = linear(*graph_, *normed, attention_head_params.v());

            Node *&key_matrix = key_matrix_layers_.at(i);
            key_matrix = key_matrix == nullptr ? k : concat(*graph_, {key_matrix, k});
            Node *&value_matrix = value_matrix_layers_.at(i);
            value_matrix = value_matrix == nullptr ? v : concat(*graph_, {value_matrix, v});

            Node *q = linear(*graph_, *normed, attention_head_params.q());
            Node *attended = dotAttention(*graph_, *key_matrix, *value_matrix, decoded_len_ + 1,
                    *q, 1, params_->headCount(), layer_params.selfFusion(), dropout_, false,
                    is_training_);
            Node *added = add(*graph_, {attended, last_layer_node});
            normed = layerNormalization(*graph_, layer_params.layerNormB(), *added);

            auto &attention_head_params_for_encoder = layer_params.encoderAttention();
            q = linear(*graph_, *normed, attention_head_params_for_encoder.q());
            attended = dotAttention(*graph_, *encoder_key_matrices_.at(i),
                    *encoder_value_matrices_.at(i), encoder_sentence_len_, *q, 1,
                     params_->headCount(), layer_params.encoderFusion(), dropout_, false,
                     is_training_);
            added = add(*graph_, {added, attended});
            normed = layerNormalization(*graph_, layer_params.layerNormC(), *added);

            Node *t = linear(*graph_, *normed, layer_params.ffnInnerParams());
            t = relu(*graph_, *t);
            t = linear(*graph_, *t, layer_params.ffnOutterParams());
            t = dropout(*graph_, *t, dropout_, is_training_);
            added = add(*graph_, {added, t});
            last_layer_node = added;
            hidden_layers_.at(i).push_back(last_layer_node);
        }
        decoded_len_++;
    }

private:
    std::vector<Node *> key_matrix_layers_, value_matrix_layers_;
    std::vector<std::vector<Node *>> hidden_layers_;
    int decoded_len_ = 0;
};

class TransformerDecoderBuilder : public TransformerDecoderBuilderAbs {
public:
    TransformerDecoderBuilder(Graph &graph, TransformerDecoderParams &params,
            Node &encoder_hiddens,
            int encoder_sentence_len,
            dtype dropout,
            bool is_training) : TransformerDecoderBuilderAbs(graph, params, encoder_hiddens,
                encoder_sentence_len, dropout, is_training) {}

    void connect(Node &inputs, int dec_sentence_len) {
        if (!prepared_) {
            std::cerr << "TransformerDecoderBuilder forward - not prepared" << std::endl;
            abort();
        }

        std::vector<int> pos_ids;
        for (int i = 0; i < dec_sentence_len; ++i) {
            pos_ids.push_back(i);
        }

        Node *pos_emb = embedding(*graph_, params_->positionalEncodingParam(), pos_ids, false);
        int row = inputs.getDim() / dec_sentence_len;
        Node *scaled_input = scaled(*graph_, inputs, ::sqrt(row));
        Node *pos_encoded = add(*graph_, {pos_emb, scaled_input});
        pos_encoded = dropout(*graph_, *pos_encoded, dropout_, is_training_);

        int layer_count = params_->layerCount();
        Node *last_layer = pos_encoded;

        for (int i = 0; i < layer_count; ++i) {
            auto &layer_params = *params_->layerParams().ptrs().at(i);
            auto &attention_head_params = layer_params.selfAttention();
            Node *normed = layerNormalization(*graph_, layer_params.layerNormA(), *last_layer,
                    dec_sentence_len);
            Node *k = linear(*graph_, *normed, attention_head_params.k());
            Node *v = linear(*graph_, *normed, attention_head_params.v());
            Node *q = linear(*graph_, *normed, attention_head_params.q());
            Node *attended = dotAttention(*graph_, *k, *v, dec_sentence_len, *q,
                    dec_sentence_len, params_->headCount(), layer_params.selfFusion(), dropout_,
                    true, is_training_);
            Node *added = add(*graph_, {attended, last_layer});
            normed = layerNormalization(*graph_, layer_params.layerNormB(), *added,
                    dec_sentence_len);

            auto &attention_head_params_for_encoder = layer_params.encoderAttention();
            q = linear(*graph_, *normed, attention_head_params_for_encoder.q());
            attended = dotAttention(*graph_, *encoder_key_matrices_.at(i),
                    *encoder_value_matrices_.at(i), encoder_sentence_len_, *q, dec_sentence_len,
                    params_->headCount(), layer_params.encoderFusion(), dropout_, false,
                    is_training_);
            added = add(*graph_, {added, attended});
            normed = layerNormalization(*graph_, layer_params.layerNormC(), *added,
                    dec_sentence_len);

            Node *t = linear(*graph_, *normed, layer_params.ffnInnerParams());
            t = relu(*graph_, *t);
            t = linear(*graph_, *t, layer_params.ffnOutterParams());
            t = dropout(*graph_, *t, dropout_, is_training_);
            added = add(*graph_, {added, t});
            last_layer = added; 
            hidden_layers_.push_back(last_layer);
        }
    }

    std::vector<Node *> &hiddenLayers() {
        return hidden_layers_;
    }

private:
    std::vector<Node *> hidden_layers_;
};

}

#endif
