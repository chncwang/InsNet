#ifndef N3LDG_PLUS_TRANSFORMER_H
#define N3LDG_PLUS_TRANSFORMER_H

#include "n3ldg-plus/operator/linear.h"
#include "n3ldg-plus/operator/layer_normalization.h"

namespace n3ldg_plus {

class AttentionHeadParams : public TunableCombination<BaseParam>
#if USE_GPU
, public cuda::TransferableComponents
#endif
{
public:
    AttentionHeadParams(const std::string &name = "");

    void init(int out_dim, int in_dim);

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
    std::vector<cuda::Transferable *> transferablePtrs() override;
#endif

protected:
    virtual std::vector<Tunable<BaseParam>*> tunableComponents() override;

private:
    LinearParam q_;
    LinearParam k_;
    LinearParam v_;
};

class TransformerEncoderLayerParams : public TunableCombination<BaseParam>
#if USE_GPU
, public cuda::TransferableComponents
#endif
{
public:
    TransformerEncoderLayerParams(const std::string &name = "");

    void init(int dim, int head_count);

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
    std::vector<cuda::Transferable *> transferablePtrs() override;
#endif

protected:
    virtual std::vector<Tunable<BaseParam>*> tunableComponents() override;

private:
    AttentionHeadParams multi_head_attention_params_;
    LinearParam heads_fusion_params_;
    LinearParam ffn_inner_params_;
    LinearParam ffn_outter_params_;
    LayerNormalizationParams layer_norm_a_;
    LayerNormalizationParams layer_norm_b_;
};

inline void initPositionalEncodingParam(Param &param, int dim, int max_sentence_len) {
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
            param.val()[pos_i][dim_i] = v;
        }
    }
#if USE_GPU
    param.val().copyFromHostToDevice();
#endif
}

template <typename T>
class TransformerParams : public TunableCombination<BaseParam>
#if USE_GPU
, public cuda::TransferableComponents
#endif
{
public:
    TransformerParams(const std::string &name = "") :
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
    std::vector<cuda::Transferable *> transferablePtrs() override {
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
, public cuda::TransferableComponents
#endif
{
public:
    TransformerDecoderLayerParams(const std::string &name = "");

    void init(int dim, int head_count);

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
    std::vector<cuda::Transferable *> transferablePtrs() override;
#endif

protected:
    virtual std::vector<Tunable<BaseParam>*> tunableComponents() override;

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

Node *dotAttention(Node& k, Node& v, int v_col, Node& q, int q_col, int head_count,
        LinearParam &fusion_param,
        dtype dropout_value,
        bool use_mask);

std::vector<Node *> transformerEncoder(Node &inputs, int sentence_len,
        TransformerEncoderParams &params,
        dtype dropout_value);

class TransformerDecoderBuilderAbs {
public:
    TransformerDecoderBuilderAbs(TransformerDecoderParams &params, Node &encoder_hiddens,
            int encoder_sentence_len,
            dtype dropout);

    virtual ~TransformerDecoderBuilderAbs() = default;

    virtual void prepare();

protected:
    Graph *graph_ = nullptr;
    TransformerDecoderParams *params_ = nullptr;

    Node *encoder_hiddens_;

    std::vector<Node *> encoder_key_matrices_;
    std::vector<Node *> encoder_value_matrices_;

    int encoder_sentence_len_;
    dtype dropout_;

    bool prepared_ = false;
};

class TransformerDecoderCellBuilder : public TransformerDecoderBuilderAbs {
public:
    TransformerDecoderCellBuilder(TransformerDecoderParams &params, Node &encoder_hiddens,
            int encoder_sentence_len,
            dtype dropout);

    const std::vector<std::vector<Node *>> &hiddenLayers() {
        return hidden_layers_;
    }

    void prepare() override;

    void step(Node &decoder_input);

private:
    std::vector<Node *> key_matrix_layers_, value_matrix_layers_;
    std::vector<std::vector<Node *>> hidden_layers_;
    int decoded_len_ = 0;
};

class TransformerDecoderBuilder : public TransformerDecoderBuilderAbs {
public:
    TransformerDecoderBuilder(TransformerDecoderParams &params, Node &encoder_hiddens,
            int encoder_sentence_len,
            dtype dropout);

    void connect(Node &inputs, int dec_sentence_len);

    std::vector<Node *> &hiddenLayers() {
        return hidden_layers_;
    }

private:
    std::vector<Node *> hidden_layers_;
};

std::vector<Node *> transformerDecoder(Node &encoder, int encoder_sentence_len, Node &input,
        int decoder_sentence_len,
        TransformerDecoderParams &params,
        dtype dropout_value);

}

#endif
