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

    LinearParams &q() {
        return q_;
    }

    LinearParams &k() {
        return k_;
    }

    LinearParams &v() {
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
    LinearParams q_;
    LinearParams k_;
    LinearParams v_;
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

    LinearParams &headsFusionParams() {
        return heads_fusion_params_;
    }

    LinearParams &ffnInnerParams() {
        return ffn_inner_params_;
    }

    LinearParams &ffnOutterParams() {
        return ffn_outter_params_;
    }

    LayerNormParams &layerNormA() {
        return layer_norm_a_;
    }

    LayerNormParams &layerNormB() {
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
    LinearParams heads_fusion_params_;
    LinearParams ffn_inner_params_;
    LinearParams ffn_outter_params_;
    LayerNormParams layer_norm_a_;
    LayerNormParams layer_norm_b_;
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

    LinearParams &selfFusion() {
        return self_fusion_;
    }

    LinearParams &encoderFusion() {
        return encoder_fusion_;
    }

    LinearParams &ffnInnerParams() {
        return ffn_inner_params_;
    }

    LinearParams &ffnOutterParams() {
        return ffn_outter_params_;
    }

    LayerNormParams &layerNormA() {
        return layer_norm_a_;
    }

    LayerNormParams &layerNormB() {
        return layer_norm_b_;
    }

    LayerNormParams &layerNormC() {
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
    LinearParams self_fusion_;
    LinearParams encoder_fusion_;
    LinearParams ffn_inner_params_;
    LinearParams ffn_outter_params_;
    LayerNormParams layer_norm_a_;
    LayerNormParams layer_norm_b_;
    LayerNormParams layer_norm_c_;
};

typedef TransformerParams<TransformerDecoderLayerParams> TransformerDecoderParams;

/// \ingroup module
/// The multi-head attention.
///
/// **The operators inside guarantee that multiheadAttention with the equal embed_dim, num_heads, Wo, dropout and use_mask will be executed in batch.**
/// For example, *multiheadAttention* in the same layer and mini-batch will be commonly executed in batch.
/// \param Q The query matrix before divided into multi-heads. Its size can be different with K and V, but should be divisble by embed_dim.
/// \param K The key matrix before divided into multi-heads. Its size should be equal to V and be divisible by embed_dim.
/// \param V The value matrix before divided into multi-heads. Its size should be equal to K and be divisible by embed_dim. **Note that multiheadAttention assumes Q, K and V are already linear transformed before they are passed so they will not be linear transformed again and as such multiheadAttention does not accept weight matrix parameters of Wq, Wk and Wv.**
/// \param embed_dim The row number of Q, K and V. It should be divisible by *num_heads*.
/// \param num_heads The head number.
/// \param Wo The weight matrix of the output linear transformation.
/// \param dropout The dropout value of the dropout following the output linear transformation.
/// \param mask Whether to mask future tokens in K, which is typically used in the Transformer decoder's self-attention. For the moment, user-defined masks are not supported yet.
/// \return The result matrix. Its size is equal to both Q.size().
Node *multiheadAttention(Node& Q, Node& K, Node& V, int embed_dim, int num_heads, LinearParams &Wo,
        dtype dropout,
        bool use_mask);

/// \ingroup module
/// The Transformer encoder.
/// 
/// **The operators inside guarantee that transformerEncoder with the same params and dropout will be executed in batch layer by layer.**
/// \param input The input matrix. Note that for the current version, the positional encoding is added inside transformerEncoder using sin and cos, which may lack flexibility.
/// \param params The Transformer encoder parameters.
/// \param dropout The dropout value. The dropout is added after self-attention and FFN, respectively.
/// \return The list of hidden matrices of each layer.
std::vector<Node *> transformerEncoder(Node &input, TransformerEncoderParams &params,
        dtype dropout);

class TransformerDecoderBuilderAbs {
public:
    TransformerDecoderBuilderAbs(TransformerDecoderParams &params, Node &encoder_hiddens,
            dtype dropout);

    virtual ~TransformerDecoderBuilderAbs() = default;

    virtual void prepare();

protected:
    Graph *graph_ = nullptr;
    TransformerDecoderParams *params_ = nullptr;

    Node *encoder_hiddens_;

    std::vector<Node *> encoder_key_matrices_;
    std::vector<Node *> encoder_value_matrices_;

    dtype dropout_;

    bool prepared_ = false;
};

class TransformerDecoderCellBuilder : public TransformerDecoderBuilderAbs {
public:
    TransformerDecoderCellBuilder(TransformerDecoderParams &params, Node &encoder_hiddens,
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
            dtype dropout);

    void connect(Node &inputs);

    std::vector<Node *> &hiddenLayers() {
        return hidden_layers_;
    }

private:
    std::vector<Node *> hidden_layers_;
};

/// \ingroup module
/// The Transformer decoder.
/// 
/// **The operators inside guarantee that transformerDecoder with the same params and dropout will be executed in batch layer by layer.**
/// \param input The input matrix. Note that for the current version, the positional encoding is added inside transformerDecoder using sin and cos, which may lack flexibility.
/// \param params The Transformer decoder parameters.
/// \param dropout The dropout value. The dropout is added after self-attention, cross-attention and FFN, respectively.
/// \return The list of hidden matrices of each layer.
std::vector<Node *> transformerDecoder(Node &encoder, Node &input,
        TransformerDecoderParams &params,
        dtype dropout_value);

}

#endif
