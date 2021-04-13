#include "n3ldg-plus/block/transformer.h"
#include "n3ldg-plus/operator/atomic.h"
#include "n3ldg-plus/operator/concat.h"
#include "n3ldg-plus/operator/split.h"
#include "n3ldg-plus/operator/add.h"
#include "n3ldg-plus/operator/embedding.h"
#include "n3ldg-plus/block/attention.h"

using std::string;
using std::function;
using std::vector;
using std::cerr;
using std::endl;

namespace n3ldg_plus {

AttentionHeadParams::AttentionHeadParams(const string &name) : q_(name + "-q"),
    k_(name + "-k"), v_(name + "-v") {}

void AttentionHeadParams::init(int out_dim, int in_dim) {
    function<dtype(int, int)> init_att = [](int out, int in) ->dtype {
        return std::sqrt(2.0 / (out * 5));
    };
    q_.init(out_dim, in_dim, false, &init_att, InitDistribution::NORM);
    k_.init(out_dim, in_dim, false, &init_att, InitDistribution::NORM);
    v_.init(out_dim, in_dim, false, &init_att, InitDistribution::NORM);
}


#if USE_GPU
vector<cuda::Transferable *> AttentionHeadParams::transferablePtrs() {
    return {&q_, &k_, &v_};
}
#endif

vector<Tunable<BaseParam>*> AttentionHeadParams::tunableComponents() {
    return {&q_, &k_, &v_};
}

TransformerEncoderLayerParams::TransformerEncoderLayerParams(const string &name) :
    multi_head_attention_params_(name + "-multi_head_attention_params"),
    heads_fusion_params_(name + "-heads_fusion_params"),
    ffn_inner_params_(name + "-ffn_inner_params"), ffn_outter_params_(name + "-ffn_outter_params"),
    layer_norm_a_(name + "-layer_norm_a"), layer_norm_b_(name + "-layer_norm_b") {}

void TransformerEncoderLayerParams::init(int dim, int head_count) {
    if (dim % head_count != 0) {
        cerr << "out_dim:" << dim << " head_count:" << head_count << endl;
        abort();
    }
    multi_head_attention_params_.init(dim, dim);

    function<dtype(int, int)> init_relu = [](int out, int in) ->dtype {
        return std::sqrt(2.0 / (out + in));
    };
    heads_fusion_params_.init(dim, dim, false, &init_relu, InitDistribution::NORM);
    ffn_inner_params_.init(4 * dim, dim, true, &init_relu, InitDistribution::NORM);
    ffn_outter_params_.init(dim, 4 * dim, true, &init_relu, InitDistribution::NORM);
    layer_norm_a_.init(dim);
    layer_norm_b_.init(dim);
}

#if USE_GPU
vector<cuda::Transferable *> TransformerEncoderLayerParams::transferablePtrs() {
    return {&multi_head_attention_params_, &heads_fusion_params_, &ffn_inner_params_,
        &ffn_outter_params_, &layer_norm_a_, &layer_norm_b_};
}
#endif

vector<Tunable<BaseParam>*> TransformerEncoderLayerParams::tunableComponents() {
    return {&multi_head_attention_params_, &heads_fusion_params_, &ffn_inner_params_,
        &ffn_outter_params_, &layer_norm_a_, &layer_norm_b_};
}

TransformerDecoderLayerParams::TransformerDecoderLayerParams(const string &name) :
    self_attention_(name + "-self_attention"), encoder_attention_(name + "-encoder_attention"),
    self_fusion_(name + "-self_fusion"), encoder_fusion_(name+"encoder_fusion"),
    ffn_inner_params_(name + "-ffn_inner_params"), ffn_outter_params_(name + "-ffn_outter_params"),
    layer_norm_a_(name + "-layer_norm_a"), layer_norm_b_(name + "-layer_norm_b"),
    layer_norm_c_(name + "-layer_norm_c") {}

void TransformerDecoderLayerParams::init(int dim, int head_count) {
    if (dim % head_count != 0) {
        cerr << "out_dim:" << dim << " head_count:" << head_count << endl;
        abort();
    }
    self_attention_.init(dim, dim);
    encoder_attention_.init(dim, dim);
    self_fusion_.init(dim, dim);
    encoder_fusion_.init(dim, dim);

    function<dtype(int, int)> init_relu = [](int out, int in) ->dtype {
        return std::sqrt(2.0 / (out + in));
    };
    ffn_inner_params_.init(4 * dim, dim, true, &init_relu);
    ffn_outter_params_.init(dim, 4 * dim, true, &init_relu);
    layer_norm_a_.init(dim);
    layer_norm_b_.init(dim);
    layer_norm_c_.init(dim);
}

#if USE_GPU
vector<cuda::Transferable *> TransformerDecoderLayerParams::transferablePtrs() {
    return {&self_attention_, &encoder_attention_, &self_fusion_, &encoder_fusion_,
        &ffn_inner_params_, &ffn_outter_params_, &layer_norm_a_, &layer_norm_b_, &layer_norm_c_};
}
#endif

vector<Tunable<BaseParam>*> TransformerDecoderLayerParams::tunableComponents() {
    return {&self_attention_, &encoder_attention_, &self_fusion_, &encoder_fusion_,
        &ffn_inner_params_, &ffn_outter_params_, &layer_norm_a_, &layer_norm_b_, &layer_norm_c_};
}

Node *dotAttention(Node& k, Node& v, int v_col, Node& q, int q_col, int head_count,
        LinearParam &fusion_param,
        dtype dropout_value,
        bool use_mask) {
    int row = q.getDim() / q_col;
    int head_dim = row / head_count;
    vector<int> offsets(head_count);
    for (int i = 0; i < head_count; ++i) {
        offsets.at(i) = i * head_dim;
    }

    BatchedNode *split_q = split(q, head_dim, offsets, q_col);
    BatchedNode *split_k = split(k, head_dim, offsets, v_col);
    BatchedNode *split_v = split(v, head_dim, offsets, v_col);
    BatchedNode *split_attended = dotAttention(*split_k, *split_v, *split_q, q_col,
            use_mask).first;
    Node *attended_matrix = concat(*split_attended, q_col);
    attended_matrix = linear(*attended_matrix, fusion_param);
    attended_matrix = dropout(*attended_matrix, dropout_value);
    return attended_matrix;
}

vector<Node *> transformerEncoder(Node &inputs, int sentence_len, TransformerEncoderParams &params,
        dtype dropout_value) {
    vector<int> pos_ids;
    pos_ids.reserve(sentence_len);
    for (int i = 0; i < sentence_len; ++i) {
        pos_ids.push_back(i);
    }

    Graph &graph = dynamic_cast<Graph &>(inputs.getNodeContainer());
    Node *pos_emb = embedding(graph, pos_ids, params.positionalEncodingParam(), false);
    Node *scaled_input = scaled(inputs, ::sqrt(inputs.getDim() / inputs.getColumn()));
    Node *pos_encoded = add({pos_emb, scaled_input});
    pos_encoded = dropout(*pos_encoded, dropout_value);

    int layer_count = params.layerCount();

    Node *last_layer = pos_encoded;
    vector<Node *> hiddens;
    hiddens.reserve(layer_count);
    for (int i = 0; i < layer_count; ++i) {
        auto &layer_params = *params.layerParams().ptrs().at(i);

        Node *normed = layerNormalization(layer_params.layerNormA(), *last_layer,
                sentence_len);
        auto &attention_head_params = layer_params.multiHeadAttentionParams();
        Node *key = linear(*normed, attention_head_params.k());
        Node *value = linear(*normed, attention_head_params.v());
        Node *q = linear(*normed, attention_head_params.q());
        Node *attended = dotAttention(*key, *value, sentence_len, *q, sentence_len,
                params.headCount(), layer_params.headsFusionParams(), dropout_value, false);
        Node *added = add({attended, last_layer});
        normed = layerNormalization(layer_params.layerNormB(), *added, sentence_len);
        Node *t = linear(*normed, layer_params.ffnInnerParams());
        t = relu(*t);
        t = linear(*t, layer_params.ffnOutterParams());
        t = dropout(*t, dropout_value);
        t = add({added, t});
        last_layer = t;
        hiddens.push_back(last_layer);
    }

    return hiddens;
}

TransformerDecoderBuilderAbs::TransformerDecoderBuilderAbs(TransformerDecoderParams &params,
        Node &encoder_hiddens,
        int encoder_sentence_len,
        dtype dropout) : params_(&params), encoder_hiddens_(&encoder_hiddens),
    encoder_sentence_len_(encoder_sentence_len), dropout_(dropout) {}

void TransformerDecoderBuilderAbs::prepare() {
    if (prepared_) {
        return;
    }
    int layer_count = params_->layerCount();
    for (int i = 0; i < layer_count; ++i) {
        auto &layer_params = *params_->layerParams().ptrs().at(i);
        auto &attention_head_params = layer_params.encoderAttention();
        Node *k = linear(*encoder_hiddens_, attention_head_params.k());
        Node *v = linear(*encoder_hiddens_, attention_head_params.v());
        encoder_key_matrices_.push_back(k);
        encoder_value_matrices_.push_back(v);
    }

    prepared_ = true;
}

TransformerDecoderCellBuilder::TransformerDecoderCellBuilder(TransformerDecoderParams &params,
        Node &encoder_hiddens,
        int encoder_sentence_len,
        dtype dropout) : TransformerDecoderBuilderAbs(params, encoder_hiddens,
            encoder_sentence_len, dropout) {
    for (int i = 0; i < params.layerCount(); ++i) {
        key_matrix_layers_.push_back(nullptr);
        value_matrix_layers_.push_back(nullptr);
    }
}

void TransformerDecoderCellBuilder::prepare() {
    if (prepared_) {
        return;
    }
    TransformerDecoderBuilderAbs::prepare();

    hidden_layers_.reserve(params_->layerCount());
    for (int i = 0; i < params_->layerCount(); ++i) {
        vector<Node *> nodes;
        hidden_layers_.push_back(nodes);
    }
}

void TransformerDecoderCellBuilder::step(Node &decoder_input) {
    if (!prepared_) {
        cerr << "TransformerDecoderBuilder forward - not prepared" << endl;
        abort();
    }
    Node *scaled_input = scaled(decoder_input,
            std::sqrt(static_cast<dtype>(decoder_input.getDim())));
    Graph &graph = dynamic_cast<Graph &>(decoder_input.getNodeContainer());
    Node *emb = embedding(graph, decoded_len_, params_->positionalEncodingParam(), false);
    Node *pos_encoded = add({scaled_input, emb});
    pos_encoded = dropout(*pos_encoded, dropout_);

    int layer_count = params_->layerCount();

    Node *last_layer_node = pos_encoded;
    for (int i = 0; i < layer_count; ++i) {
        auto &layer_params = *params_->layerParams().ptrs().at(i);
        auto &attention_head_params = layer_params.selfAttention();
        Node *normed = layerNormalization(layer_params.layerNormA(),
                *last_layer_node);
        Node *k = linear(*normed, attention_head_params.k());
        Node *v = linear(*normed, attention_head_params.v());

        Node *&key_matrix = key_matrix_layers_.at(i);
        key_matrix = key_matrix == nullptr ? k : concat({key_matrix, k});
        Node *&value_matrix = value_matrix_layers_.at(i);
        value_matrix = value_matrix == nullptr ? v : concat({value_matrix, v});

        Node *q = linear(*normed, attention_head_params.q());
        Node *attended = dotAttention(*key_matrix, *value_matrix, decoded_len_ + 1,
                *q, 1, params_->headCount(), layer_params.selfFusion(), dropout_, false);
        Node *added = add({attended, last_layer_node});
        normed = layerNormalization(layer_params.layerNormB(), *added);

        auto &attention_head_params_for_encoder = layer_params.encoderAttention();
        q = linear(*normed, attention_head_params_for_encoder.q());
        attended = dotAttention(*encoder_key_matrices_.at(i),
                *encoder_value_matrices_.at(i), encoder_sentence_len_, *q, 1,
                params_->headCount(), layer_params.encoderFusion(), dropout_, false);
        added = add({added, attended});
        normed = layerNormalization(layer_params.layerNormC(), *added);

        Node *t = linear(*normed, layer_params.ffnInnerParams());
        t = relu(*t);
        t = linear(*t, layer_params.ffnOutterParams());
        t = dropout(*t, dropout_);
        added = add({added, t});
        last_layer_node = added;
        hidden_layers_.at(i).push_back(last_layer_node);
    }
    decoded_len_++;
}

TransformerDecoderBuilder::TransformerDecoderBuilder(TransformerDecoderParams &params,
        Node &encoder_hiddens,
        int encoder_sentence_len,
        dtype dropout) : TransformerDecoderBuilderAbs(params, encoder_hiddens,
            encoder_sentence_len, dropout) {}

void TransformerDecoderBuilder::connect(Node &inputs, int dec_sentence_len) {
    if (!prepared_) {
        cerr << "TransformerDecoderBuilder forward - not prepared" << endl;
        abort();
    }

    vector<int> pos_ids;
    for (int i = 0; i < dec_sentence_len; ++i) {
        pos_ids.push_back(i);
    }

    Graph &graph = dynamic_cast<Graph &>(inputs.getNodeContainer());
    Node *pos_emb = embedding(graph, pos_ids, params_->positionalEncodingParam(), false);
    int row = inputs.getDim() / dec_sentence_len;
    Node *scaled_input = scaled(inputs, ::sqrt(row));
    Node *pos_encoded = add({pos_emb, scaled_input});
    pos_encoded = dropout(*pos_encoded, dropout_);

    int layer_count = params_->layerCount();
    Node *last_layer = pos_encoded;

    for (int i = 0; i < layer_count; ++i) {
        auto &layer_params = *params_->layerParams().ptrs().at(i);
        auto &attention_head_params = layer_params.selfAttention();
        Node *normed = layerNormalization(layer_params.layerNormA(), *last_layer,
                dec_sentence_len);
        Node *k = linear(*normed, attention_head_params.k());
        Node *v = linear(*normed, attention_head_params.v());
        Node *q = linear(*normed, attention_head_params.q());
        Node *attended = dotAttention(*k, *v, dec_sentence_len, *q, dec_sentence_len,
                params_->headCount(), layer_params.selfFusion(), dropout_, true);
        Node *added = add({attended, last_layer});
        normed = layerNormalization(layer_params.layerNormB(), *added,
                dec_sentence_len);

        auto &attention_head_params_for_encoder = layer_params.encoderAttention();
        q = linear(*normed, attention_head_params_for_encoder.q());
        attended = dotAttention(*encoder_key_matrices_.at(i), *encoder_value_matrices_.at(i),
                encoder_sentence_len_, *q, dec_sentence_len, params_->headCount(),
                layer_params.encoderFusion(), dropout_, false);
        added = add({added, attended});
        normed = layerNormalization(layer_params.layerNormC(), *added, dec_sentence_len);

        Node *t = linear(*normed, layer_params.ffnInnerParams());
        t = relu(*t);
        t = linear(*t, layer_params.ffnOutterParams());
        t = dropout(*t, dropout_);
        added = add({added, t});
        last_layer = added; 
        hidden_layers_.push_back(last_layer);
    }
}

vector<Node *> transformerDecoder(Node &encoder, int encoder_sentence_len, Node &input,
        int decoder_sentence_len,
        TransformerDecoderParams &params,
        dtype dropout_value) {
    TransformerDecoderBuilder builder(params, encoder, encoder_sentence_len, dropout_value);
    builder.prepare();
    builder.connect(input, decoder_sentence_len);
    return builder.hiddenLayers();
}

}
