#include "insnet/block/transformer.h"
#include "insnet/operator/atomic.h"
#include "insnet/operator/concat.h"
#include "insnet/operator/split.h"
#include "insnet/operator/add.h"
#include "insnet/operator/embedding.h"
#include "insnet/block/attention.h"

using std::string;
using std::function;
using std::vector;
using std::cout;
using std::cerr;
using std::endl;

namespace insnet {

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

Node *multiheadAttention(Node& q, Node& k, Node& v, int row, int head_count,
        LinearParams &fusion_param,
        dtype dropout_value,
        bool use_mask) {
    int head_dim = row / head_count;
    vector<int> offsets(head_count);
    for (int i = 0; i < head_count; ++i) {
        offsets.at(i) = i * head_dim;
    }

    int q_col = q.size() / row;
    if (q_col * row != q.size()) {
        cerr << fmt::format("dotAttention - q_col:{} row:{} q dim:{}", q_col, row, q.size()) <<
            endl;
        abort();
    }
    int v_col = v.size() / row;
    if (v_col * row != v.size()) {
        cerr << fmt::format("dotAttention - v_col:{} row:{} v dim:{}", v_col, row, v.size()) <<
            endl;
        abort();
    }

    BatchedNode *split_q = split(q, head_dim, offsets, q_col);
    BatchedNode *split_k = split(k, head_dim, offsets, v_col);
    BatchedNode *split_v = split(v, head_dim, offsets, v_col);
    BatchedNode *split_attended = dotAttention(*split_k, *split_v, *split_q, head_dim,
            use_mask).first;
    Node *attended_matrix = cat(*split_attended, q_col);
    attended_matrix = linear(*attended_matrix, fusion_param);
    attended_matrix = dropout(*attended_matrix, dropout_value);
    return attended_matrix;
}

Node *multiheadAttention(const vector<Node *> &qs, const vector<Node *> &ks,
        const vector<Node *> &vs,
        int row,
        int head_count,
        LinearParams &fusion_param,
        dtype dropout_value,
        bool use_mask) {
    int head_dim = row / head_count;
    vector<int> offsets(head_count);
    for (int i = 0; i < head_count; ++i) {
        offsets.at(i) = i * head_dim;
    }

    vector<Node *> matrices;
    matrices.reserve(qs.size());
    for (int i = 0; i < qs.size(); ++i) {
        Node &q = *qs.at(i);
        int q_col = q.size() / row;
        if (q_col * row != q.size()) {
            cerr << fmt::format("dotAttention - q_col:{} row:{} q dim:{}", q_col, row, q.size()) <<
                endl;
            abort();
        }
        Node &v = *vs.at(i);
        int v_col = v.size() / row;
        if (v_col * row != v.size()) {
            cerr << fmt::format("dotAttention - v_col:{} row:{} v dim:{}", v_col, row, v.size()) <<
                endl;
            abort();
        }
        Node &k = *ks.at(i);

        BatchedNode *split_q = split(q, head_dim, offsets, q_col);
        BatchedNode *split_k = split(k, head_dim, offsets, v_col);
        BatchedNode *split_v = split(v, head_dim, offsets, v_col);
        BatchedNode *split_attended = dotAttention(*split_k, *split_v, *split_q, head_dim,
                use_mask).first;
        Node *attended_matrix = cat(*split_attended, q_col);
        matrices.push_back(attended_matrix);
    }
    Node *merged_matrix = cat(matrices);
    Graph &graph = dynamic_cast<Graph &>(merged_matrix->getNodeContainer());
    graph.forward();
    merged_matrix = linear(*merged_matrix, fusion_param);
    graph.forward();
    merged_matrix = dropout(*merged_matrix, dropout_value);
    graph.forward();

    return merged_matrix;
}

vector<Node *> transformerEncoder(Node &inputs, TransformerEncoderParams &params,
        dtype dropout_value) {
    int hidden_dim = params.hiddenDim();
    int sentence_len = inputs.size() / hidden_dim;
    vector<int> pos_ids;
    pos_ids.reserve(sentence_len);
    for (int i = 0; i < sentence_len; ++i) {
        pos_ids.push_back(i);
    }

    Graph &graph = dynamic_cast<Graph &>(inputs.getNodeContainer());
    Node *pos_emb = embedding(graph, pos_ids, params.positionalEncodingParam(), true);
    Node *scaled_input = mul(inputs, ::sqrt(hidden_dim));
    Node *pos_encoded = add({pos_emb, scaled_input});
    pos_encoded = dropout(*pos_encoded, dropout_value);

    int layer_count = params.layerCount();

    Node *last_layer = pos_encoded;
    vector<Node *> hiddens;
    hiddens.reserve(layer_count);
    for (int i = 0; i < layer_count; ++i) {
        auto &layer_params = *params.layerParams().ptrs().at(i);

        Node *normed = layerNorm(*last_layer, layer_params.layerNormA());
        auto &attention_head_params = layer_params.multiHeadAttentionParams();
        Node *key = linear(*normed, attention_head_params.k());
        Node *value = linear(*normed, attention_head_params.v());
        Node *q = linear(*normed, attention_head_params.q());
        Node *attended = multiheadAttention(*q, *key, *value, hidden_dim, params.headCount(),
                layer_params.headsFusionParams(), dropout_value, false);
        Node *added = add({attended, last_layer});
        normed = layerNorm(*added, layer_params.layerNormB());
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

namespace {

vector<int> calLens(const vector<Node *> &nodes, int dim) {
    vector<int> ret;
    ret.reserve(nodes.size());
    for (Node *node : nodes) {
        ret.push_back(node->size() / dim);
    }
    return ret;
}

vector<int> calOffsets(const vector<int> &sen_lens) {
    vector<int> ret;
    ret.reserve(sen_lens.size());
    int offset = 0;
    for (int len : sen_lens) {
        ret.push_back(offset);
        offset += len;
    }
    return ret;
}

vector<int> calIds(const vector<int> &sen_lens) {
    vector<int> ret;
    for (int len : sen_lens) {
        for (int i = 0; i < len; ++i) {
            ret.push_back(i);
        }
    }
    return ret;
}

Node *multiheadAttention(Node &q, Node &k, Node &v, const vector<int> &sen_lens, int row,
        int head_count,
        LinearParams &fusion_param,
        dtype dropout_value,
        bool use_mask) {
    vector<Node *> sen_qs, sen_ks, sen_vs;
    sen_qs.reserve(sen_lens.size());
    sen_ks.reserve(sen_lens.size());
    sen_vs.reserve(sen_lens.size());

    vector<int> sen_offsets = calOffsets(sen_lens);

    for (int j = 0; j < sen_lens.size(); ++j) {
        int sen_len = sen_lens.at(j);
        int sen_size = sen_len * row;
        int offset = sen_offsets.at(j) * row;
        Node *sen_q = split(q, sen_size, offset);
        sen_qs.push_back(sen_q);
        Node *sen_k = split(k, sen_size, offset);
        sen_ks.push_back(sen_k);
        Node *sen_value = split(v, sen_size, offset);
        sen_vs.push_back(sen_value);
    }

    return multiheadAttention(sen_qs, sen_ks, sen_vs, row, head_count, fusion_param, dropout_value,
            false);
}

Node *multiheadAttention(Node &q, Node &k, Node &v, const vector<int> &enc_sen_lens,
        const vector<int> &dec_sen_lens,
        int row,
        int head_count,
        LinearParams &fusion_param,
        dtype dropout_value,
        bool use_mask) {
    if (enc_sen_lens.size() != dec_sen_lens.size()) {
        cerr << fmt::format("multiheadAttention enc_sen_lens size:{} dec:{}", enc_sen_lens.size(),
                dec_sen_lens.size()) << endl;
        abort();
    }

    vector<Node *> sen_qs, sen_ks, sen_vs;
    int sen_num = enc_sen_lens.size();
    sen_qs.reserve(sen_num);
    sen_ks.reserve(sen_num);
    sen_vs.reserve(sen_num);

    vector<int> enc_sen_offsets = calOffsets(enc_sen_lens);
    vector<int> dec_sen_offsets = calOffsets(dec_sen_lens);

    for (int j = 0; j < sen_num; ++j) {
        int dec_sen_len = dec_sen_lens.at(j);
        int dec_sen_size = dec_sen_len * row;
        int dec_offset = dec_sen_offsets.at(j) * row;
        Node *sen_q = split(q, dec_sen_size, dec_offset);
        sen_qs.push_back(sen_q);

        int enc_sen_len = enc_sen_lens.at(j);
        int enc_sen_size = enc_sen_len * row;
        int enc_offset = enc_sen_offsets.at(j) * row;
        Node *sen_k = split(k, enc_sen_size, enc_offset);
        sen_ks.push_back(sen_k);
        Node *sen_value = split(v, enc_sen_size, enc_offset);
        sen_vs.push_back(sen_value);
    }

    return multiheadAttention(sen_qs, sen_ks, sen_vs, row, head_count, fusion_param, dropout_value,
            false);
}

}

vector<vector<Node *>> transformerEncoder(const vector<Node *> &inputs,
        TransformerEncoderParams &params,
        dtype dropout_value) {
    Node *merged_input = cat(inputs);

    int hidden_dim = params.hiddenDim();
    vector<int> sen_lens = calLens(inputs, hidden_dim);
    vector<int> pos_ids = calIds(sen_lens);

    Graph &graph = dynamic_cast<Graph &>(inputs.front()->getNodeContainer());
    Node *pos_emb = embedding(graph, pos_ids, params.positionalEncodingParam(), true);
    graph.forward();
    Node *scaled_input = mul(*merged_input, ::sqrt(hidden_dim));
    graph.forward();
    Node *pos_encoded = add({pos_emb, scaled_input});
    graph.forward();
    pos_encoded = dropout(*pos_encoded, dropout_value);
    graph.forward();

    int layer_count = params.layerCount();
    Node *last_layer = pos_encoded;
    vector<Node *> hiddens;
    hiddens.reserve(layer_count);

    for (int i = 0; i < layer_count; ++i) {
        auto &layer_params = *params.layerParams().ptrs().at(i);

        Node *normed = layerNorm(*last_layer, layer_params.layerNormA());
        graph.forward();
        auto &attention_head_params = layer_params.multiHeadAttentionParams();
        Node *key = linear(*normed, attention_head_params.k());
        graph.forward();
        Node *value = linear(*normed, attention_head_params.v());
        graph.forward();
        Node *q = linear(*normed, attention_head_params.q());
        graph.forward();
        Node *attended = multiheadAttention(*q, *key, *value, sen_lens, hidden_dim,
                params.headCount(), layer_params.headsFusionParams(), dropout_value, false);
        Node *added = add({attended, last_layer});
        graph.forward();
        normed = layerNorm(*added, layer_params.layerNormB());
        graph.forward();
        Node *t = linear(*normed, layer_params.ffnInnerParams());
        graph.forward();
        t = relu(*t);
        graph.forward();
        t = linear(*t, layer_params.ffnOutterParams());
        graph.forward();
        t = dropout(*t, dropout_value);
        graph.forward();
        t = add({added, t});
        graph.forward();
        last_layer = t;
        hiddens.push_back(last_layer);
    }

    vector<int> sen_offsets = calOffsets(sen_lens);
    vector<vector<Node *>> ret;
    ret.reserve(layer_count);
    for (int i = 0; i < layer_count; ++i) {
        vector<Node *> sen_rets;
        sen_rets.reserve(inputs.size());
        for (int j = 0; j < inputs.size(); ++j) {
            int sen_len = sen_lens.at(j);
            int sen_offset = sen_offsets.at(j);
            Node *sen_h = split(*hiddens.at(i), sen_len * hidden_dim, sen_offset * hidden_dim);
            sen_rets.push_back(sen_h);
        }
        ret.push_back(move(sen_rets));
    }
    return ret;
}

TransformerDecoderBuilderAbs::TransformerDecoderBuilderAbs(TransformerDecoderParams &params,
        Node &encoder_hiddens,
        dtype dropout) : params_(&params), encoder_hiddens_(&encoder_hiddens), dropout_(dropout) {}

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
        dtype dropout) : TransformerDecoderBuilderAbs(params, encoder_hiddens, dropout) {
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
    Node *scaled_input = mul(decoder_input,
            std::sqrt(static_cast<dtype>(decoder_input.size())));
    Graph &graph = dynamic_cast<Graph &>(decoder_input.getNodeContainer());
    Node *emb = embedding(graph, decoded_len_, params_->positionalEncodingParam(), true);
    Node *pos_encoded = add({scaled_input, emb});
    pos_encoded = dropout(*pos_encoded, dropout_);

    int layer_count = params_->layerCount();
    int encoder_sentence_len = encoder_hiddens_->size() / params_->hiddenDim();
    if (encoder_sentence_len * params_->hiddenDim() != encoder_hiddens_->size()) {
        cerr << fmt::format("TransformerDecoderCellBuilder::step - encoder_sentence_len:{} hidden_dim:{} encoder_hiddens_ dim:{}",
                encoder_sentence_len, encoder_hiddens_->size(), encoder_hiddens_->size())
            << endl;
        abort();
    }

    Node *last_layer_node = pos_encoded;
    for (int i = 0; i < layer_count; ++i) {
        auto &layer_params = *params_->layerParams().ptrs().at(i);
        auto &attention_head_params = layer_params.selfAttention();
        Node *normed = layerNorm(*last_layer_node, layer_params.layerNormA());
        Node *k = linear(*normed, attention_head_params.k());
        Node *v = linear(*normed, attention_head_params.v());

        Node *&key_matrix = key_matrix_layers_.at(i);
        key_matrix = key_matrix == nullptr ? k : cat({key_matrix, k});
        Node *&value_matrix = value_matrix_layers_.at(i);
        value_matrix = value_matrix == nullptr ? v : cat({value_matrix, v});

        Node *q = linear(*normed, attention_head_params.q());
        int dim = params_->hiddenDim();
        Node *attended = multiheadAttention(*q, *key_matrix, *value_matrix, dim,
                params_->headCount(), layer_params.selfFusion(), dropout_, false);
        Node *added = add({attended, last_layer_node});
        normed = layerNorm(*added, layer_params.layerNormB());

        auto &attention_head_params_for_encoder = layer_params.encoderAttention();
        q = linear(*normed, attention_head_params_for_encoder.q());
        attended = multiheadAttention(*q, *encoder_key_matrices_.at(i),
                *encoder_value_matrices_.at(i), dim, params_->headCount(),
                layer_params.encoderFusion(), dropout_, false);
        added = add({added, attended});
        normed = layerNorm(*added, layer_params.layerNormC());

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
        Node &encoder_hiddens, dtype dropout) : TransformerDecoderBuilderAbs(params,
            encoder_hiddens, dropout) {}

void TransformerDecoderBuilder::connect(Node &inputs) {
    if (!prepared_) {
        cerr << "TransformerDecoderBuilder forward - not prepared" << endl;
        abort();
    }

    int dim = this->params_->hiddenDim();
    int dec_sentence_len = inputs.size() / dim;
    if (dec_sentence_len * dim != inputs.size()) {
        cerr << fmt::format("TransformerDecoderBuilder::connect - dec_sentence_len:{} dim:{} inputs dim:{}",
                dec_sentence_len, dim, inputs.size()) << endl;
        abort();
    }

    vector<int> pos_ids;
    for (int i = 0; i < dec_sentence_len; ++i) {
        pos_ids.push_back(i);
    }

    Graph &graph = dynamic_cast<Graph &>(inputs.getNodeContainer());
    Node *pos_emb = embedding(graph, pos_ids, params_->positionalEncodingParam(), false);
    int row = inputs.size() / dec_sentence_len;
    Node *scaled_input = mul(inputs, ::sqrt(row));
    Node *pos_encoded = add({pos_emb, scaled_input});
    pos_encoded = dropout(*pos_encoded, dropout_);

    int layer_count = params_->layerCount();
    Node *last_layer = pos_encoded;

    int encoder_sentence_len = encoder_hiddens_->size() / params_->hiddenDim();
    if (encoder_sentence_len * params_->hiddenDim() != encoder_hiddens_->size()) {
        cerr << fmt::format("TransformerDecoderCellBuilder::step - encoder_sentence_len:{} hidden_dim:{} encoder_hiddens_ dim:{}",
                encoder_sentence_len, encoder_hiddens_->size(), encoder_hiddens_->size())
            << endl;
        abort();
    }

    for (int i = 0; i < layer_count; ++i) {
        auto &layer_params = *params_->layerParams().ptrs().at(i);
        auto &attention_head_params = layer_params.selfAttention();
        Node *normed = layerNorm(*last_layer, layer_params.layerNormA());
        Node *k = linear(*normed, attention_head_params.k());
        Node *v = linear(*normed, attention_head_params.v());
        Node *q = linear(*normed, attention_head_params.q());
        int dim = params_->hiddenDim();
        Node *attended = multiheadAttention(*q, *k, *v, dim, params_->headCount(),
                layer_params.selfFusion(), dropout_, true);
        Node *added = add({attended, last_layer});
        normed = layerNorm(*added, layer_params.layerNormB());

        auto &attention_head_params_for_encoder = layer_params.encoderAttention();
        q = linear(*normed, attention_head_params_for_encoder.q());
        attended = multiheadAttention(*q, *encoder_key_matrices_.at(i),
                *encoder_value_matrices_.at(i), dim, params_->headCount(),
                layer_params.encoderFusion(), dropout_, false);
        added = add({added, attended});
        normed = layerNorm(*added, layer_params.layerNormC());

        Node *t = linear(*normed, layer_params.ffnInnerParams());
        t = relu(*t);
        t = linear(*t, layer_params.ffnOutterParams());
        t = dropout(*t, dropout_);
        added = add({added, t});
        last_layer = added; 
        hidden_layers_.push_back(last_layer);
    }
}

TransformerDecoderState::TransformerDecoderState(int layer) {
    for (auto *v : {&keys_, &values_}) {
        v->reserve(layer);
        for (int i = 0; i < layer; ++i) {
            v->push_back(nullptr);
        }
    }
}
int TransformerDecoderState::layerCount() const {
    if (keys_.size() != values_.size()) {
        cerr << fmt::format("TransformerDecoderState layerCount keys_ size:{} values_ size:{}",
                keys_.size(), values_.size()) << endl;
        abort();
    }
    return keys_.size();
}

vector<Node *> transformerDecoder(Node &encoder, Node &input, TransformerDecoderParams &params,
        dtype dropout_value) {
    TransformerDecoderBuilder builder(params, encoder, dropout_value);
    builder.prepare();
    builder.connect(input);
    return builder.hiddenLayers();
}

vector<vector<Node *>> transformerDecoder(const vector<Node *> &enc_hiddens,
        const vector<Node *> &inputs,
        TransformerDecoderParams &params,
        dtype dropout_value) {
    Node *merged_input = cat(inputs);

    int hidden_dim = params.hiddenDim();
    vector<int> dec_sen_lens = calLens(inputs, hidden_dim);
    vector<int> pos_ids = calIds(dec_sen_lens);

    Graph &graph = dynamic_cast<Graph &>(inputs.front()->getNodeContainer());
    Node *pos_emb = embedding(graph, pos_ids, params.positionalEncodingParam(), true);
    graph.forward();
    Node *scaled_input = mul(*merged_input, ::sqrt(hidden_dim));
    graph.forward();
    Node *pos_encoded = add({pos_emb, scaled_input});
    graph.forward();
    pos_encoded = dropout(*pos_encoded, dropout_value);
    graph.forward();

    Node *merged_enc_hidden = cat(enc_hiddens);
    graph.forward();
    int layer_count = params.layerCount();
    Node *last_layer = pos_encoded;

    vector<int> enc_sen_lens = calLens(enc_hiddens, hidden_dim);
    vector<Node *> hiddens;
    hiddens.reserve(layer_count);

    for (int i = 0; i < layer_count; ++i) {
        auto &layer_params = *params.layerParams().ptrs().at(i);
        auto &attention_head_params = layer_params.selfAttention();

        Node *normed = layerNorm(*last_layer, layer_params.layerNormA());
        graph.forward();
        Node *dec_k = linear(*normed, attention_head_params.k());
        graph.forward();
        Node *dec_v = linear(*normed, attention_head_params.v());
        graph.forward();
        Node *dec_q = linear(*normed, attention_head_params.q());
        graph.forward();
        Node *attended = multiheadAttention(*dec_q, *dec_k, *dec_v, dec_sen_lens, hidden_dim,
                params.headCount(), layer_params.selfFusion(), dropout_value, true);
        Node *added = add({attended, last_layer});
        graph.forward();
        normed = layerNorm(*added, layer_params.layerNormB());
        graph.forward();

        auto &attention_head_params_for_encoder = layer_params.encoderAttention();
        dec_q = linear(*normed, attention_head_params_for_encoder.q());
        graph.forward();
        Node *enc_k = linear(*merged_enc_hidden, attention_head_params_for_encoder.k());
        graph.forward();
        Node *enc_v = linear(*merged_enc_hidden, attention_head_params_for_encoder.v());
        graph.forward();

        attended = multiheadAttention(*dec_q, *enc_k, *enc_v, enc_sen_lens, dec_sen_lens,
                hidden_dim, params.headCount(), layer_params.encoderFusion(), dropout_value,
                false);

        added = add({added, attended});
        graph.forward();
        normed = layerNorm(*added, layer_params.layerNormC());
        graph.forward();

        Node *t = linear(*normed, layer_params.ffnInnerParams());
        graph.forward();
        t = relu(*t);
        graph.forward();
        t = linear(*t, layer_params.ffnOutterParams());
        graph.forward();
        t = dropout(*t, dropout_value);
        graph.forward();
        added = add({added, t});
        graph.forward();
        last_layer = added; 
        hiddens.push_back(last_layer);
    }

    vector<int> dec_sen_offsets = calOffsets(dec_sen_lens);
    vector<vector<Node *>> ret;
    ret.reserve(layer_count);
    for (int i = 0; i < layer_count; ++i) {
        vector<Node *> sen_rets;
        sen_rets.reserve(inputs.size());
        for (int j = 0; j < inputs.size(); ++j) {
            int sen_len = dec_sen_lens.at(j);
            int sen_offset = dec_sen_offsets.at(j);
            Node *sen_h = split(*hiddens.at(i), sen_len * hidden_dim, sen_offset * hidden_dim);
            sen_rets.push_back(sen_h);
        }
        ret.push_back(move(sen_rets));
    }
    return ret;
}

TransformerDecoderState transformerDecoder(const TransformerDecoderState &state,
        const std::vector<Node*> &encoder_keys,
        const std::vector<Node*> &encoder_values,
        Node &input,
        TransformerDecoderParams &params,
        dtype dropout_value) {
    if (state.layerCount() != params.layerCount()) {
        cerr << fmt::format("transformerDecoder - state layerCount:{} params layerCount:{}",
                state.layerCount(), params.layerCount()) << endl;
        abort();
    }
    int dim = params.hiddenDim();
    int decoded_len = state.keys().front() == nullptr ? 0 : state.keys().front()->size() / dim;
    if (decoded_len * dim != state.keys().front()->size()) {
        cerr << fmt::format("transformerDecoder decoded_len:{} dim:{} state keys front size:{}",
                decoded_len, dim, state.keys().front()->size()) << endl;
        abort();
    }
    Node *scaled_input = mul(input, std::sqrt(static_cast<dtype>(input.size())));
    Graph &graph = dynamic_cast<Graph &>(input.getNodeContainer());
    Node *emb = embedding(graph, decoded_len, params.positionalEncodingParam(), true);
    Node *pos_encoded = add({scaled_input, emb});
    pos_encoded = dropout(*pos_encoded, dropout_value);

    int layer_count = params.layerCount();
    int encoder_sentence_len = encoder_keys.front()->size() / params.hiddenDim();
    if (encoder_sentence_len * params.hiddenDim() != encoder_keys.front()->size()) {
        cerr << fmt::format("transformerDecoder - encoder_sentence_len:{} params hidden_dim:{} encoder_keys front size:{}",
                encoder_sentence_len, params.hiddenDim(), encoder_keys.front()->size()) << endl;
        abort();
    }

    vector<Node *> next_keys, next_values;
    next_keys.reserve(layer_count);
    next_values.reserve(layer_count);

    Node *last_layer_node = pos_encoded;
    for (int i = 0; i < layer_count; ++i) {
        auto &layer_params = *params.layerParams().ptrs().at(i);
        auto &attention_head_params = layer_params.selfAttention();
        Node *normed = layerNorm(*last_layer_node, layer_params.layerNormA());
        Node *k = linear(*normed, attention_head_params.k());
        Node *v = linear(*normed, attention_head_params.v());

        Node *key_matrix = state.keys().at(i);
        key_matrix = key_matrix == nullptr ? k : cat({key_matrix, k});
        next_keys.push_back(key_matrix);
        Node *value_matrix = state.values().at(i);
        value_matrix = value_matrix == nullptr ? v : cat({value_matrix, v});
        next_values.push_back(value_matrix);

        Node *q = linear(*normed, attention_head_params.q());
        int dim = params.hiddenDim();
        Node *attended = multiheadAttention(*q, *key_matrix, *value_matrix, dim,
                params.headCount(), layer_params.selfFusion(), dropout_value, false);
        Node *added = add({attended, last_layer_node});
        normed = layerNorm(*added, layer_params.layerNormB());

        auto &attention_head_params_for_encoder = layer_params.encoderAttention();
        q = linear(*normed, attention_head_params_for_encoder.q());
        attended = multiheadAttention(*q, *encoder_keys.at(i),
                *encoder_values.at(i), dim, params.headCount(),
                layer_params.encoderFusion(), dropout_value, false);
        added = add({added, attended});
        normed = layerNorm(*added, layer_params.layerNormC());

        Node *t = linear(*normed, layer_params.ffnInnerParams());
        t = relu(*t);
        t = linear(*t, layer_params.ffnOutterParams());
        t = dropout(*t, dropout_value);
        added = add({added, t});
        last_layer_node = added;
    }

    return TransformerDecoderState(next_keys, next_values);
}

}
