#ifndef UNIOP_H_
#define UNIOP_H_

#include "Param.h"
#include "MyLib.h"
#include "SparseParam.h"
#include "Node.h"
#include "Graph.h"
#include "ModelUpdate.h"
#include <cstdlib>
#include "AtomicOP.h"
#include "profiler.h"

class UniParams : public N3LDGSerializable, public TunableCombination<BaseParam>
#if USE_GPU
, public TransferableComponents
#endif
{
public:
    UniParams(const string &name) : b_(name + "-b", true), name_(name) {}

    ~UniParams() {
        if (W_ != nullptr && is_W_owner_) {
            delete W_;
        }
    }

    void init(int out_dim, int in_dim, bool use_b = true,
            const std::function<dtype(int, int)> *bound = nullptr,
            InitDistribution dist = InitDistribution::UNI) {
        if (W_ != nullptr) {
            cerr << "UniParams init already initialized" << endl;
            abort();
        }
        W_ = new Param(name_ + "-W");
        W_->init(in_dim, out_dim, bound, dist);
        is_W_owner_ = true;

        bias_enabled_ = use_b;
        if (use_b) {
            b_.init(out_dim, 1);
        }
    }

    void init(Param &W) {
        if (W_ != nullptr) {
            cerr << "UniParams init already initialized" << endl;
            abort();
        }
        W_ = &W;
        is_W_owner_ = false;
        bias_enabled_ = false;
    }

    Json::Value toJson() const override {
        Json::Value json;
        json["use_b"] = bias_enabled_;
        json["w"] = W_->toJson();
        if (bias_enabled_) {
            json["b"] = b_.toJson();
        }
        return json;
    }

    void fromJson(const Json::Value &json) override {
        bias_enabled_ = json["use_b"].asBool();
        W_->fromJson(json["w"]);
        if (bias_enabled_) {
            b_.fromJson(json["b"]);
        }
    }

    template<typename Archive>
    void serialize(Archive &ar) {
        ar(bias_enabled_, *W_);
        if (bias_enabled_) {
            ar(b_);
        }
    }

#if USE_GPU
    std::vector<n3ldg_cuda::Transferable *> transferablePtrs() override {
        std::vector<Transferable *> ptrs = {W_};
        if (bias_enabled_) {
            ptrs.push_back(&b_);
        }
        return ptrs;
    }
#endif

    Param &W() {
        return *W_;
    }

    Param &b() {
        return b_;
    }

    bool biasEnabled() const {
        return bias_enabled_;
    }

protected:
    std::vector<Tunable<BaseParam>*> tunableComponents() override {
        if (bias_enabled_) {
            return {W_, &b_};
        } else {
            return {W_};
        }
    }

private:
    Param *W_ = nullptr;
    Param b_;
    string name_;
    bool bias_enabled_ = true;
    bool is_W_owner_ = true;
};

class LinearNode : public UniInputNode, public Poolable<LinearNode> {
public:
    void initNode(int dim) override {
        init(dim);
    }

    void setNodeDim(int dim) override {
        setDim(dim);
    }

    LinearNode() : UniInputNode("linear") {}

    void setParam(UniParams &uni_params) {
        param_ = &uni_params;
    }

    void compute() override {
        abort();
    }

    void backward() override {
        abort();
    }

    Executor * generate() override;

    string typeSignature() const override {
        return Node::getNodeType() + "-" + addressToString(param_);
    }

    Param &W() {
        return param_->W();
    }

    Param *b() {
        return param_->biasEnabled() ? &param_->b() : nullptr;
    }

protected:
    virtual bool isDimLegal(const Node &input) const override {
        return true;
    }

private:
    UniParams* param_ = nullptr;
};

class LinearExecutorBase : public Executor {
protected:
    Param &W() {
        LinearNode &l = dynamic_cast<LinearNode &>(*batch.front());
        return l.W();
    }

    Param *b() {
        LinearNode &l = dynamic_cast<LinearNode &>(*batch.front());
        return l.b();
    }

    int inDim() {
        return W().outDim();
    }

    int outDim() {
        return W().inDim();
    }
};

#if USE_GPU
class LinearExecutor : public LinearExecutorBase {
public:
    void  forward() {
        int count = batch.size();
#if TEST_CUDA
        W().val.copyFromDeviceToHost();
        if (b() != nullptr) {
            b()->val.copyFromDeviceToHost();
        }
#endif
        for (int i = 0; i < count; ++i) {
            col_sum_ += batch.at(i)->getColumn();
        }

#if TEST_CUDA
        x_.init(inDim(), col_sum_);
        y_.init(outDim(), col_sum_);
        b_.init(outDim(), col_sum_);
#endif
        std::vector<dtype*> ys;
        in_vals_.reserve(batch.size());
        ys.reserve(batch.size());
        cols_.reserve(batch.size());

        for (int i = 0; i < batch.size(); ++i) {
            LinearNode *n = static_cast<LinearNode*>(batch.at(i));

            in_vals_.push_back(n->getInput().val().value);
            ys.push_back(n->val().value);
            cols_.push_back(n->getColumn());
#if TEST_CUDA
            n->getInput().val().copyFromDeviceToHost();
#endif
        }
        n3ldg_cuda::LinearForward(in_vals_, count, cols_, inDim(), outDim(), W().val.value, 
                b() == nullptr ? nullptr : b()->val.value, ys);

#if TEST_CUDA
        int col_offset = 0;
        for (int i = 0; i < count; i++) {
            LinearNode& l = dynamic_cast<LinearNode &>(*batch.at(i));
            Vec(x_.v + col_offset * inDim(), l.getInput().getDim()) = l.getInput().getVal().vec();

            col_offset += l.getColumn();
        }

        if (b() != nullptr) {
            for (int i = 0; i < col_sum_; ++i) {
                Vec(b_.v + i * outDim(), outDim()) = b()->val.vec();
            }
        }

        y_.mat() = W().val.mat().transpose() * x_.mat();
        if (b() != nullptr) {
            y_.vec() += b_.vec();
        }

        col_offset = 0;
        for (int i = 0; i < count; i++) {
            LinearNode &l = dynamic_cast<LinearNode &>(*batch.at(i));
            l.val().vec() = Vec(y_.v + col_offset * outDim(), l.getDim());
            col_offset += l.getColumn();
        }
        Executor::verifyForward();
        cout << "linear forward tested" << endl;
#endif
    }

    void backward() {
        int count = batch.size();
#if TEST_CUDA
        if (b() != nullptr) {
            n3ldg_cuda::Assert(b()->grad.verify("before linear backward b grad"));
            b()->grad.copyFromDeviceToHost();
        }
        n3ldg_cuda::Assert(W().val.verify("before linear backward W val"));
        W().val.copyFromDeviceToHost();
        n3ldg_cuda::Assert(W().grad.verify("before linear backward W grad"));
        W().grad.copyFromDeviceToHost();
        for (int i = 0; i < count; ++i) {
            LinearNode* ptr = (LinearNode*)batch[i];
            n3ldg_cuda::Assert(ptr->loss().verify("before linear backward grad"));
            ptr->loss().copyFromDeviceToHost();
            n3ldg_cuda::Assert(ptr->val().verify("before linear val"));
            ptr->val().copyFromDeviceToHost();
            n3ldg_cuda::Assert(ptr->getInput().loss().verify("before linear backward in grad"));
            ptr->getInput().loss().copyFromDeviceToHost();
        }
#endif

        std::vector<dtype*> grads, in_grads;
        grads.reserve(count);
        in_grads.reserve(count);
        for (int i = 0; i < count; ++i) {
            LinearNode* ptr = (LinearNode*)batch[i];
            grads.push_back(ptr->loss().value);
            in_grads.push_back(ptr->getInput().loss().value);
        }

        n3ldg_cuda::LinearBackward(grads, count, cols_, inDim(), outDim(), W().val.value, in_vals_,
                b() == nullptr ? nullptr : b()->grad.value, in_grads, W().grad.value);
#if TEST_CUDA
        Tensor2D lx, ly;
        lx.init(inDim(), col_sum_);
        ly.init(outDim(), col_sum_);

        int col_offset = 0;
        for (int i = 0; i < count; i++) {
            LinearNode &l = dynamic_cast<LinearNode &>(*batch.at(i));
            Vec(ly.v + col_offset * outDim(), l.getDim()) = l.getLoss().vec();
            col_offset += l.getColumn();
        }

        W().grad.mat() += x_.mat() * ly.mat().transpose();

        if (b() != nullptr) {
            for (int i = 0; i < col_sum_; ++i) {
                b()->grad.vec() += Vec(ly.v + i * outDim(), outDim());
            }
        }

        lx.mat() = W().val.mat() * ly.mat();

        col_offset = 0;
        for (int i = 0; i < count; i++) {
            LinearNode& l = (LinearNode &)(*batch.at(i));
            l.getInput().loss().vec() += Vec(lx.v + col_offset * inDim(), inDim());
            col_offset += l.getColumn();
        }

        n3ldg_cuda::Assert(W().grad.verify("LinearExecutor backward W grad"));
        if (b() != nullptr) {
            n3ldg_cuda::Assert(b()->grad.verify("backward b grad"));
        }
        for (Node * n : batch) {
            LinearNode *ptr = static_cast<LinearNode *>(n);
            n3ldg_cuda::Assert(ptr->getInput().loss().verify("backward loss"));
        }
        cout << "linear backward tested" << endl;
#endif
    }

private:
#if TEST_CUDA
    Tensor2D x_, y_, b_;
#endif
    int col_sum_ = 0;
    vector<int> cols_;
    vector<dtype *> in_vals_;
};
#else
class LinearExecutor : public LinearExecutorBase {
public:
    int calculateFLOPs() override {
        int flops = W().inDim() * W().outDim() * batch.size() * 2;
        if (b() != nullptr) {
            flops += W().inDim() * batch.size();
        }
        return flops;
    }

    void  forward() override {
        int count = batch.size();
        for (Node *node : batch) {
            col_sum_ += node->getColumn();
        }
        x_.init(inDim(), col_sum_);
        y_.init(outDim(), col_sum_);
        b_.init(outDim(), col_sum_);

        int col_offset = 0;
        for (int i = 0; i < count; i++) {
            LinearNode& l = dynamic_cast<LinearNode &>(*batch.at(i));
            Vec(x_.v + col_offset * inDim(), l.getInput().getDim()) = l.getInput().getVal().vec();

            col_offset += l.getColumn();
        }

        if (b() != nullptr) {
            for (int i = 0; i < col_sum_; ++i) {
                Vec(b_.v + i * outDim(), outDim()) = b()->val.vec();
            }
        }

        y_.mat() = W().val.mat().transpose() * x_.mat();

        if (b() != nullptr) {
            y_.vec() += b_.vec();
        }

        col_offset = 0;
        for (int i = 0; i < count; i++) {
            LinearNode &l = dynamic_cast<LinearNode &>(*batch.at(i));
            l.val().vec() = Vec(y_.v + col_offset * outDim(), l.getDim());
            col_offset += l.getColumn();
        }
    }

    void backward() override {
        Tensor2D lx, ly;
        int count = batch.size();
        lx.init(inDim(), col_sum_);
        ly.init(outDim(), col_sum_);

        int col_offset = 0;
        for (int i = 0; i < count; i++) {
            LinearNode &l = dynamic_cast<LinearNode &>(*batch.at(i));
            Vec(ly.v + col_offset * outDim(), l.getDim()) = l.getLoss().vec();
            col_offset += l.getColumn();
        }

        W().grad.mat() += x_.mat() * ly.mat().transpose();

        if (b() != nullptr) {
            for (int i = 0; i < col_sum_; ++i) {
                b()->grad.vec() += Vec(ly.v + i * outDim(), outDim());
            }
        }

        lx.mat() = W().val.mat() * ly.mat();

        col_offset = 0;
        for (int i = 0; i < count; i++) {
            LinearNode& l = (LinearNode &)(*batch.at(i));
            l.getInput().loss().vec() += Vec(lx.v + col_offset * inDim(), inDim());
            col_offset += l.getColumn();
        }
    }

private:
    int col_sum_ = 0;
    Tensor2D x_, y_, b_;
};
#endif

Executor * LinearNode::generate() {
    return new LinearExecutor();
};

class BiasParam : public Param {
public:
    BiasParam(const string &name) : Param(name, true) {}

    void init(int outDim, int inDim) override {
        cerr << "BiasParam::init - unsupported method" << endl;
        abort();
    }

    void initAsBias(int dim) {
        Param::init(dim, 1);
    }
};

class BiasNode : public UniInputNode, public Poolable<BiasNode> {
public:
    BiasNode() : UniInputNode("bias") {}

    void initNode(int dim) override {
        init(dim);
    }

    void setNodeDim(int dim) override {
        setDim(dim);
    }

    void init(int dim) override {
        UniInputNode::init(dim);
    }

    virtual string typeSignature() const override {
        return Node::typeSignature() + "-" + addressToString(bias_param_);
    }

    void compute() override {
        for (int i = 0; i < getDim(); ++i) {
            val()[i] = getInput().getVal()[i] + bias_param_->val[0][i];
        }
    }

    void backward() override {
        getInput().loss().vec() += loss().vec();
        for (int i = 0; i < getDim(); ++i) {
            bias_param_->grad[0][i] += getLoss()[i];
        }
    }

    Executor *generate() override;

    void setParam(BiasParam &param) {
        bias_param_ = &param;
        if (getDim() > bias_param_->outDim()) {
            cerr << boost::format("dim is %1%, but bias param dim is %2%") % getDim() %
                bias_param_->outDim() << endl;
            abort();
        }
    }

protected:
    virtual bool isDimLegal(const Node &input) const override {
        return input.getDim() == getDim();
    }

private:
    BiasParam *bias_param_;
    friend class BiasExecutor;
};

class BatchedBiasNode : public BatchedNodeImpl<BiasNode> {
public:
    void init(Graph &graph, BatchedNode &input, BiasParam &param) {
        allocateBatch(input.getDim(), input.batch().size());
        for (Node *node : batch()) {
            BiasNode *b = dynamic_cast<BiasNode *>(node);
            b->setParam(param);
        }
        setInputsPerNode({&input});
        afterInit(graph, {&input});
    }
};

#if USE_GPU
class BiasExecutor : public UniInputExecutor {
public:
    void forward() override {
        dtype *bias = param()->val.value;
        vector<dtype*> inputs, vals;
        for (Node *node : batch) {
            BiasNode *bias_node = static_cast<BiasNode *>(node);
            inputs.push_back(bias_node->getInput().getVal().value);
            vals.push_back(bias_node->getVal().value);
        }
        n3ldg_cuda::BiasForward(inputs, bias, batch.size(), getDim(), vals);
#if TEST_CUDA
        Executor::testForward();
        cout << "bias forward tested" << endl;
#endif
    }

    void backward() override {
        dtype *bias = param()->grad.value;
        vector<dtype *> losses, in_losses;
        for (Node *node : batch) {
            BiasNode *bias_node = static_cast<BiasNode *>(node);
            losses.push_back(bias_node->getLoss().value);
            in_losses.push_back(bias_node->getInput().getLoss().value);
        }
        n3ldg_cuda::BiasBackward(losses, batch.size(), getDim(), bias, in_losses);
#if TEST_CUDA
        UniInputExecutor::testBackward();
        cout << "count:" << batch.size() << endl;
        n3ldg_cuda::Assert(param()->grad.verify("bias backward bias grad"));
        cout << "Bias backward tested" << endl;
#endif
    }

private:
    BiasParam *param() {
        return static_cast<BiasNode*>(batch.front())->bias_param_;
    }
};
#else
class BiasExecutor : public Executor {
public:
    int calculateFLOPs() override {
        return defaultFLOPs();
    }
};
#endif

Executor *BiasNode::generate() {
    return new BiasExecutor;
}

namespace n3ldg_plus {

Node *linear(Graph &graph, Node &input, UniParams &params) {
    if (input.getDim() % params.W().outDim() != 0) {
        cerr << boost::format("linear input dim:%1% W col:%2% W col:%3%\n") % input.getDim() %
            input.getColumn() % params.W().inDim() << endl;
        abort();
    }
    int col = input.getDim() / params.W().outDim();
    int dim = params.W().inDim();
    LinearNode *uni = LinearNode::newNode(dim * col);
    uni->setColumn(col);
    uni->setParam(params);
    uni->forward(graph, input);
    return uni;
}

Node *linear(Graph &graph, Node &input, Param &param) {
    if (input.getDim() % param.outDim() != 0) {
        cerr << boost::format("linear input dim:%1% W col:%2% W col:%3%\n") % input.getDim() %
            input.getColumn() % param.inDim() << endl;
        abort();
    }

    static map<void *, UniParams *> param_map;
    auto it = param_map.find(&param);
    UniParams *uni_params;
    if (it == param_map.end()) {
        uni_params = new UniParams("uni" + addressToString(&param));
        uni_params->init(param);
        param_map.insert(make_pair(&param, uni_params));
    } else {
        uni_params = it->second;
    }

    int col = input.getDim() / param.outDim();
    int dim = param.inDim();
    bool pool = col == 1;
    LinearNode *uni = LinearNode::newNode(dim * col, pool);
    uni->setIsPooled(pool);
    uni->setColumn(col);
    uni->setParam(*uni_params);
    uni->forward(graph, input);
    return uni;
}

Node *bias(Graph &graph, Node &input, BiasParam &param) {
    int dim = input.getDim();
    BiasNode *node = BiasNode::newNode(dim);
    node->setParam(param);
    node->forward(graph, input);
    return node;
}

BatchedNode *bias(Graph &graph, BatchedNode &input, BiasParam &param) {
    BatchedBiasNode *node = new BatchedBiasNode;
    node->init(graph, input, param);
    return node;
}

}

#endif /* UNIOP_H_ */
