#ifndef UNIOP_H_
#define UNIOP_H_

/*
*  UniOP.h:
*  a simple feed forward neural operation, unary input.
*
*  Created on: Apr 22, 2017
*      Author: mszhang
*/


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
    Param W;
    Param b;
    bool bUseB = true;

    UniParams(const string &name) : W(name + "-W"), b(name + "-b", true) {}

    void init(int nOSize, int nISize, bool useB = true,
            const std::function<dtype(int, int)> *bound = nullptr,
            InitDistribution dist = InitDistribution::UNI) {
        W.init(nOSize, nISize, bound, dist);

        bUseB = useB;
        if (bUseB) {
            b.init(nOSize, 1);
        }
    }

    Json::Value toJson() const override {
        Json::Value json;
        json["use_b"] = bUseB;
        json["w"] = W.toJson();
        if (bUseB) {
            json["b"] = b.toJson();
        }
        return json;
    }

    void fromJson(const Json::Value &json) override {
        bUseB = json["use_b"].asBool();
        W.fromJson(json["w"]);
        if (bUseB) {
            b.fromJson(json["b"]);
        }
    }

#if USE_GPU
    std::vector<n3ldg_cuda::Transferable *> transferablePtrs() override {
        std::vector<Transferable *> ptrs = {&W};
        if (bUseB) {
            ptrs.push_back(&b);
        }
        return ptrs;
    }
#endif

protected:
    virtual std::vector<Tunable<BaseParam>*> tunableComponents() override {
        if (bUseB) {
            return {&W, &b};
        } else {
            return {&W};
        }
    }
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

    UniParams &getParams() {
        return *param_;
    }

    void compute() override {
        abort();
    }

    void backward() override {
        abort();
    }

    Executor * generate() override;

    string typeSignature() const override {
        return Node::typeSignature() + "-" + addressToString(param_);
    }

protected:
    virtual bool isDimLegal(const Node &input) const override {
        return input.getDim() == param_->W.inDim();
    }

private:
    UniParams* param_ = nullptr;
};

class BatchedLinearNode : public BatchedNodeImpl<LinearNode> {
public:
    void init(Graph &graph, BatchedNode &input, UniParams &params) {
        allocateBatch(params.W.outDim(), input.batch().size());
        for (Node *node : batch()) {
            LinearNode *l = dynamic_cast<LinearNode*>(node);
            l->setParam(params);
        }
        setInputsPerNode({&input});
        afterInit(graph, {&input});
    }
};

#if USE_GPU
class LinearExecutor :public Executor {
public:
    Tensor2D x, y, b;

    UniParams &params() {
        LinearNode *l = dynamic_cast<LinearNode *>(batch.front());
        return l->getParams();
    }

    int inDim() {
        return params().W.inDim();
    }

    int outDim() {
        return params().W.outDim();
    }

    void  forward() {
        int count = batch.size();
#if TEST_CUDA
        param->W.val.copyFromDeviceToHost();
        if (param->bUseB) {
            param->b.val.copyFromDeviceToHost();
        }
#endif

        x.init(inDim(), count);
        y.init(outDim(), count);
#if TEST_CUDA
        b.init(outDim, count);
#endif
        std::vector<dtype*> xs, ys;
        xs.reserve(batch.size());
        ys.reserve(batch.size());

        for (int i = 0; i < batch.size(); ++i) {
            LinearNode *n = static_cast<LinearNode*>(batch.at(i));

            xs.push_back(n->getInput()->val().value);
            ys.push_back(n->val().value);
#if TEST_CUDA
            n->in->val().copyFromDeviceToHost();
#endif
        }

        n3ldg_cuda::CopyForUniNodeForward(xs, params().b.val.value,
                x.value, y.value, count, inDim(), outDim(), params().bUseB);

        n3ldg_cuda::MatrixMultiplyMatrix(params().W.val.value, x.value, y.value,
                outDim(), inDim(), count, params().bUseB);

        std::vector<dtype*> vals;
        vals.reserve(count);
        for (Node *node : batch) {
            vals.push_back(node->val().value);
        }

        n3ldg_cuda::CopyFromOneVectorToMultiVals(y.value, vals, count, outDim());
#if TEST_CUDA
        for (int idx = 0; idx < count; idx++) {
            LinearNode* ptr = (LinearNode*)batch[idx];
            for (int idy = 0; idy < inDim; idy++) {
                x[idx][idy] = ptr->in->getVal()[idy];
            }
            if (param->bUseB) {
                for (int i = 0; i < outDim; ++i) {
                    b[idx][i] = param->b.val.v[i];
                }
            }
        }

        y.mat() = param->W.val.mat() * x.mat();
        if (param->bUseB) {
            y.vec() += b.vec();
        }

        n3ldg_cuda::Assert(x.verify("forward x"));
        n3ldg_cuda::Assert(y.verify("linear forward y"), batch.at(0)->typeSignature());

        for (int idx = 0; idx < count; idx++) {
            LinearNode* ptr = (LinearNode*)batch[idx];
            for (int idy = 0; idy < outDim; idy++) {
                ptr->val()[idy] = y[idx][idy];
            }
            n3ldg_cuda::Assert(ptr->val().verify("linear forward val"));
        }
        cout << "linear forward tested" << endl;
#endif
    }

    void backward() {
        int count = batch.size();
#if TEST_CUDA
        if (param->bUseB) {
            n3ldg_cuda::Assert(param->b.grad.verify("before linear backward b grad"));
            param->b.grad.copyFromDeviceToHost();
        }
        n3ldg_cuda::Assert(param->W.val.verify("before linear backward W val"));
        param->W.val.copyFromDeviceToHost();
        n3ldg_cuda::Assert(param->W.grad.verify("before linear backward W grad"));
        param->W.grad.copyFromDeviceToHost();
        for (int i = 0; i < count; ++i) {
            LinearNode* ptr = (LinearNode*)batch[i];
            n3ldg_cuda::Assert(ptr->loss().verify("before linear backward grad"));
            ptr->loss().copyFromDeviceToHost();
            n3ldg_cuda::Assert(ptr->val().verify("before linear val"));
            ptr->val().copyFromDeviceToHost();
            n3ldg_cuda::Assert(ptr->in->loss().verify("before linear backward in grad"));
            ptr->in->loss().copyFromDeviceToHost();
        }
#endif
        Tensor2D lx, ly;
        lx.init(inDim(), count);
        ly.init(outDim(), count);

        std::vector<dtype*> ly_vec;
        ly_vec.reserve(count);
        for (int i = 0; i < count; ++i) {
            LinearNode* ptr = (LinearNode*)batch[i];
            ly_vec.push_back(ptr->loss().value);
        }
        n3ldg_cuda::CopyFromMultiVectorsToOneVector(ly_vec, ly.value, count, outDim());
        n3ldg_cuda::MatrixMultiplyMatrix(ly.value, x.value, params().W.grad.value, outDim(), count,
                inDim(), true, true, false);
        n3ldg_cuda::MatrixMultiplyMatrix(params().W.val.value, ly.value, lx.value, inDim(),
                outDim(), count, false, false, true);
        std::vector<dtype*> losses;
        losses.reserve(count);
        for (int idx = 0; idx < count; idx++) {
            LinearNode* ptr = (LinearNode*)batch[idx];
            losses.push_back(ptr->getInput()->loss().value);
        }

        n3ldg_cuda::AddLtyToParamBiasAndAddLxToInputLossesForUniBackward(ly.value, lx.value,
                params().b.grad.value, losses, count, outDim(), inDim(), params().bUseB);

#if TEST_CUDA
        for (int idx = 0; idx < count; idx++) {
            LinearNode* ptr = (LinearNode*)batch[idx];
            for (int idy = 0; idy < outDim; idy++) {
                ly[idx][idy] = ptr->getLoss()[idy];
            }
        }

        n3ldg_cuda::Assert(x.verify("backward x"));

        param->W.grad.mat() += ly.mat() * x.mat().transpose();
        n3ldg_cuda::Assert(param->W.grad.verify("LinearExecutor backward W grad"));

        if (param->bUseB) {
            for (int idx = 0; idx < count; idx++) {
                for (int idy = 0; idy < outDim; idy++) {
                    param->b.grad.v[idy] += ly[idx][idy];
                }
            }
            n3ldg_cuda::Assert(param->b.grad.verify("backward b grad"));
        }

        lx.mat() += param->W.val.mat().transpose() * ly.mat();
        n3ldg_cuda::Assert(lx.verify("linear execute backward lx"));

        for (int idx = 0; idx < count; idx++) {
            LinearNode* ptr = (LinearNode*)batch[idx];
            for (int idy = 0; idy < inDim; idy++) {
                ptr->in->loss()[idy] += lx[idx][idy];
            }
        }

        for (Node * n : batch) {
            LinearNode *ptr = static_cast<LinearNode *>(n);
            n3ldg_cuda::Assert(ptr->in->loss().verify("backward loss"));
        }
#endif
    }
};
#else
class LinearExecutor :public Executor {
public:
    Tensor2D x, y, b;
    int inDim, outDim, count;
    UniParams* param;

    int calculateFLOPs() override {
        int flops = param->W.inDim() * param->W.outDim() * batch.size() * 2;
        if (param->bUseB) {
            flops += param->W.outDim() * batch.size();
        }
        return flops;
    }

    void  forward() override {
        count = batch.size();
        x.init(inDim, count);
        y.init(outDim, count);
        b.init(outDim, count);

        for (int idx = 0; idx < count; idx++) {
            LinearNode* ptr = (LinearNode*)batch[idx];
            memcpy(x.v + idx * inDim, ptr->in->val().v, inDim * sizeof(dtype));
            if (param->bUseB) {
                memcpy(b.v + idx * outDim, param->b.val.v, outDim * sizeof(dtype));
            }
        }

        y.mat() = param->W.val.mat() * x.mat();
        if (param->bUseB) {
            y.vec() = y.vec() + b.vec();
        }

        for (int idx = 0; idx < count; idx++) {
            LinearNode* ptr = (LinearNode*)batch[idx];
            memcpy(ptr->val().v, y.v + idx * outDim, outDim * sizeof(dtype));
        }
    }

    void backward() override {
        Tensor2D lx, ly;
        lx.init(inDim, count);
        ly.init(outDim, count);

        for (int idx = 0; idx < count; idx++) {
            LinearNode* ptr = (LinearNode*)batch[idx];
            memcpy(ly.v + idx * outDim, ptr->loss().v, outDim * sizeof(dtype));
        }

        param->W.grad.mat() += ly.mat() * x.mat().transpose();

        if (param->bUseB) {
            for (int idy = 0; idy < outDim; idy++) {
                for (int idx = 0; idx < count; idx++) {
                    param->b.grad.v[idy] += ly[idx][idy];
                }
            }
        }

        lx.mat() = param->W.val.mat().transpose() * ly.mat();

        for (int idx = 0; idx < count; idx++) {
            LinearNode* ptr = (LinearNode*)batch[idx];
            for (int idy = 0; idy < inDim; idy++) {
                ptr->in->loss()[idy] += lx[idx][idy];
            }
        }
    }
};
#endif

Executor * LinearNode::generate() {
    return new LinearExecutor();
};

class LinearWordVectorExecutor;

class LinearWordVectorNode : public UniInputNode, public Poolable<LinearWordVectorNode> {
public:
    LinearWordVectorNode() : UniInputNode("linear_word_vector_node") {}

    void initNode(int dim) override {
        init(dim);
    }

    void setNodeDim(int dim) override {
        setDim(dim);
    }

    bool isDimLegal(const Node &input) const override {
        return input.getDim() == param_->outDim();
    }

    void setParam(Param &word_vectors, int offset = 0) {
        if (offset + getDim() > word_vectors.inDim()) {
            cerr << boost::format("offset:%1% getDim():%2% word_vectors.inDim():%3%") % offset %
                getDim() % word_vectors.inDim() << endl;
            abort();
        }
        param_ = &word_vectors;
        offset_ = offset;
    }

    void compute() override {
        abort();
    }

    void backward() override {
        abort();
    }

    Executor* generate() override;

    string typeSignature() const override {
        return Node::typeSignature() + "-" + addressToString(param_) + "-" + to_string(offset_);
    }

    int getOffset() const {
        return offset_;
    }

private:
    Param *param_ = nullptr;
    friend class LinearWordVectorExecutor;
    int offset_ = 0;
};

class BatchedLinearWordVectorNode : public BatchedNodeImpl<LinearWordVectorNode> {
public:
    void init(Graph &graph, BatchedNode &input, Param &param, int dim, int offset) {
        allocateBatch(dim, input.batch().size());

        for (Node *node : batch()) {
            LinearWordVectorNode *l = dynamic_cast<LinearWordVectorNode *>(node);
            l->setParam(param, offset);
        }
        setInputsPerNode({&input});
        afterInit(graph, {&input});
    }
};

namespace n3ldg_plus {
    Node *linearWordVector(Graph &graph, Node &input, Param &param, int dim,
            int offset = 0) {
        if (dim + offset > param.inDim()) {
            cerr << boost::format("linearWordVector - dim:%1% offset%2% vocabulary_size:%3%") %
                dim % offset % param.inDim() << endl;
            abort();
        }

        if (input.getDim() != param.outDim()) {
            cerr << boost::format("LinearWordVectorNode - input dim:%1% word vector dim:%2%") %
                input.getDim() % param.outDim() << endl;
            abort();
        }

        LinearWordVectorNode *node =  LinearWordVectorNode::newNode(dim);
        node->setParam(param, offset);
        node->forward(graph, input);
        return node;
    }

    BatchedNode *linearWordVector(Graph &graph, BatchedNode &input, Param &param, int dim,
            int offset = 0) {
        if (dim + offset > param.inDim()) {
            cerr << boost::format("linearWordVector - dim:%1% offset%2% vocabulary_size:%3%") %
                dim % offset % param.inDim() << endl;
            abort();
        }

        if (input.getDim() != param.outDim()) {
            cerr << boost::format("LinearWordVectorNode - input dim:%1% word vector dim:%2%") %
                input.getDim() % param.outDim() << endl;
            abort();
        }
        BatchedLinearWordVectorNode *node = new BatchedLinearWordVectorNode;
        node->init(graph, input, param, dim, offset);
        return node;
    }
}

#if USE_GPU

class LinearWordVectorExecutor : public UniInputExecutor {
public:
    Tensor2D x, y;
    int inDim, outDim;
    Param *param;

    void forward() {
        int count = batch.size();

        x.init(inDim, count);
        y.init(outDim, count);
        std::vector<dtype*> xs, ys;
        xs.reserve(batch.size());
        ys.reserve(batch.size());
#if TEST_CUDA
        param->val.copyFromDeviceToHost();
#endif

        for (int i = 0; i < batch.size(); ++i) {
            LinearWordVectorNode *n = static_cast<LinearWordVectorNode*>(batch.at(i));
#if TEST_CUDA
            n->getInput()->val().copyFromDeviceToHost();
#endif
            xs.push_back(n->getInput()->val().value);
            ys.push_back(n->val().value);
        }

        n3ldg_cuda::CopyForUniNodeForward(xs, nullptr, x.value, y.value, count, inDim, outDim,
                false);
        int offset = static_cast<LinearWordVectorNode*>(batch.front())->offset_;
        n3ldg_cuda::MatrixMultiplyMatrix(param->val.value + offset * inDim, x.value, y.value,
                outDim, inDim, count, false, false, true);

        std::vector<dtype*> vals;
        vals.reserve(count);
        for (Node *node : batch) {
            vals.push_back(node->val().value);
        }

        n3ldg_cuda::CopyFromOneVectorToMultiVals(y.value, vals, count, outDim);
#if TEST_CUDA
        for (int i = 0; i < count; i++) {
            LinearWordVectorNode* ptr = (LinearWordVectorNode*)batch.at(i);
            memcpy(x.v + i * inDim, ptr->getInput()->val().v, inDim * sizeof(dtype));
        }
        Mat scoped_matrix(param->val.mat().data() + offset * inDim, inDim, outDim);
        y.mat() = scoped_matrix.transpose() * x.mat();
        n3ldg_cuda::Assert(x.verify("forward x"));
        n3ldg_cuda::Assert(y.verify("linear word forward y"));

        for (int idx = 0; idx < count; idx++) {
            LinearNode* ptr = (LinearNode*)batch[idx];
            for (int idy = 0; idy < outDim; idy++) {
                ptr->val()[idy] = y[idx][idy];
            }
            n3ldg_cuda::Assert(ptr->val().verify("linear forward val"));
        }
        cout << "LinearWordVectorExecutor forward tested" << endl;
#endif
    }

    void backward() {
        int count = batch.size();
        Tensor2D lx, ly;
        lx.init(inDim, count);
        ly.init(outDim, count);
#if TEST_CUDA
        param->grad.copyFromDeviceToHost();
#endif

        std::vector<dtype*> ly_vec;
        ly_vec.reserve(count);
        for (int i = 0; i < count; ++i) {
            LinearWordVectorNode* ptr = (LinearWordVectorNode*)batch.at(i);
#if TEST_CUDA
            ptr->loss().copyFromDeviceToHost();
            ptr->val().copyFromDeviceToHost();
#endif
            ly_vec.push_back(ptr->loss().value);
        }
        n3ldg_cuda::CopyFromMultiVectorsToOneVector(ly_vec, ly.value, count, outDim);
        int offset = static_cast<LinearWordVectorNode*>(batch.front())->offset_;
        n3ldg_cuda::MatrixMultiplyMatrix(x.value, ly.value, param->grad.value + inDim * offset,
                inDim, count, outDim, true, true, false);
        n3ldg_cuda::MatrixMultiplyMatrix(param->val.value + offset * inDim, ly.value, lx.value,
                inDim, outDim, count, false, false, false);
        std::vector<dtype*> losses;
        losses.reserve(count);
        for (int idx = 0; idx < count; idx++) {
            LinearWordVectorNode* ptr = (LinearWordVectorNode*)batch[idx];
            losses.push_back(ptr->getInput()->loss().value);
        }

        n3ldg_cuda::AddLtyToParamBiasAndAddLxToInputLossesForUniBackward(ly.value, lx.value,
                nullptr, losses, count, outDim, inDim, false);
#if TEST_CUDA
        for (int idx = 0; idx < count; idx++) {
            LinearWordVectorNode* ptr = (LinearWordVectorNode*)batch[idx];
            memcpy(ly.v + idx * outDim, ptr->loss().v, outDim * sizeof(dtype));
            ptr->loss().copyFromDeviceToHost();
        }

        n3ldg_cuda::Assert(x.verify("backward x"));
        n3ldg_cuda::Assert(ly.verify("backward ly"));

        auto scoped_grad = x.mat() * ly.mat().transpose();
        MatrixXdtype full_grad(inDim, param->inDim()), left(inDim, offset),
                     right(inDim, param->inDim() - offset - outDim);
        left.setZero();
        right.setZero();
        full_grad << left, scoped_grad, right;
        param->grad.mat() += full_grad;
        function<void(void)> print = [&]()->void {
            cerr << "outdim:" << outDim << " param indim:" << param->inDim() << " offset:" <<
                offset << " indim:" << inDim << endl;
//            cerr << "cpu:" << endl << param->grad.toString() << endl << "gpu:" << endl;
//            param->grad.print();
        };
        n3ldg_cuda::Assert(param->grad.verify("LinearWordVectorExecutor backward W grad"), "",
                print);

        Mat scoped_matrix(param->val.mat().data() + offset * inDim, inDim, outDim);
        lx.mat() = scoped_matrix * ly.mat();
        n3ldg_cuda::Assert(lx.verify("linear word vector execute backward lx"));

        for (int idx = 0; idx < count; idx++) {
            LinearWordVectorNode* ptr = (LinearWordVectorNode*)batch[idx];
            for (int idy = 0; idy < inDim; idy++) {
                ptr->getInput()->loss()[idy] += lx[idx][idy];
            }
        }

        for (Node * n : batch) {
            LinearWordVectorNode *ptr = static_cast<LinearWordVectorNode *>(n);
            n3ldg_cuda::Assert(ptr->getInput()->loss().verify("backward loss"));
        }
        cout << "LinearWordVectorNode backward tested" << endl;
#endif
    }
};

#else

class LinearWordVectorExecutor : public Executor {
public:
    Tensor2D x, y;
    int inDim, outDim;
    Param *param;

    int calculateFLOPs() override {
        LinearWordVectorNode *node = static_cast<LinearWordVectorNode*>(batch.front());
        return node->getDim() * node->getInput()->getDim() * batch.size() * 2;
    }

    void forward() override {
        int count = batch.size();
        x.init(inDim, count);
        y.init(outDim, count);

        for (int i = 0; i < count; i++) {
            LinearWordVectorNode* ptr = (LinearWordVectorNode*)batch.at(i);
            memcpy(x.v + i * inDim, ptr->getInput()->val().v, inDim * sizeof(dtype));
        }
        int offset = static_cast<LinearWordVectorNode*>(batch.front())->offset_;
        Mat scoped_matrix(param->val.mat().data() + offset * inDim, inDim, outDim);
        y.mat() = scoped_matrix.transpose() * x.mat();

        for (int i = 0; i < count; i++) {
            LinearWordVectorNode* ptr = (LinearWordVectorNode*)batch.at(i);
            memcpy(ptr->val().v, y.v + i * outDim, outDim * sizeof(dtype));
        }
    }

    void backward() override {
        Tensor2D lx, ly;
        int count = batch.size();
        lx.init(inDim, count);
        ly.init(outDim, count);

        for (int idx = 0; idx < count; idx++) {
            LinearWordVectorNode* ptr = (LinearWordVectorNode*)batch[idx];
            memcpy(ly.v + idx * outDim, ptr->loss().v, outDim * sizeof(dtype));
        }

        int offset = static_cast<LinearWordVectorNode*>(batch.front())->offset_;
        auto scoped_grad = x.mat() * ly.mat().transpose();
        MatrixXdtype full_grad(inDim, param->inDim()), left(inDim, offset),
                     right(inDim, param->inDim() - offset - outDim);
        left.setZero();
        right.setZero();
        full_grad << left, scoped_grad, right;
        param->grad.mat() += full_grad;

        Mat scoped_matrix(param->val.mat().data() + offset * inDim, inDim, outDim);
        lx.mat() = scoped_matrix * ly.mat();

        for (int idx = 0; idx < count; idx++) {
            LinearWordVectorNode* ptr = (LinearWordVectorNode*)batch[idx];
            for (int idy = 0; idy < inDim; idy++) {
                ptr->getInput()->loss()[idy] += lx[idx][idy];
            }
        }
    }
};

#endif

Executor* LinearWordVectorNode::generate() {
    LinearWordVectorExecutor* exec = new LinearWordVectorExecutor();
    exec->batch.push_back(this);
    exec->inDim = param_->outDim();
    exec->outDim = getDim();
    exec->param = param_;
    return exec;
}

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
            val()[i] = getInput()->getVal()[i] + bias_param_->val[0][i];
        }
    }

    void backward() override {
        getInput()->loss().vec() += loss().vec();
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
            inputs.push_back(bias_node->getInput()->getVal().value);
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
            in_losses.push_back(bias_node->getInput()->getLoss().value);
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
    int dim = params.W.outDim();
    LinearNode *uni = LinearNode::newNode(dim);
    uni->setParam(params);
    uni->forward(graph, input);
    return uni;
}

BatchedNode *linear(Graph &graph, BatchedNode &input, UniParams &params) {
    BatchedLinearNode *node = new BatchedLinearNode;
    node->init(graph, input, params);
    return node;
}

Node *uni(Graph &graph, UniParams &params, Node &input, ActivatedEnum activated_type =
        ActivatedEnum::TANH) {
    int dim = params.W.outDim();

    Node *uni = linear(graph, input, params);

    UniInputNode *activated;
    if (activated_type == ActivatedEnum::TANH) {
        activated = TanhNode::newNode(dim);
    } else if (activated_type == ActivatedEnum::SIGMOID) {
        activated = SigmoidNode::newNode(dim);
    } else if (activated_type == ActivatedEnum::RELU) {
        activated = ReluNode::newNode(dim);
    } else {
        cerr << "uni - unsupported activated " << activated << endl;
        abort();
    }

    activated->forward(graph, *uni);

    return activated;
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
