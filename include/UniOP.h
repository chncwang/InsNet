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

    void init(int nOSize, int nISize, bool useB = true) {
        W.init(nOSize, nISize);

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

    virtual std::string name() const {
        return "UniParams";
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

class LinearNode : public Node {
public:
    PNode in;
    UniParams* param;

    LinearNode() : Node("linear") {
        in = NULL;
        param = NULL;
    }

    void setParam(UniParams &uni_params) {
        this->setParam(&uni_params);
    }

    void setParam(UniParams* paramInit) {
        if (paramInit->bUseB) {
            assert(paramInit->W.outDim() == paramInit->b.outDim());
        }
        param = paramInit;
    }

    void forward(Graph &graph, Node &x) {
        if (x.getDim() != param->W.inDim()) {
            cerr << boost::format("input dim:%1% node in dim:%2%") % x.getDim() % param->W.inDim()
                << endl;
            abort();
        }
        in = &x;
        in->addParent(this);
        graph.addNode(this);
    }

    void compute() override {
        abort();
    }

    void backward() override {
        abort();
    }

    PExecutor generate() override;

    bool typeEqual(PNode other) override {
        bool result = Node::typeEqual(other);
        if (!result) return false;
        LinearNode* conv_other = (LinearNode*)other;
        if (param != conv_other->param) {
            return false;
        }

        return true;
    }

    string typeSignature() const override {
        return Node::typeSignature() + "-" + addressToString(param);
    }
};

#if USE_GPU
class LinearExecutor :public Executor {
public:
    Tensor2D x, y, b;
    int inDim, outDim, count;
    UniParams* param;

    void  forward() {
        int count = batch.size();
#if TEST_CUDA
        param->W.val.copyFromDeviceToHost();
        if (param->bUseB) {
            param->b.val.copyFromDeviceToHost();
        }
#endif

        x.init(inDim, count);
        y.init(outDim, count);
#if TEST_CUDA
        b.init(outDim, count);
#endif
        std::vector<dtype*> xs, ys;
        xs.reserve(batch.size());
        ys.reserve(batch.size());

        for (int i = 0; i < batch.size(); ++i) {
            LinearNode *n = static_cast<LinearNode*>(batch.at(i));

            xs.push_back(n->in->val().value);
            ys.push_back(n->val().value);
#if TEST_CUDA
            n->in->val().copyFromDeviceToHost();
#endif
        }

        n3ldg_cuda::CopyForUniNodeForward(xs, param->b.val.value,
                x.value, y.value, count, inDim, outDim, param->bUseB);

        n3ldg_cuda::MatrixMultiplyMatrix(param->W.val.value, x.value, y.value,
                outDim, inDim, count, param->bUseB);

        std::vector<dtype*> vals;
        vals.reserve(count);
        for (Node *node : batch) {
            vals.push_back(node->val().value);
        }

        n3ldg_cuda::CopyFromOneVectorToMultiVals(y.value, vals, count, outDim);
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
        n3ldg_cuda::Assert(y.verify("linear forward y"), batch.at(0)->getNodeName());

        for (int idx = 0; idx < count; idx++) {
            LinearNode* ptr = (LinearNode*)batch[idx];
            for (int idy = 0; idy < outDim; idy++) {
                ptr->val()[idy] = y[idx][idy];
            }
            n3ldg_cuda::Assert(ptr->val().verify("linear forward val"));
        }
#endif
    }

    void backward() {
        int count = batch.size();
        Tensor2D lx, ly;
        lx.init(inDim, count);
        ly.init(outDim, count);

        std::vector<dtype*> ly_vec;
        ly_vec.reserve(count);
        for (int i = 0; i < count; ++i) {
            LinearNode* ptr = (LinearNode*)batch[i];
            ly_vec.push_back(ptr->loss().value);
        }
        n3ldg_cuda::CopyFromMultiVectorsToOneVector(ly_vec, ly.value, count, outDim);
        n3ldg_cuda::MatrixMultiplyMatrix(ly.value, x.value,
                param->W.grad.value, outDim, count, inDim, true, true, false);
        n3ldg_cuda::MatrixMultiplyMatrix(param->W.val.value, ly.value,
                lx.value, inDim, outDim, count, false, false, true);
        std::vector<dtype*> losses;
        losses.reserve(count);
        for (int idx = 0; idx < count; idx++) {
            LinearNode* ptr = (LinearNode*)batch[idx];
            losses.push_back(ptr->in->loss().value);
        }

        n3ldg_cuda::AddLtyToParamBiasAndAddLxToInputLossesForUniBackward(
                ly.value, lx.value, param->b.grad.value, losses, count,
                outDim, inDim, param->bUseB);

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

    void  forward() {
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

    void backward() {
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

PExecutor LinearNode::generate() {
    LinearExecutor* exec = new LinearExecutor();
    exec->batch.push_back(this);
    exec->inDim = param->W.inDim();
    exec->outDim = param->W.outDim();
    exec->param = param;
    return exec;
};

class LinearWordVectorExecutor;

class LinearWordVectorNode : public UniInputNode {
public:
    LinearWordVectorNode() : UniInputNode("linear_word_vector_node") {}

    bool isDimLegal(const Node &input) const override {
        return input.getDim() == param_->outDim();
    }

    void setParam(SparseParam &word_vectors, int offset = 0) {
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

    bool typeEqual(PNode other) override {
        LinearWordVectorNode* conv_other = (LinearWordVectorNode*)other;
        return Node::typeEqual(other) && param_ == conv_other->param_ &&
            offset_ == conv_other->offset_;
    }

    string typeSignature() const override {
        return Node::typeSignature() + "-" + addressToString(param_) + "-" + to_string(offset_);
    }

    int getOffset() const {
        return offset_;
    }

private:
    SparseParam *param_ = nullptr;
    friend class LinearWordVectorExecutor;
    int offset_ = 0;
};

#if USE_GPU

class LinearWordVectorExecutor : public UniInputExecutor {
public:
    Tensor2D x, y;
    int inDim, outDim;
    SparseParam *param;

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
    SparseParam *param;

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

namespace n3ldg_plus {

Node *uni(Graph &graph, UniParams &params, Node &input, ActivatedEnum activated_type =
        ActivatedEnum::TANH) {
    int dim = params.W.outDim();

    LinearNode *uni(new LinearNode);
    uni->init(dim);
    uni->setParam(params);
    uni->forward(graph, input);

    UniInputNode *activated;
    if (activated_type == ActivatedEnum::TANH) {
        activated = new TanhNode;
    } else if (activated_type == ActivatedEnum::SIGMOID) {
        activated = new SigmoidNode;
    } else {
        cerr << "unsupported activated " << activated << endl;
        abort();
    }

    activated->init(dim);
    activated->forward(graph, *uni);

    return activated;
}

}

#endif /* UNIOP_H_ */
