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
#include "Node.h"
#include "Graph.h"
#include "ModelUpdate.h"
#include <cstdlib>
#include "profiler.h"

class UniParams : public N3LDGSerializable
#if USE_GPU
, public TransferableComponents 
#endif
{
public:
    Param W;
    Param b;
    bool bUseB = true;

    UniParams() : b(true) {}

    void exportAdaParams(ModelUpdate& ada) {
        ada.addParam(&W);
        if (bUseB) {
            ada.addParam(&b);
        }
    }

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
};


class UniNode : public Node {
public:
    PNode in;
    UniParams* param;
    dtype(*activate)(const dtype&);   // activation function
    dtype(*derivate)(const dtype&, const dtype&);  // derivation function of activation function
    Tensor1D ty, lty;

    UniNode() : Node("uni") {
        in = NULL;
        activate = ftanh;
        derivate = dtanh;
        param = NULL;
    }

    ~UniNode() {
        in = NULL;
    }

    void init(int ndim) override {
        Node::init(ndim);
        ty.init(ndim);
        lty.init(ndim);
    }


    void setParam(UniParams& paramInit) {
        param = &paramInit;
    }

    void setFunctions(dtype(*f)(const dtype&), dtype(*f_deri)(const dtype&, const dtype&)) {
        activate = f;
        derivate = f_deri;
    }

    void forward(Graph &graph, Node &x) {
        this->forward(&graph, &x);
    }

    void forward(Graph *cg, PNode x) {
        in = x;
        in->addParent(this);
        cg->addNode(this);
    }

    void compute() override {
        ty.mat() = param->W.val.mat() * in->val().mat();
        if (param->bUseB) {
            ty.vec() += param->b.val.vec();
        }
        val().vec() = ty.vec().unaryExpr(ptr_fun(activate));
    }

    void backward() override {
        lty.vec() = loss().vec() * ty.vec().binaryExpr(val().vec(), ptr_fun(derivate));
        param->W.grad.mat() += lty.mat() * in->val().tmat();
        if (param->bUseB) {
            param->b.grad.vec() += lty.vec();
        }
        in->loss().mat() += param->W.val.mat().transpose() * lty.mat();
    }

    PExecutor generate() override;

    // better to rewrite for deep understanding
    bool typeEqual(PNode other) override {
        bool result = Node::typeEqual(other);
        if (!result) return false;

        UniNode* conv_other = (UniNode*)other;
        if (param != conv_other->param) {
            return false;
        }
        if (activate != conv_other->activate || derivate != conv_other->derivate) {
            return false;
        }

        return true;
    }

    size_t typeHashCode() const override {
        void *act = reinterpret_cast<void*>(activate);
        void *de = reinterpret_cast<void*>(derivate);
        return Node::typeHashCode() ^ ::typeHashCode(param) ^ ::typeHashCode(act) ^
            (::typeHashCode(de) << 1);
    }
};


// non-linear feed-forward node
// input nodes should be specified by forward function
// for input variables, we exploit column vector,
// which means a concrete input vector x_i is represented by x(0, i), x(1, i), ..., x(n, i)
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
            cerr << boost::format("input dim:%1% node in dim:%2%") % x.getDim() % param->W.inDim() << endl;
            abort();
        }
        in = &x;
        in->addParent(this);
        graph.addNode(this);
    }

    void compute() override {
        val().mat() = param->W.val.mat() * in->val().mat();
    }

    void backward() override {
        param->W.grad.mat() += loss().mat() * in->val().tmat();
        in->loss().mat() += param->W.val.mat().transpose() * loss().mat();
    }

    PExecutor generate() override;

    // better to rewrite for deep understanding
    bool typeEqual(PNode other) override {
        bool result = Node::typeEqual(other);
        if (!result) return false;
        LinearNode* conv_other = (LinearNode*)other;
        if (param != conv_other->param) {
            return false;
        }

        return true;
    }

    size_t typeHashCode() const override {
        return Node::typeHashCode() ^ ::typeHashCode(param);
    }
};


class UniExecutor :public Executor {
  public:
    Tensor2D x, ty, y, b;
    int inDim, outDim;
    UniParams* param;
    dtype(*activate)(const dtype&);   // activation function
    dtype(*derivate)(const dtype&, const dtype&);  // derivation function of activation function

    void  forward() {
        int count = batch.size();
        ty.init(outDim, count);
        x.init(inDim, count);
        y.init(outDim, count);
#if TEST_CUDA || !USE_GPU
        b.init(outDim, count);
#endif

#if USE_GPU
        std::vector<dtype*> xs, ys;
        xs.reserve(batch.size());
        ys.reserve(batch.size());


        for (int i = 0; i < batch.size(); ++i) {
            UniNode *n = static_cast<UniNode*>(batch.at(i));

            xs.push_back(n->in->val().value);
            ys.push_back(n->val().value);
        }

        n3ldg_cuda::CopyForUniNodeForward(xs, param->b.val.value,
                x.value, ty.value, count, inDim, outDim, param->bUseB);

        n3ldg_cuda::MatrixMultiplyMatrix(param->W.val.value, x.value,
                ty.value, outDim, inDim, count, param->bUseB);

        n3ldg_cuda::ActivatedEnum activatedEnum = ToActivatedEnum(activate);
        n3ldg_cuda::Activated(activatedEnum, ty.value, ys, y.value, outDim);

#if TEST_CUDA
        for (int idx = 0; idx < count; idx++) {
            UniNode* ptr = (UniNode*)batch[idx];
            n3ldg_cuda::Assert(ptr->in->val().verify("Uni forward in"));
            for (int idy = 0; idy < inDim; idy++) {
                x[idx][idy] = ptr->in->getVal()[idy];
            }
            if (param->bUseB) {
                for (int idy = 0; idy < outDim; idy++) {
                    b[idx][idy] = param->b.val.v[idy];
                }
            }
        }
        n3ldg_cuda::Assert(x.verify("forward x"));

        ty.mat() = param->W.val.mat() * x.mat();

        if (param->bUseB) {
            ty.vec() = ty.vec() + b.vec();
        }

        y.vec() = ty.vec().unaryExpr(ptr_fun(activate));

        for (int idx = 0; idx < count; idx++) {
            UniNode* ptr = (UniNode*)batch[idx];
            for (int idy = 0; idy < outDim; idy++) {
                ptr->val()[idy] = y[idx][idy];
            }
        }

        for (int i = 0; i < count; ++i) {
            n3ldg_cuda::Assert(batch[i]->val().verify("forward batch i val"));
        }

        n3ldg_cuda::Assert(ty.verify("forward ty"));
        n3ldg_cuda::Assert(y.verify("Uni forward y"));
#endif
#else
        n3ldg_cuda::Profiler &profiler = n3ldg_cuda::Profiler::Ins();
        profiler.BeginEvent("uni forward");
        profiler.BeginEvent("uni merge");
        for (int idx = 0; idx < count; idx++) {
            UniNode* ptr = (UniNode*)batch[idx];
            memcpy(x.v + idx * inDim, ptr->in->val().v, inDim * sizeof(dtype));
            if (param->bUseB) {
                memcpy(b.v + idx * outDim, param->b.val.v, outDim * sizeof(dtype));
            }
        }
        profiler.EndEvent();

        ty.mat() = param->W.val.mat() * x.mat();

        if (param->bUseB) {
            ty.vec() = ty.vec() + b.vec();
        }

        y.vec() = ty.vec().unaryExpr(ptr_fun(activate));

        profiler.BeginEvent("uni split");
        for (int idx = 0; idx < count; idx++) {
            UniNode* ptr = (UniNode*)batch[idx];
            memcpy(ptr->val().v, y.v + idx * outDim, outDim * sizeof(dtype));
        }
        profiler.EndEvent();

        profiler.EndEvent();
#endif
    }

    void backward() {
        int count = batch.size();
        Tensor2D lx, lty, ly;
#if USE_GPU
        lx.init(inDim, count);
        lty.init(outDim, count);
        ly.init(outDim, count);

        std::vector<dtype*> ly_vec;
        ly_vec.reserve(count);
        for (int i = 0; i < count; ++i) {
            UniNode* ptr = (UniNode*)batch[i];
            ly_vec.push_back(ptr->loss().value);
        }
        n3ldg_cuda::ActivatedEnum activatedEnum = ToActivatedEnum(activate);
        n3ldg_cuda::CalculateLtyForUniBackward(activatedEnum, ly_vec, ty.value,
                y.value, lty.value, count, outDim);
#if TEST_CUDA
        n3ldg_cuda::Assert(param->W.grad.verify(
                    "uni backward W grad init"));
#endif
        n3ldg_cuda::MatrixMultiplyMatrix(lty.value, x.value,
                param->W.grad.value, outDim, count, inDim, true, true, false);
#if TEST_CUDA
        n3ldg_cuda::Assert(param->W.val.verify("uni W.val init"));
#endif
        n3ldg_cuda::MatrixMultiplyMatrix(param->W.val.value, lty.value,
                lx.value, inDim, outDim, count, false, false, true);
        std::vector<dtype*> losses;
        losses.reserve(count);
        for (int idx = 0; idx < count; idx++) {
            UniNode* ptr = (UniNode*)batch[idx];
            losses.push_back(ptr->in->loss().value);
        }

#if TEST_CUDA
        n3ldg_cuda::Assert(
                param->b.grad.verify("uni backward param b init"));
#endif
        n3ldg_cuda::AddLtyToParamBiasAndAddLxToInputLossesForUniBackward(
                lty.value, lx.value, param->b.grad.value, losses, count,
                outDim, inDim, param->bUseB);

#if TEST_CUDA
        for (int idx = 0; idx < count; idx++) {
            UniNode* ptr = (UniNode*)batch[idx];
            for (int idy = 0; idy < outDim; idy++) {
                ly[idx][idy] = ptr->getLoss()[idy];
            }
        }

        n3ldg_cuda::Assert(x.verify("backward x"));
        lty.vec() = ly.vec() * ty.vec().binaryExpr(y.vec(), ptr_fun(derivate));
        n3ldg_cuda::Assert(lty.verify("backward lty"));

        param->W.grad.mat() += lty.mat() * x.mat().transpose();
        n3ldg_cuda::Assert(param->W.grad.verify("backward W grad"));

        if (param->bUseB) {
            for (int idx = 0; idx < count; idx++) {
                for (int idy = 0; idy < outDim; idy++) {
                    param->b.grad.v[idy] += lty[idx][idy];
                }
            }
        }
        n3ldg_cuda::Assert(param->b.grad.verify("backward b grad"));

        lx.mat() += param->W.val.mat().transpose() * lty.mat();
        n3ldg_cuda::Assert(lx.verify("backward lx"));

        for (int idx = 0; idx < count; idx++) {
            UniNode* ptr = (UniNode*)batch[idx];
            for (int idy = 0; idy < inDim; idy++) {
                ptr->in->loss()[idy] += lx[idx][idy];
            }
        }

        for (Node * n : batch) {
            UniNode *ptr = static_cast<UniNode *>(n);
            n3ldg_cuda::Assert(ptr->in->loss().verify("uni backward loss"));
        }
#endif
#else
        lx.init(inDim, count);
        lty.init(outDim, count);
        ly.init(outDim, count);
        for (int idx = 0; idx < count; idx++) {
            UniNode* ptr = (UniNode*)batch[idx];
            memcpy(ly.v + idx * outDim, ptr->loss().v, outDim * sizeof(dtype));
        }

        lty.vec() = ly.vec() * ty.vec().binaryExpr(y.vec(), ptr_fun(derivate));
        param->W.grad.mat() += lty.mat() * x.mat().transpose();

        if (param->bUseB) {
            for (int idy = 0; idy < outDim; idy++) {
                for (int idx = 0; idx < count; idx++) {
                    param->b.grad.v[idy] += lty[idx][idy];
                }
            }
        }

        lx.mat() += param->W.val.mat().transpose() * lty.mat();

        for (int idx = 0; idx < count; idx++) {
            UniNode* ptr = (UniNode*)batch[idx];
            for (int idy = 0; idy < inDim; idy++) {
                ptr->in->loss()[idy] += lx[idx][idy];
            }
        }
#endif
    }
};

PExecutor UniNode::generate() {
    UniExecutor* exec = new UniExecutor();
    exec->batch.push_back(this);
    exec->inDim = param->W.inDim();
    exec->outDim = param->W.outDim();
    exec->param = param;
    exec->activate = activate;
    exec->derivate = derivate;
    return exec;
};

#if USE_GPU
class LinearExecutor :public Executor {
public:
    Tensor2D x, y, b;
    int inDim, outDim, count;
    UniParams* param;

    void  forward() {
        int count = batch.size();

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

struct LinearWordVectorNode : public Node {
    Node *input;
    SparseParam *param;

    LinearWordVectorNode() : Node("linear_word_vector_node"), input(nullptr), param(nullptr) {}

    void setParam(SparseParam &word_vectors) {
        param = &word_vectors;
    }

    void forward(Graph &graph, Node &in) {
        input = &in;
        in.addParent(this);
        graph.addNode(this);
    }

    void compute() override {
        abort();
    }

    void backward() override {
        abort();
    }

    Executor* generate() override;

    bool typeEqual(PNode other) override {
        bool result = Node::typeEqual(other);
        if (!result) return false;
        LinearWordVectorNode* conv_other = (LinearWordVectorNode*)other;
        if (param != conv_other->param) {
            return false;
        }

        return true;
    }

    size_t typeHashCode() const override {
        return Node::typeHashCode() ^ ::typeHashCode(param);
    }
};

#if USE_GPU

struct LinearWordVectorExecutor : public Executor {
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

        for (int i = 0; i < batch.size(); ++i) {
            LinearNode *n = static_cast<LinearNode*>(batch.at(i));

            xs.push_back(n->in->val().value);
            ys.push_back(n->val().value);
        }

        n3ldg_cuda::CopyForUniNodeForward(xs, nullptr, x.value, y.value, count, inDim, outDim,
                false);
        n3ldg_cuda::MatrixMultiplyMatrix(param->val.value, x.value, y.value, outDim, inDim, count,
                false, false, true);

        std::vector<dtype*> vals;
        vals.reserve(count);
        for (Node *node : batch) {
            vals.push_back(node->val().value);
        }

        n3ldg_cuda::CopyFromOneVectorToMultiVals(y.value, vals, count, outDim);
#if TEST_CUDA
        for (int idx = 0; idx < count; idx++) {
            LinearWordVectorNode* ptr = (LinearWordVectorNode*)batch[idx];
            for (int idy = 0; idy < inDim; idy++) {
                x[idx][idy] = ptr->input->val()[idy];
            }
        }

        y.mat() = param->val.mat().transpose() * x.mat();
        n3ldg_cuda::Assert(x.verify("forward x"));
        n3ldg_cuda::Assert(y.verify("linear word forward y"));

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
            UniNode* ptr = (UniNode*)batch[i];
            ly_vec.push_back(ptr->loss().value);
        }
        n3ldg_cuda::CopyFromMultiVectorsToOneVector(ly_vec, ly.value, count, outDim);
        n3ldg_cuda::MatrixMultiplyMatrix(x.value, ly.value, param->grad.value, inDim, count,
                outDim, true, true, false);
        n3ldg_cuda::MatrixMultiplyMatrix(param->val.value, ly.value, lx.value, inDim, outDim,
                count, false, false, false);
        std::vector<dtype*> losses;
        losses.reserve(count);
        for (int idx = 0; idx < count; idx++) {
            UniNode* ptr = (UniNode*)batch[idx];
            losses.push_back(ptr->in->loss().value);
        }

        n3ldg_cuda::AddLtyToParamBiasAndAddLxToInputLossesForUniBackward(ly.value, lx.value,
                nullptr, losses, count, outDim, inDim, false);
#if TEST_CUDA
        for (int idx = 0; idx < count; idx++) {
            Node* ptr = batch[idx];
            for (int idy = 0; idy < outDim; idy++) {
                ly[idx][idy] = ptr->getLoss()[idy];
            }
        }

        n3ldg_cuda::Assert(x.verify("backward x"));
        n3ldg_cuda::Assert(ly.verify("backward x"));

        param->grad.mat() += x.mat() * ly.mat().transpose();
        n3ldg_cuda::Assert(param->grad.verify("LinearExecutor backward W grad"));

        lx.mat() += param->val.mat() * ly.mat();
        n3ldg_cuda::Assert(lx.verify("linear execute backward lx"));

        for (int idx = 0; idx < count; idx++) {
            LinearWordVectorNode* ptr = (LinearWordVectorNode*)batch[idx];
            for (int idy = 0; idy < inDim; idy++) {
                ptr->input->loss()[idy] += lx[idx][idy];
            }
        }

        for (Node * n : batch) {
            LinearWordVectorNode *ptr = static_cast<LinearWordVectorNode *>(n);
            n3ldg_cuda::Assert(ptr->input->loss().verify("backward loss"));
        }
#endif
    }
};

#else

struct LinearWordVectorExecutor : public Executor {
    Tensor2D x, y;
    int inDim, outDim;
    SparseParam *param;

    void forward() {
        int count = batch.size();
        x.init(inDim, count);
        y.init(outDim, count);

        for (int i = 0; i < count; i++) {
            LinearWordVectorNode* ptr = (LinearWordVectorNode*)batch.at(i);
            memcpy(x.v + i * inDim, ptr->input->val().v, inDim * sizeof(dtype));
        }
        y.mat() = param->val.mat().transpose() * x.mat();

        for (int i = 0; i < count; i++) {
            LinearWordVectorNode* ptr = (LinearWordVectorNode*)batch[i];
            memcpy(ptr->val().v, y.v + i * outDim, outDim * sizeof(dtype));
        }
    }

    void backward() {
        Tensor2D lx, ly;
        int count = batch.size();
        lx.init(inDim, count);
        ly.init(outDim, count);

        for (int idx = 0; idx < count; idx++) {
            LinearWordVectorNode* ptr = (LinearWordVectorNode*)batch[idx];
            memcpy(ly.v + idx * outDim, ptr->loss().v, outDim * sizeof(dtype));
        }

        param->grad.mat() += x.mat() * ly.mat().transpose();

        lx.mat() = param->val.mat() * ly.mat();

        for (int idx = 0; idx < count; idx++) {
            LinearWordVectorNode* ptr = (LinearWordVectorNode*)batch[idx];
            for (int idy = 0; idy < inDim; idy++) {
                ptr->input->loss()[idy] += lx[idx][idy];
            }
        }
    }
};

#endif

Executor* LinearWordVectorNode::generate() {
    LinearWordVectorExecutor* exec = new LinearWordVectorExecutor();
    exec->batch.push_back(this);
    exec->inDim = param->outDim();
    exec->outDim = param->inDim();
    exec->param = param;
    return exec;
}

#endif /* UNIOP_H_ */
