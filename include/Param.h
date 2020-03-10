/*
 * Param.h
 *
 *  Created on: Jul 25, 2016
 *      Author: mason
 */

#ifndef PARAM_H_
#define PARAM_H_

#include "Eigen/Dense"
#include "BaseParam.h"
#include "MyTensor.h"
#if USE_GPU
#include "N3LDG_cuda.h"
#endif
#include "Node.h"

// Notice: aux is an auxiliary variable to help parameter updating
class Param : public BaseParam {
public:
    Tensor2D aux_square;
    Tensor2D aux_mean;
    int iter;
    bool is_fixed = false;

    Param(const string &name, bool is_bias = false) : BaseParam(name, is_bias) {}

    virtual void init(int outDim, int inDim) override {
        init(outDim, inDim, nullptr);
    }

    void init(int outDim, int inDim, const std::function<dtype(int, int)> *cal_bound) {
#if USE_GPU
        val.initOnMemoryAndDevice(outDim, inDim);
        aux_square.initOnMemoryAndDevice(outDim, inDim);
        aux_mean.initOnMemoryAndDevice(outDim, inDim);
#else
        val.init(outDim, inDim);
        aux_square.init(outDim, inDim);
        aux_mean.init(outDim, inDim);
#endif
        grad.init(outDim, inDim);
        if (isBias()) {
            val.assignAll(0.0f);
        } else {
            dtype bound = cal_bound == nullptr ? sqrt(6.0 / (outDim + inDim + 1)) :
                (*cal_bound)(outDim, inDim);
            val.random(bound);
        }
        iter = 0;
#if USE_GPU
        n3ldg_cuda::Memset(grad.value, outDim * inDim, 0.0f);
        n3ldg_cuda::Memset(aux_square.value, outDim * inDim, 0.0f);
        n3ldg_cuda::Memset(aux_mean.value, outDim * inDim, 0.0f);
#endif
    }

#if USE_GPU
    std::vector<n3ldg_cuda::Transferable *> transferablePtrs() override {
        auto v = BaseParam::transferablePtrs();
        v.push_back(&aux_square);
        v.push_back(&aux_mean);
        return v;
    }

    virtual std::string name() const {
        return "Param";
    }
#endif

    int outDim() override {
        return val.row;
    }

    int inDim() override {
        return val.col;
    }

    void clearGrad() override {
#if USE_GPU
        n3ldg_cuda::Memset(grad.value, grad.size, 0.0f);
#if TEST_CUDA
        grad.zero();
        n3ldg_cuda::Assert(grad.verify("Param clearGrad"));
#endif
#else
        grad.zero();
#endif
    }

    void updateAdagrad(dtype alpha, dtype reg, dtype eps) override {
        if (is_fixed) {
            return;
        }
#if USE_GPU
        n3ldg_cuda::UpdateAdagrad(val.value, grad.value, val.row, val.col,
                aux_square.value, alpha, reg, eps);
#if TEST_CUDA
        if (!isBias()) grad.vec() = grad.vec() + val.vec() * reg;
        aux_square.vec() = aux_square.vec() + grad.vec().square();
        val.vec() = val.vec() - grad.vec() * alpha / (aux_square.vec() + eps).sqrt();
        n3ldg_cuda::Assert(val.verify("Param adagrad"));
#endif
#else
        if (!isBias()) grad.vec() = grad.vec() + val.vec() * reg;
        aux_square.vec() = aux_square.vec() + grad.vec().square();
        val.vec() = val.vec() - grad.vec() * alpha / (aux_square.vec() + eps).sqrt();
#endif
    }

    void updateAdam(dtype belta1, dtype belta2, dtype alpha, dtype reg, dtype eps) override {
        if (is_fixed) {
            return;
        }
#if USE_GPU
#if TEST_CUDA
        n3ldg_cuda::Assert(val.verify("Param adam begin val"));
        n3ldg_cuda::Assert(grad.verify("Param adam begin grad"));
        n3ldg_cuda::Assert(aux_mean.verify("Param adam begin aux_mean"));
        n3ldg_cuda::Assert(aux_square.verify("Param adam begin aux_square"));
#endif
        n3ldg_cuda::UpdateAdam(val.value, grad.value, val.row, val.col, isBias(),
                aux_mean.value,
                aux_square.value,
                iter,
                belta1,
                belta2,
                alpha,
                reg,
                eps);
#if TEST_CUDA
        if (!isBias()) grad.vec() = grad.vec() + val.vec() * reg;
        aux_mean.vec() = belta1 * aux_mean.vec() + (1 - belta1) * grad.vec();
        aux_square.vec() = belta2 * aux_square.vec() + (1 - belta2) * grad.vec().square();
        dtype lr_t = alpha * sqrt(1 - pow(belta2, iter + 1)) / (1 - pow(belta1, iter + 1));
        val.vec() = val.vec() - aux_mean.vec() * lr_t / (aux_square.vec() + eps).sqrt();
        n3ldg_cuda::Assert(val.verify("Param adam"));
#endif
#else
        if (!isBias()) grad.vec() = grad.vec() + val.vec() * reg;
        aux_mean.vec() = belta1 * aux_mean.vec() + (1 - belta1) * grad.vec();
        aux_square.vec() = belta2 * aux_square.vec() + (1 - belta2) * grad.vec().square();
        dtype lr_t = alpha * sqrt(1 - pow(belta2, iter + 1)) / (1 - pow(belta1, iter + 1));
        val.vec() = val.vec() - aux_mean.vec() * lr_t / (aux_square.vec() + eps).sqrt();
#endif
        iter++;
    }

    void updateAdamW(dtype belta1, dtype belta2, dtype alpha, dtype reg, dtype eps) override {
        if (is_fixed) {
            return;
        }
#if USE_GPU
#if TEST_CUDA
        n3ldg_cuda::Assert(val.verify("Param adam begin val"));
        n3ldg_cuda::Assert(grad.verify("Param adam begin grad"));
        n3ldg_cuda::Assert(aux_mean.verify("Param adam begin aux_mean"));
        n3ldg_cuda::Assert(aux_square.verify("Param adam begin aux_square"));
#endif
        n3ldg_cuda::UpdateAdamW(val.value, grad.value, val.row, val.col, isBias(), aux_mean.value,
                aux_square.value, iter, belta1, belta2, alpha, reg, eps);
#if TEST_CUDA
        aux_mean.vec() = belta1 * aux_mean.vec() + (1 - belta1) * grad.vec();
        aux_square.vec() = belta2 * aux_square.vec() + (1 - belta2) * grad.vec().square();
        dtype lr_t = alpha * sqrt(1 - pow(belta2, iter + 1)) / (1 - pow(belta1, iter + 1));
        val.vec() = (1 - (isBias() ? 0.0f : reg)) * val.vec() -
            aux_mean.vec() * lr_t / (aux_square.vec() + eps).sqrt();
        n3ldg_cuda::Assert(val.verify("Param adam"));
#endif
#else
        aux_mean.vec() = belta1 * aux_mean.vec() + (1 - belta1) * grad.vec();
        aux_square.vec() = belta2 * aux_square.vec() + (1 - belta2) * grad.vec().square();
        dtype lr_t = alpha * sqrt(1 - pow(belta2, iter + 1)) / (1 - pow(belta1, iter + 1));
        val.vec() = (1 - (isBias() ? 0.0f : reg)) * val.vec() -
            aux_mean.vec() * lr_t / (aux_square.vec() + eps).sqrt();
#endif
        iter++;
    }

    void randpoint(int& idx, int &idy) override {
        std::vector<int> idRows, idCols;
        for (int i = 0; i < val.row; i++)
            idRows.push_back(i);
        for (int i = 0; i < val.col; i++)
            idCols.push_back(i);

        random_shuffle(idRows.begin(), idRows.end());
        random_shuffle(idCols.begin(), idCols.end());

        idy = idRows.at(0);
        idx = idCols.at(0);
    }

    dtype squareGradNorm() override {
#if USE_GPU && !TEST_CUDA
        return n3ldg_cuda::SquareSum(grad.value, grad.size);
#elif USE_GPU && TEST_CUDA
        cout << "squareGradNorm - param name:" << this->getParamName() << endl;
        n3ldg_cuda::Assert(grad.verify("squareGradNorm grad"));
        dtype cuda = n3ldg_cuda::SquareSum(grad.value, grad.size);
        dtype sumNorm = 0.0;
        for (int i = 0; i < grad.size; i++) {
            sumNorm += grad.v[i] * grad.v[i];
        }
        if (!isEqual(sumNorm, cuda)) {
            std::cout << "cpu:" << sumNorm << " cuda:" << cuda << std::endl;
        }
        return sumNorm;
#else
        dtype sumNorm = 0.0;
        for (int i = 0; i < grad.size; i++) {
            sumNorm += grad.v[i] * grad.v[i];
        }
        return sumNorm;
#endif
    }

    void rescaleGrad(dtype scale) override {
#if USE_GPU
        n3ldg_cuda::Rescale(grad.value, grad.size, scale);
#if TEST_CUDA
        grad.vec() = grad.vec() * scale;
        n3ldg_cuda::Assert(grad.verify("Param rescaleGrad"));
#endif
#else
        grad.vec() = grad.vec() * scale;
#endif
    }

    virtual Json::Value toJson() const override {
        Json::Value json;
        json["val"] = val.toJson();
        json["aux_square"] = aux_square.toJson();
        json["aux_mean"] = aux_mean.toJson();
        json["iter"] = iter;
        json["is_fixed"] = is_fixed;
        return json;
    }

    virtual void fromJson(const Json::Value &json) override {
        val.fromJson(json["val"]);
        aux_square.fromJson(json["aux_square"]);
        aux_mean.fromJson(json["aux_mean"]);
        iter = json["iter"].asInt();
        is_fixed = json["is_fixed"].asBool();
    }

    void value(const int& featId, Tensor1D& out) {
        if (out.dim != val.row) {
            cerr << boost::format("Param value - out dim is %1% param row is %2%") % out.dim %
                val.row << endl;
            abort();
        }
        memcpy(out.v, val[featId], val.row * sizeof(dtype));
    }

    void loss(const int& featId, const Tensor1D& loss) {
        assert(loss.dim == val.row);
        for (int idx = 0; idx < val.row; idx++) {
            grad[featId][idx] += loss[idx];
        }
    }
};

template<typename ParamType>
struct ParamArray : public N3LDGSerializable, public TunableCombination<BaseParam>
#if USE_GPU
, public TransferableComponents
#endif
{
    ParamArray(const string &nam) : name(nam) {}

    vector<shared_ptr<ParamType>> params;
    string name;

    vector<ParamType *> ptrs() const {
        vector<ParamType *> results;
        for (auto &p : params) {
            results.push_back(p.get());
        }
        return results;
    }

    int size() const {
        return params.size();
    }

    void init(int layer, function<void(ParamType &, int)> &init_param) {
        for (int i = 0; i < layer; ++i) {
            shared_ptr<ParamType> param(new ParamType(name + std::to_string(i)));
            init_param(*param, i);
            params.push_back(param);
        }
    }

    Json::Value toJson() const override {
        Json::Value json;
        json[name + "-size"] = params.size();
        for (int i = 0; i < params.size(); ++i) {
            json[name + std::to_string(i)] = params.at(i)->toJson();
        }
        return json;
    }

    void fromJson(const Json::Value &json) override {
        int size = json[name + "-size"].asInt();
        for (int i = 0; i < size; ++i) {
            ParamType *param = params.at(i).get();
            param->fromJson(json[name + std::to_string(i)]);
        }
    }

#if USE_GPU
    std::vector<n3ldg_cuda::Transferable *> transferablePtrs() override {
        std::vector<n3ldg_cuda::Transferable *> results;
        for (auto &p : params) {
            results.push_back(p.get());
        }
        return results;
    }
#endif

protected:
    virtual std::vector<Tunable<BaseParam> *> tunableComponents() override {
        std::vector<Tunable<BaseParam> *> results;
        for (auto &p : params) {
            results.push_back(p.get());
        }
        return results;
    }
};

#endif /* PARAM_H_ */
