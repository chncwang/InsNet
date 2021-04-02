#ifndef SPARSEPARAM_H_
#define SPARSEPARAM_H_

#include "n3ldg-plus/param/base-param.h"
#include "n3ldg-plus/base/tensor.h"
#include "n3ldg-plus/util/nrmat.h"
#include "n3ldg-plus/computation-graph/node.h"

namespace n3ldg_plus {

class SparseParam : public BaseParam {
public:
    Tensor2D aux_square;
    Tensor2D aux_mean;
    nr::NRVec<bool> indexers;
    nr::NRVec<int> last_update;

    SparseParam(const std::string &name = "sparse") : BaseParam(name) {}

#if USE_GPU
    n3ldg_cuda::BoolArray dIndexers;
    n3ldg_cuda::IntArray dIters;

    void copyFromHostToDevice() override {
        BaseParam::copyFromHostToDevice();
        n3ldg_cuda::MyCudaMemcpy(dIters.value, last_update.c_buf(), sizeof(int) * dIters.len,
                cudaMemcpyHostToDevice);
        aux_square.copyFromHostToDevice();
        aux_mean.copyFromHostToDevice();
    }

    void copyFromDeviceToHost() override {
        BaseParam::copyFromDeviceToHost();
        n3ldg_cuda::MyCudaMemcpy(last_update.c_buf(), dIters.value, sizeof(int) * dIters.len,
                cudaMemcpyDeviceToHost);
        aux_square.copyFromDeviceToHost();
        aux_mean.copyFromDeviceToHost();
    }

    virtual std::string name() const {
        return "SparseParam";
    }
#endif

    // allow sparse and dense parameters have different parameter initization methods
    void init(int outDim, int inDim) override {
        //not in the aligned memory pool
#if USE_GPU
        val.initOnMemoryAndDevice(outDim, inDim);
        aux_square.initOnMemoryAndDevice(outDim, inDim);
        aux_mean.initOnMemoryAndDevice(outDim, inDim);
#else
        val_.init(outDim, inDim);
        aux_square.init(outDim, inDim);
        aux_mean.init(outDim, inDim);
#endif
        dtype bound = std::sqrt(6.0 / (outDim + inDim));
        val_.random(bound);
        grad_.init(outDim, inDim);
        indexers.resize(inDim);
        indexers = false;
        last_update.resize(inDim);
        last_update = 0;
#if USE_GPU
        dIndexers.init(indexers.c_buf(), indexers.size());
        dIters.init(last_update.c_buf(), last_update.size());
        n3ldg_cuda::Memset(grad.value, grad.size, 0.0f);
        n3ldg_cuda::Memset(aux_square.value, inDim * outDim, 0.0f);
        n3ldg_cuda::Memset(aux_mean.value, inDim * outDim, 0.0f);
#endif
    }

    void clearGrad() override {
#if USE_GPU
        n3ldg_cuda::Memset(grad.value, grad.size, 0.0f);
        n3ldg_cuda::Memset(dIndexers.value, grad.col, false);
#if TEST_CUDA
        int inDim = indexers.size();
        for (int index = 0; index < inDim; index++) {
            for (int idx = 0; idx < grad.row; idx++) {
                grad[index][idx] = 0;
            }
        }
        indexers = false;
        n3ldg_cuda::Assert(grad.verify("SparseParam clearGrad"));
        n3ldg_cuda::Assert(n3ldg_cuda::Verify(indexers.c_buf(),
                    dIndexers.value, grad.col, "SparseParam indexers"));
#endif
#else
        int inDim = indexers.size();
        for (int index = 0; index < inDim; index++) {
            for (int idx = 0; idx < grad_.row; idx++) {
                grad_[index][idx] = 0;
            }
        }
        indexers = false;
#endif
    }

    int outDim() override {
        return val_.row;
    }

    int inDim() override {
        return val_.col;
    }

    void adagrad(dtype alpha, dtype reg, dtype eps) override {
#if USE_GPU
        n3ldg_cuda::UpdateAdagrad(val.value, grad.value, indexers.size(),
                grad.col, aux_square.value, dIndexers.value, alpha, reg, eps);
#if TEST_CUDA
        int inDim = indexers.size();
        for (int index = 0; index < inDim; index++) {
            if (!indexers[index]) continue;
            for (int idx = 0; idx < grad.row; idx++) {
                grad[index][idx] = grad[index][idx] + val[index][idx] * reg;
                aux_square[index][idx] = aux_square[index][idx] + grad[index][idx] * grad[index][idx];
                val[index][idx] = val[index][idx] - grad[index][idx] * alpha / sqrt(aux_square[index][idx] + eps);
            }
        }

        n3ldg_cuda::Assert(val.verify("SparseParam updateAdagrad"));
#endif
#else
        int inDim = indexers.size();
        for (int index = 0; index < inDim; index++) {
            if (!indexers[index]) continue;
            for (int idx = 0; idx < grad_.row; idx++) {
                grad_[index][idx] = grad_[index][idx] + val_[index][idx] * reg;
                aux_square[index][idx] = aux_square[index][idx] + grad_[index][idx] *
                    grad_[index][idx];
                val_[index][idx] = val_[index][idx] - grad_[index][idx] * alpha /
                    std::sqrt(aux_square[index][idx] + eps);
            }
        }
#endif
    }

    void adam(dtype belta1, dtype belta2, dtype alpha, dtype reg, dtype eps) override {
#if USE_GPU
        n3ldg_cuda::UpdateAdam(val.value, grad.value, grad.row, indexers.size(), aux_mean.value,
                aux_square.value, dIndexers.value, dIters.value, belta1, belta2, alpha, reg, eps);
#if TEST_CUDA
        dtype lr_t;
        int inDim = indexers.size();
        for (int index = 0; index < inDim; index++) {
            if (!indexers[index]) continue;
            for (int idx = 0; idx < grad.row; idx++) {
                grad[index][idx] = grad[index][idx] + val[index][idx] * reg;
                aux_mean[index][idx] = belta1 * aux_mean[index][idx] + (1 - belta1) * grad[index][idx];
                aux_square[index][idx] = belta2 * aux_square[index][idx] + (1 - belta2) * grad[index][idx] * grad[index][idx];
                lr_t = alpha * sqrt(1 - pow(belta2, last_update[index] + 1)) / (1 - pow(belta1, last_update[index] + 1));
                val[index][idx] = val[index][idx] - aux_mean[index][idx] * lr_t / sqrt(aux_square[index][idx] + eps);
            }
            last_update[index]++;
        }

        n3ldg_cuda::Assert(val.verify("SparseParam updateAdam"));
#endif
#else
        dtype lr_t;
        int inDim = indexers.size();
        for (int index = 0; index < inDim; index++) {
            if (!indexers[index]) continue;
            for (int idx = 0; idx < grad_.row; idx++) {
                grad_[index][idx] = grad_[index][idx] + val_[index][idx] * reg;
                aux_mean[index][idx] = belta1 * aux_mean[index][idx] + (1 - belta1) * grad_[index][idx];
                aux_square[index][idx] = belta2 * aux_square[index][idx] + (1 - belta2) * grad_[index][idx] * grad_[index][idx];
                lr_t = alpha * std::sqrt(1 - std::pow(belta2, last_update[index] + 1)) / (1 - std::pow(belta1, last_update[index] + 1));
                val_[index][idx] = val_[index][idx] - aux_mean[index][idx] * lr_t / std::sqrt(aux_square[index][idx] + eps);
            }
            last_update[index]++;
        }
#endif
    }

    void adamW(dtype belta1, dtype belta2, dtype alpha, dtype reg, dtype eps) override {
        adam(belta1, belta2, alpha, reg, eps); // TODO
    }

    void randpoint(int& idx, int &idy) override {
        std::vector<int> idRows, idCols;
        int inDim = indexers.size();
        for (int index = 0; index < inDim; index++) {
            if (!indexers[index]) continue;
            idCols.push_back(index);
        }

        for (int i = 0; i < val_.row; i++) {
            idRows.push_back(i);
        }

        random_shuffle(idRows.begin(), idRows.end());
        random_shuffle(idCols.begin(), idCols.end());

        idx = idCols.at(0);
        idy = idRows.at(0);
    }

    dtype gradSquareSum() override {
#if USE_GPU && !TEST_CUDA
        dtype result = n3ldg_cuda::SquareSum(grad.value, dIndexers.value,
                indexers.size(), val.row);
        return result;
#elif USE_GPU && TEST_CUDA
        grad.copyFromDeviceToHost();
        dtype sumNorm = 0.0;
        int inDim = indexers.size();
        for (int index = 0; index < inDim; index++) {
            if (!indexers[index]) continue;
            for (int idx = 0; idx < val.row; idx++) {
                sumNorm += grad[index][idx] * grad[index][idx];
            }
        }


        n3ldg_cuda::Assert(n3ldg_cuda::Verify(indexers.c_buf(),
                    dIndexers.value,
                    indexers.size(),
                    "sparse squareGradNorm"));
        n3ldg_cuda::Assert(grad.verify("squareGradNorm grad"));
        dtype cuda = n3ldg_cuda::SquareSum(grad.value, dIndexers.value, inDim,
                val.row);
        n3ldg_cuda::Assert(isEqual(cuda, sumNorm));

        return sumNorm;
#else
        dtype sumNorm = 0.0;
        int inDim = indexers.size();
        for (int index = 0; index < inDim; index++) {
            if (!indexers[index]) continue;
            for (int idx = 0; idx < val_.row; idx++) {
                sumNorm += grad_[index][idx] * grad_[index][idx];
            }
        }

        return sumNorm;
#endif
    }

    void rescaleGrad(dtype scale) override {
#if USE_GPU
        n3ldg_cuda::Rescale(grad_.value, grad_.size, scale);
#if TEST_CUDA
        int inDim = indexers.size();
        for (int index = 0; index < inDim; index++) {
            for (int idx = 0; idx < val.row; idx++) {
                grad_[index][idx] = grad_[index][idx] * scale;
            }
        }
        n3ldg_cuda::Assert(grad_.verify("SparseParam rescaleGrad"));
#endif
#else
        int inDim = indexers.size();
        for (int index = 0; index < inDim; index++) {
            if (!indexers[index]) continue;
            for (int idx = 0; idx < val_.row; idx++) {
                grad_[index][idx] = grad_[index][idx] * scale;
            }
        }
#endif
    }

    void val_ue(const int& featId, Tensor1D& out) {
        assert(out.dim == val_.row);
        memcpy(out.v, val_[featId], val_.row * sizeof(dtype));
    }

    void val_ue(const std::vector<int>& featIds, Tensor1D& out) {
        assert(out.dim == val_.row);
        int featNum = featIds.size();
        int featId;
        for (int i = 0; i < featNum; i++) {
            featId = featIds[i];
            for (int idx = 0; idx < val_.row; idx++) {
                out[idx] += val_[featId][idx];
            }
        }
    }

    void loss(const int& featId, const Tensor1D& loss) {
        assert(loss.dim == val_.row);
        indexers[featId] = true;
        for (int idx = 0; idx < val_.row; idx++) {
            grad_[featId][idx] += loss[idx];
        }
    }

    void loss(const std::vector<int>& featIds, const Tensor1D& loss) {
        assert(loss.dim == val_.row);
        int featNum = featIds.size();
        int featId;
        for (int i = 0; i < featNum; i++) {
            featId = featIds[i];
            indexers[featId] = true;
            for (int idx = 0; idx < val_.row; idx++) {
                grad_[featId][idx] += loss[idx];
            }
        }
    }

    template<typename Archive>
    void serialize(Archive &ar) {
        ar(val_, aux_square, aux_mean);
    }
};

}

#endif /* SPARSEPARAM_H_ */
