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
#if USE_GPU
#include "N3LDG_cuda.h"
#endif

// Notice: aux is an auxiliary variable to help parameter updating
class Param : public BaseParam {
  public:
    Tensor2D aux_square;
    Tensor2D aux_mean;
    int iter;

    // allow sparse and dense parameters have different parameter initialization methods
    void initial(int outDim, int inDim) {
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
        dtype bound = sqrt(6.0 / (outDim + inDim + 1));
        val.random(bound);
        iter = 0;
#if USE_GPU
        n3ldg_cuda::Memset(grad.value, outDim * inDim, 0.0f);
        n3ldg_cuda::Memset(aux_square.value, outDim * inDim, 0.0f);
        n3ldg_cuda::Memset(aux_mean.value, outDim * inDim, 0.0f);
#endif
    }

#if USE_GPU
    std::vector<n3ldg_cuda::Transferable *> transferablePtrs() {
        auto v = BaseParam::transferablePtrs();
        v.push_back(&aux_square);
        v.push_back(&aux_mean);
        return v;
    }

    virtual std::string name() const {
        return "Param";
    }
#endif

    int outDim() {
        return val.row;
    }

    int inDim() {
        return val.col;
    }

    void clearGrad() {
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

    void updateAdagrad(dtype alpha, dtype reg, dtype eps) {
#if USE_GPU
        n3ldg_cuda::UpdateAdagrad(val.value, grad.value, val.row, val.col,
                aux_square.value, alpha, reg, eps);
#if TEST_CUDA
        if (val.col > 1 && val.row > 1)grad.vec() = grad.vec() + val.vec() * reg;
        aux_square.vec() = aux_square.vec() + grad.vec().square();
        val.vec() = val.vec() - grad.vec() * alpha / (aux_square.vec() + eps).sqrt();
        n3ldg_cuda::Assert(val.verify("Param adagrad"));
#endif
#else
        if (val.col > 1 && val.row > 1)grad.vec() = grad.vec() + val.vec() * reg;
        aux_square.vec() = aux_square.vec() + grad.vec().square();
        val.vec() = val.vec() - grad.vec() * alpha / (aux_square.vec() + eps).sqrt();
#endif
    }

    void updateAdam(dtype belta1, dtype belta2, dtype alpha, dtype reg, dtype eps) {
#if USE_GPU
#if TEST_CUDA
        n3ldg_cuda::Assert(val.verify("Param adam begin val"));
        n3ldg_cuda::Assert(grad.verify("Param adam begin grad"));
        n3ldg_cuda::Assert(aux_mean.verify("Param adam begin aux_mean"));
        n3ldg_cuda::Assert(aux_square.verify("Param adam begin aux_square"));
#endif
        n3ldg_cuda::UpdateAdam(val.value, grad.value, val.row, val.col,
                aux_mean.value,
                aux_square.value,
                iter,
                belta1,
                belta2,
                alpha,
                reg,
                eps);
#if TEST_CUDA
        if (val.col > 1 && val.row > 1)grad.vec() = grad.vec() + val.vec() * reg;
        aux_mean.vec() = belta1 * aux_mean.vec() + (1 - belta1) * grad.vec();
        aux_square.vec() = belta2 * aux_square.vec() + (1 - belta2) * grad.vec().square();
        dtype lr_t = alpha * sqrt(1 - pow(belta2, iter + 1)) / (1 - pow(belta1, iter + 1));
        val.vec() = val.vec() - aux_mean.vec() * lr_t / (aux_square.vec() + eps).sqrt();
        n3ldg_cuda::Assert(val.verify("Param adam"));
#endif
#else
        if (val.col > 1 && val.row > 1)grad.vec() = grad.vec() + val.vec() * reg;
        aux_mean.vec() = belta1 * aux_mean.vec() + (1 - belta1) * grad.vec();
        aux_square.vec() = belta2 * aux_square.vec() + (1 - belta2) * grad.vec().square();
        dtype lr_t = alpha * sqrt(1 - pow(belta2, iter + 1)) / (1 - pow(belta1, iter + 1));
        val.vec() = val.vec() - aux_mean.vec() * lr_t / (aux_square.vec() + eps).sqrt();
#endif
        iter++;
    }

    void randpoint(int& idx, int &idy) {
        //select indexes randomly
        std::vector<int> idRows, idCols;
        idRows.clear();
        idCols.clear();
        for (int i = 0; i < val.row; i++)
            idRows.push_back(i);
        for (int i = 0; i < val.col; i++)
            idCols.push_back(i);

        random_shuffle(idRows.begin(), idRows.end());
        random_shuffle(idCols.begin(), idCols.end());

        idy = idRows[0];
        idx = idCols[0];
    }

    dtype squareGradNorm() {
#if USE_GPU && !TEST_CUDA
        return n3ldg_cuda::SquareSum(grad.value, grad.size);
#elif USE_GPU && TEST_CUDA
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

    void rescaleGrad(dtype scale) {
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

    void save(std::ostream &os)const {
        val.save(os);
        aux_square.save(os);
        aux_mean.save(os);
        os << iter << endl;
    }

    void load(std::istream &is) {
        val.load(is);
        aux_square.load(is);
        aux_mean.load(is);
        is >> iter;
    }
};

#endif /* PARAM_H_ */
