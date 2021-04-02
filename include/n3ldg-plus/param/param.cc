#include "n3ldg-plus/param/param.h"

using std::function;
using std::vector;
using std::cerr;

namespace n3ldg_plus {

void Param::init(int outDim, int inDim, const function<dtype(int, int)> *cal_bound,
        InitDistribution dist) {
#if USE_GPU
    val.initOnMemoryAndDevice(outDim, inDim);
    aux_square_.initOnMemoryAndDevice(outDim, inDim);
    aux_mean_.initOnMemoryAndDevice(outDim, inDim);
#else
    val_.init(outDim, inDim);
    aux_square_.init(outDim, inDim);
    aux_mean_.init(outDim, inDim);
#endif
    grad_.init(outDim, inDim);
    if (isBias()) {
        val_.assignAll(0.0f);
    } else {
        dtype bound = cal_bound == nullptr ? sqrt(6.0 / (outDim + inDim)) :
            (*cal_bound)(outDim, inDim);
        if (dist == InitDistribution::UNI) {
            val_.random(bound);
        } else {
            val_.randomNorm(bound);
        }
    }

#if USE_GPU
    n3ldg_cuda::Memset(grad.value, outDim * inDim, 0.0f);
    n3ldg_cuda::Memset(aux_square.value, outDim * inDim, 0.0f);
    n3ldg_cuda::Memset(aux_mean.value, outDim * inDim, 0.0f);
#endif
}

void Param::adagrad(dtype alpha, dtype reg, dtype eps) {
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
    if (!isBias()) grad_.vec() = grad_.vec() + val_.vec() * reg;
    aux_square_.vec() = aux_square_.vec() + grad_.vec().square();
    val_.vec() = val_.vec() - grad_.vec() * alpha / (aux_square_.vec() + eps).sqrt();
#endif
}

void Param::adam(dtype belta1, dtype belta2, dtype alpha, dtype reg, dtype eps) {
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
    if (!isBias()) grad_.vec() = grad_.vec() + val_.vec() * reg;
    aux_mean_.vec() = belta1 * aux_mean_.vec() + (1 - belta1) * grad_.vec();
    aux_square_.vec() = belta2 * aux_square_.vec() + (1 - belta2) * grad_.vec().square();
    dtype lr_t = alpha * sqrt(1 - pow(belta2, iter_ + 1)) / (1 - pow(belta1, iter_ + 1));
    val_.vec() = val_.vec() - aux_mean_.vec() * lr_t / (aux_square_.vec() + eps).sqrt();
#endif
    iter_++;
}

void Param::adamW(dtype belta1, dtype belta2, dtype alpha, dtype reg, dtype eps) {
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
    aux_mean_.vec() = belta1 * aux_mean_.vec() + (1 - belta1) * grad_.vec();
    aux_square_.vec() = belta2 * aux_square_.vec() + (1 - belta2) * grad_.vec().square();
    dtype lr_t = alpha * sqrt(1 - pow(belta2, iter_ + 1)) / (1 - pow(belta1, iter_ + 1));
    val_.vec() = (1 - (isBias() ? 0.0f : reg)) * val_.vec() -
        aux_mean_.vec() * lr_t / (aux_square_.vec() + eps).sqrt();
#endif
    iter_++;
}

void Param::randpoint(int& idx, int &idy) {
    vector<int> idRows, idCols;
    for (int i = 0; i < val_.row; i++)
        idRows.push_back(i);
    for (int i = 0; i < val_.col; i++)
        idCols.push_back(i);

    random_shuffle(idRows.begin(), idRows.end());
    random_shuffle(idCols.begin(), idCols.end());

    idy = idRows.at(0);
    idx = idCols.at(0);
}

dtype Param::gradSquareSum() {
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
        cout << "cpu:" << sumNorm << " cuda:" << cuda << endl;
    }
    return sumNorm;
#else
    dtype sumNorm = 0.0;
    for (int i = 0; i < grad_.size; i++) {
        sumNorm += grad_.v[i] * grad_.v[i];
    }
    return sumNorm;
#endif
}

}
