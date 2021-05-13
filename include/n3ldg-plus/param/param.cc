#include "n3ldg-plus/param/param.h"
#include "n3ldg-plus/cuda/n3ldg_plus_cuda.h"
#include "n3ldg-plus/util/util.h"

using std::function;
using std::vector;
using std::cerr;
using std::cout;
using std::endl;

namespace n3ldg_plus {

void Param::init(int outDim, int inDim, const function<dtype(int, int)> *cal_bound,
        InitDistribution dist) {
#if USE_GPU
    val_.initOnMemoryAndDevice(outDim, inDim);
    aux_square_.initOnMemoryAndDevice(outDim, inDim);
    aux_mean_.initOnMemoryAndDevice(outDim, inDim);
#else
    val_.init(outDim, inDim);
    aux_square_.init(outDim, inDim);
    aux_mean_.init(outDim, inDim);
#endif
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
    cuda::Memset(aux_square_.value, outDim * inDim, 0.0f);
    cuda::Memset(aux_mean_.value, outDim * inDim, 0.0f);
#endif
}

void Param::rescaleGrad(dtype scale) {
#if USE_GPU
        cuda::Rescale(grad_->value, grad_->size, scale);
#if TEST_CUDA
        grad_->vec() = grad_->vec() * scale;
        cuda::Assert(grad_->verify("Param rescaleGrad"));
#endif
#else
        grad_->vec() = grad_->vec() * scale;
#endif
}

void Param::adagrad(dtype alpha, dtype reg, dtype eps) {
#if USE_GPU
    cuda::UpdateAdagrad(val_.value, grad_->value, val_.row, val_.col,
            aux_square_.value, alpha, reg, eps);
#if TEST_CUDA
    if (!isBias()) grad_->vec() = grad_->vec() + val_.vec() * reg;
    aux_square_.vec() = aux_square_.vec() + grad_->vec().square();
    val_.vec() = val_.vec() - grad_->vec() * alpha / (aux_square_.vec() + eps).sqrt();
    cuda::Assert(val_.verify("Param adagrad"));
#endif
#else
    if (!isBias()) grad_->vec() = grad_->vec() + val_.vec() * reg;
    aux_square_.vec() = aux_square_.vec() + grad_->vec().square();
    val_.vec() = val_.vec() - grad_->vec() * alpha / (aux_square_.vec() + eps).sqrt();
#endif
}

void Param::adam(dtype belta1, dtype belta2, dtype alpha, dtype reg, dtype eps) {
#if USE_GPU
#if TEST_CUDA
    cuda::Assert(val_.verify("Param adam begin val"));
    cuda::Assert(grad_->verify("Param adam begin grad"));
    cuda::Assert(aux_mean_.verify("Param adam begin aux_mean"));
    cuda::Assert(aux_square_.verify("Param adam begin aux_square"));
#endif
    cuda::UpdateAdam(val_.value, grad_->value, val_.row, val_.col, isBias(),
            aux_mean_.value,
            aux_square_.value,
            iter_,
            belta1,
            belta2,
            alpha,
            reg,
            eps);
#if TEST_CUDA
    if (!isBias()) grad_->vec() = grad_->vec() + val_.vec() * reg;
    aux_mean_.vec() = belta1 * aux_mean_.vec() + (1 - belta1) * grad_->vec();
    aux_square_.vec() = belta2 * aux_square_.vec() + (1 - belta2) * grad_->vec().square();
    dtype lr_t = alpha * sqrt(1 - pow(belta2, iter_ + 1)) / (1 - pow(belta1, iter_ + 1));
    val_.vec() = val_.vec() - aux_mean_.vec() * lr_t / (aux_square_.vec() + eps).sqrt();
    cuda::Assert(val_.verify("Param adam"));
#endif
#else
    if (!isBias()) grad_->vec() = grad_->vec() + val_.vec() * reg;
    aux_mean_.vec() = belta1 * aux_mean_.vec() + (1 - belta1) * grad_->vec();
    aux_square_.vec() = belta2 * aux_square_.vec() + (1 - belta2) * grad_->vec().square();
    dtype lr_t = alpha * sqrt(1 - pow(belta2, iter_ + 1)) / (1 - pow(belta1, iter_ + 1));
    val_.vec() = val_.vec() - aux_mean_.vec() * lr_t / (aux_square_.vec() + eps).sqrt();
#endif
    iter_++;
}

void Param::adamW(dtype belta1, dtype belta2, dtype alpha, dtype reg, dtype eps) {
#if USE_GPU
#if TEST_CUDA
    cuda::Assert(val_.verify("Param adam begin val"));
    cuda::Assert(grad_->verify("Param adam begin grad"));
    cuda::Assert(aux_mean_.verify("Param adam begin aux_mean"));
    cuda::Assert(aux_square_.verify("Param adam begin aux_square"));
#endif
    cuda::UpdateAdamW(val_.value, grad_->value, val_.row, val_.col, isBias(), aux_mean_.value,
            aux_square_.value, iter_, belta1, belta2, alpha, reg, eps);
#if TEST_CUDA
    aux_mean_.vec() = belta1 * aux_mean_.vec() + (1 - belta1) * grad_->vec();
    aux_square_.vec() = belta2 * aux_square_.vec() + (1 - belta2) * grad_->vec().square();
    dtype lr_t = alpha * sqrt(1 - pow(belta2, iter_ + 1)) / (1 - pow(belta1, iter_ + 1));
    val_.vec() = (1 - (isBias() ? 0.0f : reg)) * val_.vec() -
        aux_mean_.vec() * lr_t / (aux_square_.vec() + eps).sqrt();
    cuda::Assert(val_.verify("Param adam"));
#endif
#else
    aux_mean_.vec() = belta1 * aux_mean_.vec() + (1 - belta1) * grad_->vec();
    aux_square_.vec() = belta2 * aux_square_.vec() + (1 - belta2) * grad_->vec().square();
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
    return cuda::SquareSum(grad_->value, grad_->size);
#elif USE_GPU && TEST_CUDA
    cout << "squareGradNorm - param name:" << this->getParamName() << endl;
    cuda::Assert(grad_->verify("squareGradNorm grad"));
    dtype cuda = cuda::SquareSum(grad_->value, grad_->size);
    dtype sumNorm = 0.0;
    for (int i = 0; i < grad_->size; i++) {
        sumNorm += grad_->v[i] * grad_->v[i];
    }
    if (!isEqual(sumNorm, cuda)) {
        cout << "cpu:" << sumNorm << " cuda:" << cuda << endl;
    }
    return sumNorm;
#else
    dtype sumNorm = 0.0;
    for (int i = 0; i < grad_->size; i++) {
        sumNorm += grad_->v[i] * grad_->v[i];
    }
    return sumNorm;
#endif
}

}
