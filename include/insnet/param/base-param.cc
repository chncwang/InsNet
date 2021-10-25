#include "base-param.h"

#if USE_GPU
#include "insnet/cuda/insnet_cuda.h"
#endif

using std::make_unique;

namespace insnet {

bool BaseParam::initAndZeroGrad() {
    if (grad_ != nullptr) {
        return false;
    }
    grad_ = make_unique<Tensor2D>();
    grad_->init(val_.row, val_.col);
# if USE_GPU
    int size = val_.row * val_.col;
    cuda::Memset(grad_->value, size, 0.0f);
#if TEST_CUDA
    for (int index = 0; index < size; index++) {
        grad_->v[index] = 0;
    }
    cuda::Assert(grad_->verify("BaseParam clearGrad"));
#endif
#endif
    return true;
}

}
