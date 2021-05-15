#include "base-param.h"

#if USE_GPU
#include "n3ldg-plus/cuda/n3ldg_plus_cuda.h"
#endif

using std::make_unique;

namespace n3ldg_plus {

void BaseParam::initAndZeroGrad() {
    if (grad_ != nullptr) {
        return;
    }
    grad_ = make_unique<Tensor2D>();
    grad_->init(val_.row, val_.col);
# if USE_GPU
    int size = val_.row * val_.col;
    cuda::Memset(grad_->value, size, 0.0f);
#endif
}

}
