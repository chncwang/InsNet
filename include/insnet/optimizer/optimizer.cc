#include "insnet/optimizer/optimizer.h"

namespace insnet {

void Optimizer::step() {
    for (BaseParam *param : params_) {
        if (param->isGradInitialized()) {
            optimize(*param);
        }
    }

    for (BaseParam *p : params_) {
        p->releaseGrad();
    }
}

void Optimizer::clipGrad(dtype clip_value) {
    dtype sum = 0;
    for (int idx = 0; idx < params_.size(); idx++) {
        sum += params_.at(idx)->gradSquareSum();
    }
    dtype sqrt_sum = sqrt(sum);
    if (sqrt_sum > clip_value) {
        dtype scale = clip_value / sqrt_sum;
        for (int idx = 0; idx < params_.size(); idx++) {
            params_[idx]->rescaleGrad(scale);
        }
    }
}

}
