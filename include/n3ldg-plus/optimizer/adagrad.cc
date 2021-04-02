#include "n3ldg-plus/optimizer/adagrad.h"

namespace n3ldg_plus {

void AdagradOptimizer::step() {
    for (int idx = 0; idx < params_.size(); idx++) {
        params_[idx]->adagrad(lr_, reg_, eps_);
        params_[idx]->clearGrad();
    }
}

}
