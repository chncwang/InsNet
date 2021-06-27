#include "insnet/optimizer/adagrad.h"

namespace insnet {

void AdagradOptimizer::optimize() {
    for (int idx = 0; idx < params_.size(); idx++) {
        params_[idx]->adagrad(lr_, reg_, eps_);
    }
}

}
