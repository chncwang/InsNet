#include "n3ldg-plus/optimizer/adam.h"

namespace n3ldg_plus {

void AdamOptimzer::step() {
    for (int idx = 0; idx < params_.size(); idx++) {
        params_[idx]->adam(beta1_, beta2_, lr_, l2_penalty_, eps_);
        params_[idx]->clearGrad();
    }
}

}
