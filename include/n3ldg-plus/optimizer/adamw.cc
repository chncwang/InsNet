#include "n3ldg-plus/optimizer/adamw.h"

namespace n3ldg_plus {

void AdamWOptimzer::step() {
    for (int idx = 0; idx < params_.size(); idx++) {
        params_[idx]->updateAdamW(beta1_, beta2_, lr_, weight_decay_, eps_);
        params_[idx]->clearGrad();
    }
}

}
