#include "insnet/optimizer/adamw.h"

namespace insnet {

void AdamWOptimzer::optimize() {
    for (int idx = 0; idx < params_.size(); idx++) {
        params_[idx]->adamW(beta1_, beta2_, lr_, weight_decay_, eps_);
    }
}

}
