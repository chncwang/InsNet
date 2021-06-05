#include "insnet/optimizer/adam.h"

namespace insnet {

void AdamOptimzer::optimize() {
    for (int idx = 0; idx < params_.size(); idx++) {
        params_[idx]->adam(beta1_, beta2_, lr_, l2_penalty_, eps_);
    }
}

}
