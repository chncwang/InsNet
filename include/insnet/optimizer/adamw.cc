#include "insnet/optimizer/adamw.h"

namespace insnet {

void AdamWOptimizer::optimize(BaseParam &param) {
    param.adamW(beta1_, beta2_, lr_, weight_decay_, eps_);
}

}
