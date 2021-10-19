#include "insnet/optimizer/adam.h"

namespace insnet {

void AdamOptimizer::optimize(BaseParam &param) {
    param.adam(beta1_, beta2_, lr_, l2_penalty_, eps_);
}

}
