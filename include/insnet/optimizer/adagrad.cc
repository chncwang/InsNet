#include "insnet/optimizer/adagrad.h"

namespace insnet {

void AdagradOptimizer::optimize(BaseParam &param) {
    param.adagrad(lr_, reg_, eps_);
}

}
