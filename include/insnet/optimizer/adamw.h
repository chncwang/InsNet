#ifndef INSNET_ADAMW_OPTIMIZER_H
#define INSNET_ADAMW_OPTIMIZER_H

#include "insnet/optimizer/optimizer.h"

namespace insnet {

class AdamWOptimizer : public Optimizer {
public:
    AdamWOptimizer(const std::vector<BaseParam *> &params, dtype learning_rate = 1e-2,
            dtype beta1 = 0.9, dtype beta2 = 0.999, dtype eps = 1e-8, dtype weight_decay = 1e-2) :
        Optimizer(params, learning_rate), beta1_(beta1), beta2_(beta2), eps_(eps),
        weight_decay_(weight_decay) {}

    void optimize() override;

private:
    dtype beta1_, beta2_, eps_, weight_decay_;
};

}

#endif
