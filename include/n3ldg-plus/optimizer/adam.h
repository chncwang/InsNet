#ifndef N3LDG_PLUS_ADAM_OPTIMIZER_H
#define N3LDG_PLUS_ADAM_OPTIMIZER_H

#include "n3ldg-plus/optimizer/optimizer.h"

namespace n3ldg_plus {

class AdamOptimzer : public Optimizer {
public:
    AdamOptimzer(const std::vector<BaseParam *> &params, dtype learning_rate = 1e-3,
            dtype beta1 = 0.9, dtype beta2 = 0.999, dtype eps = 1e-8, dtype l2_penalty = 0) :
        Optimizer(params, learning_rate), beta1_(beta1), beta2_(beta2), eps_(eps),
        l2_penalty_(l2_penalty) {}

    void optimize() override;

private:
    dtype beta1_, beta2_, eps_, l2_penalty_;
};

}

#endif
