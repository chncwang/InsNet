#ifndef N3LDG_PLUS_ADAGRAD_OPTIMIZER_H
#define N3LDG_PLUS_ADAGRAD_OPTIMIZER_H

#include "n3ldg-plus/optimizer/optimizer.h"

namespace n3ldg_plus {

class AdagradOptimizer : public Optimizer {
public:
    AdagradOptimizer(std::vector<BaseParam *> &params, dtype learning_rate=1e-2,
            dtype l2_penalty=0, dtype eps=1e-10) : Optimizer(params, learning_rate),
    reg_(l2_penalty), eps_(eps) {}

    void step() override;

private:
    dtype reg_, eps_;
};

}

#endif
