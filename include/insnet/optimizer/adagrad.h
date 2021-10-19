#ifndef INSNET_ADAGRAD_OPTIMIZER_H
#define INSNET_ADAGRAD_OPTIMIZER_H

#include "insnet/optimizer/optimizer.h"

namespace insnet {

class AdagradOptimizer : public Optimizer {
public:
    AdagradOptimizer(const std::vector<BaseParam *> &params, dtype learning_rate=1e-2,
            dtype l2_penalty=0, dtype eps=1e-10) : Optimizer(params, learning_rate),
    reg_(l2_penalty), eps_(eps) {}

    void optimize(BaseParam &param) override;

private:
    dtype reg_, eps_;
};

}

#endif
