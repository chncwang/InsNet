#ifndef N3LDG_PLUS_GRU_H
#define N3LDG_PLUS_GRU_H

#include "n3ldg-plus/operator/linear.h"

namespace n3ldg_plus {

struct GRUParam : public TunableCombination<BaseParam>
#if USE_GPU
, public cuda::TransferableComponents
#endif
{
    LinearParam update_input;
    LinearParam update_hidden;
    LinearParam reset_input;
    LinearParam reset_hidden;
    LinearParam candidate_input;
    LinearParam candidate_hidden;

    GRUParam(const std::string &name);

    template<typename Archive>
    void serialize(Archive &ar) {
        ar(update_input, update_hidden, reset_input, reset_hidden, candidate_input,
                candidate_hidden);
    }

    void init(int out_size, int in_size);

    int inDim() {
        return update_input.W().outDim();
    }

    int outDim() {
        return update_input.W().inDim();
    }

#if USE_GPU
    std::vector<Transferable *> transferablePtrs() override;
#endif

protected:
    std::vector<Tunable<BaseParam> *> tunableComponents() override;
};

class GRUBuilder {
public:
    int size() const {
        return hiddens_.size();
    }

    const std::vector<Node *> &hiddens() {
        return hiddens_;
    }

    void step(GRUParam &gru_params, Node &input, Node &h0, dtype dropout, bool is_training);

private:
    std::vector<Node*> hiddens_;
};

}

#endif
