#include "n3ldg-plus/operator/embedding.h"

using std::vector;
using std::string;

namespace n3ldg_plus {

template <typename ParamType>
class BatchedLookupNode;
template <typename ParamType>
class LookupExecutor;

template <typename ParamType>
class LookupNode : public Node, public Poolable<LookupNode<ParamType>> {
public:
    LookupNode() : Node("lookup") {}

    void setNodeDim(int dim) override {
        setDim(dim);
    }

    void setParam(Embedding<ParamType>& param) {
        param_ = param;
    }

    void setParam(ParamType &table) {
        param_ = &table;
    }

    void connect(Graph &graph, const vector<int> &ids) {
        ids_ = ids;
        setColumn(ids.size());
        graph.addNode(this);
    }

    Executor* generate() override;

    string typeSignature() const override {
        return Node::getNodeType() + "-" + addressToString(param_) +
            (should_backward_ ? "-backward" : "nobackward");
    }

    void compute() override {
        int dim = size() / ids_.size();
        int i = 0;
        for (int id : ids_) {
            Vec(val().v + i++ * dim, dim) = Vec(param_->val()[id], dim);
        }
    }

    void backward() override {
        if (should_backward_) {
            int dim = size() / ids_.size();
            int i = 0;
            for (int id : ids_) {
                Vec(param_->grad()[id], dim) += Vec(grad().v + i++ * dim, dim);
            }
        }
    }

    void setShouldBackward(bool should_backward) {
        should_backward_ = should_backward;
    }

protected:
    int forwardOnlyInputValSize() override {
        return {};
    }

    bool isValForwardOnly() const override {
        return true;
    }

private:
    ParamType* param_ = nullptr;
    vector<int> ids_;
    bool should_backward_ = true;

    friend class BatchedLookupNode<ParamType>;
    friend class LookupExecutor<ParamType>;
};

template <typename ParamType>
Node *embedding(Graph &graph, const vector<int> &ids, ParamType &lookup,
        bool freeze = false) {
    LookupNode<ParamType>* input_lookup =
        LookupNode<ParamType>::newNode(lookup.outDim() * ids.size());
    input_lookup->setShouldBackward(!freeze);
    input_lookup->setParam(lookup);
    input_lookup->connect(graph, ids);
    return input_lookup;
}

template <typename ParamType>
Node *embedding(Graph &graph, int id, ParamType &lookup, bool freeze = false) {
    vector<int> ids = {id};
    return embedding(graph, ids, lookup, freeze);
}

template <typename ParamType>
Node *embedding(Graph &graph, const vector<string> &words, Embedding<ParamType> &lookup, int dim,
        bool freeze = false) {
    using namespace std;
    vector<int> ids;
    ids.reserve(words.size());
    for (const string &word : words) {
        int id;
        if (!lookup.findElemId(word)) {
            if (lookup.nUNKId < 0) {
                cerr << "nUNKId is negative:" << lookup.nUNKId << endl;
                abort();
            }
            id = lookup.nUNKId;
        } else {
            id = lookup.getElemId(word);
        }
        ids.push_back(id);
    }
    LookupNode<ParamType>* input_lookup = LookupNode<ParamType>::newNode(dim * words.size());
    input_lookup->setParam(lookup.E);
    input_lookup->connect(graph, ids);
    input_lookup->setShouldBackward(!freeze);
    return input_lookup;
}

template <typename ParamType>
Node *embedding(Graph &graph, const string &word, Embedding<ParamType> &lookup, int dim,
        bool freeze = false) {
    using namespace std;
    vector<string> words = {word};
    return embedding(graph, words, lookup, dim, freeze);
}

template <typename ParamType>
Node *embedding(Graph &graph, const vector<string> &words, Embedding<ParamType> &lookup,
        bool freeze = false) {
    return embedding(graph, words, lookup, lookup.nDim, freeze);
}

template <typename ParamType>
Node *embedding(Graph &graph, const string &word, Embedding<ParamType> &lookup,
        bool freeze = false) {
    using namespace std;
    vector<string> words = {word};
    return embedding(graph, words, lookup, freeze);
}

template<typename ParamType>
#if USE_GPU
class LookupExecutor :public Executor {
public:
    void  forward() override {
        int count = batch.size();
        vector<int> cols;
        cols.reserve(count);
        for (Node *node : batch) {
            LookupNode<ParamType> &l = dynamic_cast<LookupNode<ParamType> &>(*node);
            int col = l.getColumn();
            cols.push_back(col);
        }
        max_col_ = *max_element(cols.begin(), cols.end());
        vector<int> ids, backward_switches;
        ids.reserve(count * max_col_);
        backward_switches.reserve(count);
        vector<dtype*> vals;
        vals.reserve(count);
        for (Node *node : batch) {
            LookupNode<ParamType> &l = dynamic_cast<LookupNode<ParamType> &>(*node);
            for (int id : l.ids_) {
                ids.push_back(id);
            }
            for (int i = 0; i < max_col_ - l.ids_.size(); ++i) {
                ids.push_back(-1);
            }
            vals.push_back(l.getVal().value);
        }
        id_arr_.init(ids.data(), ids.size());
        col_arr_.init(cols.data(), count);
        int row = getRow();
        cuda::LookupForward(id_arr_.value, param().val().value, count, row, col_arr_.value,
                max_col_, vals);

#if TEST_CUDA
        testForward();
#endif
    }

    void backward() override {
        if (!shouldBackward()) {
            return;
        }

        param().initAndZeroGrad();

        int count = batch.size();
        vector<dtype*> grads;
        grads.reserve(count);
        for (Node *n : batch) {
            grads.push_back(n->grad().value);
        }
        genericBackward(grads);
#if TEST_CUDA
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->backward();
        }

        cuda::Assert(param().grad().verify("lookup backward grad"));
#endif
    }

private:
    void genericBackward(vector<dtype*> &);

    ParamType &param() {
        return *dynamic_cast<LookupNode<ParamType> &>(*batch.front()).param_;
    }

    bool shouldBackward() {
        return dynamic_cast<LookupNode<ParamType> &>(*batch.front()).should_backward_;
    }

    cuda::IntArray id_arr_, backward_switch_arr_, col_arr_;
    int max_col_;
};

template<>
void LookupExecutor<SparseParam>::genericBackward(vector<dtype*> &grads) {
        cuda::LookupBackward(id_arr_.value, grads, batch.size(), getRow(), col_arr_.value,
                max_col_, param().grad().value, param().dIndexers->value);
}

template<>
void LookupExecutor<Param>::genericBackward(vector<dtype*> &grads) {
        cuda::LookupBackward(id_arr_.value, grads, batch.size(), getRow(), col_arr_.value,
                max_col_, param().grad().value, nullptr);
}

#else
class LookupExecutor :public Executor {
public:
    int calculateFLOPs() override {
        return 0;
    }

    int calculateActivations() override {
        return 0;
    }

    void backward() override {
        param().initAndZeroGrad();
        Executor::backward();
    }

    ParamType &param() {
        return *dynamic_cast<LookupNode<ParamType> &>(*batch.front()).param_;
    }
};
#endif

template<typename ParamType>
Executor* LookupNode<ParamType>::generate() {
    return new LookupExecutor<ParamType>();
}

Node *embedding(Graph &graph, const vector<int> &ids, Param &lookup,
        bool freeze) {
    return embedding<Param>(graph, ids, lookup, freeze);
}

Node *embedding(Graph &graph, int id, Param &lookup, bool freeze) {
    return embedding<Param>(graph, id, lookup, freeze);
}

Node *embedding(Graph &graph, const vector<string> &words, Embedding<Param> &lookup,
        bool freeze) {
    return embedding<Param>(graph, words, lookup, freeze);
}

Node *embedding(Graph &graph, const vector<string> &words, Embedding<SparseParam> &lookup,
        bool freeze) {
    return embedding<SparseParam>(graph, words, lookup, freeze);
}

Node *embedding(Graph &graph, const string &word, Embedding<Param> &lookup, bool freeze) {
    return embedding<Param>(graph, word, lookup, freeze);
}

Node *embedding(Graph &graph, const string &word, Embedding<SparseParam> &lookup,
        bool freeze) {
    return embedding<SparseParam>(graph, word, lookup, freeze);
}

}
