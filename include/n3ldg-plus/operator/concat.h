#ifndef CONCAT
#define CONCAT

#include "n3ldg-plus/util/util.h"
#include "n3ldg-plus/computation-graph/graph.h"
#if USE_GPU
#include "N3LDG_cuda.h"
#endif
#include "n3ldg-plus/util/profiler.h"

namespace n3ldg_plus {

class ConcatNode : public Node, public Poolable<ConcatNode> {
public:
    ConcatNode() : Node("concat") {}

    void initNode(int dim) override {
        init(dim);
    }

    void setNodeDim(int dim) override {
        setDim(dim);
    }

    void clear() override {
        in_rows_.clear();
        ins_.clear();
        Node::clear();
    }

    void setInputs(const std::vector<Node *> &ins) override {
        int in_size = ins.size();
        int cur_dim = 0;
        in_rows_.reserve(in_size);
        for (int i = 0; i < in_size; ++i) {
            in_rows_.push_back(ins.at(i)->getDim() / getColumn());
            cur_dim += in_rows_.at(i);
        }
        if (cur_dim * getColumn() != getDim()) {
            std::cerr << "input dim size not match" << cur_dim << "\t" << getDim() << std::endl;
            abort();
        }
        ins_ = ins;
    }

    void connect(Graph &cg, const std::vector<Node *> &x) {
        if (x.empty()) {
            std::cerr << "empty inputs for concat" << std::endl;
            abort();
        }

        setInputs(x);
        afterForward(cg, x);
    }

    PExecutor generate() override;

    std::string typeSignature() const override {
        std::string hash_code = Node::getNodeType() + "-" + std::to_string(in_rows_.size());
        for (int dim : in_rows_) {
            hash_code += "-" + std::to_string(dim);
        }
        return hash_code;
    }

    void compute() override {
        int in_size = ins_.size();
        int row = getDim() / getColumn();
        for (int i = 0; i < getColumn(); ++i) {
            int offset = 0;
            for (int j = 0; j < in_size; ++j) {
                Vec(val().v + i * row + offset, in_rows_.at(j)) =
                    Vec(ins_.at(j)->val().v + i * in_rows_.at(j), in_rows_.at(j));
                offset += in_rows_[j];
            }
        }
    }

    void backward() override {
        int in_size = ins_.size();
        int row = getDim() / getColumn();
        for (int i = 0; i < getColumn(); ++i) {
            int offset = 0;
            for (int j = 0; j < in_size; ++j) {
                Vec(ins_[j]->loss().v + i * in_rows_.at(j), in_rows_.at(j)) +=
                    Vec(getLoss().v + i * row + offset, in_rows_.at(j));
                offset += in_rows_[j];
            }
        }
    }

private:
    std::vector<int> in_rows_;
    std::vector<Node *> ins_;

    friend class ConcatExecutor;
};

#if USE_GPU
class ConcatExecutor : public Executor {
public:
    void forward() override {
        int count = batch.size();

        std::vector<dtype*> in_vals, vals;
        in_vals.reserve(inCount() * count);
        vals.reserve(count);
        cols_.reserve(count);
        for (Node *node : batch) {
            ConcatNode *concat = static_cast<ConcatNode*>(node);
            for (Node *in : concat->ins_) {
                in_vals.push_back(in->val().value);
            }
            vals.push_back(node->getVal().value);
            cols_.push_back(concat->getColumn());
        }

        ConcatNode &first = dynamic_cast<ConcatNode &>(*batch.front());
        row_ = first.getDim() / first.getColumn();
        n3ldg_cuda::ConcatForward(in_vals, static_cast<ConcatNode*>(batch.at(0))->in_rows_, vals,
                count, inCount(), row_, cols_);
#if TEST_CUDA
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->compute();
            n3ldg_cuda::Assert(batch[idx]->val().verify("concat forward"));
        }
        cout << "concat forward tested" << endl;
#endif
    }

    void backward() override {
        int count = batch.size();
        std::vector<dtype*> in_losses, losses;
        in_losses.reserve(inCount() * count);
        losses.reserve(count);
        for (Node *node : batch) {
            ConcatNode *concat = static_cast<ConcatNode*>(node);
            for (Node *in : concat->ins_) {
                in_losses.push_back(in->loss().value);
            }
            losses.push_back(node->loss().value);
        }

        n3ldg_cuda::ConcatBackward(in_losses, static_cast<ConcatNode*>(batch.at(0))->in_rows_,
                losses, count, inCount(), row_, cols_);
#if TEST_CUDA
        for (int idx = 0; idx < count; idx++) {
            batch[idx]->backward();
        }
        for (int idx = 0; idx < count; idx++) {
            for (int j = 0; j < inCount(); ++j) {
                n3ldg_cuda::Assert(static_cast<ConcatNode *>(batch[idx])->
                        ins_.at(j)->loss().verify("concat backward"));
            }
        }
#endif
    }

private:
    int inCount() {
        return dynamic_cast<ConcatNode *>(batch.front())->ins_.size();
    }

    vector<int> cols_;
    int row_;
};
#else
class ConcatExecutor : public Executor {
public:
    int calculateFLOPs() override {
        return 0;
    }

    int calculateActivations() override {
        return 0;
    }
};
#endif

PExecutor ConcatNode::generate() {
    return new ConcatExecutor();
}

class BatchedConcatNode : public BatchedNodeImpl<ConcatNode> {
public:
    void init(Graph &graph, const std::vector<BatchedNode *> &ins) {
        int dim = 0;
        for (BatchedNode *node : ins) {
            dim += node->getDim();
        }
        allocateBatch(dim, ins.front()->batch().size());
        setInputsPerNode(ins);
        afterInit(graph, ins);
    }
};

class ScalarConcatNode : public Node, public Poolable<ScalarConcatNode> {
public:
    ScalarConcatNode() : Node("scalar_concat") {}

    void setNodeDim(int dim) override {
        setDim(dim);
    }

    void initNode(int dim) override {
        init(dim);
    }

    void forward(Graph &graph, const std::vector<Node *> &ins) {
        if (ins.size() != getDim()) {
            std::cerr << "ScalarConcatNode forward - ins size error" << std::endl;
            std::cerr << fmt::format("ins size:{} dim:{}", ins.size(), getDim());
            abort();
        }

        ins_ = ins;
        for (Node *n : ins) {
            if (n->getDim() != 1) {
                std::cerr << "ScalarConcatNode forward - non scalar found" << std::endl;
                abort();
            }
            n->addParent(this);
        }
        graph.addNode(this);
    }

    Executor *generate() override;

    void compute() override {
        int i = 0;
        for (Node *in : ins_) {
            val()[i++] = in->getVal()[0];
        }
    }

    void backward() override {
        int i = 0;
        for (Node *in : ins_) {
            in->loss()[0] += getLoss()[i++];
        }
    }

    const std::vector<Node *> ins() const {
        return ins_;
    }

    std::string typeSignature() const override {
        return getNodeType();
    }

private:
    std::vector<Node *> ins_;
};

#if USE_GPU
class ScalarConcatExecutor : public Executor {
public:
    void forward() override {
        vector<dtype *> in_vals, vals;
        for (Node *node : batch) {
            dims_.push_back(node->getDim());
        }
        max_dim_ = *max_element(dims_.begin(), dims_.end());
        for (Node *node : batch) {
            ScalarConcatNode *concat = static_cast<ScalarConcatNode *>(node);
            for (Node *in : concat->ins()) {
                in_vals.push_back(in->getVal().value);
            }
            for (int i = 0; i < max_dim_ - concat->ins().size(); ++i) {
                in_vals.push_back(nullptr);
            }
            vals.push_back(node->getVal().value);
        }
        n3ldg_cuda::ScalarConcatForward(in_vals, batch.size(), dims_, max_dim_, vals);
#if TEST_CUDA
        Executor::testForward();
#endif
    }

    void backward() override {
        vector<dtype *> losses, in_losses;
        for (Node *node : batch) {
            ScalarConcatNode *concat = static_cast<ScalarConcatNode *>(node);
            for (Node *in : concat->ins()) {
                in_losses.push_back(in->getLoss().value);
            }
            for (int i = 0; i < max_dim_ - concat->ins().size(); ++i) {
                in_losses.push_back(nullptr);
            }
            losses.push_back(node->getLoss().value);
        }
#if TEST_CUDA
        auto get_inputs = [](Node &node) {
            ScalarConcatNode &concat = static_cast<ScalarConcatNode&>(node);
            vector<pair<Node *, string>> results;
            for (Node *n : concat.ins()) {
                results.push_back(make_pair(n, "input"));
            }
            return results;
        };
//        cout << "test before scalar concat" << endl;
        Executor::testBeforeBackward(get_inputs);
#endif
//        cout << "gpu loss:";
//        batch.front()->getLoss().print();
//        for (int dim : dims_) {
//            cout << "dim:" << dim;
//        }
        n3ldg_cuda::ScalarConcatBackward(losses, batch.size(), dims_, max_dim_, in_losses);
#if TEST_CUDA
//        cout << "batch:" << this->batch.size() << endl;
//        cout << "loss:" << batch.front()->getLoss().toString() << endl;
        Executor::testBackward(get_inputs);
//        cout << "scalar concat tested" << endl;
#endif
    }

private:
    vector<int> dims_;
    int max_dim_;
};
#else
class ScalarConcatExecutor : public Executor {
public:
    int calculateFLOPs() override {
        return 0;
    }
};
#endif

Executor *ScalarConcatNode::generate() {
    return new ScalarConcatExecutor;
}

Node *concat(Graph &graph, const std::vector<Node*> &inputs, int col = 1) {
    int dim = 0;
    for (Node *in : inputs) {
        dim += in->getDim();
    }
    ConcatNode *concat = ConcatNode::newNode(dim);
    concat->setColumn(col);
    concat->connect(graph, inputs);
    return concat;
}

Node *concat(Graph &graph, BatchedNode &inputs, int col = 1) {
    int dim = 0;
    for (Node *in : inputs.batch()) {
        dim += in->getDim();
    }
    ConcatNode *concat = ConcatNode::newNode(dim, col == 1);
    concat->setColumn(col);
    concat->setInputs(inputs.batch());
    inputs.addParent(concat);
    graph.addNode(concat);
    return concat;
}

BatchedNode *concatInBatch(Graph &graph, const std::vector<BatchedNode *> &inputs) {
    BatchedConcatNode *node = new BatchedConcatNode;
    node->init(graph, inputs);
    return node;
}

Node *scalarConcat(Graph &graph, std::vector<Node *> &inputs) {
    ScalarConcatNode *concat = ScalarConcatNode::newNode(inputs.size());
    concat->forward(graph, inputs);
    return concat;
}

}

#endif
