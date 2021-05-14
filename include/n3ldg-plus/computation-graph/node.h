#ifndef N3LDG_PLUS_NODE_H
#define N3LDG_PLUS_NODE_H

#include <string>
#include <vector>
#include <set>
#include <map>
#include <memory>
#include <iostream>
#include "fmt/core.h"
#include "n3ldg-plus/base/tensor.h"
#include "n3ldg-plus/cuda/n3ldg_plus_cuda.h"

namespace n3ldg_plus {

class Executor;
class NodeAbs;

enum ModelStage {
    TRAINING = 0,
    INFERENCE = 1
};

class NodeContainer {
public:
    NodeContainer(ModelStage model_stage = ModelStage::TRAINING) : model_stage_(model_stage) {}

    virtual void addNode(NodeAbs *node) = 0;

    ModelStage getModelStage() const {
        return model_stage_;
    }

private:
    ModelStage model_stage_;
};

std::string addressToString(const void* p);

class Node;

class NodeAbs {
public:
    NodeAbs(const std::string &node_type) : node_type_(node_type) {}
    virtual ~NodeAbs() = default;

    virtual Executor* generate() = 0;
    virtual std::string typeSignature() const = 0;
    virtual int getDim() const = 0;

    virtual const std::string &getNodeType() const {
        return node_type_;
    }

    const std::string &cachedTypeSig() const;

    virtual void clear();

    virtual std::vector<Node *> &batch() {
        std::cerr << "NodeAbs unsupported op" << std::endl;
        abort();
    }

    int getDegree() const {
        return degree_;
    }

    void setDegree(int degree) {
        degree_ = degree;
    }

    int getDepth() const {
        return depth_;
    }

    virtual void addParent(NodeAbs* parent);

    const std::vector<NodeAbs *> getParents() const {
        return parents_;
    }

    virtual bool isBatched() const = 0;

    virtual bool isPooled() const = 0;

    virtual NodeAbs &topologicalNode() = 0;

    std::string toString() const {
        return fmt::format("node_type;");
    }

    void setNodeContainer(NodeContainer &container) {
        node_container_ = &container;
    }

    NodeContainer &getNodeContainer() const {
        return *node_container_;
    }

private:
    std::string node_type_;
    int degree_ = 0;
    int depth_ = 0;
    std::vector<NodeAbs *> parents_;
    mutable std::string type_sig_;
    NodeContainer *node_container_ = nullptr;
};

#if USE_GPU
using cuda::Tensor1D;
using cuda::Tensor2D;
#else
using cpu::Tensor1D;
using cpu::Tensor2D;
#endif

class Node : public NodeAbs {
public:
    virtual void compute() = 0;
    virtual void backward() = 0;

    virtual std::string typeSignature() const override;

    const Tensor1D &getVal() const {
        return val_;
    }

    Tensor1D &val() {
        return val_;
    }

    const Tensor1D &getGrad() const {
        return grad_;
    }

    Tensor1D &grad() {
        return grad_;
    }

    int getDim() const override {
        return dim_;
    }

    int getId() const {
        return id_;
    }

    virtual void clear() override;

    Node (const Node &) = delete;

    virtual ~Node() = default;

    int getColumn() const {
        return column_;
    }

    int getRow() const {
        return getDim() / column_;
    }

    Mat valMat() {
        return Mat(val_.v, getRow(), column_);
    }

    Mat gradMat() {
        return Mat(grad_.v, getRow(), column_);
    }

    void setColumn(int column);

    virtual bool isBatched() const override {
        return false;
    }

    virtual bool isPooled() const override {
        return is_pooled_;
    }

    virtual NodeAbs &topologicalNode() override {
        return *batched_node_;
    }

    void setBatchedNode(NodeAbs *node) {
        batched_node_ = node;
    }

    void setIsPooled(bool is_pooled) {
        is_pooled_ = is_pooled;
    }

    virtual void setInputs(const std::vector<Node*> &inputs);

    void clearInputVals(bool force);

    void clearVal(bool force);

    void clearGrad();

    int inputSize() const {
        return input_dims_.size();
    }

    std::vector<Tensor1D *> &inputVals() {
        return input_vals_;
    }

    const std::vector<const std::string *> &inputTypes() const {
        return input_types_;
    }

    const std::vector<int> &inputIds() const {
        return input_ids_;
    }

protected:
    void afterConnect(const std::vector<Node*> &ins);

    std::string isVectorSig() const;

    Node(const std::string &node_type, int dim = 0);

    virtual void setDim(int dim);

    virtual bool isValForwardOnly() const = 0;

    virtual int forwardOnlyInputValSize() = 0;

    std::vector<Tensor1D *> input_vals_;
    std::vector<Tensor1D *> input_grads_;
    std::vector<int> input_dims_;
    std::vector<const std::string *> input_types_;
    std::vector<int> input_ids_;

private:
    Tensor1D val_;
    Tensor1D grad_;
    int dim_;
    int column_ = 1;
    NodeAbs *batched_node_;
    bool is_pooled_ = true;
    int id_ = 0;

    friend class Executor;
};

class BatchedNode : public NodeAbs {
public:
    virtual std::string typeSignature() const override;

    bool isBatched() const override {
        return true;
    }

    bool isPooled() const override {
        return false;
    }

    NodeAbs &topologicalNode() override {
        return *this;
    }

    virtual void clear() override;

    int getDim() const override {
        return batch_.front()->getDim();
    }

    BatchedNode();

    virtual ~BatchedNode();

    std::string shape() const;

    virtual const std::string &getNodeType() const override;

    std::vector<Node *> &batch() override {
        return batch_;
    }

    const std::vector<Node *> &batch() const {
        return batch_;
    }

    Executor* generate() override {
        return batch_.front()->generate();
    }

    const std::vector<int> &getDims() const;

protected:
    void afterInit(const std::vector<BatchedNode *> &ins);

    void setInputsPerNode(const std::vector<BatchedNode *> &batched_inputs);

private:
    std::vector<Node *> batch_;
    mutable std::vector<int> *dims_ = nullptr;
    mutable std::string node_type_;
};

template<typename NodeType>
class BatchedNodeImpl : public BatchedNode {
public:
    BatchedNodeImpl() = default;

protected:
    void allocateBatch(int dim, int size) {
        if (!batch().empty()) {
            std::cerr << "batch not empty" << std::endl;
            abort();
        }
        auto v = NodeType::newNodeVector(dim, size);
        batch().reserve(v.size());
        for (auto *x : v) {
            x->setBatchedNode(this);
            batch().push_back(x);
        }
    }

    void allocateBatch(const std::vector<int> &dims) {
        if (!batch().empty()) {
            std::cerr << "batch not empty" << std::endl;
            abort();
        }

        batch().reserve(dims.size());
        for (int dim : dims) {
            auto node = NodeType::newNode(dim);
            node->setBatchedNode(this);
            batch().push_back(node);
        }
    }
};


inline std::map<std::vector<Node *> *, int *> &globalPoolReferences() {
    static std::map<std::vector<Node *> *, int *> o;
    return o;
}

inline bool &globalPoolEnabled() {
    static bool pool_enabled = true;
    return pool_enabled;
}

inline bool &globalLimitedDimEnabled() {
    static bool enabled = false;
    return enabled;
}

inline int NextTwoIntegerPowerNumber(int number) {
    int result = 1;
    while (number > result) {
        result <<= 1;
    }
    return result;
}

template <typename T>
class Poolable {
public:
    static std::vector<T *> newNodeVector(int key, int size) {
        if (key <= 0) {
            std::cerr << "newNode key:" << key << std::endl;
            abort();
        }
        std::vector<T *> results(size);
        if (!globalPoolEnabled()) {
            for (int i = 0; i < size; ++i) {
                T *node = new T;
                node->setNodeDim(key);
                node->setIsPooled(false);
                results.at(i) = node;
            }
            return results;
        }

        if (pool_.empty()) {
            globalPoolReferences().insert(std::make_pair(&pool_, &used_count_));
        }

        if (used_count_ > pool_.size()) {
            abort();
        } else if (pool_.size() < used_count_ + size) {
            int vsize = pool_.size();
            for (int i = 0; i < used_count_ + size - vsize; ++i) {
                T *node = new T;
                pool_.push_back(node);
            }
        }

        std::vector<T *> nodes(size);
        nodes.reserve(size);
        for (int i = 0; i < size; ++i) {
            T *node = dynamic_cast<T*>(pool_.at(used_count_ + i));
            node->setNodeDim(key);
            static_cast<Node *>(node)->clear();
            nodes.at(i) = node;
        }

        used_count_ += size;

        return nodes;
    }

    static T *newNode(int key) {
        if (key <= 0) {
            std::cerr << "newNode key:" << key << std::endl;
            abort();
        }
        if (!globalPoolEnabled()) {
            T *node = new T;
            node->setIsPooled(false);
            node->setNodeDim(key);
            node->setBatchedNode(node);
            return node;
        }

        if (pool_.empty()) {
            globalPoolReferences().insert(std::make_pair(&pool_, &used_count_));
        }

        T *node;
        if (used_count_ > pool_.size()) {
            abort();
        } else if (pool_.size() == used_count_) {
            node = new T;
            node->setNodeDim(key);
            node->setBatchedNode(node);
            pool_.push_back(node);
            ++used_count_;
        } else {
            node = dynamic_cast<T*>(pool_.at(used_count_));
            node->clear();
            node->setNodeDim(key);
            node->setBatchedNode(node);
            ++used_count_;
        }
        return node;
    }

    virtual void setNodeDim(int dim) = 0;

private:
    static std::vector<Node *> pool_;
    static int used_count_;
};

template<typename T>
std::vector<Node *> Poolable<T>::pool_;
template<typename T>
int Poolable<T>::used_count_ = 0;

void validateEqualNodeDims(const std::vector<Node *> &nodes);

class UniInputNode : public Node {
public:
    UniInputNode(const std::string &node_type) : Node(node_type) {}

    virtual std::string typeSignature() const override;

    void connect(Node &input);

    Tensor1D &inputVal() {
        return *input_vals_.front();
    }

    Tensor1D &getInputVal() const {
        return *input_vals_.front();
    }

    Tensor1D &inputGrad() {
        return *input_grads_.front();
    }

    int inputDim() const {
        return input_dims_.front();
    }

protected:
    virtual bool isDimLegal(const Node &input) const = 0;

    virtual bool isInputValForwardOnly() const = 0;

    virtual int forwardOnlyInputValSize() override;

private:
    friend class UniInputExecutor;
};

template<typename T>
std::vector<Node*> toNodePointers(const std::vector<T *> &vec) {
    std::vector<Node *> results(vec.size());
    int i = 0;
    for (T *p : vec) {
        results.at(i++) = p;
    }
    return results;
}

void initAndZeroGrads(std::vector<Node*> &nodes);

class Executor {
public:
    std::vector<Node *> batch;
    std::vector<NodeAbs *> topo_nodes;
    virtual ~Executor() = default;

#if USE_GPU
    std::vector<dtype *> getVals();

    std::vector<dtype *> getGrads();
#else
    virtual int calculateFLOPs() = 0;

    virtual int calculateActivations();
#endif

    int getDim() const {
        return dynamic_cast<Node *>(batch.back())->getDim();
    }

    int getRow() const {
        Node &node = dynamic_cast<Node &>(*batch.front());
        return node.getDim() / node.getColumn();
    }

    const std::string &getNodeType() const {
        return batch.front()->getNodeType();
    }

    std::string getSignature() const {
        return batch.front()->typeSignature();
    }

    int getCount() const {
        return batch.size();
    }

    void forwardFully();

    void backwardFully();

    virtual void backward();

protected:
    virtual void forward();

    int defaultFLOPs();

#if TEST_CUDA
    void testForward();

    void verifyForward();

    void testForwardInpputs();

    void verifyBackward();

    void testBackward();

    void testBeforeBackward();
#endif
};

}

#endif
