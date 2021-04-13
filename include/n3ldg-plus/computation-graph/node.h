#ifndef N3LDG_PLUS_NODE_H
#define N3LDG_PLUS_NODE_H

#include <string>
#include <vector>
#include <set>
#include <map>
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

    virtual std::string getNodeType() const {
        return node_type_;
    }

    std::string cachedTypeSig() const;

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

    const Tensor1D &getLoss() const {
        return loss_;
    }

    Tensor1D &loss() {
        return loss_;
    }

    int getDim() const override {
        return dim_;
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
        return Mat(val().v, getRow(), column_);
    }

    Mat gradMat() {
        return Mat(loss().v, getRow(), column_);
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

    virtual void setInputs(const std::vector<Node*> &inputs) {}

protected:
    void afterConnect(const std::vector<Node*> &ins);

    std::string isVectorSig() const;

    Node(const std::string &node_type, int dim = 0);

    virtual void setDim(int dim) {
        dim_ = dim;
    }

    virtual void init(int ndim);

private:
    Tensor1D val_;
    Tensor1D loss_;
    int dim_;
    int column_ = 1;
    NodeAbs *batched_node_;
    bool is_pooled_ = true;
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

    virtual std::string getNodeType() const override;

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
};

template<typename NodeType>
class BatchedNodeImpl : public BatchedNode {
public:
    BatchedNodeImpl() = default;

protected:
    void allocateBatch(int dim, int size, bool pool = true) {
        if (!batch().empty()) {
            std::cerr << "batch not empty" << std::endl;
            abort();
        }
        auto v = NodeType::newNodeVector(dim, size, pool);
        batch().reserve(v.size());
        for (auto *x : v) {
            x->setBatchedNode(this);
            batch().push_back(x);
        }
    }

    void allocateBatch(const std::vector<int> &dims, bool pool = true) {
        if (!batch().empty()) {
            std::cerr << "batch not empty" << std::endl;
            abort();
        }

        batch().reserve(dims.size());
        for (int dim : dims) {
            auto node = NodeType::newNode(dim, pool);
            node->setBatchedNode(this);
            node->setIsPooled(pool);
            batch().push_back(node);
        }
    }
};


inline std::set<std::pair<std::vector<Node *>, int> *>& globalPoolReferences() {
    static std::set<std::pair<std::vector<Node *>, int> *> o;
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

constexpr int BIG_VECTOR_SIZE = 1024 * 16;

template <typename T>
class Poolable {
public:
    static std::vector<T *> newNodeVector(int key, int size, bool pool = true) {
        if (key <= 0) {
            std::cerr << "newNode key:" << key << std::endl;
            abort();
        }
        std::vector<T *> results(size);
        if (!globalPoolEnabled() || (!pool && key >= BIG_VECTOR_SIZE)) {
            for (int i = 0; i < size; ++i) {
                T *node = new T;
                node->initNode(key);
                node->setIsPooled(false);
                results.at(i) = node;
            }
            return results;
        }

        int original_key = key;
        if (globalLimitedDimEnabled()) {
            key = NextTwoIntegerPowerNumber(key);
        }

        auto it = pool_.find(key);
        if (it == pool_.end()) {
            pool_.insert(make_pair(key, make_pair(std::vector<Node *>(), 0)));
            it = pool_.find(key);
            globalPoolReferences().insert(&it->second);
        }
        auto &p = it->second;
        std::vector<Node *> &v = p.first;
        if (p.second > v.size()) {
            abort();
        } else if (v.size() < p.second + size) {
            int vsize = v.size();
            for (int i = 0; i < p.second + size - vsize; ++i) {
                T *node = new T;
                node->initNode(key);
                v.push_back(node);
            }
        }

        std::vector<T *> nodes(size);
        nodes.reserve(size);
        for (int i = 0; i < size; ++i) {
            T *node = dynamic_cast<T*>(v.at(p.second + i));
            node->setNodeDim(original_key);
            static_cast<Node *>(node)->clear();
            nodes.at(i) = node;
        }

        p.second += size;

        return nodes;
    }

    static T *newNode(int key, bool pool = true) {
        if (key <= 0) {
            std::cerr << "newNode key:" << key << std::endl;
            abort();
        }
        if (!globalPoolEnabled() || (!pool && key >= BIG_VECTOR_SIZE)) {
            T *node = new T;
            node->initNode(key);
            node->setIsPooled(false);
            node->setNodeDim(key);
            node->setBatchedNode(node);
            return node;
        }
        int original_key = key;
        if (globalLimitedDimEnabled()) {
            key = NextTwoIntegerPowerNumber(key);
        }

        std::map<int, std::pair<std::vector<Node *>, int>>::iterator it;
        if (last_key_ == key) {
            it = last_it_;
        } else {
            it = pool_.find(key);
            if (it == pool_.end()) {
                pool_.insert(make_pair(key, make_pair(std::vector<Node *>(), 0)));
                it = pool_.find(key);
                globalPoolReferences().insert(&it->second);
            }
            last_it_ = it;
            last_key_ = key;
        }
        auto &p = it->second;
        std::vector<Node *> &v = p.first;
        T *node;
        if (p.second > v.size()) {
            abort();
        } else if (v.size() == p.second) {
            node = new T;
            node->initNode(key);
            node->setNodeDim(original_key);
            node->setBatchedNode(node);
            v.push_back(node);
            ++p.second;
        } else {
            node = dynamic_cast<T*>(v.at(p.second));
            node->setNodeDim(original_key);
            node->clear();
            node->setBatchedNode(node);
            ++p.second;
        }
        return node;
    }

    virtual void initNode(int dim) = 0;
    virtual void setNodeDim(int dim) = 0;

private:
    static std::map<int, std::pair<std::vector<Node *>, int>> pool_;
    static std::map<int, std::pair<std::vector<Node *>, int>>::iterator last_it_;
    static int last_key_;
};

template<typename T>
std::map<int, std::pair<std::vector<Node *>, int>> Poolable<T>::pool_;
template<typename T>
std::map<int, std::pair<std::vector<Node *>, int>>::iterator Poolable<T>::last_it_;
template<typename T>
int Poolable<T>::last_key_;

void validateEqualNodeDims(const std::vector<Node *> &nodes);

class UniInputNode : public Node {
public:
    UniInputNode(const std::string &node_type) : Node(node_type) {}

    virtual std::string typeSignature() const override;

    virtual void setInputs(const std::vector<Node *> &ins) override {
        input_ = ins.front();
    }

    void connect(Node &input);

    Node &getInput() const {
        return *input_;
    }

protected:
    virtual bool isDimLegal(const Node &input) const = 0;

private:
    Node *input_;
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

#if USE_GPU
void clearNodes(std::vector<Node*> &nodes);
#endif

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

    std::string getNodeType() const {
        return batch.front()->getNodeType();
    }

    std::string getSignature() const {
        return batch.front()->typeSignature();
    }

    int getCount() const {
        return batch.size();
    }

    void forwardFully();

    void backwardFully() {
        backward();
    }

    virtual void backward();

protected:
    virtual void forward();

    int defaultFLOPs();

#if TEST_CUDA
    void testForward();

    void verifyForward();

    void testForwardInpputs(const function<vector<Node*>(Node &node)> &get_inputs);

    void testForwardInpputs(const function<vector<pair<Node*, string>>(Node &node)> &get_inputs);

    void verifyBackward(const function<vector<pair<Node*, string>>(Node &node)> &get_inputs);

    void testBackward(const function<vector<pair<Node*, string>>(Node &node)> &get_inputs);

    void testBeforeBackward(const function<vector<pair<Node*, string>>(Node &node)> &get_inputs);
#endif
};

class UniInputExecutor : public Executor {
protected:
#if TEST_CUDA
    void testForwardInpputs();

    void testBeforeBackward();

    void verifyBackward();

    void testBackward();
#endif
};

}

#endif
