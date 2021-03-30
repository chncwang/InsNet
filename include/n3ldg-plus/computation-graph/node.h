#ifndef BasicNode
#define BasicNode

#include <iomanip>
#include <functional>
#include <string>
#include <tuple>
#include <memory>
#include <utility>
#include <vector>
#include <set>
#include "n3ldg-plus/base/tensor.h"
#include "n3ldg-plus/util/util.h"
#include "n3ldg-plus/util/profiler.h"

namespace n3ldg_plus {

#if USE_GPU
#include "N3LDG_cuda.h"
using n3ldg_cuda::Tensor1D;
using n3ldg_cuda::Tensor2D;
#else
using cpu::Tensor1D;
using cpu::Tensor2D;
#endif

dtype fexp(const dtype& x) {
    return exp(x);
}

dtype flog(const dtype& x) {
    return log(x);
}

dtype dequal(const dtype& x, const dtype& y) {
    return 1;
}

dtype dtanh(const dtype& x, const dtype& y) {
    return (1 + y) * (1 - y);
}

dtype dleaky_relu(const dtype& x, const dtype& y) {
    if (x < 0) return 0.1;
    return 1;
}

dtype dselu(const dtype& x, const dtype& y) {
    dtype lambda = 1.0507009873554804934193349852946;
    dtype alpha = 1.6732632423543772848170429916717;
    if (x <= 0) return lambda * alpha + y;
    return lambda;
}

dtype dsigmoid(const dtype& x, const dtype& y) {
    return (1 - y) * y;
}

dtype drelu(const dtype& x, const dtype& y) {
    if (y <= 0) return 0;
    return 1;
}

dtype dexp(const dtype& x, const dtype& y) {
    return y;
}

dtype dlog(const dtype& x, const dtype& y) {
    if(x < 0.001) return 1000;
    return 1.0 / x;
}

dtype dsqrt(dtype y) {
    return 0.5 / y;
}

//useful functions
dtype fequal(const dtype& x) {
    return x;
}

dtype ftanh(const dtype& x) {
    return tanh(x);
}

dtype fsigmoid(const dtype& x) {
    return 1.0 / (1.0 + exp(-x));
}

dtype frelu(const dtype& x) {
    if (x <= 0) return 0;
    return x;
}

dtype fleaky_relu(const dtype& x) {
    if (x < 0) return (0.1*x);
    return x;
}

dtype fselu(const dtype& x) {
    dtype lambda = 1.0507009873554804934193349852946;
    dtype alpha = 1.6732632423543772848170429916717;
    if (x <= 0) return lambda * alpha * (exp(x) - 1);
    return lambda * x;
}

dtype fsqrt(const dtype &x) {
    return sqrt(x);
}

class Executor;
class NodeAbs;

class NodeContainer {
public:
    virtual void addNode(NodeAbs *node) = 0;
};

std::string addressToString(const void* p) {
    std::stringstream ss;
    ss << p;  
    return ss.str();
}

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

    std::string cachedTypeSig() const {
        if (type_sig_.empty()) {
            type_sig_ = typeSignature();
        }
        return type_sig_;
    }

    virtual void clear() {
        degree_ = 0;
        depth_ = 0;
        type_sig_.clear();
        parents_.clear();
    }

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

    virtual void addParent(NodeAbs* parent) {
        NodeAbs &topo = topologicalNode();
        if (topo.degree_ >= 0) {
            topo.parents_.push_back(parent);
            parent->degree_++;
            parent->depth_ = std::max(topo.depth_ + 1, parent->depth_);
        }
    }

    const std::vector<NodeAbs *> getParents() const {
        return parents_;
    }

    virtual bool isBatched() const = 0;

    virtual bool isPooled() const = 0;

    virtual NodeAbs &topologicalNode() = 0;

    std::string toString() const {
        return fmt::format("node_type;");
    }

private:
    std::string node_type_;
    int degree_ = 0;
    int depth_ = 0;
    std::vector<NodeAbs *> parents_;
    mutable std::string type_sig_;
};

class Node : public NodeAbs {
public:
    virtual void compute() = 0;
    virtual void backward() = 0;

    virtual std::string typeSignature() const override {
        return getNodeType() + "-" + std::to_string(dim_) + "-";
    }

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

    virtual void clear() override {
#if !USE_GPU || TEST_CUDA
        loss_.zero();
#endif
        batched_node_ = this;
        column_ = 1;
        NodeAbs::clear();
    }

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

    void setColumn(int column) {
        if (getDim() % column != 0) {
            std::cerr << fmt::format("MatrixNode setColumn - dim:{} column:{}\n", getDim(),
                    column);
            abort();
        }
        column_ = column;
    }

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
    void afterForward(NodeContainer &container, const std::vector<Node*> &ins) {
        for (Node *in : ins) {
            in->addParent(this);
        }
        container.addNode(this);
    }

    std::string isVectorSig() const {
        return column_ == 1 ? "-vector-" : "-matrix-";
    }

    Node(const std::string &node_type, int dim = 0) : NodeAbs(node_type), dim_(dim) {}

    virtual void setDim(int dim) {
        dim_ = dim;
    }

    virtual void init(int ndim) {
        if (ndim <= 0) {
            std::cerr << fmt::format("Node init - dim is less than 0:{} type:{}\n", ndim,
                    getNodeType());
            abort();
        }
        dim_ = ndim;
        val_.init(dim_);
        loss_.init(dim_);
    }

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
    virtual std::string typeSignature() const override {
        return "Batched-" + batch_.front()->typeSignature();
    }

    bool isBatched() const override {
        return true;
    }

    bool isPooled() const override {
        return false;
    }

    NodeAbs &topologicalNode() override {
        return *this;
    }

    virtual void clear() override {
        batch_.clear();
        NodeAbs::clear();
    }

    int getDim() const override {
        return batch_.front()->getDim();
    }

    BatchedNode() : NodeAbs("") {}

    virtual ~BatchedNode() {
        if (dims_ != nullptr) {
            delete dims_;
        }
    }

    std::string shape() const {
        bool dims_same = true;
        for (int i = 1; i < batch().size(); ++i) {
            if (batch().front()->getDim() != batch().at(i)->getDim()) {
                dims_same = false;
                break;
            }
        }
        if (dims_same) {
            return fmt::format("batch size:{} dim:{}", batch().size(), getDim());
        } else {
            std::string str = fmt::format("batch size:{} dims:", batch().size());
            for (int dim : getDims()) {
                str += std::to_string(dim) + ",";
            }
            return str;
        }
    }

    virtual std::string getNodeType() const override {
        return "Batched-" + batch_.front()->getNodeType();
    }

    std::vector<Node *> &batch() override {
        return batch_;
    }

    const std::vector<Node *> &batch() const {
        return batch_;
    }

    Executor* generate() override {
        return batch_.front()->generate();
    }

    const std::vector<int> &getDims() const {
        if (dims_ == nullptr) {
            dims_ = new std::vector<int>(batch_.size());
            int i = 0;
            for (Node *node : batch_) {
                dims_->at(i++) = node->getDim();
            }
        }
        return *dims_;
    }

protected:
    void afterInit(NodeContainer &graph, const std::vector<BatchedNode *> &ins) {
        for (NodeAbs *x : ins) {
            x->addParent(this);
        }
        graph.addNode(this);
    }

    void setInputsPerNode(const std::vector<BatchedNode *> &batched_inputs) {
        for (int i = 0; i < batch_.size(); ++i) {
            std::vector<Node *> ins(batched_inputs.size());
            int j = 0;
            for (BatchedNode *in : batched_inputs) {
                ins.at(j++) = in->batch().at(i);
            }
            batch().at(i)->setInputs(ins);
        }
    }

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

std::set<std::pair<std::vector<Node *>, int> *>& globalPoolReferences() {
    static std::set<std::pair<std::vector<Node *>, int> *> o;
    return o;
}

bool &globalPoolEnabled() {
    static bool pool_enabled = true;
    return pool_enabled;
}

bool &globalLimitedDimEnabled() {
    static bool enabled = false;
    return enabled;
}

int NextTwoIntegerPowerNumber(int number) {
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

void validateEqualNodeDims(const std::vector<Node *> &nodes) {
    for (int i = 1; i < nodes.size(); ++i) {
        if (nodes.at(i)->getDim() != nodes.front()->getDim()) {
            std::cerr << fmt::format(
                    "validateEqualNodeDims - first node size is {}, but {}st is {}",
                nodes.size(), i, nodes.front()->getDim());
            abort();
        }
    }
}

auto cpu_get_node_val = [](Node *node) {
    return node->val().v;
};

auto cpu_get_node_loss = [](Node *node) {
    return node->loss().v;
};

#if USE_GPU

auto gpu_get_node_val = [](Node *node) {
    return node->val().value;
};

auto gpu_get_node_loss = [](Node *node) {
    return node->loss().value;
};

#endif

class UniInputNode : public Node {
public:
    UniInputNode(const std::string &node_type) : Node(node_type) {}

    virtual std::string typeSignature() const override {
        return Node::typeSignature() + "-" + std::to_string(input_->getDim()) + "-";
    }

    virtual void setInputs(const std::vector<Node *> &ins) override {
        input_ = ins.front();
    }

    void forward(NodeContainer &container, Node &input) {
        if (!isDimLegal(input)) {
            std::cerr << fmt::format("dim:%1% input dim:%2%\n", Node::getDim(), input.getDim());
            abort();
        }
        std::vector<Node*> ins = {&input};
        setInputs(ins);
        Node::afterForward(container, ins);
    }

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
void clearNodes(std::vector<Node*> &nodes) {
    n3ldg_cuda::Profiler &profiler = n3ldg_cuda::Profiler::Ins();
    profiler.BeginEvent("clearNodes");
    std::vector<dtype*> grads(nodes.size());
    vector<int> dims(nodes.size());
    int i = 0;
    for (Node *n : nodes) {
        grads.at(i) = n->getLoss().value;
        dims.at(i++) = n->getDim();
    }
    n3ldg_cuda::BatchMemset(grads, grads.size(), dims, 0.0f);
#if TEST_CUDA
    for (Node *node : nodes) {
        node->loss().verify("clearNodes");
    }
#endif
    profiler.EndEvent();
}
#endif

class Executor {
public:
    std::vector<Node *> batch;
    std::vector<NodeAbs *> topo_nodes;
    virtual ~Executor() = default;

#if USE_GPU
    vector<dtype *> getVals() {
        vector<dtype *> vals(batch.size());
        int i = 0;
        for (NodeAbs *node : batch) {
            Node *x = dynamic_cast<Node *>(node);
            vals.at(i++) = x->getVal().value;
        }
        return vals;
    }

    vector<dtype *> getGrads() {
        vector<dtype *> grads(batch.size());
        int i = 0;
        for (NodeAbs *node : batch) {
            Node *x = dynamic_cast<Node *>(node);
            grads.at(i++) = x->getLoss().value;
        }
        return grads;
    }
#else
    virtual int calculateFLOPs() = 0;

    virtual int calculateActivations() {
        int sum = 0;
        for (Node *node : batch) {
            sum += node->getDim();
        }
        return sum;
    }
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

    void forwardFully() {
        n3ldg_cuda::Profiler &profiler = n3ldg_cuda::Profiler::Ins();
        profiler.BeginEvent(batch.front()->getNodeType() + " forward");
        forward();

        profiler.EndCudaEvent();
        for (NodeAbs *node : topo_nodes) {
            node->setDegree(-1);
        }
    }

    void backwardFully() {
        n3ldg_cuda::Profiler &profiler = n3ldg_cuda::Profiler::Ins();
        profiler.BeginEvent(getNodeType() + " backward");
        backward();
        profiler.EndEvent();
    }

    virtual void backward() {
        for (NodeAbs *node : batch) {
            Node *x = dynamic_cast<Node *>(node);
            x->backward();
        }
    }

protected:
    virtual void forward() {
        for (NodeAbs *node : batch) {
            Node *x = dynamic_cast<Node *>(node);
            x->compute();
        }
    }

    int defaultFLOPs() {
        int sum = 0;
        for (NodeAbs *node : batch) {
            Node *x = dynamic_cast<Node *>(node);
            sum += x->getDim();
        }
        return sum;
    }

#if TEST_CUDA
    void testForward() {
        Executor::forward();
        verifyForward();
    }

    void verifyForward() {
        int i = 0;
        for (NodeAbs *node : batch) {
            Node *x = dynamic_cast<Node *>(node);
            if(!x->getVal().verify((getNodeType() + " forward").c_str())) {
                cout << "cpu:" << endl;
                cout << x->getVal().toJson();
                cout << "gpu:" << endl;
                x->getVal().print();
                throw n3ldg_cuda::CudaVerificationException(i);
            }
            ++i;
        }
    }

    void testForwardInpputs(const function<vector<Node*>(Node &node)> &get_inputs) {
        for (NodeAbs *node : batch) {
            Node *x = dynamic_cast<Node *>(node);
            vector<Node*> inputs = get_inputs(*x);
            for (Node *input : inputs) {
                n3ldg_cuda::Assert(input->getVal().verify((getNodeType() +
                                " forward input").c_str()));
            }
        }
    }

    void testForwardInpputs(const function<vector<pair<Node*, string>>(Node &node)> &get_inputs) {
        for (NodeAbs *node : batch) {
            Node *x = dynamic_cast<Node *>(node);
            auto inputs = get_inputs(*x);
            for (auto &input : inputs) {
                n3ldg_cuda::Assert(input.first->getVal().verify((getNodeType() +
                                " forward input").c_str()));
            }
        }
    }

    void verifyBackward(const function<vector<pair<Node*, string>>(Node &node)> &get_inputs) {
        for (NodeAbs *node : batch) {
            Node *x = dynamic_cast<Node *>(node);
            auto inputs = get_inputs(*x);
            for (pair<Node*, string> &input : inputs) {
                if (!input.first->getLoss().verify((getNodeType() +
                                " backward " + input.second).c_str())) {
                    cout << "cpu:" << endl << input.first->getLoss().toString() << endl;;
                    cerr << "gpu:" << endl;
                    input.first->getLoss().print();
                    cerr << input.second << endl;
                    abort();
                }
            }
        }
    }

    void testBackward(const function<vector<pair<Node*, string>>(Node &node)> &get_inputs) {
        Executor::backward();
        verifyBackward(get_inputs);
        cout << batch.front()->cachedTypeSig() << " backward tested" << endl;
    }

    void testBeforeBackward(const function<vector<pair<Node*, string>>(Node &node)> &get_inputs) {
        for (NodeAbs *node : batch) {
            Node *x = dynamic_cast<Node *>(node);
            auto inputs = get_inputs(*x);
            for (pair<Node*, string> &input : inputs) {
                n3ldg_cuda::Assert(input.first->getLoss().verify((getNodeType() + " backward " +
                                input.second).c_str()));
            }
        }
    }
#endif
};

auto get_inputs = [](Node &node) {
    UniInputNode &uni_input = static_cast<UniInputNode&>(node);
    std::vector<Node*> inputs = {&uni_input.getInput()};
    return inputs;
};

void printZeroGrads(Node &node, int level = 0) {
    bool all_zero = true;
    for (int i = 0; i < node.getDim(); ++i) {
        if (std::abs(node.getLoss()[i]) > 1e-6) {
            all_zero = false;
            break;
        }
    }
    if (all_zero) {
        std::cout << "type sig:" << node.cachedTypeSig() << " level:" << level << std::endl;
        std::cout << "t parents count:" << node.topologicalNode().getParents().size() << std::endl;
        for (NodeAbs *t_p : node.topologicalNode().getParents()) {
            std::cout << "parent sig:" << t_p->cachedTypeSig() << std::endl;
            for (Node *b : t_p->batch()) {
                printZeroGrads(*b, level + 1);
                break;
            }
        }
    }
};

class UniInputExecutor : public Executor {
protected:
#if TEST_CUDA
    void testForwardInpputs() {
        for (NodeAbs *node : batch) {
            Node *x = dynamic_cast<Node *>(node);
            vector<Node*> inputs = get_inputs(*x);
            for (Node *input : inputs) {
                n3ldg_cuda::Assert(input->getVal().verify((getNodeType() +
                                " forward input").c_str()));
            }
        }
    }

    void testBeforeBackward() {
        auto get_inputs = [](Node &node) {
            UniInputNode &uni_input = static_cast<UniInputNode&>(node);
            vector<pair<Node*, string>> inputs = {make_pair(uni_input.input_, "input")};
            return inputs;
        };
        Executor::testBeforeBackward(get_inputs);
    }

    void verifyBackward() {
        auto get_inputs = [](Node &node) {
            UniInputNode &uni_input = static_cast<UniInputNode&>(node);
            vector<pair<Node*, string>> inputs = {make_pair(uni_input.input_, "input")};
            return inputs;
        };
        Executor::verifyBackward(get_inputs);
    }

    void testBackward() {
        auto get_inputs = [](Node &node) {
            UniInputNode &uni_input = static_cast<UniInputNode&>(node);
            vector<pair<Node*, string>> inputs = {make_pair(uni_input.input_, "input")};
            return inputs;
        };
        Executor::testBackward(get_inputs);
    }
#endif
};

typedef  Executor* PExecutor;

#if USE_GPU

typedef dtype N3LDGActivated(const dtype &x);
ActivatedEnum ToActivatedEnum(N3LDGActivated func) {
    if (func == ftanh) {
        return ActivatedEnum::TANH;
    } else if (func == fsigmoid) {
        return ActivatedEnum::SIGMOID;
    } else if (func == frelu) {
        return ActivatedEnum::RELU;
    } else if (func == fleaky_relu) {
        return ActivatedEnum::LEAKY_RELU;
    } else if (func == fselu) {
        return ActivatedEnum::SELU;
    } else {
        abort();
    }
}

#endif

}

#endif