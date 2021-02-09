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
#include "MyTensor.h"
#include "MyLib.h"
#include "profiler.h"
#include <boost/format.hpp>
#if USE_GPU
#include "N3LDG_cuda.h"
using n3ldg_cuda::Tensor1D;
using n3ldg_cuda::Tensor2D;
#else
using n3ldg_cpu::Tensor1D;
using n3ldg_cpu::Tensor2D;
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
    if (x <= 0) return 0;
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

string addressToString(const void* p) {
    std::stringstream ss;
    ss << p;  
    return ss.str();
}

class Node;

class NodeAbs {
public:
    NodeAbs(const string &node_type) : node_type_(node_type) {}
    virtual ~NodeAbs() = default;

    virtual Executor* generate() = 0;
    virtual string typeSignature() const = 0;
    virtual int getDim() const = 0;

    virtual string getNodeType() const {
        return node_type_;
    }

    string cachedTypeSig() const {
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

    virtual vector<Node *> &batch() {
        cerr << "NodeAbs unsupported op" << endl;
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

    const vector<NodeAbs *> getParents() const {
        return parents_;
    }

    virtual bool isBatched() const = 0;

    virtual NodeAbs &topologicalNode() = 0;

    string toString() const {
        return toJson().toStyledString();
    }

    virtual Json::Value toJson() const {
        Json::Value json;
        json["node_type"] = node_type_;
        json["degree"] = degree_;
        json["depth"] = depth_;
        int i = 0;
        for (NodeAbs *node : parents_) {
            json["parent" + to_string(i++)] = node->toJson();
        }
        return json;
    }

private:
    string node_type_;
    int degree_ = 0;
    int depth_ = 0;
    std::vector<NodeAbs *> parents_;
    mutable std::string type_sig_;
};

class Node : public NodeAbs {
public:
    virtual void compute() = 0;
    virtual void backward() = 0;

    virtual string typeSignature() const override {
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
            cerr << boost::format("MatrixNode setColumn - dim:%1% column:%2%") % getDim() % column
                << endl;
            abort();
        }
        column_ = column;
    }

    virtual bool isBatched() const override {
        return false;
    }

    virtual NodeAbs &topologicalNode() override {
        return *batched_node_;
    }

    void setBatchedNode(NodeAbs *node) {
        batched_node_ = node;
    }

    virtual void setInputs(const vector<Node*> &inputs) {}

    Json::Value toJson() const override {
        Json::Value json = NodeAbs::toJson();
        json["dim"] = dim_;
        json["col"] = column_;
        json["batched_node"] = batched_node_;
        return json;
    }

protected:
    void afterForward(NodeContainer &container, const vector<Node*> &ins) {
        for (Node *in : ins) {
            in->addParent(this);
        }
        container.addNode(this);
    }

    Node(const string &node_type, int dim = 0) : NodeAbs(node_type), dim_(dim) {}

    virtual void setDim(int dim) {
        dim_ = dim;
    }

    virtual void init(int ndim) {
        if (ndim <= 0) {
            cerr << boost::format("Node init - dim is less than 0:%1% type:%2%") % ndim %
                getNodeType() << endl;
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
};

class BatchedNode : public NodeAbs {
public:
    virtual string typeSignature() const override {
        return "Batched-" + batch_.front()->typeSignature();
    }

    bool isBatched() const override {
        return true;
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

    virtual string getNodeType() const override {
        return "Batched-" + batch_.front()->getNodeType();
    }

    vector<Node *> &batch() override {
        return batch_;
    }

    Executor* generate() override {
        return batch_.front()->generate();
    }

    const vector<int> &getDims() const {
        if (dims_ == nullptr) {
            dims_ = new vector<int>(batch_.size());
            int i = 0;
            for (Node *node : batch_) {
                dims_->at(i++) = node->getDim();
            }
        }
        return *dims_;
    }

protected:
    void afterInit(NodeContainer &graph, const vector<BatchedNode *> &ins) {
        for (NodeAbs *x : ins) {
            x->addParent(this);
        }
        graph.addNode(this);
    }

    void setInputsPerNode(const vector<BatchedNode *> &batched_inputs) {
        for (int i = 0; i < batch_.size(); ++i) {
            vector<Node *> ins(batched_inputs.size());
            int j = 0;
            for (BatchedNode *in : batched_inputs) {
                ins.at(j++) = in->batch().at(i);
            }
            batch().at(i)->setInputs(ins);
        }
    }

    Json::Value toJson() const override {
        Json::Value json = NodeAbs::toJson();
        int i = 0;
        for (Node *node : batch_) {
            json["batch" + to_string(i++)] = node->toJson();
        }
        return json;
    }

private:
    vector<Node *> batch_;
    mutable vector<int> *dims_ = nullptr;
};

template<typename NodeType>
class BatchedNodeImpl : public BatchedNode {
public:
    BatchedNodeImpl() = default;

protected:
    void allocateBatch(int dim, int size) {
        if (!batch().empty()) {
            cerr << "batch not empty" << endl;
            abort();
        }
        auto v = NodeType::newNodeVector(dim, size);
        batch().reserve(v.size());
        for (auto *x : v) {
            x->setBatchedNode(this);
            batch().push_back(x);
        }
    }

    void allocateBatch(const vector<int> &dims) {
        if (!batch().empty()) {
            cerr << "batch not empty" << endl;
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

set<pair<vector<Node *>, int> *>& globalPoolReferences() {
    static set<pair<vector<Node *>, int> *> o;
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

template <typename T>
class Poolable {
public:
    static vector<T *> newNodeVector(int key, int size) {
        if (key <= 0) {
            cerr << "newNode key:" << key << endl;
            abort();
        }
        vector<T *> results(size);
        if (!globalPoolEnabled()) {
            for (int i = 0; i < size; ++i) {
                T *node = new T;
                node->initNode(key);
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
            pool_.insert(make_pair(key, make_pair(vector<Node *>(), 0)));
            it = pool_.find(key);
            globalPoolReferences().insert(&it->second);
        }
        auto &p = it->second;
        vector<Node *> &v = p.first;
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

        vector<T *> nodes(size);
        nodes.reserve(size);
        for (int i = 0; i < size; ++i) {
//            cout << boost::format("v.size:%1% p.second:%2% i:%3% size:%4%") % v.size() % p.second %
//                i % size << endl;
            T *node = dynamic_cast<T*>(v.at(p.second + i));
            node->setNodeDim(original_key);
            static_cast<Node *>(node)->clear();
            nodes.at(i) = node;
        }

        p.second += size;

        return nodes;
    }

    static T *newNode(int key) {
        if (key <= 0) {
            cerr << "newNode key:" << key << endl;
            abort();
        }
        if (!globalPoolEnabled()) {
            T *node = new T;
            node->initNode(key);
            node->setNodeDim(key);
            node->setBatchedNode(node);
            return node;
        }
        int original_key = key;
        if (globalLimitedDimEnabled()) {
            key = NextTwoIntegerPowerNumber(key);
        }

        map<int, pair<vector<Node *>, int>>::iterator it;
        if (last_key_ == key) {
            it = last_it_;
        } else {
            it = pool_.find(key);
            if (it == pool_.end()) {
                pool_.insert(make_pair(key, make_pair(vector<Node *>(), 0)));
                it = pool_.find(key);
                globalPoolReferences().insert(&it->second);
            }
            last_it_ = it;
            last_key_ = key;
        }
        auto &p = it->second;
        vector<Node *> &v = p.first;
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
    static map<int, pair<vector<Node *>, int>> pool_;
    static map<int, pair<vector<Node *>, int>>::iterator last_it_;
    static int last_key_;
};

template<typename T>
map<int, pair<vector<Node *>, int>> Poolable<T>::pool_;
template<typename T>
map<int, pair<vector<Node *>, int>>::iterator Poolable<T>::last_it_;
template<typename T>
int Poolable<T>::last_key_;

void validateEqualNodeDims(const vector<Node *> &nodes) {
    for (int i = 1; i < nodes.size(); ++i) {
        if (nodes.at(i)->getDim() != nodes.front()->getDim()) {
            cerr << boost::format(
                    "validateEqualNodeDims - first node size is %1%, but %2%st is %3%") %
                nodes.size() % i % nodes.front()->getDim() << endl;
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
    UniInputNode(const string &node_type) : Node(node_type) {}

    virtual string typeSignature() const override {
        return Node::typeSignature() + "-" + to_string(input_->getDim()) + "-";
    }

    void setInputs(const vector<Node *> &ins) override {
        input_ = ins.front();
    }

    void forward(NodeContainer &container, Node &input) {
        if (!isDimLegal(input)) {
            cerr << boost::format("dim:%1% input dim:%2%") % Node::getDim() % input.getDim() <<
                endl;
            abort();
        }
        vector<Node*> ins = {&input};
        setInputs(ins);
        Node::afterForward(container, ins);
    }

    Node *getInput() const {
        return input_;
    }

    Json::Value toJson() const override {
        Json::Value json = NodeAbs::toJson();
        json["input"] = input_ == nullptr ? nullptr : input_->toJson();
        return json;
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
    std::vector<dtype*> val_and_losses(nodes.size());
    vector<int> dims(nodes.size());
    int i = 0;
    for (Node *n : nodes) {
        val_and_losses.at(i) = n->getLoss().value;
        dims.at(i++) = n->getDim();
    }
    n3ldg_cuda::BatchMemset(val_and_losses, val_and_losses.size(), dims, 0.0f);
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

    string getNodeType() const {
        return batch.front()->getNodeType();
    }

    string getSignature() const {
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

        for (NodeAbs *node : batch) {
            Node *x = dynamic_cast<Node *>(node);
            if(!x->getVal().verify((getNodeType() + " forward").c_str())) {
                cout << "cpu:" << endl;
                cout << x->getVal().toJson();
                cout << "gpu:" << endl;
                x->getVal().print();
                abort();
            }
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

    void testBackward(const function<vector<pair<Node*, string>>(Node &node)> &get_inputs) {
        Executor::backward();

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
    vector<Node*> inputs = {uni_input.getInput()};
    return inputs;
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

#endif
