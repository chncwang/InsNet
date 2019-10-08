#ifndef BasicNode
#define BasicNode

/*
*  Node.h:
*  basic processing unit in a neural network
*  (1) we have a node structure to build user graph
*  (2) we have a execute structure to merge similar nodes that can be execute together
*  The real forward and backward are defined in Executor.
*  Every operation should define a node class and a execute class together.
*
*  Created on: Apr 21, 2017
*      Author: mszhang
*/
#include <iomanip>
#include <functional>
#include <string>
#include <tuple>
#include <memory>
#include <utility>
#include <vector>
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

//derive function
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

class Executor;
class Node;

class NodeContainer {
public:
    virtual void addNode(Node *node) = 0;
};

string addressToString(const void* p) {
    std::stringstream ss;
    ss << p;  
    return ss.str();
}

class Node {
public:
    Node(const string &node_type, int dim = 0) : dim_(dim) {
        degree_ = 0;
        node_type_ = node_type;
    }

    virtual ~Node() = default;

    virtual void init(int ndim) {
        if (ndim <= 0) {
            cerr << "dim is less than 0:" << ndim << endl;
            abort();
        }
        dim_ = ndim;
        val_.init(dim_);
        loss_.init(dim_);
    }

#if USE_GPU
    virtual void initOnHostAndDevice(int ndim) {
        dim_ = ndim;
        val_.initOnMemoryAndDevice(ndim);
        loss_.initOnMemoryAndDevice(ndim);
        n3ldg_cuda::Memset(val_.value, dim_, 0.0f);
        n3ldg_cuda::Memset(loss_.value, dim_, 0.0f);
    }
#endif

    virtual void compute() = 0;
    virtual void backward() = 0;

    virtual Executor* generate() = 0;

    virtual bool typeEqual(Node* other) {
        if (node_type_.compare(other->node_type_) != 0) {
            return false;
        }
        if (dim_ != other->dim_) {
            return false;
        }
        return true;
    }

    virtual string typeSignature() const {
        return node_type_ + "-" + std::to_string(dim_);
    }

    virtual void addParent(Node* parent) {
        if (degree_ >= 0) {
            parents_.push_back(parent);
            parent->degree_++;
            parent->depth_ = std::max(depth_ + 1, parent->depth_);
        }
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

    int getDim() const {
        return dim_;
    }

    int getDegree() const {
        return degree_;
    }

    void setDegree(int degree) {
        degree_ = degree;
    }

    const string &getNodeName() const {
        return node_name_;
    }

    void setNodeName(const string &node_name) {
        node_name_ = node_name;
    }

    void setNodeIndex(int node_index) {
        node_index_ = node_index;
    }
    
    int getNodeIndex() const {
        return node_index_;
    }

    int getDepth() const {
        return depth_;
    }

    const string &getNodeType() const {
        return node_type_;
    }

    const vector<Node*> getParents() const {
        return parents_;
    }
protected:
    void afterForward(NodeContainer &container, vector<Node*> &ins) {
        for (Node *in : ins) {
            in->addParent(this);
        }
        container.addNode(this);
    }

private:
    std::vector<Node*> parents_;
    Tensor1D val_;
    Tensor1D loss_;
    int dim_;
    int degree_ = 0;
    int depth_ = 0;
    string node_type_;
    string node_name_;
    int node_index_;
};

typedef Node* PNode;

//class CompositionNode : public Node {
//public:
//    virtual void compute() {
//        abort();
//    }

//    virtual void backward() {
//        abort();
//    }

//    virtual Executor* generate() {
//        return nullptr;
//    }

//    static unique_ptr<CompositionNode> instance() {
//        return unique_ptr<CompositionNode>(new CompositionNode);
//    }

//private:
//    CompositionNode() : Node("") {}
//};

class UniInputNode : public Node {
public:
    UniInputNode(const string &node_type) : Node(node_type) {}

    virtual bool typeEqual(Node *other) override {
        UniInputNode *o = static_cast<UniInputNode*>(other);
        return Node::typeEqual(other) && input_->getDim() == o->input_->getDim();
    }

    virtual string typeSignature() const override {
        return Node::typeSignature() + "-" + to_string(input_->getDim());
    }

    void forward(NodeContainer &container, Node &input) {
        if (!isDimLegal(input)) {
            cerr << boost::format("dim:%1% input dim:%2%") % getDim() % input.getDim() << endl;
            abort();
        }
        input_ = &input;
        vector<Node*> ins = {input_};
        afterForward(container, ins);
    }

    Node *getInput() const {
        return input_;
    }

protected:
    virtual bool isDimLegal(const Node &input) const = 0;

private:
    Node *input_;
    friend class UniInputExecutor;
};

template<typename T>
std::vector<Node*> toNodePointers(const std::vector<T *> &vec) {
    std::vector<Node *> results;
    for (T *p : vec) {
        results.push_back(p);
    }
    return results;
}

/* *
 * return tuple<exp, pair<max_i, max>, sum>
 * */
std::tuple<std::unique_ptr<Tensor1D>, std::pair<int, dtype>, dtype> toExp(const Node &node) {
    dtype max = node.getVal().v[0];
    int max_j = 0;
    for (int j = 1; j < node.getDim(); ++j) {
        if (node.getVal().v[j] > max) {
            max = node.getVal().v[j];
            max_j = j;
        }
    }

    std::unique_ptr<Tensor1D> exp(new Tensor1D);
    exp->init(node.getDim());
    exp->vec() = (node.getVal().vec() - max).exp();
    dtype sum = static_cast<Eigen::Tensor<dtype, 0>>(exp->vec().sum())(0);
    return std::make_tuple(std::move(exp), std::make_pair(max_j, max), sum);
}

#if USE_GPU
void clearNodes(std::vector<Node*> &nodes, int dim) {
    n3ldg_cuda::Profiler &profiler = n3ldg_cuda::Profiler::Ins();
    profiler.BeginEvent("clearNodes");
    std::vector<dtype*> val_and_losses;
    val_and_losses.reserve(2 * nodes.size());
    for (Node *n : nodes) {
        val_and_losses.push_back(n->getVal().value);
        val_and_losses.push_back(n->getLoss().value);
    }
    n3ldg_cuda::BatchMemset(val_and_losses, val_and_losses.size(), dim,
            0.0f);
    profiler.EndEvent();
}
#endif

class Executor {
public:
    std::vector<PNode> batch;
#if USE_GPU
    void *graph_info;
#endif

    virtual ~Executor() = default;

    int getDim() const {
        return batch.at(batch.size() - 1)->getDim();
    }

    string getNodeType() const {
        return batch.front()->getNodeType();
    }

    void forwardFully() {
        Node *first = batch.front();
        for (int i = 1; i < batch.size(); ++i) {
            if (!first->typeEqual(batch.at(i))) {
                cerr << "type not equal in the same batch - node_type:" << first->getNodeType() <<
                    endl;
                abort();
            }
        }

        n3ldg_cuda::Profiler &profiler = n3ldg_cuda::Profiler::Ins();
        profiler.BeginEvent(getNodeType() + " forward");
        forward();
        profiler.EndCudaEvent();
        for (Node *node : batch) {
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
        for (Node *node : batch) {
            node->backward();
        }
    }

    virtual bool addNode(PNode in) {
        if (in == nullptr) {
            cerr << "in is nullptr" << endl;
            abort();
        }
        if (batch.empty()) {
            return false;
        }

        if (batch[0]->typeEqual(in)) {
            batch.push_back(in);
            return true;
        }

        return false;
    }

protected:
    virtual void forward() {
        for (Node *node : batch) {
            node->compute();
        }
    }

#if TEST_CUDA
    void testForward() {
        Executor::forward();

        for (Node *node : batch) {
            n3ldg_cuda::Assert(node->getVal().verify((getNodeType() + " forward").c_str()));
        }
    }

    void testForwardInpputs(const function<vector<Node*>(Node &node)> &get_inputs) {
        for (Node *node : batch) {
            vector<Node*> inputs = get_inputs(*node);
            for (Node *input : inputs) {
                n3ldg_cuda::Assert(input->getVal().verify((getNodeType() +
                                " forward input").c_str()));
            }
        }
    }

    void testBackward(const function<vector<pair<Node*, string>>(Node &node)> &get_inputs) {
        Executor::backward();

        for (Node *node : batch) {
            auto inputs = get_inputs(*node);
            for (pair<Node*, string> &input : inputs) {
                n3ldg_cuda::Assert(input.first->getLoss().verify((getNodeType() +
                                " backward " + input.second).c_str()));
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
        for (Node *node : batch) {
            vector<Node*> inputs = get_inputs(*node);
            for (Node *input : inputs) {
                n3ldg_cuda::Assert(input->getVal().verify((getNodeType() +
                                " forward input").c_str()));
            }
        }
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
