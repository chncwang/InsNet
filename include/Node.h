#ifndef BasicNode
#define BasicNode

/*
*  Node.h:
*  basic processing unit in a neural network
*  (1) we have a node structure to build user graph
*  (2) we have a execute structure to merge similar nodes that can be execute together
*  The real forward and backward are defined in Execute.
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



class Execute;

class Node {
public:
    std::vector<Node*> parents;
    Tensor1D val;
    Tensor1D loss;
    int dim;
    int degree;
    int depth = 0;
    string node_type;
    string node_name;
    int node_index;

    Node() : node_type("interface") {
        dim = 0;
        degree = 0;
        parents.clear();
    }

    virtual ~Node() = default;

    virtual void init(int ndim) {
        if (ndim <= 0) {
            cerr << "dim is less than 0:" << ndim << endl;
            abort();
        }
        dim = ndim;
        val.init(dim);
        loss.init(dim);
#if USE_GPU
        n3ldg_cuda::Memset(val.value, dim, 0.0f);
        n3ldg_cuda::Memset(loss.value, dim, 0.0f);
#endif
    }

#if USE_GPU
    virtual void initOnHostAndDevice(int ndim) {
        dim = ndim;
        val.initOnMemoryAndDevice(ndim);
        loss.initOnMemoryAndDevice(ndim);
        n3ldg_cuda::Memset(val.value, dim, 0.0f);
        n3ldg_cuda::Memset(loss.value, dim, 0.0f);
    }
#endif

    virtual void compute() = 0;
    virtual void backward() = 0;

    virtual Execute* generate() = 0;

    virtual bool typeEqual(Node* other) {
        if (node_type.compare(other->node_type) != 0) {
            return false;
        }
        if (dim != other->dim) {
            return false;
        }
        return true;
    }

    virtual size_t typeHashCode() const {
        return std::hash<std::string>{}(node_type) ^ std::hash<int>{}(dim);
    }

    virtual void addParent(Node* parent) {
        if (degree >= 0) {
            parents.push_back(parent);
            parent->degree++;
            parent->depth = std::max(depth + 1, parent->depth);
        }
    }
};

typedef Node* PNode;

template<typename T>
std::vector<Node*> toNodePointers(std::vector<T *> &vec) {
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
    dtype max = node.val.v[0];
    int max_j = 0;
    for (int j = 1; j < node.dim; ++j) {
        if (node.val.v[j] > max) {
            max = node.val.v[j];
            max_j = j;
        }
    }

    std::unique_ptr<Tensor1D> exp(new Tensor1D);
    exp->init(node.dim);
    exp->vec() = (node.val.vec() - max).exp();
    dtype sum = static_cast<Eigen::Tensor<dtype, 0>>(exp->vec().sum())(0);

    return std::make_tuple(std::move(exp), std::make_pair(max_j, max), sum);
}

#if USE_GPU
void clearNodes(std::vector<Node*> &nodes, int dim) {
    std::vector<dtype*> val_and_losses;
    val_and_losses.reserve(2 * nodes.size());
    for (Node *n : nodes) {
        val_and_losses.push_back(n->val.value);
        val_and_losses.push_back(n->loss.value);
    }
    n3ldg_cuda::BatchMemset(val_and_losses, val_and_losses.size(), dim,
            0.0f);
}
#endif

class Execute {
public:
    std::vector<PNode> batch;
#if USE_GPU
    void *graph_info;
#endif

    virtual ~Execute() = default;

    void forwardFully() {
        forward();
        for (Node *node : batch) {
            node->degree = -1;
        }
    }

    void backwardFully() {
        backward();
    }

    virtual void backward() {
        for (Node *node : batch) {
            node->backward();
        }
    }

    virtual bool addNode(PNode in) {
        if (batch.empty()) {
            std::cout << "empty batch, strange...." << std::endl;
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
};

typedef  Execute* PExecute;

#if USE_GPU

typedef dtype N3LDGActivated(const dtype &x);
n3ldg_cuda::ActivatedEnum ToActivatedEnum(N3LDGActivated func) {
    using n3ldg_cuda::ActivatedEnum;
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
