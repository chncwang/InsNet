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

// one Node means a vector
// the col should be 1, because we aimed for NLP only
class Node {
  public:
    std::vector<Node*> parents;
  public:
    Tensor1D val;
    Tensor1D loss;
  public:
    int dim;
    int degree;
    int depth = 0;
    string node_type;
    string node_name;

  public:
    Tensor1D drop_mask;
    dtype drop_value;
    int node_index;

  public:
    Node() : node_type("interface") {
        dim = 0;
        degree = 0;
        parents.clear();
        drop_value = -1;
    }

    virtual ~Node() = default;

  public:
//    virtual void clearValue() {
//        if (val.v != NULL) {
//            val = 0;
//        }
//        if (loss.v != NULL) {
//            loss = 0;
//        }
//#if !USE_GPU || TEST_CUDA
//        if (drop_value > 0) drop_mask = 1;
//#endif
//        degree = 0;
//        parents.clear();
//    }

    virtual void init(int ndim, dtype dropout) {
        if (ndim <= 0) {
            abort();
        }
        dim = ndim;
        val.init(dim);
        loss.init(dim);
        drop_mask.init(dim);
#if USE_GPU
        n3ldg_cuda::Memset(val.value, dim, 0.0f);
        n3ldg_cuda::Memset(loss.value, dim, 0.0f);
#endif
        if (dropout > 0 && dropout <= 1) {
            drop_value = dropout;
        } else {
            drop_value = -1;
        }
        parents.clear();
    }

#if USE_GPU
    virtual void initOnHostAndDevice(int ndim, dtype dropout) {
        dim = ndim;
        val.initOnMemoryAndDevice(ndim);
        loss.initOnMemoryAndDevice(ndim);
        drop_mask.init(dim);
        n3ldg_cuda::Memset(val.value, dim, 0.0f);
        n3ldg_cuda::Memset(loss.value, dim, 0.0f);
        if (dropout > 0 && dropout <= 1) {
            drop_value = dropout;
        } else {
            drop_value = -1;
        }
        parents.clear();
    }
#endif

    virtual void generate_dropmask(dtype drop_factor) {
        int dropNum = (int)(dim * drop_value * drop_factor);
        std::vector<int> tmp_masks(dim);
        for (int idx = 0; idx < dim; idx++) {
            tmp_masks[idx] = idx < dropNum ? 0 : 1;
        }
        random_shuffle(tmp_masks.begin(), tmp_masks.end());
        for (int idx = 0; idx < dim; idx++) {
            drop_mask[idx] = tmp_masks[idx];
        }
    }

    void forward_drop(bool bTrain, dtype drop_factor) {
        if (drop_value > 0) {
            if (bTrain) {
#if !TEST_CUDA
                generate_dropmask(drop_factor);
#endif
            } else {
                drop_mask = 1 - drop_value * drop_factor;
            }
            val.vec() = val.vec() * drop_mask.vec();
        }
    }

    void backward_drop() {
        if (drop_value > 0) {
            loss.vec() = loss.vec() * drop_mask.vec();
        }
    }

  public:
    virtual void compute() = 0;
    virtual void backward() = 0;

    virtual Execute* generate(bool bTrain, dtype cur_drop_factor) = 0;

    virtual bool typeEqual(Node* other) {
        if (node_type.compare(other->node_type) != 0) {
            return false;
        }
        if (dim != other->dim) {
            return false;
        }
        if (!isEqual(drop_value, other->drop_value)) {
            return false;
        }
        return true;
    }

    virtual size_t typeHashCode() const {
        return std::hash<std::string>{}(node_type) ^ std::hash<int>{}(dim) ^
            (std::hash<int>{}((int)(10000 * drop_value)) << 1);
    }

  public:
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
std::vector<Node*> toNodePointers(const std::vector<std::shared_ptr<T>> &vec) {
    std::vector<Node *> results;
    for (const std::shared_ptr<T> &p : vec) {
        results.push_back(p.get());
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
    bool bTrain;
    std::vector<PNode> batch;
    dtype drop_factor;
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

    virtual void backward() {
        for (Node *node : batch) {
            node->backward_drop();
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

    dtype dynamicDropValue() const {
        return drop_factor * batch.at(0)->drop_value;
    }

    dtype initialDropValue() const {
        return batch.at(0)->drop_value;
    }

#if USE_GPU
    void CalculateDropMask(int count, int dim,
            const Tensor2D &mask) {
        if (bTrain && initialDropValue() > 0) {
            n3ldg_cuda::CalculateDropoutMask(dynamicDropValue(), count, dim,
                    mask.value);
        }
    }
#endif
protected:
    virtual void forward() {
        for (Node *node : batch) {
            node->compute();
            node->forward_drop(bTrain, drop_factor);
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
