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
#include "MyTensor.h"
#if USE_GPU
#include "n3ldg_cuda.h"
using n3ldg_cuda::Tensor1D;
using n3ldg_cuda::Tensor2D;
#else
using n3ldg_cpu::Tensor1D;
using n3ldg_cpu::Tensor2D;
#endif

class Execute;

#if USE_GPU
struct NodeInfo {
    dtype *val;
    dtype *loss;
    std::vector<dtype *> input_vals;
    std::vector<dtype *> input_losses;
    int64_t input_count = -1;
    std::vector<int64_t> input_dims;

    NodeInfo() = default;
    NodeInfo(const NodeInfo &) = default;
    NodeInfo(NodeInfo &&) = default;
};

int GraphToMemory(const std::vector<std::vector<NodeInfo>> &graph,
        void *memory, std::vector<int> &offsets, int size) {
    assert(offsets.empty());
    int offset = 0;
    char *m = (char*)memory;
    for (const std::vector<NodeInfo> &vec : graph) {
        offsets.push_back(offset);
        for (const NodeInfo &node_info : vec) {
            *(dtype**)(m + offset) = node_info.val;
            offset += sizeof(node_info.val);
        }
        for (const NodeInfo &node_info : vec) {
            *(dtype**)(m + offset) = node_info.loss;
            offset += sizeof(node_info.loss);
        }
        int max_input_count = 0;
        for (const NodeInfo &node_info : vec) {
            if (node_info.input_vals.size() > max_input_count) {
                max_input_count = node_info.input_vals.size();
            }
        }
        for (const NodeInfo &node_info : vec) {
            if (!node_info.input_vals.empty()) {
                int len = node_info.input_vals.size() *
                    sizeof(node_info.input_vals.at(0));
                memcpy((void*)(m + offset), node_info.input_vals.data(), len);
                offset += max_input_count * sizeof(node_info.input_vals.at(0));
            }
        }
        for (const NodeInfo &node_info : vec) {
            if (!node_info.input_losses.empty()) {
                int len = node_info.input_losses.size() *
                        sizeof(node_info.input_losses.at(0));
                memcpy((void*)(m + offset),
                        node_info.input_losses.data(), len);
                offset += max_input_count *
                    sizeof(node_info.input_losses.at(0));
            }
        }
        for (const NodeInfo &node_info : vec) {
            if (node_info.input_count != -1) {
                *(int64_t*)(m + offset) = node_info.input_count;
                offset += sizeof(node_info.input_count);
            }
        }
        const NodeInfo &node_info = vec.at(0);
        if (!node_info.input_dims.empty()) {
            int len = node_info.input_dims.size() *
                    sizeof(node_info.input_dims.at(0));
            memcpy((void*)(m + offset), node_info.input_dims.data(), len);
            offset += max_input_count * sizeof(node_info.input_dims.at(0));
        }
    }
    if (offset > size) {
        std::cout << "actual_size is " << offset <<
            " but allocated size is " << size << std::endl;
        abort();
    }

//    std::cout << "graph size:" << offset << std::endl;
//    int i = 0;
//    for (int i = 0; i < offsets.size(); ++i) {
//        int offset = offsets.at(i);
//        std::cout << "offset:" << offset << std::endl;
//        char *m = (char*)memory + offset;
//        std::cout << "val:" << (void*)m << std::endl;
//        for (int j = 0; j < graph.at(i).size(); ++j) {
//            std::cout << "memory:" << *(dtype**)(m + j * sizeof(dtype*)) <<
//                " node:" << graph.at(i).at(j).val << std::endl;
//        }
//        m += graph.at(i).size() * sizeof(dtype*);
//        std::cout << "loss:" << (void*)m << std::endl;
//        for (int j = 0; j < graph.at(i).size(); ++j) {
//            std::cout << "memory:" << *(dtype**)(m + j * sizeof(dtype*)) <<
//                " node:" << graph.at(i).at(j).loss << std::endl;
//        }
//        m += graph.at(i).size() * sizeof(dtype*);

//        int max_input_count = 0;
//        for (const NodeInfo &node_info : graph.at(i)) {
//            if (node_info.input_vals.size() > max_input_count) {
//                max_input_count = node_info.input_vals.size();
//            }
//        }
//        std::cout << "max_input_count when decoding:" << max_input_count <<
//            std::endl;

//        std::cout << "input val:" << (void*)m << std::endl;
//        for (int j = 0; j < graph.at(i).size(); ++j) {
//            int input_size = graph.at(i).at(j).input_vals.size();
//            for (int k = 0; k < input_size; ++k) {
//                std::cout << "memory:" <<
//                    *(dtype**)(m + (j * max_input_count + k) * sizeof(dtype*))
//                    << " node:" << graph.at(i).at(j).input_vals.at(k) <<
//                    std::endl;
//            }
//        }
//        m += max_input_count * graph.at(i).size() * sizeof(dtype*);

//        std::cout << "input loss:" << (void*)m << std::endl;
//        for (int j = 0; j < graph.at(i).size(); ++j) {
//            int input_size = graph.at(i).at(j).input_losses.size();
//            for (int k = 0; k < input_size; ++k) {
//                std::cout << "memory:" <<
//                    *(dtype**)(m + (j * max_input_count + k) * sizeof(dtype*))
//                    << " node:" << graph.at(i).at(j).input_losses.at(k) <<
//                    std::endl;
//            }
//        }
//        m += max_input_count * graph.at(i).size() * sizeof(dtype*);

//        std::cout << "input count:" << (void*)m << std::endl;
//        bool contain_input_count = false;
//        for (int j = 0; j < graph.at(i).size(); ++j) {
//            int input_size = graph.at(i).at(j).input_count;
//            if (input_size != -1) {
//                contain_input_count = true;
//                std::cout << "memory:" << *(int64_t*)(m + j * sizeof(int64_t))
//                    << " node:" << graph.at(i).at(j).input_count << std::endl;
//            }
//        }
//        if (contain_input_count) {
//            m += graph.at(i).size() * sizeof(int64_t);
//        }

//        std::cout << "input dim:" << (void*)m << std::endl;
//        int input_size = graph.at(i).at(0).input_dims.size();
//        for (int k = 0; k < input_size; ++k) {
//            std::cout << "memory:" <<
//                *(int64_t*)(m + k * sizeof(int64_t))
//                << " node:" << graph.at(i).at(0).input_dims.at(k) <<
//                std::endl;
//        }
//    }

    return offset;
}

#endif

// one Node means a vector
// the col should be 1, because we aimed for NLP only
class Node {
  public:
    vector<Node*> parents;
  public:
    Tensor1D val;
    Tensor1D loss;
  public:
    int dim;
    int degree;
    string node_type;

  public:
    Tensor1D drop_mask;
    dtype drop_value;

  public:
    Node() {
        dim = 0;
        degree = 0;
        parents.clear();
        node_type = "interface";
        drop_value = -1;
    }

    virtual ~Node() = default;

  public:
    virtual inline void clearValue() {
#if !USE_GPU || TEST_CUDA
        val = 0;
        loss = 0;
        if (drop_value > 0) drop_mask = 1;
#endif
        degree = 0;
        parents.clear();
    }

    virtual inline void init(int ndim, dtype dropout) {
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

    virtual void generate_dropmask(dtype drop_factor) {
        int dropNum = (int)(dim * drop_value * drop_factor);
        vector<int> tmp_masks(dim);
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
        degree = -1;
    }

    inline void backward_drop() {
        if (drop_value > 0) {
            loss.vec() = loss.vec() * drop_mask.vec();
        }
    }

  public:
    virtual inline void compute() = 0;
    virtual inline void backward() = 0;

    virtual inline Execute* generate(bool bTrain, dtype cur_drop_factor) = 0;

    virtual bool typeEqual(Node* other) {
        if (node_type.compare(other->node_type) != 0) {
            return false;
        }
#if USE_GPU
        if (dim != other->dim) {
            return false;
        }
        if (!isEqual(drop_value, other->drop_value)) {
            return false;
        }
#endif
        return true;
    }

    virtual size_t typeHashCode() const {
        return std::hash<std::string>{}(node_type) ^ std::hash<int>{}(dim) ^
            (std::hash<int>{}((int)(10000 * drop_value)) << 1);
    }

  public:
    virtual inline void addParent(Node* parent) {
        if (degree >= 0) {
            parents.push_back(parent);
            parent->degree++;
        }
    }

#if USE_GPU
    virtual void toNodeInfo(NodeInfo &node_info) const {
        node_info.val = val.value;
        node_info.loss = loss.value;
    }
#endif
};

typedef  Node* PNode;


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
    vector<PNode> batch;
    dtype drop_factor;
#if USE_GPU
    void *graph_info;
#endif

    virtual ~Execute() = default;

    virtual void forward() = 0;
    virtual void backward() = 0;
    virtual void clearValue() {
        for (PNode p : batch) {
            p->clearValue();
        }
#if USE_GPU
        clearNodes(batch, batch.at(0)->dim);
#endif
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
