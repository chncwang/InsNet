#include "n3ldg-plus/computation-graph/node.h"
#include "n3ldg-plus/base/memory.h"
#include "n3ldg-plus/util/profiler.h"
#include <functional>

using std::string;
using std::to_string;
using std::stringstream;
using std::max;
using std::cerr;
using std::cout;
using std::endl;
using std::vector;
using std::function;
using std::pair;
using std::make_pair;
using std::make_shared;
using std::to_string;

namespace n3ldg_plus {

string addressToString(const void* p) {
    stringstream ss;
    ss << p;  
    return ss.str();
}

const string &NodeAbs::cachedTypeSig() const {
    if (type_sig_.empty()) {
        type_sig_ = typeSignature();
    }
    return type_sig_;
}

void NodeAbs::clear() {
    degree_ = 0;
    depth_ = 0;
    type_sig_.clear();
    parents_.clear();
}

void NodeAbs::addParent(NodeAbs* parent) {
    NodeAbs &topo = topologicalNode();
    if (topo.degree_ >= 0) {
        topo.parents_.push_back(parent);
        parent->degree_++;
        parent->depth_ = max(topo.depth_ + 1, parent->depth_);
    }
}

string Node::typeSignature() const {
    return getNodeType() + "-" + to_string(dim_) + "-";
}

void Node::setDim(int dim) {
    if (dim <= 0) {
        cerr << fmt::format("Node::setDim - dim:{}", dim) << endl;
        abort();
    }
    dim_ = dim;
}

void Node::clear() {
    val_.ref_count_ = 1;
    batched_node_ = this;
    column_ = 1;
    input_dims_.clear();
    input_vals_.clear();
    input_grads_.clear();
    NodeAbs::clear();
}

void Node::setColumn(int column) {
    if (getDim() % column != 0) {
        cerr << fmt::format("MatrixNode setColumn - dim:{} column:{}\n", getDim(),
                column);
        abort();
    }
    column_ = column;
}

void Node::afterConnect(const vector<Node*> &ins) {
    NodeContainer &container = ins.front()->getNodeContainer();
    for (Node *in : ins) {
        if (&container != &in->getNodeContainer()) {
            cerr << "Node afterConnect - inconsist containers found\\n";
            abort();
        }
        in->addParent(this);
    }
    container.addNode(this);
}

string Node::isVectorSig() const {
    return column_ == 1 ? "-vector-" : "-matrix-";
}

Node::Node(const string &node_type, int dim) : NodeAbs(node_type), dim_(dim) {
    static int id;
    cout << fmt::format("Node::Node id:{}", id) << endl;
    id_ = id++;
}

void Node::setInputs(const std::vector<Node*> &inputs) {
    if (!input_vals_.empty() || !input_grads_.empty() || !input_dims_.empty()) {
        cerr << fmt::format("Node::setInputs input_vals_ size:{} input_grads_ size:{} input_dims_ size:{}\n",
                input_vals_.size(), input_grads_.size(), input_dims_.size());
        abort();
    }

    int size = inputs.size();
    input_vals_.reserve(size);
    input_grads_.reserve(size);
    input_dims_.reserve(size);
    input_types_.reserve(size);
    input_ids_.reserve(size);

    for (Node *input : inputs) {
        input->val_.retain();
        input_vals_.push_back(&input->val_);
        input_grads_.push_back(&input->grad_);
        input_dims_.push_back(input->dim_);
        input_types_.push_back(&input->getNodeType());
        input_ids_.push_back(input->getId());
    }
}

void Node::clearInputVals(bool force) {
    int begin = force ? forwardOnlyInputValSize() : 0;
    int end = force ? inputSize() : forwardOnlyInputValSize();
    for (int i = begin; i < end; ++i) {
        input_vals_.at(i)->release();
    }
}

void Node::clearVal(bool force) {
    if (force || isValForwardOnly()) {
        val_.release();
    }
}

void Node::clearGrad() {
    grad_.releaseMemory();
}

string BatchedNode::typeSignature() const {
    return "Batched-" + batch_.front()->typeSignature();
}

void BatchedNode::clear() {
    batch_.clear();
    NodeAbs::clear();
}

BatchedNode::BatchedNode() : NodeAbs("") {}

BatchedNode::~BatchedNode() {
    if (dims_ != nullptr) {
        delete dims_;
    }
}

string BatchedNode::shape() const {
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
        string str = fmt::format("batch size:{} dims:", batch().size());
        for (int dim : getDims()) {
            str += to_string(dim) + ",";
        }
        return str;
    }
}

const string &BatchedNode::getNodeType() const {
    if (node_type_.empty()) {
        node_type_ = "Batched-" + batch_.front()->getNodeType();
    }
    return node_type_;
}

const vector<int> &BatchedNode::getDims() const {
    if (dims_ == nullptr) {
        dims_ = new vector<int>(batch_.size());
        int i = 0;
        for (Node *node : batch_) {
            dims_->at(i++) = node->getDim();
        }
    }
    return *dims_;
}

void BatchedNode::afterInit(const vector<BatchedNode *> &ins) {
    NodeContainer &container = ins.front()->getNodeContainer();
    for (NodeAbs *in : ins) {
        if (&container != &in->getNodeContainer()) {
            cerr << "Node afterConnect - inconsist containers found\\n";
            abort();
        }
        in->addParent(this);
    }
    container.addNode(this);
}

void BatchedNode::setInputsPerNode(const vector<BatchedNode *> &batched_inputs) {
    for (int i = 0; i < batch_.size(); ++i) {
        vector<Node *> ins(batched_inputs.size());
        int j = 0;
        for (BatchedNode *in : batched_inputs) {
            ins.at(j++) = in->batch().at(i);
        }
        batch().at(i)->setInputs(ins);
    }
}

void validateEqualNodeDims(const vector<Node *> &nodes) {
    for (int i = 1; i < nodes.size(); ++i) {
        if (nodes.at(i)->getDim() != nodes.front()->getDim()) {
            cerr << fmt::format(
                    "validateEqualNodeDims - first node size is {}, but {}st is {}",
                nodes.size(), i, nodes.front()->getDim());
            abort();
        }
    }
}

string UniInputNode::typeSignature() const {
    int input_dim = input_vals_.front()->dim;
    return Node::typeSignature() + "-" + to_string(input_dim) + "-";
}

void UniInputNode::connect(Node &input) {
    if (!isDimLegal(input)) {
        cerr << fmt::format("dim:%1% input dim:%2%\n", Node::getDim(), input.getDim());
        abort();
    }
    vector<Node*> ins = {&input};
    setInputs(ins);
    Node::afterConnect(ins);
}

int UniInputNode::forwardOnlyInputValSize() {
    return isInputValForwardOnly() ? 1 : 0;
}

void initAndZeroGrads(vector<Node *> &nodes) {
    int size = nodes.size();
    vector<cpu::Tensor1D *> grads;
    grads.reserve(size);
    vector<int> dims;
    dims.reserve(size);
    vector<string> sigs;
    sigs.reserve(size);
    for (Node *node : nodes) {
        if (!node->getGrad().isInitialized()) {
            grads.push_back(&node->grad());
            dims.push_back(node->getDim());
            sigs.push_back(node->cachedTypeSig());
        }
    }

    initAndZeroTensors(grads, dims, sigs);
}

#if USE_GPU
vector<dtype *> Executor::getVals() {
    vector<dtype *> vals(batch.size());
    int i = 0;
    for (NodeAbs *node : batch) {
        Node *x = dynamic_cast<Node *>(node);
        vals.at(i++) = x->getVal().value;
    }
    return vals;
}

vector<dtype *> Executor::getGrads() {
    vector<dtype *> grads(batch.size());
    int i = 0;
    for (NodeAbs *node : batch) {
        Node *x = dynamic_cast<Node *>(node);
        grads.at(i++) = x->getGrad().value;
    }
    return grads;
}
#else
int Executor::calculateActivations() {
    int sum = 0;
    for (Node *node : batch) {
        sum += node->getDim();
    }
    return sum;
}
#endif

void Executor::forwardFully() {
    Profiler &profiler = Profiler::Ins();
    profiler.BeginEvent("memory_management");

    int size_sum = 0;
    for (Node *node : batch) {
        size_sum += node->getDim();
    }
    
    auto memory_container = memoryContainer(size_sum * sizeof(dtype));

    for (Node *node : batch) {
        node->val().init(node->getDim(), memory_container);
    }
    profiler.EndEvent();

    profiler.BeginEvent(getNodeType() + "-forward");
    forward();
    profiler.EndCudaEvent();

    profiler.BeginEvent("memory_management");
    for (NodeAbs *node : topo_nodes) {
        node->setDegree(-1);
    }

    for (Node *node : batch) {
        if (!node->topologicalNode().getParents().empty()) {
            node->clearVal(false);
        }
        node->clearInputVals(false);
    }
    profiler.EndEvent();
}

void Executor::backwardFully() {
    Profiler &profiler = Profiler::Ins();
    profiler.BeginEvent("memory_management");
    int size = 0;
    for (Node *node : batch) {
        size += node->inputSize();
    }

    vector<cpu::Tensor1D *> grads;
    grads.reserve(size);
    vector<int> dims;
    dims.reserve(size);
    vector<string> sigs;
    sigs.reserve(size);

    for (Node *node : batch) {
        for (int i = 0; i < node->inputSize(); ++i) {
            Tensor1D &input_grad = *node->input_grads_.at(i);
            if (!input_grad.isInitialized()) {
                grads.push_back(&input_grad);
                dims.push_back(node->input_dims_.at(i));
                sigs.push_back(node->cachedTypeSig());
            }
        }
    }

    profiler.EndEvent();
    initAndZeroTensors(grads, dims, sigs);

    profiler.BeginEvent(getNodeType() + "-backward");
    backward();
    profiler.EndCudaEvent();

    profiler.BeginEvent("memory_management");
    for (Node *node : batch) {
        node->clearVal(true);
        node->clearInputVals(true);
        node->clearGrad();
    }
    profiler.EndEvent();
}

void Executor::backward() {
    for (NodeAbs *node : batch) {
        Node *x = dynamic_cast<Node *>(node);
        x->backward();
    }
}

void Executor::forward() {
    for (NodeAbs *node : batch) {
        Node *x = dynamic_cast<Node *>(node);
        x->compute();
    }
}

int Executor::defaultFLOPs() {
    int sum = 0;
    for (NodeAbs *node : batch) {
        Node *x = dynamic_cast<Node *>(node);
        sum += x->getDim();
    }
    return sum;
}

#if TEST_CUDA
void Executor::testForward() {
    Executor::forward();
    verifyForward();
}

void Executor::verifyForward() {
    int i = 0;
    for (NodeAbs *node : batch) {
        Node *x = dynamic_cast<Node *>(node);
        cout << fmt::format("i:{} dim:{}", i, node->getDim()) << endl;
        if(!x->getVal().verify((getNodeType() + " forward").c_str())) {
            cout << "cpu:" << endl;
            cout << x->getVal().toString();
            cout << "gpu:" << endl;
            x->getVal().print();
            throw cuda::CudaVerificationException(i);
        }
        ++i;
    }
}

void Executor::testForwardInpputs() {
    for (NodeAbs *node : batch) {
        Node *x = dynamic_cast<Node *>(node);
        for (Tensor1D *input : x->input_vals_) {
            cuda::Assert(input->verify((getNodeType() + " forward input").c_str()));
        }
    }
}

void Executor::verifyBackward() {
    int j = 0;
    for (NodeAbs *node : batch) {
        Node *x = dynamic_cast<Node *>(node);
        int i = 0;
        for (Tensor1D *input_grad : x->input_grads_) {
            if (!input_grad->verify((getNodeType() + " backward " + to_string(i++)).c_str())) {
                cout << fmt::format("{}th node dim:{}", j, node->getDim()) << endl;
                cout << "cpu:" << endl << input_grad->toString() << endl;;
                cerr << "gpu:" << endl;
                input_grad->print();
                abort();
            }
        }
        ++j;
    }
}

void Executor::testBackward() {
    Executor::backward();
    verifyBackward();
    cout << batch.front()->cachedTypeSig() << " backward tested" << endl;
}

void Executor::testBeforeBackward() {
    for (NodeAbs *node : batch) {
        Node *x = dynamic_cast<Node *>(node);
        int i = 0;
        for (Tensor1D *input_grad : x->input_grads_) {
            string msg = fmt::format("{} backward {}", getNodeType(), i++);
            cuda::Assert(input_grad->verify(msg.c_str()));
        }
    }
}
#endif

}
