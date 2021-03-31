#include "n3ldg-plus/computation-graph/node.h"

using std::string;
using std::to_string;
using std::stringstream;
using std::max;
using std::cerr;
using std::vector;

namespace n3ldg_plus {

string addressToString(const void* p) {
    stringstream ss;
    ss << p;  
    return ss.str();
}

string NodeAbs::cachedTypeSig() const {
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

void Node::clear() {
#if !USE_GPU || TEST_CUDA
    loss_.zero();
#endif
    batched_node_ = this;
    column_ = 1;
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

void Node::afterConnect(NodeContainer &container, const vector<Node*> &ins) {
    for (Node *in : ins) {
        in->addParent(this);
    }
    container.addNode(this);
}

string Node::isVectorSig() const {
    return column_ == 1 ? "-vector-" : "-matrix-";
}

Node::Node(const string &node_type, int dim) : NodeAbs(node_type), dim_(dim) {}

void Node::init(int ndim) {
    if (ndim <= 0) {
        cerr << fmt::format("Node init - dim is less than 0:{} type:{}\n", ndim,
                getNodeType());
        abort();
    }
    dim_ = ndim;
    val_.init(dim_);
    loss_.init(dim_);
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

string BatchedNode::getNodeType() const {
    return "Batched-" + batch_.front()->getNodeType();
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

void BatchedNode::afterInit(NodeContainer &graph, const vector<BatchedNode *> &ins) {
    for (NodeAbs *x : ins) {
        x->addParent(this);
    }
    graph.addNode(this);
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
    return Node::typeSignature() + "-" + to_string(input_->getDim()) + "-";
}

void UniInputNode::connect(NodeContainer &container, Node &input) {
    if (!isDimLegal(input)) {
        cerr << fmt::format("dim:%1% input dim:%2%\n", Node::getDim(), input.getDim());
        abort();
    }
    vector<Node*> ins = {&input};
    setInputs(ins);
    Node::afterConnect(container, ins);
}

#if USE_GPU
void clearNodes(vector<Node*> &nodes) {
    n3ldg_cuda::Profiler &profiler = n3ldg_cuda::Profiler::Ins();
    profiler.BeginEvent("clearNodes");
    vector<dtype*> grads(nodes.size());
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
        grads.at(i++) = x->getLoss().value;
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
    forward();

    for (NodeAbs *node : topo_nodes) {
        node->setDegree(-1);
    }
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

void Executor::testForwardInpputs(const function<vector<Node*>(Node &node)> &get_inputs) {
    for (NodeAbs *node : batch) {
        Node *x = dynamic_cast<Node *>(node);
        vector<Node*> inputs = get_inputs(*x);
        for (Node *input : inputs) {
            n3ldg_cuda::Assert(input->getVal().verify((getNodeType() +
                            " forward input").c_str()));
        }
    }
}

void Executor::testForwardInpputs(const function<vector<pair<Node*,
        string>>(Node &node)> &get_inputs) {
    for (NodeAbs *node : batch) {
        Node *x = dynamic_cast<Node *>(node);
        auto inputs = get_inputs(*x);
        for (auto &input : inputs) {
            n3ldg_cuda::Assert(input.first->getVal().verify((getNodeType() +
                            " forward input").c_str()));
        }
    }
}

void Executor::verifyBackward(
        const function<vector<pair<Node*, string>>(Node &node)> &get_inputs) {
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

void Executor::testBackward(const function<vector<pair<Node*, string>>(Node &node)> &get_inputs) {
    Executor::backward();
    verifyBackward(get_inputs);
    cout << batch.front()->cachedTypeSig() << " backward tested" << endl;
}

void Executor::testBeforeBackward(
        const function<vector<pair<Node*, string>>(Node &node)> &get_inputs) {
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

#if TEST_CUDA
void UniInputExecutor::testForwardInpputs() {
    for (NodeAbs *node : batch) {
        Node *x = dynamic_cast<Node *>(node);
        vector<Node*> inputs = get_inputs(*x);
        for (Node *input : inputs) {
            n3ldg_cuda::Assert(input->getVal().verify((getNodeType() + " forward input").c_str()));
        }
    }
}

void UniInputExecutor::testBeforeBackward() {
    auto get_inputs = [](Node &node) {
        UniInputNode &uni_input = static_cast<UniInputNode&>(node);
        vector<pair<Node*, string>> inputs = {make_pair(uni_input.input_, "input")};
        return inputs;
    };
    Executor::testBeforeBackward(get_inputs);
}

void UniInputExecutor::verifyBackward() {
    auto get_inputs = [](Node &node) {
        UniInputNode &uni_input = static_cast<UniInputNode&>(node);
        vector<pair<Node*, string>> inputs = {make_pair(uni_input.input_, "input")};
        return inputs;
    };
    Executor::verifyBackward(get_inputs);
}

void UniInputExecutor::testBackward() {
    auto get_inputs = [](Node &node) {
        UniInputNode &uni_input = static_cast<UniInputNode&>(node);
        vector<pair<Node*, string>> inputs = {make_pair(uni_input.input_, "input")};
        return inputs;
    };
    Executor::testBackward(get_inputs);
}
#endif

}
