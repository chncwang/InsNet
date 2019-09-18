#ifndef N3LDG_PLUS_LAYER
#define N3LDG_PLUS_LAYER

#include "Eigen/Dense"
#include "MyLib.h"
#include "Node.h"
#include "Graph.h"

#include <vector>

using namespace std;

class Layer {
public:
    Layer() = default;
    Layer(const Layer &layer) = delete;
    Layer(const vector<Node *> &nodes) : nodes_(nodes) {}

    const vector<Node*> &getNodes() const {
        return nodes_;
    }

    Layer &forward(Graph &graph, const Layer &input) {
        innerForward(graph, input);
        return *this;
    }

protected:
    virtual void innerForward(Graph &graph, const Layer &input)  = 0;

    void setNodes(const vector<Node*> &nodes) {
        nodes_ = nodes;
    }

private:
    vector<Node*> nodes_;
};

#endif
