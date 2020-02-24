#ifndef N3LDG_PLUS_LAYER_NORMALIZATION_H
#define N3LDG_PLUS_LAYER_NORMALIZATION_H

#include "MyLib.h"
#include "Node.h"
#include "Graph.h"
#include "Pooling.h"
#include "Sub.h"
#include "PMultiOP.h"

vector<Node *> layerNormalization(Graph &graph, vector<Node *> &input_layer) {
    Node *avg = n3ldg_plus::averagePool(graph, input_layer);
    int len = input_layer.size();
    vector<Node *> square_nodes;
    square_nodes.reserve(len);
    for (Node *input : input_layer) {
        Node *sub = n3ldg_plus::sub(graph, *input, *avg);
        Node *square = n3ldg_plus::pointwiseMultiply(graph, *sub, *sub);
        square_nodes.push_back(square);
    }
}

#endif
