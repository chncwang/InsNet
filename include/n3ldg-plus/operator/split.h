#ifndef N3LDG_PLUS_SPLIT_H
#define N3LDG_PLUS_SPLIT_H

#include "n3ldg-plus/computation-graph/graph.h"

namespace n3ldg_plus {

/// Returns a contiguous region of a matrix.
///
/// For example, split([0.1, 0.2, 0.3, 0.4], 2, 2) will return [0.3, 0.4] and split([0.1, 0.2, 0.3, 0.4], 1, 1, 2) will return [0.2, 0.4].
///
/// **All the operators will be executed in batch.**
Node* split(Node &input, int result_row, int row_offset, int input_col = 1);

BatchedNode *split(Node &input, int row, const std::vector<int> &offsets, int col = 1);

}

#endif
