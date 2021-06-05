#ifndef INSNET_SPLIT_H
#define INSNET_SPLIT_H

#include "insnet/computation-graph/graph.h"

namespace insnet {

/// Returns a contiguous region of a matrix.
///
/// For example, split([0.1, 0.2, 0.3, 0.4], 2, 2) will return [0.3, 0.4] and split([0.1, 0.2, 0.3, 0.4], 1, 1, 2) will return [0.2, 0.4].
///
/// **All the operators will be executed in batch.**
/// \param input The input tensor.
/// \param result_row The result tensor's row number. It should be no greater than *input.size() / input_col*.
/// \param row_offset The row-wise offset where the split begins.
/// \param input_col The column number of the input matrix.
/// \return The result tensor. Its size is *result_row \* input_col*.
Node* split(Node &input, int result_row, int row_offset, int input_col = 1);

BatchedNode *split(Node &input, int row, const std::vector<int> &offsets, int col = 1);

}

#endif
