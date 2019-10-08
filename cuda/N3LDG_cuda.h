#ifndef N3LDG_CUDA_N3LDG_CUDA_H
#define N3LDG_CUDA_N3LDG_CUDA_H

#include "Def.h"

#include <iostream>
#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <vector>
#include <cmath>

namespace n3ldg_cuda {

struct NumberPointerArray {
    dtype **value = nullptr;
    int len = 0;

    NumberPointerArray() = default;
    NumberPointerArray(NumberPointerArray&&) {
        abort();
    }
    NumberPointerArray(const NumberPointerArray &) {
        abort();
    }
    void init(dtype **host_arr, int len);
    ~NumberPointerArray();
};

struct PageLockedNumberPointerArray {
    dtype **value = nullptr;
    int len = 0;

    PageLockedNumberPointerArray() = default;
    PageLockedNumberPointerArray(PageLockedNumberPointerArray&&) {
        abort();
    }
    PageLockedNumberPointerArray(const PageLockedNumberPointerArray &) {
        abort();
    }
    void init(dtype **host_arr, int len);
    ~PageLockedNumberPointerArray();

    dtype **GetDevicePointer() const;
};

struct NumberPointerPointerArray {
    dtype ***value = nullptr;
    int len = 0;

    NumberPointerPointerArray() = default;
    NumberPointerPointerArray(NumberPointerPointerArray&&) {
        abort();
    }
    NumberPointerPointerArray(const NumberPointerPointerArray &) {
        abort();
    }
    void init(dtype ***host_arr, int len);
    ~NumberPointerPointerArray();
};

struct NumberArray {
    dtype *value = nullptr;
    int len = 0;

    NumberArray() = default;
    NumberArray(NumberArray&&) {
        abort();
    }
    NumberArray(const NumberArray &) = delete;
    void init(dtype *host_arr, int len);
    void init(int len);
    ~NumberArray();
};

struct IntPointerArray {
    int **value = nullptr;
    int len = 0;

    IntPointerArray() = default;
    IntPointerArray(IntPointerArray&&) {
        abort();
    }
    IntPointerArray(const IntPointerArray &) = delete;
    void init(int **host_arr, int len);
    ~IntPointerArray();
};

struct DeviceNumber {
    dtype *value = nullptr;
    dtype v = 0.0f;

    DeviceNumber() = default;
    DeviceNumber(DeviceNumber &&) {
        abort();
    }
    DeviceNumber(const DeviceNumber&) {
        abort();
    }

    void init();
    void copyFromDeviceToHost();
    ~DeviceNumber();
};

struct DeviceInt {
    int *value = nullptr;
    int v = 0;

    DeviceInt() = default;
    DeviceInt(DeviceInt &&) {
        abort();
    }
    DeviceInt(const DeviceInt&) {
        abort();
    }

    void init();
    void copyFromDeviceToHost();
    void copyFromHostToDevice();
    ~DeviceInt();
};

struct IntArray {
    int *value = nullptr;
    int len = 0;

    IntArray() = default;
    IntArray(IntArray&&) {
        abort();
    }
    IntArray(const IntArray &) {
        abort();
    }
    void init(int *host_arr, int len);
    void init(int len);
    ~IntArray();
};

struct PageLockedIntArray {
    int *value = nullptr;
    int len = 0;

    PageLockedIntArray() = default;
    PageLockedIntArray(PageLockedIntArray&&) {
        abort();
    }
    PageLockedIntArray(const PageLockedIntArray &) {
        abort();
    }
    void init(int *host_arr, int len);
    void init(int len);
    ~PageLockedIntArray();
};

struct BoolArray {
    bool *value = nullptr;
    bool *v = nullptr;
    int len = 0;

    BoolArray() = default;
    BoolArray(BoolArray&&) {
        abort();
    }
    BoolArray(const BoolArray &) {
        abort();
    }
    void init(bool *host_arr, int len);
    void copyFromHost(bool *host_arr);
    void copyToHost(bool *host_arr);
    ~BoolArray();
};

bool Verify(bool *host, bool *device, int len, const char* message);
bool Verify(int *host, int *device, int len, const char* message);

void Assert(bool v, const std::string &message = "",
        const std::function<void(void)> &call = []() {});
void Memset(dtype *p, int len, dtype value);
void Memset(bool *p, int len, bool value);
void *Malloc(int size);
void BatchMemset(const std::vector<dtype*> &vec, int count, int dim,
        dtype value);
void PrintNums(const dtype* p, int len);
void PrintInts(const int* p, int len);

void InitCuda(int device_id = 0, float memory_in_gb = 0.0f);
void EndCuda();

cudaError_t MyCudaMemcpy(void *dest, const void *src, size_t count, cudaMemcpyKind kind);
void CopyFromMultiVectorsToOneVector(const std::vector<dtype*> &src, dtype *dest, int count,
        int len);
void CopyFromOneVectorToMultiVals(const dtype *src, std::vector<dtype*> &vals,
        int count,
        int len);
void CopyFromHostToDevice(const std::vector<dtype*> &src,
        std::vector<dtype*> &dest, int count, int dim);
void CopyFromDeviceToHost(const std::vector<dtype*> &src,
        std::vector<dtype*> &dest, int count, int dim);

enum PoolingEnum {
    MAX,
    MIN,
    SUM,
    AVG
};

void Activated(ActivatedEnum activated, const dtype *src,
        const std::vector<dtype*>& dest,
        dtype* dest2,
        int len);
void TanhForward(ActivatedEnum activated, const std::vector<dtype*> &xs,
        int count,
        int dim,
        std::vector<dtype*> &ys);
void TanhBackward(ActivatedEnum activated, const std::vector<dtype*> &losses,
        const std::vector<dtype*> &vals,
        int count,
        int dim,
        std::vector<dtype*> &in_losses);
void ExpForward(const std::vector<const dtype*> &inputs, int count, int dim,
        std::vector<dtype*> &results);
void ExpBackward(const std::vector<const dtype*> &losses, const std::vector<const dtype*> &vals,
        int count,
        int dim,
        std::vector<dtype*> input_losses);
void DropoutForward(const std::vector<dtype*> &xs, int count, int dim,
        bool is_training,
        const dtype *drop_mask,
        dtype drop_factor,
        std::vector<dtype*> &ys);
void DropoutBackward(const std::vector<dtype*> &losses,
        const std::vector<dtype*> &vals,
        int count,
        int dim,
        bool is_training,
        const dtype *drop_mask,
        dtype drop_factor,
        std::vector<dtype*> &in_losses);
void BucketForward(const std::vector<dtype> input, int count, int dim, std::vector<dtype*> &ys);
void CopyForUniNodeForward(const std::vector<dtype*> &xs, const dtype* b,
        dtype* xs_dest,
        dtype* b_dest,
        int count,
        int x_len,
        int b_len,
        bool use_b);
void CopyForBiNodeForward(const std::vector<dtype*>& x1s,
        const std::vector<dtype *>& x2s,
        const dtype *b,
        dtype *x1s_dest,
        dtype *x2s_dest,
        dtype *b_dest,
        int count,
        int x1_len,
        int x2_len,
        bool use_b,
        int b_len);
void MatrixMultiplyMatrix(dtype *W, dtype *x, dtype *y, int row, int col,
        int count,
        bool useb,
        bool should_x_transpose = false,
        bool should_W_transpose = false);


void CalculateLtyForUniBackward(ActivatedEnum activated,
        const std::vector<dtype*> &ly,
        const dtype *ty,
        const dtype *y,
        dtype *lty,
        int count,
        int dim);
void AddLtyToParamBiasAndAddLxToInputLossesForUniBackward(const dtype *lty,
        const dtype *lx, dtype *b, std::vector<dtype*> &losses, int count,
        int out_dim, int in_dim, bool use_b);
void AddLtyToParamBiasAndAddLxToInputLossesForBiBackward(const dtype *lty,
        const dtype *lx1,
        const dtype *lx2,
        dtype *b,
        std::vector<dtype*> &losses1,
        std::vector<dtype*> &losses2,
        int count,
        int out_dim,
        int in_dim1,
        int in_dim2,
        bool use_b);
void CalculateDropoutMask(dtype dropout_ratio, int count, int dim, dtype *mask);
void ConcatForward(const std::vector<dtype*> &in_vals,
        const std::vector<int> &in_dims,
        std::vector<dtype*> &vals,
        int count,
        int in_count,
        int out_dim);
void ConcatBackward(const std::vector<dtype*> &in_losses,
        const std::vector<int> &in_dims,
        std::vector<dtype*> &losses,
        int count,
        int in_count,
        int out_dim);
void LookupForward(const std::vector<int> &xids, const dtype *vocabulary,
        int count,
        int dim,
        std::vector<dtype*> &vals);
void LookupBackward(const std::vector<int> &xids, int unknown_id,
        bool fine_tune,
        const std::vector<dtype*> &losses,
        int count,
        int dim,
        dtype *grad,
        bool *indexers);
void PoolForward(PoolingEnum pooling, const std::vector<dtype*> &in_vals,
        std::vector<dtype*> &vals,
        int count,
        const std::vector<int> &in_counts,
        int dim,
        int *hit_inputs);
void PoolBackward(const std::vector<dtype*> &losses,
        std::vector<dtype*> &in_losses,
        const std::vector<int> &in_counts,
        const int *hit_inputs,
        int count,
        int dim);
void SumPoolForward(PoolingEnum pooling, const std::vector<dtype*> &in_vals,
        int count,
        int dim,
        const std::vector<int> &in_counts,
        std::vector<dtype*> &vals);
void SumPoolBackward(PoolingEnum pooling, const std::vector<dtype*> &losses,
        const std::vector<int> &in_counts,
        int count,
        int dim,
        std::vector<dtype*> &in_losses);
void ScalarAttentionForward(const std::vector<dtype*> &ins,
        const std::vector<dtype*> &unnormeds,
        const std::vector<int> &in_counts, int count, int dim,
        std::vector<dtype*> &masks, std::vector<dtype*> &vals);
void ScalarAttentionBackward(const std::vector<dtype*> &losses,
        const std::vector<dtype*> &in_vals,
        const std::vector<dtype*> &masks,
        const std::vector<int> &in_counts,
        int count,
        int dim,
        std::vector<dtype*> &in_losses,
        std::vector<dtype*> &unnormed_losses);
void VectorAttentionForward(const std::vector<dtype*> &ins,
        const std::vector<dtype*> &unnormeds,
        const std::vector<int> &in_counts, int count, int dim,
        std::vector<dtype*> &masks, std::vector<dtype*> &vals);
void VectorAttentionBackward(const std::vector<dtype*> &losses,
        const std::vector<dtype*> &in_vals,
        const std::vector<dtype*> &masks,
        const std::vector<int> &in_counts,
        int count,
        int dim,
        std::vector<dtype*> &in_losses,
        std::vector<dtype*> &unnormed_losses);
void PMultiForward(const std::vector<dtype*> &ins1,
        const std::vector<dtype*> &ins2,
        int count,
        int dim,
        std::vector<dtype*> &vals);
void DivForwartd(const std::vector<const dtype*> numerators,
        const std::vector<const dtype*> denominators,
        int count,
        int dim,
        std::vector<dtype*> &results);
void DivBackward(const std::vector<const dtype*> &losses,
        const std::vector<const dtype*> &denominator_vals,
        const std::vector<const dtype*> &numerator_vals,
        int count,
        int dim,
        std::vector<dtype*> &numerator_losses,
        std::vector<dtype*> &denominator_losses);
void SplitForward(const std::vector<const dtype*> &inputs, const std::vector<int> &offsets,
        int count,
        int dim,
        std::vector<dtype*> &results);
void SplitBackward(const std::vector<const dtype*> &losses, const std::vector<int> offsets,
        int count,
        int dim,
        const std::vector<dtype*> &input_losses);
void SubForward(const std::vector<const dtype*> &minuend,
        const std::vector<const dtype*> &subtrahend,
        int count,
        int dim,
        std::vector<dtype*> &results);
void SubBackward(const std::vector<const dtype*> &losses, int count, int dim,
        std::vector<dtype*> &minuend_losses,
        std::vector<dtype*> &subtrahend_losses);
void PMultiBackward(const std::vector<dtype*> &losses,
        const std::vector<dtype*> &in_vals1,
        const std::vector<dtype*> &in_vals2,
        int count,
        int dim,
        std::vector<dtype*> &in_losses1,
        std::vector<dtype*> &in_losses2);
void PDotForward(const std::vector<dtype*> &ins1,
        const std::vector<dtype*> &ins2,
        int count,
        int dim,
        std::vector<dtype*> &vals);
void PDotBackward(const std::vector<dtype*> &losses,
        const std::vector<dtype*> &in_vals1,
        const std::vector<dtype*> &in_vals2,
        int count,
        int dim,
        std::vector<dtype*> &in_losses1,
        std::vector<dtype*> &in_losses2);
void PAddForward(const std::vector<std::vector<dtype*>> &ins, int count,
        int dim,
        int in_count,
        std::vector<dtype*> &vals);
void PAddBackward(const std::vector<dtype*> &losses, int count, int dim,
        int in_count,
        std::vector<std::vector<dtype*>> &in_losses);
void SoftMaxLoss(const std::vector<dtype*> &vals, std::vector<dtype*> &losses,
        int *correct_count,
        const std::vector<int> &answers,
        int batchsize,
        int count,
        int dim);
void MaxScalarForward(const std::vector<const dtype*> &inputs, int count, int dim,
        std::vector<dtype*> &results,
        std::vector<int> &max_indexes);
void MaxScalarBackward(const std::vector<const dtype *> &losses, const std::vector<int> &indexes,
        int count,
        const std::vector<dtype*> &input_losses);
void VectorSumForward(const std::vector<const dtype *> &inputs, int count, int dim,
        std::vector<dtype*> &results);
void VectorSumBackward(const std::vector<const dtype*> &losses, int count, int dim,
        std::vector<dtype*> &input_losses);
void ScalarToVectorForward(const std::vector<const dtype*> &inputs, int count, int dim,
        std::vector<dtype*> &results);
void ScalarToVectorBackward(const std::vector<const dtype*> &losses, int count, int dim,
        std::vector<dtype*> &input_losses);
int Predict(const dtype* val, int dim);
std::pair<dtype, std::vector<int>> SoftMaxLoss(const std::vector<const dtype *> &vals_vector,
        int count,
        int dim,
        const std::vector<int> &gold_answers,
        int batchsize,
        const std::vector<dtype *> &losses_vector);
dtype SquareSum(const dtype *v, int len);
dtype SquareSum(const dtype *v, const bool *indexers, int count, int dim);
void Rescale(dtype *v, int len, dtype scale);
void UpdateAdam(dtype *val, dtype *grad, int row, int col, bool is_bias, dtype *aux_mean,
        dtype *aux_square,
        int iter,
        dtype belta1,
        dtype belta2,
        dtype alpha,
        dtype reg,
        dtype eps);
void UpdateAdam(dtype *val, dtype *grad, int row, int col, dtype *aux_mean,
        dtype *aux_square,
        const bool *indexers,
        int *iters,
        dtype belta1,
        dtype belta2,
        dtype alpha,
        dtype reg,
        dtype eps);
void UpdateAdamW(dtype *val, dtype *grad, int row, int col, bool is_bias, dtype *aux_mean,
        dtype *aux_square,
        int iter,
        dtype belta1,
        dtype belta2,
        dtype alpha,
        dtype reg,
        dtype eps);
void UpdateAdagrad(dtype *val, dtype *grad, int row, int col,
        dtype *aux_square,
        dtype alpha,
        dtype reg,
        dtype eps);
void UpdateAdagrad(dtype *val, dtype *grad, int row, int col,
        dtype *aux_square,
        const bool *indexers,
        dtype alpha,
        dtype reg,
        dtype eps);
void *GraphHostAlloc();
}

#endif
