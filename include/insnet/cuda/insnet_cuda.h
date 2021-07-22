#if USE_GPU
#ifndef N3LDG_CUDA_N3LDG_CUDA_H
#define N3LDG_CUDA_N3LDG_CUDA_H

#include "insnet/base/def.h"

#include "memory_pool.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <string>
#include <cmath>
#include <exception>

namespace insnet {
namespace cuda {

class CudaVerificationException: public std::exception {
public:
    CudaVerificationException(int index) : index_(index) {}

    int getIndex() const {
        return index_;
    }

private:
    int index_;
};

template<typename T>
struct GPUArray {
    T *value = nullptr;
    int len = 0;

    GPUArray() = default;
    GPUArray(GPUArray<T>&&) {
        abort();
    }
    GPUArray(const GPUArray &) {
        abort();
    }

    void init(T *host_arr, int len);
    void init(int len);
    ~GPUArray();

    std::vector<T> toCpu() const;
};

enum MyCudaMemcpyKind {
    HOST_TO_DEVICE = 0,
    DEVICE_TO_HOST = 1,
    DEVICE_TO_DEVICE = 2
};

void MyCudaMemcpy(void *dest, void *src, size_t count, MyCudaMemcpyKind kind);
void CallCuda(int status);

template <typename T>
void GPUArray<T>::init(T *host_arr, int len) {
    if (value != nullptr) {
        MemoryPool::Ins().Free(value);
        value = nullptr;
    }
    MemoryPool::Ins().Malloc((void**)&value, len * sizeof(T));
    MyCudaMemcpy(value, host_arr, len * sizeof(T), MyCudaMemcpyKind::HOST_TO_DEVICE);
    this->len = len;
}

template <typename T>
void GPUArray<T>::init(int len) {
    if (value != nullptr) {
        MemoryPool::Ins().Free(value);
        value = nullptr;
    }
    MemoryPool::Ins().Malloc((void**)&value, len * sizeof(T));
    this->len = len;
}

template <typename T>
std::vector<T> GPUArray<T>::toCpu() const {
    std::vector<T> result;
    result.resize(len);
    if (value == nullptr) {
        std::cerr << "GPUArray::toCpu - value is nullptr" << std::endl;
        abort();
    }
    MyCudaMemcpy(result.data(), value, sizeof(T) * len, MyCudaMemcpyKind::DEVICE_TO_HOST);
    return result;
}

template <typename T>
GPUArray<T>::~GPUArray() {
    if (value != nullptr) {
        MemoryPool::Ins().Free(value);
        value = nullptr;
    }
}

typedef GPUArray<dtype> NumberArray;
typedef GPUArray<bool> BoolArray;
typedef GPUArray<bool *> BoolPointerArray;
typedef GPUArray<dtype *> NumberPointerArray;
typedef GPUArray<dtype **> NumberPointerPointerArray;
typedef GPUArray<int> IntArray;
typedef GPUArray<int *> IntPointerArray;

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

bool Verify(bool *host, bool *device, int len, const char* message);
bool Verify(int *host, int *device, int len, const char* message);

void Assert(bool v, const std::string &message = "",
        const std::function<void(void)> &call = []() {});
void Memset(dtype *p, int len, dtype value);
void Memset(bool *p, int len, bool value);
void *Malloc(int size);
void BatchMemset(std::vector<dtype*> &vec, int count, const std::vector<int> &dims, dtype value);
void PrintNums(dtype* p, int len);
void PrintInts(int* p, int len);

void initCuda(int device_id = 0, float memory_in_gb = 0.0f);

void CopyFromMultiVectorsToOneVector(std::vector<dtype*> &src, dtype *dest, int count,
        int len);
void CopyFromOneVectorToMultiVals(dtype *src, std::vector<dtype*> &vals,
        int count,
        int len);
void CopyFromHostToDevice(std::vector<dtype*> &src,
        std::vector<dtype*> &dest, int count, int dim);
void CopyFromDeviceToHost(std::vector<dtype*> &src,
        std::vector<dtype*> &dest, int count, int dim);

enum PoolingEnum {
    MAX,
    MIN,
    SUM,
    AVG
};

void ActivationForward(ActivatedEnum activated, std::vector<dtype*> &xs,
        int count,
        std::vector<int> &dims,
        std::vector<dtype*> &ys);
void ActivationBackward(ActivatedEnum activated,
        std::vector<dtype*> &losses,
        std::vector<dtype*> &vals,
        int count,
        std::vector<int> &dims,
        std::vector<dtype*> &in_losses);
void DropoutForward(std::vector<dtype*> &xs, int count, std::vector<int> &dims,
        int max_dim,
        std::vector<int> &offsets,
        bool is_training,
        dtype *drop_mask,
        dtype drop_factor,
        std::vector<dtype*> &ys);
void DropoutBackward(std::vector<dtype*> &grads, int count, std::vector<int> &dims,
        int max_dim,
        std::vector<int> &offsets,
        bool is_training,
        dtype *drop_mask,
        dtype drop_factor,
        std::vector<dtype*> &in_grads);
void BucketForward(std::vector<dtype> input, int count, int dim,
        std::vector<dtype*> &ys);
void MatrixMultiplyMatrix(dtype *W, dtype *x, dtype *y,
        int row,
        int col,
        int count,
        bool useb,
        bool should_x_transpose = false,
        bool should_W_transpose = false);
void LinearForward(dtype **in_val_arr, int count, std::vector<int> &in_cols,
        int in_row,
        int out_row,
        dtype *W,
        dtype *bias,
        std::vector<dtype *> &vals);
void LinearBackward(std::vector<dtype *> &grads, int count, std::vector<int> &cols, int in_row,
        int out_row,
        dtype *W_val,
        dtype **in_val_arr,
        dtype *bias_grad,
        std::vector<dtype *> &in_grads,
        dtype *W_grad);
void CalculateDropoutMask(dtype dropout_ratio, int dim, dtype *mask);
void ConcatForward(std::vector<dtype*> &in_vals, std::vector<int> &in_dims,
        std::vector<dtype*> &vals,
        int count,
        int in_count,
        int out_dim,
        std::vector<int> &cols);
void ConcatBackward(std::vector<dtype*> &in_grads, std::vector<int> &in_rows,
        std::vector<dtype*> &grads,
        int count,
        int in_count,
        int out_row,
        std::vector<int> &cols);
void ScalarConcatForward(std::vector<dtype *> &ins, int count, std::vector<int> &dims,
        int max_dim,
        std::vector<dtype *> &results);
void ScalarConcatBackward(std::vector<dtype *> &losses, int count,
        std::vector<int> &dims,
        int max_dim,
        std::vector<dtype *> in_losses);
void LookupForward(int *ids, dtype *vocabulary, int count, int row, int *cols,
        int max_col,
        std::vector<dtype*> &vals);
void LookupBackward(int *ids, std::vector<dtype*> &grads, int count, int row,
        int *cols,
        int max_col,
        dtype *param_grad,
        bool *indexers);
void PoolForward(PoolingEnum pooling, std::vector<dtype*> &in_vals,
        std::vector<dtype*> &vals,
        int count,
        std::vector<int> &in_counts,
        int dim,
        int *hit_inputs);
void PoolBackward(std::vector<dtype*> &losses,
        std::vector<dtype*> &in_losses,
        std::vector<int> &in_counts,
        int *hit_inputs,
        int count,
        int dim);
void SumPoolForward(PoolingEnum pooling, std::vector<dtype*> &in_vals,
        int count,
        int dim,
        std::vector<int> &in_counts,
        std::vector<dtype*> &vals);
void SumPoolBackward(PoolingEnum pooling, std::vector<dtype*> &losses,
        std::vector<int> &in_counts,
        int count,
        int dim,
        std::vector<dtype*> &in_losses);
void MatrixConcatForward(std::vector<dtype*> &in_vals, int count, int in_dim,
        std::vector<int> &in_counts,
        std::vector<dtype*> &vals);
void MatrixConcatBackward(std::vector<dtype *> &grads, int count, int in_dim,
        std::vector<int> &in_counts,
        std::vector<dtype *> &in_grads);
void TranMatrixMulVectorForward(std::vector<dtype *> &matrices,
        std::vector<dtype *> &vectors,
        int count,
        std::vector<int> &cols,
        int row,
        std::vector<dtype *> &vals);
void TranMatrixMulVectorBackward(std::vector<dtype *> &grads,
        std::vector<dtype *> &matrix_vals,
        std::vector<dtype *> &vector_vals,
        int count,
        std::vector<int> &cols,
        int row,
        std::vector<dtype *> &matrix_grads,
        std::vector<dtype *> &vector_grads);
void TranMatrixMulMatrixForward(std::vector<dtype *> &input_a_vals,
        std::vector <dtype *> &input_b_vals,
        int count,
        std::vector<int> &a_cols,
        std::vector<int> &b_cols,
        int row,
        bool use_lower_triangle_mask,
        std::vector<dtype *> &vals);
void TranMatrixMulMatrixBackward(std::vector<dtype *> &grads,
        std::vector<dtype *> &a_vals,
        std::vector<dtype *> &b_vals,
        int count,
        std::vector<int> &a_cols,
        std::vector<int> &b_cols,
        int row,
        std::vector<dtype *> &a_grads,
        std::vector<dtype *> &b_grads);
void MatrixMulMatrixForward(std::vector<dtype *> &a,
        std::vector<dtype *> &b,
        int count,
        std::vector<int> &ks,
        std::vector<int> &b_cols,
        int row,
        std::vector<dtype *> &vals);
void MatrixMulMatrixBackward(std::vector<dtype *> &grads,
        std::vector<dtype *> &a_vals,
        std::vector<dtype *> &b_vals,
        int count,
        std::vector<int> &ks,
        std::vector<int> &b_cols,
        int row,
        std::vector<dtype *> &a_grads,
        std::vector<dtype *> &b_grads);
void MatrixAndVectorMultiForward(std::vector<dtype *> &matrices,
        std::vector<dtype *> &vectors,
        int count,
        int row,
        std::vector<int> &cols,
        std::vector<dtype *> &vals);
void MatrixAndVectorMultiBackward(std::vector<dtype *> &grads,
        std::vector<dtype *> &matrices,
        std::vector<dtype *> &vectors,
        int count,
        int row,
        std::vector<int> &cols,
        std::vector<dtype *> &matrix_grads,
        std::vector<dtype *> &vector_grads);
void PMultiForward(std::vector<dtype*> &ins1,
        std::vector<dtype*> &ins2,
        int count,
        int dim,
        std::vector<dtype*> &vals);
void FullDivForward(std::vector<dtype*> &numerators,
        std::vector<dtype*> &denominators,
        int count,
        std::vector<int> &dims,
        std::vector<dtype*> &results);
void FullDivBackward(std::vector<dtype*> &grads,
        std::vector<dtype*> &denominator_vals,
        std::vector<dtype*> &numerator_vals,
        int count,
        std::vector<int> &dims,
        std::vector<dtype*> &numerator_grads,
        std::vector<dtype*> &denominator_grads);
void SplitForward(std::vector<dtype*> &inputs, std::vector<int> &offsets, int count,
        std::vector<int> &rows,
        std::vector<int> &in_rows,
        std::vector<int> &cols,
        std::vector<dtype*> &results);
void SplitBackward(std::vector<dtype*> &grads, std::vector<int> offsets, int count,
        std::vector<int> &rows,
        std::vector<int> &in_rows,
        std::vector<int> &cols,
        std::vector<dtype*> &input_grads);
void SubForward(std::vector<dtype*> &minuend,
        std::vector<dtype*> &subtrahend,
        int count,
        std::vector<int> &dims,
        std::vector<dtype*> &results);
void SubBackward(std::vector<dtype*> &losses, int count, std::vector<int> &dims,
        std::vector<dtype*> &minuend_losses,
        std::vector<dtype*> &subtrahend_losses);
void PMultiBackward(std::vector<dtype*> &losses,
        std::vector<dtype*> &in_vals1,
        std::vector<dtype*> &in_vals2,
        int count,
        int dim,
        std::vector<dtype*> &in_losses1,
        std::vector<dtype*> &in_losses2);
void PAddForward(std::vector<dtype*> &ins, int count, std::vector<int> &dims,
        int max_dim,
        int in_count,
        std::vector<dtype*> &vals,
        IntArray &dim_arr);
void PAddBackward(std::vector<dtype*> &grads, int count, int max_dim, int in_count,
        std::vector<dtype*> &in_grads,
        IntArray &dim_arr);
dtype CrossEntropyLoss(std::vector<dtype *> &vals,
        const std::vector<std::vector<int>> &answers,
        int count,
        int row,
        dtype factor,
        std::vector<dtype *> &grads);
dtype MultiCrossEntropyLoss(std::vector<dtype*> &vals, std::vector<std::vector<int> *> &answers,
        int count, int dim,
        dtype factor,
        std::vector<dtype*> &losses);
dtype KLCrossEntropyLoss(std::vector<dtype*> &vals,
        std::vector<std::vector<dtype> *> &answers,
        int count,
        int dim,
        dtype factor,
        std::vector<dtype*> &losses);
void MaxScalarForward(std::vector<dtype*> &inputs, int count, int head_count,
        std::vector<int> &head_dims,
        std::vector<dtype*> &results,
        std::vector<int> *max_indexes = nullptr);
void MaxScalarBackward(std::vector<dtype *> &losses, std::vector<int> &indexes,
        int count,
        std::vector<dtype*> &input_losses);
void VectorSumForward(std::vector<dtype *> &inputs, int count, int col,
        std::vector<int> &dims,
        std::vector<dtype*> &results);
void VectorSumBackward(std::vector<dtype*> &losses, int count, int col,
        std::vector<int> &dims,
        std::vector<dtype*> &input_losses);
void SoftmaxForward(std::vector<dtype *> &in_vals, int count, int *rows, int max_row,
        int *cols,
        int max_col,
        dtype **vals);
void SoftmaxBackward(std::vector<dtype *> &grads, dtype **vals, int count,
        int *rows, int max_row,
        int *cols,
        int max_col,
        int *offsets,
        int dim_sum,
        std::vector<dtype *> &in_grads);
void LogSoftmaxForward(dtype **in_vals, int count, int row, int *cols, int max_col, dtype **vals);
void LogSoftmaxBackward(dtype **grads, dtype **vals, int count, int row, int *cols, int max_col,
        dtype **in_grads);
void ScaledForward(std::vector<dtype *> &in_vals, int count, std::vector<int> &dims,
        std::vector<dtype> &factors,
        std::vector<dtype *> &vals);
void ScaledBackward(std::vector<dtype *> &grads, int count, std::vector<int> &dims,
        std::vector<dtype> &factors,
        std::vector<dtype *> &in_grads);
void ScalarToVectorForward(std::vector<dtype*> &inputs, int count, int input_col,
        std::vector<int> &rows,
        std::vector<dtype*> &results);
void ScalarToVectorBackward(std::vector<dtype*> &losses, int count, int input_col,
        std::vector<int> &rows,
        std::vector<dtype*> &input_losses);
void BiasForward(std::vector<dtype*> &in_vals, dtype *bias, int count,
        int dim,
        std::vector<dtype *> &vals);
void BiasBackward(std::vector<dtype *> &losses, int count, int dim,
        dtype *bias_loss,
        std::vector<dtype *> input_losses);
void StandardLayerNormForward(dtype **in_vals, int count, int row, int *cols,
        int max_col,
        dtype **vals,
        dtype *sds);
void StandardLayerNormBackward(dtype **grads, int count, int row, int *cols,
        int col_sum,
        int max_col,
        int *col_offsets,
        int *dims,
        int *dim_offsets,
        dtype **vals,
        dtype *sds,
        dtype **in_grads);
void PointwiseLinearForward(dtype **in_vals, int count, int row, int *cols,
        int max_col, dtype *g,
        dtype *b,
        dtype **vals);
void PointwiseLinearBackward(dtype **grads, dtype **in_vals,
        dtype *g_vals, int count, int row,
        int *cols,
        int max_col,
        int col_sum,
        int *dims,
        int *dim_offsets,
        dtype **in_grads,
        dtype *g_grads,
        dtype *bias_grads);
void BroadcastForward(dtype **in_vals, int count, int in_dim, int *ns, int max_n, dtype **vals);
void BroadcastBackward(dtype **grads, int count, int in_dim, int *ns, int max_n, dtype **in_grads);
std::vector<std::vector<int>> Predict(std::vector<dtype*> &vals, int count, std::vector<int> &cols,
        int row);

void ParamForward(dtype *param, int size, dtype *val);
void ParamBackward(dtype *grad, int size, dtype *param_grad);

int Predict(dtype* val, int dim);
void Max(dtype **v, int count, int dim, int *max_indexes, dtype *max_vals);
std::pair<dtype, std::vector<int>> SoftMaxLoss(
        std::vector<dtype *> &vals_vector,
        int count,
        int dim,
        std::vector<int> &gold_answers,
        int batchsize,
        std::vector<dtype *> &losses_vector);
dtype SquareSum(dtype *v, int len);
dtype SquareSum(dtype *v, bool *indexers, int count, int dim);
void Rescale(dtype *v, int len, dtype scale);
void UpdateAdam(dtype *val, dtype *grad, int row, int col, bool is_bias,
        dtype *aux_mean,
        dtype *aux_square,
        int iter,
        dtype belta1,
        dtype belta2,
        dtype alpha,
        dtype reg,
        dtype eps);
void UpdateAdam(dtype *val, dtype *grad, int row, int col,
        dtype *aux_mean,
        dtype *aux_square,
        bool *indexers,
        int *iters,
        dtype belta1,
        dtype belta2,
        dtype alpha,
        dtype reg,
        dtype eps);
void UpdateAdamW(dtype *val, dtype *grad, int row, int col, bool is_bias,
        dtype *aux_mean,
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
        bool *indexers,
        dtype alpha,
        dtype reg,
        dtype eps);
void *GraphHostAlloc();

}
}

#endif
#endif
