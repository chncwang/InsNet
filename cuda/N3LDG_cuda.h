#ifndef N3LDG_CUDA_N3LDG_CUDA_H
#define N3LDG_CUDA_N3LDG_CUDA_H

#include "Def.h"

#include "Memory_cuda.h"
#include "host_allocate.h"
#include <iostream>
#include <cassert>
#include <helper_cuda.h>
#include <vector>
#include <cmath>

using std::vector;

namespace n3ldg_cuda {

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

    void init(const T *host_arr, int len, cudaStream_t *stream);
    void init(int len);
    ~GPUArray();

    PageLockedVector<T> toCpu(cudaStream_t *stream) const;
};

cudaError_t MyCudaMemcpy(void *dest, const void *src, size_t count, cudaMemcpyKind kind,
        cudaStream_t *stream);
void CallCuda(cudaError_t status);

template <typename T>
void GPUArray<T>::init(const T *host_arr, int len, cudaStream_t *stream) {
    if (value != nullptr) {
        CallCuda(DeviceMemoryPool::Ins().Free(value));
        value = nullptr;
    }
    CallCuda(DeviceMemoryPool::Ins().Malloc((void**)&value, len * sizeof(T)));
    CallCuda(MyCudaMemcpy(value, host_arr, len * sizeof(T), cudaMemcpyHostToDevice, stream));
    this->len = len;
}

template <typename T>
void GPUArray<T>::init(int len) {
    if (value != nullptr) {
        CallCuda(DeviceMemoryPool::Ins().Free(value));
        value = nullptr;
    }
    CallCuda(DeviceMemoryPool::Ins().Malloc((void**)&value, len * sizeof(T)));
    this->len = len;
}

template <typename T>
PageLockedVector<T> GPUArray<T>::toCpu(cudaStream_t *stream) const {
    PageLockedVector<T> result;
    result.resize(len);
    if (value == nullptr) {
        cerr << "GPUArray::toCpu - value is nullptr" << endl;
        abort();
    }
    CallCuda(MyCudaMemcpy(result.data(), value, sizeof(T) * len, cudaMemcpyDeviceToHost, stream));
    return result;
}

template <typename T>
GPUArray<T>::~GPUArray() {
    if (value != nullptr) {
        CallCuda(DeviceMemoryPool::Ins().Free(value));
        value = nullptr;
    }
}

typedef GPUArray<dtype> NumberArray;
typedef GPUArray<bool> BoolArray;
typedef GPUArray<const bool *> BoolPointerArray;
typedef GPUArray<const dtype *> NumberPointerArray;
typedef GPUArray<const dtype *const *> NumberPointerPointerArray;
typedef GPUArray<int> IntArray;
typedef GPUArray<const int *> IntPointerArray;

template <typename T>
void deleteGPUArray(void *userdata) {
    GPUArray<T> *arr = static_cast<GPUArray<T> *>(userdata);
    delete arr;
}

template<typename T>
struct DeviceValue {
    T *value = nullptr;
    T v = 0;

    DeviceValue() = default;
    DeviceValue(DeviceValue &&) {
        abort();
    }
    DeviceValue(const DeviceValue&) {
        abort();
    }

    void init() {
        if (value != nullptr) {
            CallCuda(DeviceMemoryPool::Ins().Free(value));
            value = nullptr;
        }
        CallCuda(DeviceMemoryPool::Ins().Malloc((void**)&value, sizeof(T)));
    }

    void copyFromDeviceToHost(cudaStream_t *stream) {
        CallCuda(MyCudaMemcpy(&v, value, sizeof(T), cudaMemcpyDeviceToHost, stream));
    }

    void copyFromHostToDevice(cudaStream_t *stream) {
        CallCuda(MyCudaMemcpy(value, &v, sizeof(T), cudaMemcpyHostToDevice, stream));
    }

    ~DeviceValue() {
        if (value != NULL) {
            CallCuda(DeviceMemoryPool::Ins().Free(value));
        }
    }
};

typedef DeviceValue<dtype> DeviceNumber;
typedef DeviceValue<int> DeviceInt;

template<typename T>
void deleteDeviceValue(void *userdata) {
    DeviceValue<T> *v = static_cast<DeviceValue<T>*>(userdata);
    delete v;
}

bool Verify(bool *host, bool *device, int len, const char* message);
bool Verify(int *host, int *device, int len, const char* message);

void Assert(bool v, const std::string &message = "",
        const std::function<void(void)> &call = []() {});
void Memset(dtype *p, int len, dtype value, cudaStream_t *stream);
void Memset(bool *p, int len, bool value, cudaStream_t *stream);
void *Malloc(int size);
void BatchMemset(const PageLockedVector<dtype*> &vec, int count, const PageLockedVector<int> &dims, dtype value,
        cudaStream_t *stream);
void PrintNums(const dtype* p, int len);
void PrintInts(const int* p, int len);

void InitCuda(int device_id = 0, float memory_in_gb = 0.0f);
void EndCuda();

void CopyFromMultiVectorsToOneVector(const PageLockedVector<dtype*> &src, dtype *dest, int count,
        int len,
        cudaStream_t *stream);
void CopyFromOneVectorToMultiVals(const dtype *src, PageLockedVector<dtype*> &vals,
        int count,
        int len,
        cudaStream_t *stream);
enum PoolingEnum {
    MAX,
    MIN,
    SUM,
    AVG
};

void ActivationForward(ActivatedEnum activated, const PageLockedVector<const dtype*> &xs,
        int count,
        const PageLockedVector<int> &dims,
        PageLockedVector<dtype*> &ys,
        cudaStream_t *stream);
void ActivationBackward(ActivatedEnum activated, const PageLockedVector<const dtype*> &losses,
        const PageLockedVector<dtype*> &vals,
        int count,
        const PageLockedVector<int> &dims,
        PageLockedVector<dtype*> &in_losses,
        cudaStream_t *stream);
void DropoutForward(const PageLockedVector<dtype*> &xs, int count, int dim,
        bool is_training,
        const dtype *drop_mask,
        dtype drop_factor,
        PageLockedVector<dtype*> &ys,
        cudaStream_t *stream);
void DropoutBackward(const PageLockedVector<dtype*> &losses,
        const PageLockedVector<dtype*> &vals,
        int count,
        int dim,
        bool is_training,
        const dtype *drop_mask,
        dtype drop_factor,
        PageLockedVector<dtype*> &in_losses,
        cudaStream_t *stream);
void BucketForward(const PageLockedVector<dtype> input, int count, int dim, PageLockedVector<dtype*> &ys,
        cudaStream_t *stream);
void CopyForUniNodeForward(const n3ldg_cuda::PageLockedVector<dtype*> &xs, const dtype* b,
        dtype* xs_dest,
        dtype* b_dest,
        int count,
        int x_len,
        int b_len,
        bool use_b,
        cudaStream_t *stream);
void MatrixMultiplyMatrix(dtype *W, dtype *x, dtype *y, int row, int col,
        int count,
        bool useb,
        cudaStream_t *stream,
        bool should_x_transpose = false,
        bool should_W_transpose = false);
void AddLtyToParamBiasAndAddLxToInputLossesForUniBackward(const dtype *lty, const dtype *lx,
        dtype *b,
        PageLockedVector<dtype*> &losses,
        int count,
        int out_dim,
        int in_dim,
        bool use_b,
        cudaStream_t *stream);
void AddLtyToParamBiasAndAddLxToInputLossesForBiBackward(const dtype *lty,
        const dtype *lx1,
        const dtype *lx2,
        dtype *b,
        PageLockedVector<dtype*> &losses1,
        PageLockedVector<dtype*> &losses2,
        int count,
        int out_dim,
        int in_dim1,
        int in_dim2,
        bool use_b,
        cudaStream_t *stream);
void CalculateDropoutMask(dtype dropout_ratio, int count, int dim, dtype *mask);
void ConcatForward(const PageLockedVector<dtype*> &in_vals,
        const PageLockedVector<int> &in_dims,
        PageLockedVector<dtype*> &vals,
        int count,
        int in_count,
        int out_dim,
        cudaStream_t *stream);
void ConcatBackward(const PageLockedVector<dtype*> &in_losses,
        const PageLockedVector<int> &in_dims,
        PageLockedVector<dtype*> &losses,
        int count,
        int in_count,
        int out_dim,
        cudaStream_t *stream);
void ScalarConcatForward(const PageLockedVector<dtype *> &ins, int count, const PageLockedVector<int> &dims,
        int max_dim,
        const PageLockedVector<dtype *> &results,
        cudaStream_t *stream);
void ScalarConcatBackward(const PageLockedVector<dtype *> &losses, int count, const PageLockedVector<int> &dims,
        int max_dim,
        const PageLockedVector<dtype *> in_losses,
        cudaStream_t *stream);
void LookupForward(const PageLockedVector<int> &xids, const dtype *vocabulary,
        int count,
        int dim,
        PageLockedVector<dtype*> &vals,
        cudaStream_t *stream);
void LookupBackward(const PageLockedVector<int> &xids, int unknown_id,
        bool fine_tune,
        const PageLockedVector<dtype*> &losses,
        int count,
        int dim,
        dtype *grad,
        bool *indexers,
        cudaStream_t *stream);
void LookupBackward(const PageLockedVector<int> &xids, int unknown_id,
        bool fine_tune,
        const PageLockedVector<dtype*> &losses,
        int count,
        int dim,
        dtype *grad,
        cudaStream_t *stream);
void ParamRowForward(const dtype *param, int row_index, int param_row_count, int count, int dim,
        PageLockedVector<dtype*> &vals,
        cudaStream_t *stream);
void PoolForward(PoolingEnum pooling, const PageLockedVector<dtype*> &in_vals,
        PageLockedVector<dtype*> &vals,
        int count,
        const PageLockedVector<int> &in_counts,
        int dim,
        int *hit_inputs,
        cudaStream_t *stream);
void PoolBackward(const PageLockedVector<dtype*> &losses,
        PageLockedVector<dtype*> &in_losses,
        const PageLockedVector<int> &in_counts,
        const int *hit_inputs,
        int count,
        int dim,
        cudaStream_t *stream);
void SumPoolForward(PoolingEnum pooling, const PageLockedVector<dtype*> &in_vals,
        int count,
        int dim,
        const PageLockedVector<int> &in_counts,
        PageLockedVector<dtype*> &vals,
        cudaStream_t *stream);
void SumPoolBackward(PoolingEnum pooling, const PageLockedVector<dtype*> &losses,
        const PageLockedVector<int> &in_counts,
        int count,
        int dim,
        PageLockedVector<dtype*> &in_losses,
        cudaStream_t *stream);
void PMultiForward(const PageLockedVector<dtype*> &ins1,
        const PageLockedVector<dtype*> &ins2,
        int count,
        int dim,
        PageLockedVector<dtype*> &vals,
        cudaStream_t *stream);
void DivForward(const PageLockedVector<const dtype*> numerators, const PageLockedVector<const dtype*> denominators,
        int count,
        const PageLockedVector<int> &dims,
        PageLockedVector<dtype*> &results,
        cudaStream_t *stream);
void DivBackward(const PageLockedVector<const dtype*> &losses, const PageLockedVector<const dtype*> &denominator_vals,
        const PageLockedVector<const dtype*> &numerator_vals,
        int count,
        const PageLockedVector<int> &dims,
        PageLockedVector<dtype*> &numerator_losses,
        PageLockedVector<dtype*> &denominator_losses,
        cudaStream_t *stream);
void SplitForward(const PageLockedVector<const dtype*> &inputs, const PageLockedVector<int> &offsets,
        int count,
        int dim,
        PageLockedVector<dtype*> &results,
        cudaStream_t *stream);
void SplitBackward(const PageLockedVector<const dtype*> &losses, const PageLockedVector<int> offsets,
        int count,
        int dim,
        const PageLockedVector<dtype*> &input_losses,
        cudaStream_t *stream);
void SubForward(const PageLockedVector<const dtype*> &minuend,
        const PageLockedVector<const dtype*> &subtrahend,
        int count,
        const PageLockedVector<int> &dims,
        PageLockedVector<dtype*> &results,
        cudaStream_t *stream);
void SubBackward(const PageLockedVector<const dtype*> &losses, int count, const PageLockedVector<int> &dims,
        PageLockedVector<dtype*> &minuend_losses,
        PageLockedVector<dtype*> &subtrahend_losses,
        cudaStream_t *stream);
void PMultiBackward(const PageLockedVector<dtype*> &losses,
        const PageLockedVector<dtype*> &in_vals1,
        const PageLockedVector<dtype*> &in_vals2,
        int count,
        int dim,
        PageLockedVector<dtype*> &in_losses1,
        PageLockedVector<dtype*> &in_losses2,
        cudaStream_t *stream);
void PDotForward(const PageLockedVector<dtype*> &ins1,
        const PageLockedVector<dtype*> &ins2,
        int count,
        int dim,
        PageLockedVector<dtype*> &vals,
        cudaStream_t *stream);
void PDotBackward(const PageLockedVector<dtype*> &losses,
        const PageLockedVector<dtype*> &in_vals1,
        const PageLockedVector<dtype*> &in_vals2,
        int count,
        int dim,
        PageLockedVector<dtype*> &in_losses1,
        PageLockedVector<dtype*> &in_losses2,
        cudaStream_t *stream);
void PAddForward(const PageLockedVector<PageLockedVector<dtype*>> &ins, int count,
        int dim,
        int in_count,
        PageLockedVector<dtype*> &vals,
        cudaStream_t *stream);
void PAddBackward(const PageLockedVector<dtype*> &losses, int count, int dim,
        int in_count,
        PageLockedVector<PageLockedVector<dtype*>> &in_losses,
        cudaStream_t *stream);
dtype CrossEntropyLoss(const PageLockedVector<dtype *> &vals, const PageLockedVector<int> &answers, int count,
        dtype batchsize,
        PageLockedVector<dtype *> &losses,
        cudaStream_t *stream);
dtype MultiCrossEntropyLoss(const PageLockedVector<dtype*> &vals, const PageLockedVector<PageLockedVector<int>> &answers,
        int count,
        int dim,
        dtype factor,
        const PageLockedVector<dtype*> &losses,
        cudaStream_t *stream);
dtype KLCrossEntropyLoss(const PageLockedVector<dtype*> &vals,
        const PageLockedVector<shared_ptr<PageLockedVector<dtype>>> &answers,
        int count,
        int dim,
        dtype factor,
        const PageLockedVector<dtype*> &losses,
        cudaStream_t *stream);
void MaxScalarForward(const PageLockedVector<const dtype*> &inputs, int count, const PageLockedVector<int> &dims,
        PageLockedVector<dtype*> &results,
        PageLockedVector<int> &max_indexes,
        cudaStream_t *stream);
void MaxScalarBackward(const PageLockedVector<const dtype *> &losses, const PageLockedVector<int> &indexes,
        int count,
        const PageLockedVector<dtype*> &input_losses,
        cudaStream_t *stream);
void VectorSumForward(const PageLockedVector<const dtype *> &inputs, int count, const PageLockedVector<int> &dims,
        PageLockedVector<dtype*> &results,
        cudaStream_t *stream);
void VectorSumBackward(const PageLockedVector<const dtype*> &losses, int count, const PageLockedVector<int> &dims,
        PageLockedVector<dtype*> &input_losses,
        cudaStream_t *stream);
void ScalarToVectorForward(const PageLockedVector<const dtype*> &inputs, int count, const PageLockedVector<int> &dims,
        PageLockedVector<dtype*> &results,
        cudaStream_t *stream);
void ScalarToVectorBackward(const PageLockedVector<const dtype*> &losses, int count, const PageLockedVector<int> &dims,
        PageLockedVector<dtype*> &input_losses,
        cudaStream_t *stream);
void BiasForward(const PageLockedVector<dtype*> &in_vals, const dtype *bias, int count, int dim,
        const PageLockedVector<dtype *> &vals,
        cudaStream_t *stream);
void BiasBackward(const PageLockedVector<dtype *> &losses, int count, int dim, dtype *bias_loss,
        const PageLockedVector<dtype *> input_losses,
        cudaStream_t *stream);
PageLockedVector<int> Predict(const PageLockedVector<dtype*> &vals, int count, int dim,
        cudaStream_t *stream);
//int Predict(const dtype* val, int dim);
void Max(const dtype *const *v, int count, int dim, int *max_indexes, dtype *max_vals,
        cudaStream_t *stream);
dtype SquareSum(const dtype *v, int len, cudaStream_t *stream);
dtype SquareSum(const dtype *v, const bool *indexers, int count, int dim, cudaStream_t *stream);
void Rescale(dtype *v, int len, dtype scale, cudaStream_t *stream);
void UpdateAdam(dtype *val, dtype *grad, int row, int col, bool is_bias, dtype *aux_mean,
        dtype *aux_square,
        int iter,
        dtype belta1,
        dtype belta2,
        dtype alpha,
        dtype reg,
        dtype eps,
        cudaStream_t *stream);
void UpdateAdam(dtype *val, dtype *grad, int row, int col, dtype *aux_mean,
        dtype *aux_square,
        const bool *indexers,
        int *iters,
        dtype belta1,
        dtype belta2,
        dtype alpha,
        dtype reg,
        dtype eps,
        cudaStream_t *stream);
void UpdateAdamW(dtype *val, dtype *grad, int row, int col, bool is_bias, dtype *aux_mean,
        dtype *aux_square,
        int iter,
        dtype belta1,
        dtype belta2,
        dtype alpha,
        dtype reg,
        dtype eps,
        cudaStream_t *stream);
void UpdateAdagrad(dtype *val, dtype *grad, int row, int col,
        dtype *aux_square,
        dtype alpha,
        dtype reg,
        dtype eps,
        cudaStream_t *stream);
void UpdateAdagrad(dtype *val, dtype *grad, int row, int col,
        dtype *aux_square,
        const bool *indexers,
        dtype alpha,
        dtype reg,
        dtype eps,
        cudaStream_t *stream);
void *GraphHostAlloc();
}

#endif
