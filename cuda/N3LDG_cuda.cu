#include "N3LDG_cuda.h"
#include <array>
#include <boost/format.hpp>
#include <cstdlib>
#include <cstddef>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cublas_v2.h>
#include "Printf_cuda.cuh"
#include "Printf_cuda.cu"
#include "Memory_cuda.h"
#include <curand.h>
#include <curand_kernel.h>
#include "cnmem.h"
#include <string>
#include <utility>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <thread>
#include <numeric>
#include <memory>
#include "profiler.h"
#include "MyTensor-def.h"

namespace n3ldg_cuda {

using namespace std;
using boost::format;

#if USE_FLOAT
#define cuda_sqrt(x) sqrtf(x)
#define cuda_pow(x, y) powf(x, y)
#define cuda_tanh(x) tanhf(x)
#define cuda_exp(x) __expf(x)
#define cuda_log(x) logf(x)
#else
#define cuda_sqrt(x) sqrt(x)
#define cuda_pow(x, y) pow(x, y)
#define cuda_tanh(x) tanh(x)
#define cuda_exp(x) exp(x)
#define cuda_log(x) log(x)
#endif

#define KERNEL_LOG

#ifdef KERNEL_LOG
#define  KernelPrintLine(format, ...)\
{\
    cuPrintf("block:x=%d,y=%d thread:x=%d,y=%d "#format"\n", blockIdx.x,\
            blockIdx.y, threadIdx.x, threadIdx.y,__VA_ARGS__);\
}
#else
#define KernelPrintLine(format, ...)
#endif

constexpr int TPB = 1024;
constexpr int BLOCK_COUNT = 56;

void CallCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        cerr << "cuda error:" << cudaGetErrorString(status) << endl;
        abort();
    }
}

void CheckCudaError() {
    //cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "cuda error:" << cudaGetErrorName(error) << std::endl;
        std::cerr << "cuda error:" << cudaGetErrorString(error) << std::endl;
        abort();
    }
}

void CallCnmem(cnmemStatus_t status) {
    assert(status == CNMEM_STATUS_SUCCESS);
}

void CallCublas(cublasStatus_t status) {
    //cudaDeviceSynchronize();
    if (status != CUBLAS_STATUS_SUCCESS) {
        cerr << "cublas status:" << status << endl;
        abort();
    }
}

void CallCurand(curandStatus status) {
    assert(status == CURAND_STATUS_SUCCESS);
}

cublasHandle_t& GetCublasHandle(cudaStream_t *stream) {
    static cublasHandle_t *handle;
    if (handle == nullptr) {
        handle = new cublasHandle_t;
        CallCublas(cublasCreate(handle));
    }
    CallCublas(cublasSetStream(*handle, *stream));
    return *handle;
}

cudaError_t MyCudaMemcpy(void *dest, const void *src, size_t count, cudaMemcpyKind kind,
        cudaStream_t *stream) {
    if (stream == nullptr) {
        cudaDeviceSynchronize();
    }
    cudaError_t e = stream == nullptr ? cudaMemcpyAsync(dest, src, count, kind) :
        cudaMemcpyAsync(dest, src, count, kind, *stream);
    CallCuda(e);
    //if (stream == nullptr) {
    cudaDeviceSynchronize();
    //}
    return e;
}

int NextTwoIntegerPowerNumber(int number) {
    int result = 1;
    while (number > result) {
        result <<= 1;
    }
    return result;
}

template <>
vector<bool> GPUArray<bool>::toCpu(cudaStream_t *stream) const {
    bool *cpu_arr = new bool[len];
    CallCuda(MyCudaMemcpy(cpu_arr, value, sizeof(bool) * len, cudaMemcpyDeviceToHost, stream));
    vector<bool> result;
    result.resize(len);
    for (int i = 0; i < len; ++i) {
        result.at(i) = cpu_arr[i];
    }
    delete[] cpu_arr;
    return result;
}

void DeviceInt::init() {
    if (value != NULL) {
        CallCuda(MemoryPool::Ins().Free(value));
        value = NULL;
    }
    CallCuda(MemoryPool::Ins().Malloc((void**)&value, sizeof(int)));
}

void DeviceInt::copyFromDeviceToHost(cudaStream_t *stream) {
    CallCuda(MyCudaMemcpy(&v, value, sizeof(int), cudaMemcpyDeviceToHost, stream));
}

void DeviceInt::copyFromHostToDevice(cudaStream_t *stream) {
    CallCuda(MyCudaMemcpy(value, &v, sizeof(int), cudaMemcpyHostToDevice, stream));
}

DeviceInt::~DeviceInt() {
    if (value != NULL) {
        CallCuda(MemoryPool::Ins().Free(value));
    }
}

void DeviceNumber::init() {
    if (value != NULL) {
        CallCuda(MemoryPool::Ins().Free(value));
        value = NULL;
    }
    CallCuda(MemoryPool::Ins().Malloc((void**)&value, sizeof(int)));
}

void DeviceNumber::copyFromDeviceToHost(cudaStream_t *stream) {
    CallCuda(MyCudaMemcpy(&v, value, sizeof(dtype), cudaMemcpyDeviceToHost, stream));
}

DeviceNumber::~DeviceNumber() {
    if (value != NULL) {
        CallCuda(MemoryPool::Ins().Free(value));
    }
}

void Tensor1D::init(int dim) {
    initOnDevice(dim);
#if TEST_CUDA
    v = new dtype[dim];
    zero();
#endif
}

void Tensor1D::initOnMemoryAndDevice(int dim) {
    initOnDevice(dim);
    v = new dtype[dim];
    zero();
}

void Tensor1D::initOnDevice(int dim) {
    CallCuda(MemoryPool::Ins().Malloc((void**)&value, dim * sizeof(dtype)));
    this->dim = dim;
}

void Tensor1D::initOnMemory(int len) {
    v = new dtype[dim];
    zero();
}

Tensor1D::Tensor1D(const Tensor1D &t) {
    dim = t.dim;
    memcpy(v, t.v, dim *sizeof(dtype));
    CallCuda(MyCudaMemcpy(value, t.value, dim * sizeof(dtype), cudaMemcpyDeviceToDevice, nullptr));
}

Tensor1D::~Tensor1D() {
    if (value != NULL) {
        CallCuda(MemoryPool::Ins().Free(value));
    }
}

void Tensor1D::print() const {
    cout << "dim:" << dim << endl;
    PrintNums(value, dim);
}

void Tensor1D::copyFromHostToDevice(void *stream) {
    assert(v != NULL);
    assert(value != NULL);
    CallCuda(MyCudaMemcpy(value, v, dim * sizeof(dtype), cudaMemcpyHostToDevice,
                (cudaStream_t*)stream));
}

void Tensor1D::copyFromDeviceToHost(void *stream) {
    CallCuda(MyCudaMemcpy(v, value, dim * sizeof(dtype), cudaMemcpyDeviceToHost,
                (cudaStream_t*)stream));
}

__device__ int DeviceDefaultIndex();
__device__ int DeviceDefaultStep();
int DefaultBlockCount(int len);

__global__ void KernelCheckIsNumber(const dtype *v, int dim, int *error) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *error = 0;
    }
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < dim; i += step) {
        if (v[i] != v[i]) {
            *error = 1;
            return;
        }
    }
}

void CheckIsNumber(const dtype *v, int dim) {
    int block_count = DefaultBlockCount(dim);
    DeviceInt error;
    error.init();
    KernelCheckIsNumber<<<block_count, TPB>>>(v, dim, error.value);
    CheckCudaError();
    error.copyFromDeviceToHost(nullptr);
    if (error.v != 0) {
        cerr << "nan checked!" << endl;
        abort();
    }
}

void Tensor1D::checkIsNumber() const {
    n3ldg_cuda::CheckIsNumber(value, dim);
}

void Tensor2D::initOnMemoryAndDevice(int row, int col) {
    initOnDevice(row, col);
    v = new dtype[row * col];
    zero();
}

void Tensor2D::init(int row, int col) {
    initOnDevice(row, col);
#if TEST_CUDA
    v = new dtype[row * col];
    zero();
#endif
}

void Tensor2D::initOnDevice(int row, int col) {
    CallCuda(MemoryPool::Ins().Malloc((void**)&value,
                row * col * sizeof(dtype)));
    this->row = row;
    this->col = col;
    this->size = row * col;
}

Tensor2D::Tensor2D(const Tensor2D &t) {
    row = t.row;
    col = t.col;
    memcpy(v, t.v, sizeof(dtype) * row * col);
    CallCuda(MyCudaMemcpy(value, t.value, sizeof(dtype) * row * col,
                cudaMemcpyDeviceToDevice, nullptr));
}

Tensor2D::~Tensor2D() {
    if (value != NULL) {
        CallCuda(MemoryPool::Ins().Free(value));
    }
}

void Tensor2D::print() const {
    cout << "row:" << row << " col:" << col << endl;
    PrintNums(value, size);
}

void Tensor2D::copyFromHostToDevice(void *stream) {
    CallCuda(MyCudaMemcpy(value, v, size * sizeof(dtype), cudaMemcpyHostToDevice,
                (cudaStream_t*)stream));
}

void Tensor2D::copyFromDeviceToHost(void *stream) {
    CallCuda(MyCudaMemcpy(v, value, size * sizeof(dtype), cudaMemcpyDeviceToHost,
                (cudaStream_t*)stream));
}

void Assert(bool v, const std::string &message, const function<void(void)> &call) {
#if TEST_CUDA
    if (!v) {
        std::cerr << message << std::endl;
        call();
        abort();
    }
#endif
}

__device__ void DeviceAtomicAdd(dtype* address, dtype value) {
    float old = value;  
    float new_old;
    do {
        new_old = atomicExch(address, 0.0);
        new_old += old;
    } while ((old = atomicExch(address, new_old))!=0.0);
};

__device__ dtype cuda_dexp(dtype y) {
    return y;
}

__device__ dtype cuda_dtanh(dtype y) {
    return 1.0f - y * y;
}

__device__ dtype cuda_sigmoid(dtype x) {
    return 1.0f / (1.0f + cuda_exp(-x));
}

__device__ dtype cuda_dsigmoid(dtype y) {
    return y * (1.0f - y);
}

__device__ dtype cuda_relu(dtype x) {
    return x > 0.0f ? x : 0.0f;
}

__device__ dtype cuda_drelu(dtype x) {
    return x > 0.0f ? 1 : 0.0f;
}

__device__ dtype cuda_leaky_relu(dtype x) {
    return x > 0.0f ? x : -0.1f * x;
}

__device__ dtype cuda_dleaky_relu(dtype x) {
    return x > 0.0f ? 1.0f : -0.1f;
}

const dtype SELU_LAMBDA = 1.0507009873554804934193349852946;
const dtype SELU_ALPHA = 1.6732632423543772848170429916717;

__device__ dtype cuda_selu(dtype x) {
    return x <= 0.0f ? SELU_LAMBDA * SELU_ALPHA * (cuda_exp(x) - 1.0f) :
        SELU_LAMBDA * x;
}

__device__ dtype cuda_dselu(dtype x, dtype y) {
    return x <= 0.0f ? SELU_LAMBDA * SELU_ALPHA + y : SELU_LAMBDA;
}

__device__ int DeviceDefaultIndex() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ int DeviceDefaultStep() {
    return gridDim.x * blockDim.x;
}

__device__ dtype DeviceAbs(dtype d) {
    return d > 0 ? d : -d;
}

int DefaultBlockCount(int len) {
    int block_count = (len - 1 + TPB) /
        TPB;
    return std::min(block_count, BLOCK_COUNT);
}

int DefaultBlockCountWithoutLimit(int len) {
    return (len - 1 + TPB) / TPB;
}

//__global__ void KernelZero(dtype *v, int len) {
//    int index = blockIdx.x * blockDim.x + threadIdx.x;
//    if (index >= len) {
//        return;
//    }
//    v[index] = 0;
//}
//
//void Zero(dtype *v, int len) {
//    int block_count = (len - 1 + TPB) /
//        TPB;
//    KernelZero<<<block_count, TPB>>>(v, len);
//    CheckCudaError();
//}

__global__ void PrintPointers(void **p, int len) {
    for (int i = 0; i < len; ++i) {
        printf("%p\n", p[i]);
    }
}

__global__ void KernelPrintNums(const dtype* p, int len) {
    for (int i = 0; i < len; ++i) {
        printf("%d %f\n", i, p[i]);
    }
}

void PrintNums(const dtype* p, int len) {
    KernelPrintNums<<<1, 1>>>(p, len);
    cudaDeviceSynchronize();
    CheckCudaError();
}

__global__ void KernelPrintNums(const dtype *const *p, int index, int len) {
    for (int i = 0; i < len; ++i) {
        printf("%d %f\n", i, p[index][i]);
    }
}

void PrintNums(const dtype *const *p, int count_i, int len) {
    KernelPrintNums<<<1, 1>>>(p, count_i, len);
    cudaDeviceSynchronize();
    CheckCudaError();
}

__global__ void KernelPrintInts(const int* p, int len) {
    for (int i = 0; i < len; ++i) {
        printf("%d\n", p[i]);
    }
}

void PrintInts(const int* p, int len) {
    KernelPrintInts<<<1, 1>>>(p, len);
    cudaDeviceSynchronize();
    CheckCudaError();
}

void InitCuda(int device_id, float memory_in_gb) {
    std::cout << "device_id:" << device_id << std::endl;
    CallCuda(cudaSetDeviceFlags(cudaDeviceMapHost));

#if DEVICE_MEMORY == 0
    cnmemDevice_t device;
    device.size = 10000000000;
    device.device = device_id;
    cnmemInit(1, &device, CNMEM_FLAGS_DEFAULT);
#else
    CallCuda(cudaSetDevice(device_id));
#endif
    CallCuda(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
    CallCuda(cudaPrintfInit());
    MemoryPool::Ins().Init(memory_in_gb);
}

void EndCuda() {
    cudaPrintfEnd();
    Profiler::Ins().Print();
}

__global__ void KernelCopyFromOneVectorToMultiVectors(const dtype *src,
        dtype *const *dest, int count, int len) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count * len; i += step) {
        int count_i = i / len;
        int len_i = i % len;
        dest[count_i][len_i] = src[i];
    }
}

void CopyFromOneVectorToMultiVals(const dtype *src, std::vector<dtype*> &vals,
        int count,
        int len,
        cudaStream_t *stream) {
    NumberPointerArray val_arr;
    val_arr.init((dtype**)vals.data(), vals.size(), stream);
    int block_count = (len * count - 1 + TPB) / TPB;
    block_count = std::min(block_count, BLOCK_COUNT);
    KernelCopyFromOneVectorToMultiVectors<<<block_count, TPB, 0, *stream>>>(src,
            (dtype *const *)val_arr.value, count, len);
    CheckCudaError();
}

__global__ void KernelCopyFromMultiVectorsToOneVector(const dtype **src, dtype *dest, int count,
        int len) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count * len; i += step) {
        int count_i = i / len;
        int len_i = i % len;
        dest[i] = src[count_i][len_i];
    }
}

void CopyFromMultiVectorsToOneVector(const std::vector<dtype*> &src,
        dtype *dest,
        int count,
        int len,
        cudaStream_t *stream) {
    NumberPointerArray src_arr;
    src_arr.init((dtype**)src.data(), src.size(), stream);
    int block_count = DefaultBlockCount(len * count);
    KernelCopyFromMultiVectorsToOneVector<<<block_count, TPB, 0, *stream>>>(
            (const dtype**)src_arr.value, dest, count, len);
    CheckCudaError();
}

__global__ void KernelActivationForward(ActivatedEnum activated, const dtype *const *xs,
        int count,
        int *dims,
        int max_dim,
        dtype *const *ys) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < max_dim * count; i += step) {
        int count_i = i / max_dim;
        int dim_i = i % max_dim;
        if (dim_i < dims[count_i]) {
            if (activated == ActivatedEnum::TANH) {
                ys[count_i][dim_i] = cuda_tanh(xs[count_i][dim_i]);
            } else if (activated == ActivatedEnum::SIGMOID) {
                ys[count_i][dim_i] = cuda_sigmoid(xs[count_i][dim_i]);
            } else if (activated == ActivatedEnum::EXP) {
                ys[count_i][dim_i] = cuda_exp(xs[count_i][dim_i]);
            } else if (activated == ActivatedEnum::RELU) {
                ys[count_i][dim_i] = cuda_relu(xs[count_i][dim_i]);
            } else {
                printf("KernelActivationForward - error enum\n");
                assert(false);
            }
        }
    }
}

void ActivationForward(ActivatedEnum activated, const std::vector<const dtype*> &xs,
        int count,
        const vector<int> &dims,
        std::vector<dtype*> &ys,
        cudaStream_t *stream) {
    int max_dim = *max_element(dims.begin(), dims.end());
    NumberPointerArray x_arr, y_arr;
    x_arr.init((dtype**)xs.data(), xs.size(), stream);
    y_arr.init((dtype**)ys.data(), ys.size(), stream);
    int block_count = DefaultBlockCount(count * max_dim);

    IntArray dim_arr;
    dim_arr.init(dims.data(), dims.size(), stream);

    KernelActivationForward<<<block_count, TPB, 0, *stream>>>(activated,
            (const dtype* const *)x_arr.value, count, dim_arr.value, max_dim,
            (dtype *const *)y_arr.value);
    CheckCudaError();
}

__global__ void KernelActivationBackward(ActivatedEnum activated,
        const dtype *const *losses,
        const dtype *const *vals,
        int count,
        int *dims,
        int max_dim,
        dtype* const* in_losses) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < max_dim * count; i += step) {
        int count_i = i / max_dim;
        int dim_i = i % max_dim;
        if (dim_i < dims[count_i]) {
            dtype l;
            if (activated == ActivatedEnum::TANH) {
                l = cuda_dtanh(vals[count_i][dim_i]);
            } else if (activated == ActivatedEnum::SIGMOID) {
                l = cuda_dsigmoid(vals[count_i][dim_i]);
            } else if (activated == ActivatedEnum::EXP) {
                l = cuda_dexp(vals[count_i][dim_i]);
            } else if (activated == ActivatedEnum::RELU) {
                l = cuda_drelu(vals[count_i][dim_i]);
            } else {
                printf("KernelActivationBackward - error enum\n");
                assert(false);
            }
            dtype v = l * losses[count_i][dim_i];
            DeviceAtomicAdd(in_losses[count_i] + dim_i, v);
        }
    }
}

void ActivationBackward(ActivatedEnum activated, const std::vector<const dtype*> &losses,
        const std::vector<dtype*> &vals,
        int count,
        const vector<int> &dims,
        std::vector<dtype*> &in_losses,
        cudaStream_t *stream) {
    int max_dim = *max_element(dims.begin(), dims.end());
    NumberPointerArray loss_arr, val_arr, in_loss_arr;
    loss_arr.init((dtype**)losses.data(), losses.size(), stream);
    val_arr.init((dtype**)vals.data(), vals.size(), stream);
    in_loss_arr.init((dtype**)in_losses.data(), in_losses.size(), stream);
    int block_count = DefaultBlockCount(count * max_dim);
    IntArray dim_arr;
    dim_arr.init(dims.data(), dims.size(), stream);
    KernelActivationBackward<<<block_count, TPB, 0, *stream>>>(activated, loss_arr.value,
            val_arr.value, count, dim_arr.value, max_dim, (dtype *const *)in_loss_arr.value);
    CheckCudaError();
}

__global__ void KernelDropoutForward(const dtype *const *xs, int count, int dim,
        bool is_training,
        const dtype* drop_mask,
        dtype drop_factor,
        dtype *const *ys) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < dim * count; i += step) {
        int count_i = i / dim;
        int dim_i = i % dim;
        if (is_training) {
            if (drop_mask[i] < drop_factor) {
                ys[count_i][dim_i] = 0.0f;
            } else {
                ys[count_i][dim_i] = xs[count_i][dim_i];
            }
        } else {
            ys[count_i][dim_i] = (1 - drop_factor) * xs[count_i][dim_i];
        }
    }
}

void DropoutForward(const std::vector<dtype*> &xs, int count, int dim,
        bool is_training,
        const dtype *drop_mask,
        dtype drop_factor,
        std::vector<dtype*> &ys,
        cudaStream_t *stream) {
    if (drop_factor < 0 || drop_factor >= 1.0f) {
        std::cerr << "drop value is " << drop_factor << std::endl;
        abort();
    }
    NumberPointerArray x_arr, y_arr;
    x_arr.init((dtype**)xs.data(), xs.size(), stream);
    y_arr.init((dtype**)ys.data(), ys.size(), stream);
    int block_count = DefaultBlockCount(count * dim);
    KernelDropoutForward<<<block_count, TPB, 0, *stream>>>(x_arr.value, count, dim, is_training,
            drop_mask, drop_factor, (dtype *const *)y_arr.value);
    CheckCudaError();
}

__global__ void KernelDropoutBackward(const dtype *const *losses, const dtype *const *vals,
        int count,
        int dim,
        bool is_training,
        const dtype* drop_mask,
        dtype drop_factor,
        dtype *const *in_losses) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < dim * count; i += step) {
        int count_i = i / dim;
        int dim_i = i % dim;
        if (is_training) {
            if (drop_mask[i] >= drop_factor) {
                DeviceAtomicAdd(in_losses[count_i] + dim_i, losses[count_i][dim_i]);
            }
        } else {
            DeviceAtomicAdd(in_losses[count_i] + dim_i,
                    (1 - drop_factor) * losses[count_i][dim_i]);
        }
    }
}

void DropoutBackward(const std::vector<dtype*> &losses,
        const std::vector<dtype*> &vals,
        int count,
        int dim,
        bool is_training,
        const dtype *drop_mask,
        dtype drop_factor,
        std::vector<dtype*> &in_losses,
        cudaStream_t *stream) {
    if (drop_factor < 0 || drop_factor >= 1) {
        std::cerr << "drop value is " << drop_factor << std::endl;
        abort();
    }
    NumberPointerArray loss_arr, val_arr, in_loss_arr;
    loss_arr.init((dtype**)losses.data(), losses.size(), stream);
    val_arr.init((dtype**)vals.data(), vals.size(), stream);
    in_loss_arr.init((dtype**)in_losses.data(), in_losses.size(), stream);
    int block_count = DefaultBlockCount(count * dim);
    KernelDropoutBackward<<<block_count, TPB, 0, *stream>>>(loss_arr.value, val_arr.value, count,
            dim, is_training, drop_mask, drop_factor, (dtype *const *)in_loss_arr.value);
    CheckCudaError();
}

__global__ void KernelBucketForward(const dtype *input, int count, int dim, dtype *const *ys) {
    int index = DeviceDefaultIndex();
    for (int i = index; i < count * dim; i+= DeviceDefaultStep()) {
        int count_i = i / dim;
        int dim_i = i % dim;
        ys[count_i][dim_i] = input[count_i * dim + dim_i];
    }
}

void BucketForward(const std::vector<dtype> input, int count, int dim, std::vector<dtype*> &ys,
        cudaStream_t *stream) {
    NumberArray input_arr;
    NumberPointerArray ys_arr;
    input_arr.init((dtype*)input.data(), input.size(), stream);
    ys_arr.init((dtype**)ys.data(), ys.size(), stream);
    int block_count = DefaultBlockCount(count * dim);
    KernelBucketForward<<<block_count, TPB, 0, *stream>>>((const dtype*)input_arr.value, count,
            dim, (dtype *const *)ys_arr.value);
    CheckCudaError();
}

__global__ void KernelCopyForUniNodeForward(const dtype** xs, const dtype* b,
        dtype* xs_dest,
        dtype* b_dest,
        int count,
        int x_len,
        int b_len,
        bool use_b) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int step = gridDim.x * blockDim.x;
    int x_total_len = count * x_len;
    int b_total_len = count * b_len;
    for (int i = index; i < x_total_len + b_total_len; i += step) {
        if (i < x_total_len) {
            int count_i = i / x_len;
            int len_i = i % x_len;
            xs_dest[i] = xs[count_i][len_i];
        } else if (use_b) {
            int b_i = i - x_total_len;
            int len_i = b_i % b_len;
            b_dest[b_i] = b[len_i];
        }
    }
}

void CopyForUniNodeForward(const std::vector<dtype*> &xs, const dtype* b, dtype* xs_dest,
        dtype* b_dest,
        int count,
        int x_len,
        int b_len,
        bool use_b,
        cudaStream_t *stream) {
    NumberPointerArray x_arr;
    x_arr.init((dtype**)xs.data(), xs.size(), stream);
    int len = x_len + b_len;
    int block_count = std::min((count * len - 1 + TPB) / TPB, 56);
    KernelCopyForUniNodeForward<<<block_count, TPB, 0, *stream>>>(
            (const dtype**)x_arr.value, (const dtype*)b, xs_dest, b_dest,
            count, x_len, b_len, use_b);
    CheckCudaError();
}

void MatrixMultiplyMatrix(dtype *W, dtype *x, dtype *y, int row, int col, int count, bool useb,
        cudaStream_t *stream,
        bool should_x_transpose,
        bool should_W_transpose) {
    cublasHandle_t &handle = GetCublasHandle(stream);
    dtype alpha = 1;
    dtype beta = useb? 1 : 0;
    cublasOperation_t x_op = should_x_transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
    int ldx = should_x_transpose ? count : col;
    cublasOperation_t W_op = should_W_transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
    int ldw = should_W_transpose ? col : row;
#if USE_FLOAT
    CallCublas(cublasSgemm(handle, W_op, x_op, row, count, col, &alpha, W, ldw, x, ldx, &beta, y,
                row));
#else
    CallCublas(cublasDgemm(handle, W_op, x_op, row, count, col, &alpha, W, ldw, x, ldx, &beta, y,
                row));
#endif
}

__global__ void KernelVerify(dtype *host, dtype *device, int len,
        const char *message, bool *success) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < len; i += step) {
        dtype loss = host[index] - device[index];
        if (DeviceAbs(loss) > 0.001 && DeviceAbs(loss) > 0.001 * DeviceAbs(host[index])) {
            *success = false;
            KernelPrintLine("KernelVerify: host:%f device:%f loss:%f",
                    host[index],
                    device[index],
                    loss);
        }
    }
}

bool Verify(dtype *host, dtype *device, int len, const char* message) {
    NumberArray arr;
    arr.init(host, len, nullptr);
    int block_count = DefaultBlockCount(len);
    char *m = NULL;
    CallCuda(MemoryPool::Ins().Malloc((void**)&m,
                (strlen(message) + 1) * sizeof(char)));
    CallCuda(MyCudaMemcpy(m, message,
                (strlen(message) + 1) * sizeof(char), cudaMemcpyHostToDevice, nullptr));
    bool success = true;
    bool *dev_success = NULL;
    CallCuda(MemoryPool::Ins().Malloc((void**)&dev_success, 8 * sizeof(bool)));
    CallCuda(MyCudaMemcpy(dev_success, &success, sizeof(bool),
                cudaMemcpyHostToDevice, nullptr));
    KernelVerify<<<block_count, TPB>>>(arr.value, device, len, m, dev_success);
    CheckCudaError();
    CallCuda(MyCudaMemcpy(&success, dev_success, sizeof(bool),
                cudaMemcpyDeviceToHost, nullptr));
    MemoryPool::Ins().Free(dev_success);
    MemoryPool::Ins().Free(m);
    cudaDeviceSynchronize();
    cudaPrintfDisplay(stdout, true);
    if (!success) {
        cout << message << endl;
    }
    return success;
}

__global__ void KernelVerify(bool *host, bool *device, int len,
        const char *message, bool *success) {
    int index = DeviceDefaultIndex();
    if (index < len) {
        if (host[index] != device[index]) {
            *success = false;
            printf("KernelVerify %s: host:%d device:%d \n", message,
                    host[index],
                    device[index]);
            KernelPrintLine("KernelVerify: host:%d device:%d", host[index],
                    device[index]);
        }
    }
}

bool Verify(bool *host, bool *device, int len, const char* message) {
    BoolArray arr;
    arr.init(host, len, nullptr);
    int block_count = (len + TPB - 1) / TPB;
    char *m = NULL;
    CallCuda(MemoryPool::Ins().Malloc((void**)&m,
                (strlen(message) + 1) * sizeof(char)));
    CallCuda(MyCudaMemcpy(m, message, (strlen(message) + 1) * sizeof(char),
                cudaMemcpyHostToDevice, nullptr));
    bool success = true;
    bool *dev_success = NULL;
    CallCuda(MemoryPool::Ins().Malloc((void**)&dev_success, 8 * sizeof(bool)));
    CallCuda(MyCudaMemcpy(dev_success, &success, sizeof(bool),
                cudaMemcpyHostToDevice, nullptr));
    KernelVerify<<<block_count, TPB>>>(arr.value, device, len, m, dev_success);
    CheckCudaError();
    CallCuda(MyCudaMemcpy(&success, dev_success, sizeof(bool),
                cudaMemcpyDeviceToHost, nullptr));
    MemoryPool::Ins().Free(dev_success);
    MemoryPool::Ins().Free(m);
    cudaDeviceSynchronize();
    cudaPrintfDisplay(stdout, true);
    return success;
}

__global__ void KernelVerify(int *host, int *device, int len,
        const char *message, bool *success) {
    int index = DeviceDefaultIndex();
    if (index < len) {
        if (host[index] != device[index]) {
            *success = false;
            printf("KernelVerify %s: host:%d device:%d \n", message,
                    host[index],
                    device[index]);
            KernelPrintLine("KernelVerify: host:%d device:%d", host[index],
                    device[index]);
        }
    }
}

bool Verify(int *host, int *device, int len, const char* message) {
    IntArray arr;
    arr.init(host, len, nullptr);
    int block_count = (len + TPB - 1) / TPB;
    char *m = NULL;
    CallCuda(MemoryPool::Ins().Malloc((void**)&m,
                (strlen(message) + 1) * sizeof(char)));
    CallCuda(MyCudaMemcpy(m, message,
                (strlen(message) + 1) * sizeof(char), cudaMemcpyHostToDevice, nullptr));
    bool success = true;
    bool *dev_success = NULL;
    CallCuda(MemoryPool::Ins().Malloc((void**)&dev_success, sizeof(bool)));
    CallCuda(MyCudaMemcpy(dev_success, &success, sizeof(bool),
                cudaMemcpyHostToDevice, nullptr));
    KernelVerify<<<block_count, TPB>>>(arr.value, device, len, m, dev_success);
    CheckCudaError();
    CallCuda(MyCudaMemcpy(&success, dev_success, sizeof(bool),
                cudaMemcpyDeviceToHost, nullptr));
    MemoryPool::Ins().Free(dev_success);
    MemoryPool::Ins().Free(m);
    cudaDeviceSynchronize();
    cudaPrintfDisplay(stdout, true);
    return success;
}

constexpr int MAX_BLOCK_POWER = 100;

MemoryPool& MemoryPool::Ins() {
    static MemoryPool *p;
    if (p == NULL) {
        p = new MemoryPool;
        p->free_blocks_.resize(MAX_BLOCK_POWER + 1);
        p->busy_blocks_.reserve(10000);
    }
    return *p;
}

void appendFreeBlock(const MemoryBlock &memory_block,
        vector<map<void*, MemoryBlock>> &free_blocks,
        int i,
        const unordered_map<void*, MemoryBlock> &busy_blocks) {
    if (memory_block.size != (1 << i)) {
        cerr << boost::format("incorrect block size %1%, but i is %2%") % memory_block.size % i <<
            endl;
        abort();
    }
    free_blocks.at(i).insert(make_pair(memory_block.p, memory_block));
}

cudaError_t MemoryPool::Malloc(void **p, int size) {
    cudaError_t error_code;
    assert(*p == NULL);
    Profiler &profiler = Profiler::Ins();
    profiler.BeginEvent("Malloc");
#if DEVICE_MEMORY == 0
    CallCnmem(cnmemMalloc(p, size, NULL));
    profiler.EndEvent();
    error_code = cudaSuccess;
    cudaDeviceSynchronize();
#elif DEVICE_MEMORY == 1
    error_code = cudaMalloc(p, size);
    profiler.EndEvent();
    cudaDeviceSynchronize();
#else
    int fit_size = 1;
    int n = 0;
    while (fit_size < size) {
        fit_size <<= 1;
        ++n;
    }
    cudaError_t status = cudaErrorMemoryAllocation;
    while (status != cudaSuccess) {
        if (free_blocks_.at(n).empty()) {
            int higher_power = n + 1;
            while (higher_power <= MAX_BLOCK_POWER && free_blocks_.at(higher_power).empty()) {
                ++higher_power;
            }
            if (higher_power > MAX_BLOCK_POWER) {
                while (status != cudaSuccess) {
                    status = cudaMalloc(p, fit_size);
                    //cout << "status:" << status << endl;
                }
                CallCuda(status);
//                cout << "MemoryPool malloc - before sync" << endl;
                //cudaDeviceSynchronize();
//                cout << "MemoryPool malloc - after sync" << endl;
                MemoryBlock block(*p, fit_size);
                busy_blocks_.insert(std::make_pair(*p, block));
            } else {
                auto &v = free_blocks_.at(higher_power);
                MemoryBlock &to_split = v.rbegin()->second;
                int half_size = to_split.size >> 1;
                void *half_address = static_cast<void*>(static_cast<char*>(to_split.p) +
                        half_size);
                MemoryBlock low_block(to_split.p, half_size, to_split.buddy),
                            high_block(half_address, half_size, to_split.p);
                v.erase(v.rbegin()->first);
                appendFreeBlock(low_block, free_blocks_, higher_power - 1, busy_blocks_);
                appendFreeBlock(high_block, free_blocks_, higher_power - 1, busy_blocks_);
            }
        } else {
            status = cudaSuccess;
            int this_size = free_blocks_.at(n).size();
            MemoryBlock &block = free_blocks_.at(n).rbegin()->second;
            *p = block.p;
            busy_blocks_.insert(std::make_pair(block.p, block));
            free_blocks_.at(n).erase(free_blocks_.at(n).rbegin()->first);
        }
    }
    profiler.EndEvent();

    error_code = status;
#endif
    return error_code;
}

std::pair<const MemoryBlock *, const MemoryBlock *> lowerAndhigherBlocks(const MemoryBlock &a,
        const MemoryBlock &b) {
    if (a.size != b.size) {
        cerr << "a.size is not equal to b.size" << endl;
        abort();
    }
    int distance = static_cast<char*>(a.p) - static_cast<char*>(b.p);
    if (distance == 0) {
        cerr << "block a and b has the same address" << endl;
        abort();
    }
    const MemoryBlock &low = distance > 0 ? b : a;
    const MemoryBlock &high = distance > 0 ? a : b;
    return std::make_pair(&low, &high);
}

bool isBuddies(const MemoryBlock &a, const MemoryBlock &b) {
    if (a.size != b.size) {
        return false;
    }
    auto pair = lowerAndhigherBlocks(a, b);
    return pair.second->buddy == pair.first->p &&
        ((char*)pair.second->p - (char*)pair.first->p) == a.size;
}

MemoryBlock mergeBlocks(const MemoryBlock &a, const MemoryBlock &b) {
    if (a.size != b.size) {
        cerr << "sizes of memory blocks to merge not equal" << endl;
        abort();
    }

    auto pair = lowerAndhigherBlocks(a, b);
    if ((char*)pair.second->p - (char*)pair.first->p != a.size ||
            (a.p != b.buddy && a.buddy != b.p)) {
        cerr << "a and b are not buddies" << endl;
        cerr << boost::format("a:%1%\nb:%2%") % a.toString() % b.toString() << endl;
        abort();
    }
    MemoryBlock block(pair.first->p, pair.first->size << 1, pair.first->buddy);
    return block;
}

void returnFreeBlock(const MemoryBlock &block, vector<map<void*, MemoryBlock>> &free_blocks,
        int power,
        const unordered_map<void*, MemoryBlock> &busy_blocks) {
    Profiler &profiler = Profiler::Ins();
    profiler.BeginEvent("returnFreeBlock");
    MemoryBlock current_block = block;
    for (int i = power; i <= MAX_BLOCK_POWER; ++i) {
        map<void*, MemoryBlock> &v = free_blocks.at(i);
        void *free_p = (char*)current_block.p - (char*)current_block.buddy == current_block.size ?
            current_block.buddy : (void*)((char*)current_block.p + current_block.size);
        auto it = v.find(free_p);
        if (it == v.end() || (it->second.p != current_block.buddy &&
                    it->second.buddy != current_block.p)) {
            appendFreeBlock(current_block, free_blocks, i, busy_blocks);
            break;
        } else {
            MemoryBlock merged_block = mergeBlocks(it->second, current_block);
            current_block = merged_block;
            v.erase(it);
        }
    }
    profiler.EndEvent();
}

cudaError_t MemoryPool::Free(void *p) {
    Profiler &profiler = Profiler::Ins();
    profiler.BeginEvent("Free");
#if DEVICE_MEMORY == 0
    CallCnmem(cnmemFree(p, NULL));
    profiler.EndEvent();
#elif DEVICE_MEMORY == 1
    cudaError_t r = cudaFree(p);
    profiler.EndEvent();
    return r;
#else
    auto it = busy_blocks_.find(p);
    if (it == busy_blocks_.end()) {
        cerr << "cannot find busy block " << p << endl;
        abort();
    }
    int size = it->second.size;
    int n = 0;
    while (size > 1) {
        size >>= 1;
        ++n;
    }
    if (it->second.size != (1 << n)) {
        cerr << boost::format("size:%1% n:%2%") % it->second.size % n << endl;
        abort();
    }

    auto block = it->second;
    busy_blocks_.erase(it);
    returnFreeBlock(block, free_blocks_, n, busy_blocks_);
    it = busy_blocks_.find(p);
    if (it != busy_blocks_.end()) {
        cerr << "can find erased block " << p << endl;
        abort();
    }

    profiler.EndEvent();
    if (busy_blocks_.find(p) != busy_blocks_.end()) {
        cerr << boost::format("Malloc - find freed p in busy blocks") << endl;
    }
    return cudaSuccess;
#endif
}

void Profiler::EndCudaEvent() {
    //cudaDeviceSynchronize();
    EndEvent();
}

__global__ void KernelAddLtyToParamBiasAndAddLxToInputLossesForUniBackward(
        const dtype *lty,
        const dtype *lx,
        dtype *b,
        dtype *const *losses,
        int count,
        int out_dim,
        int in_dim,
        volatile dtype *block_sums,
        int *global_block_count,
        bool use_b) {
    __shared__ volatile dtype shared_arr[TPB];

    int count_i = blockIdx.y * blockDim.x + threadIdx.x;
    int dim_i = blockIdx.x;
    if (dim_i < out_dim) {
        if (use_b) {
            if (threadIdx.x == 0 && blockIdx.y == 0) {
                global_block_count[dim_i] = 0;
            }
            int lty_index = count_i * out_dim + dim_i;
            shared_arr[threadIdx.x] = count_i < count ? lty[lty_index] : 0.0f;
            __syncthreads();

            for (int i = (TPB >> 1); i > 0; i>>=1) {
                if (threadIdx.x < i) {
                    shared_arr[threadIdx.x] += shared_arr[threadIdx.x + i];
                }
                __syncthreads();
            }

            if (threadIdx.x == 0) {
                block_sums[gridDim.y * blockIdx.x + blockIdx.y] =
                    shared_arr[0];
                if (atomicAdd(global_block_count + dim_i, 1) ==
                        gridDim.y - 1) {
                    dtype sum = 0.0;
                    for (int i = 0; i < gridDim.y; ++i) {
                        sum += block_sums[gridDim.y * blockIdx.x + i];
                    }
                    DeviceAtomicAdd(b + dim_i, sum);
                }
            }
        }
    } else {
        if (count_i < count) {
            dim_i -= out_dim;
            int lx_index = dim_i + count_i * in_dim;
            DeviceAtomicAdd(losses[count_i] + dim_i, lx[lx_index]);
        }
    }
}

void AddLtyToParamBiasAndAddLxToInputLossesForUniBackward(const dtype *lty,
        const dtype *lx,
        dtype *b,
        std::vector<dtype*> &losses,
        int count,
        int out_dim,
        int in_dim,
        bool use_b,
        cudaStream_t *stream) {
    int block_y = (count - 1 + TPB) / TPB;
    dim3 block_dim(out_dim + in_dim, block_y, 1);
    NumberPointerArray loss_arr;
    loss_arr.init(losses.data(), count, stream);
    Tensor1D block_sums;
    block_sums.init(block_y * out_dim);
    IntArray global_block_count_arr;
    global_block_count_arr.init(out_dim);
    KernelAddLtyToParamBiasAndAddLxToInputLossesForUniBackward<<<block_dim, TPB, 0, *stream>>>(lty,
            lx, b, (dtype *const *)loss_arr.value, count, out_dim, in_dim, block_sums.value,
            global_block_count_arr.value, use_b);
    CheckCudaError();
}

__global__ void KernelAddLtyToParamBiasAndAddLxToInputLossesForBiBackward(
        const dtype *lty,
        const dtype *lx1,
        const dtype *lx2,
        dtype *b,
        dtype *const *losses1,
        dtype *const *losses2,
        int count,
        int out_dim,
        int in_dim1,
        int in_dim2,
        bool use_b,
        volatile dtype *block_sums,
        int *global_block_count) {
    __shared__ volatile dtype shared_arr[TPB];

    int count_i = blockIdx.y * blockDim.x + threadIdx.x;
    int dim_i = blockIdx.x;
    if (dim_i < out_dim) {
        if (threadIdx.x == 0 && blockIdx.y == 0) {
            global_block_count[dim_i] = 0;
        }
        //int lty_index = dim_i * count + count_i;
        int lty_index = dim_i + count_i * out_dim;
        shared_arr[threadIdx.x] = count_i < count ? lty[lty_index] : 0.0f;
        __syncthreads();

        for (int i = (TPB >> 1); i > 0; i>>=1) {
            if (threadIdx.x < i) {
                shared_arr[threadIdx.x] += shared_arr[threadIdx.x + i];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            block_sums[gridDim.y * blockIdx.x + blockIdx.y] = shared_arr[0];
            if (atomicAdd(global_block_count + dim_i, 1) == gridDim.y - 1) {
                dtype sum = 0.0;
                for (int i = 0; i < gridDim.y; ++i) {
                    sum += block_sums[gridDim.y * blockIdx.x + i];
                }
                if (use_b) {
                    DeviceAtomicAdd(b + dim_i, sum);
                }
            }
        }
    } else if (dim_i < out_dim + in_dim1) {
        if (count_i < count) {
            dim_i -= out_dim;
            int lx_index = dim_i + count_i * in_dim1;
            DeviceAtomicAdd(losses1[count_i] + dim_i, lx1[lx_index]);
        }
    } else {
        if (count_i < count) {
            dim_i -= (out_dim + in_dim1);
            int lx_index = dim_i + count_i * in_dim2;
            DeviceAtomicAdd(losses2[count_i] + dim_i, lx2[lx_index]);
        }
    }
}

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
        bool use_b,
        cudaStream_t *stream) {
    int block_y = (count - 1 + TPB) / TPB;
    dim3 block_dim(out_dim + in_dim1 + in_dim2, block_y, 1);
    NumberPointerArray loss1_arr;
    loss1_arr.init(losses1.data(), count, stream);
    NumberPointerArray loss2_arr;
    loss2_arr.init(losses2.data(), count, stream);
    Tensor1D block_sums;
    block_sums.init(block_y * out_dim);
    IntArray global_block_count_arr;
    global_block_count_arr.init(out_dim);
    KernelAddLtyToParamBiasAndAddLxToInputLossesForBiBackward<<<block_dim, TPB, 0, *stream>>>(lty,
            lx1, lx2, b, (dtype *const *)loss1_arr.value, (dtype *const *)loss2_arr.value, count,
            out_dim, in_dim1, in_dim2, use_b, block_sums.value, global_block_count_arr.value);
    CheckCudaError();
}

constexpr int MAX_BATCH_COUNT = 1000000;

__global__ void KernelInitCurandStates(curandState_t *states) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int step = gridDim.x * blockDim.x;
    for (int i = index; i < MAX_BATCH_COUNT; i += step) {
        curand_init(0, i, 0, &states[i]);
    }
}

curandState_t *GetCurandStates() {
    static curandState_t *states;
    if (states == NULL) {
        MemoryPool &pool = MemoryPool::Ins();
        CallCuda(pool.Malloc((void**)&states, sizeof(curandState_t) *
                    MAX_BATCH_COUNT));
        KernelInitCurandStates<<<BLOCK_COUNT, TPB>>>( states);
        CheckCudaError();
    }
    return states;
}

curandGenerator_t &GetGenerator() {
    static curandGenerator_t gen;
    static bool init;
    if (!init) {
        CallCurand(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
        CallCurand(curandSetPseudoRandomGeneratorSeed(gen, 0));
        init = true;
    }
    return gen;
}

void CalculateDropoutMask(dtype drop_factor, int count, int dim, dtype* mask) {
    curandGenerator_t &gen = GetGenerator();
    CallCurand(curandGenerateUniform(gen, mask, count * dim));
}

__global__ void KernelConcatForward(const dtype *const *ins, int *in_dims,
        dtype *const *outs,
        int count,
        int in_count,
        int out_dim) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();

    for (int i = index; i < out_dim * count; i += step) {
        int out_dim_i = i % out_dim;
        int count_i = i / out_dim;
        int in_dim_sum = 0;
        int last_in_dim_sum;
        int offset_j = 0;
        for (int j = 0; j < in_count; ++j) {
            last_in_dim_sum = in_dim_sum;
            in_dim_sum += in_dims[j];
            offset_j = j;
            if (out_dim_i < in_dim_sum) {
                break;
            }
        }
        int in_dim_i = out_dim_i - last_in_dim_sum;
        dtype v = ins[count_i * in_count + offset_j][in_dim_i];
        outs[count_i][out_dim_i] = v;
    }
}

void ConcatForward(const std::vector<dtype*> &in_vals,
        const std::vector<int> &in_dims,
        std::vector<dtype*> &vals,
        int count,
        int in_count,
        int out_dim,
        cudaStream_t *stream) {
    int len = count * out_dim;
    int block_count = std::min(BLOCK_COUNT, (len - 1 + TPB) / TPB);
    NumberPointerArray in_val_arr, val_arr;
    in_val_arr.init((dtype**)in_vals.data(), in_vals.size(), stream);
    val_arr.init((dtype**)vals.data(), vals.size(), stream);
    IntArray in_dim_arr;
    in_dim_arr.init((int*)in_dims.data(), in_dims.size(), stream);

    KernelConcatForward<<<block_count, TPB, 0, *stream>>>(in_val_arr.value,
            in_dim_arr.value, (dtype *const *)val_arr.value, count, in_count, out_dim);
    CheckCudaError();
}

__global__ void KernelConcatBackward(dtype *const *in_losses, int *in_dims,
        const dtype *const *out_losses,
        int count,
        int in_count,
        int out_dim) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < out_dim * count; i += step) {
        int out_dim_i = i % out_dim;
        int count_i = i / out_dim;
        int in_dim_sum = 0;
        int last_in_dim_sum;
        int offset_j = 0;
        for (int j = 0; j < in_count; ++j) {
            last_in_dim_sum = in_dim_sum;
            in_dim_sum += in_dims[j];
            offset_j = j;
            if (out_dim_i < in_dim_sum) {
                break;
            }
        }
        int in_dim_i = out_dim_i - last_in_dim_sum;
        DeviceAtomicAdd(in_losses[count_i * in_count + offset_j] +
                in_dim_i, out_losses[count_i][out_dim_i]);
    }
}

void ConcatBackward(const std::vector<dtype*> &in_losses,
        const std::vector<int> &in_dims,
        std::vector<dtype*> &losses,
        int count,
        int in_count,
        int out_dim,
        cudaStream_t *stream) {
    int len = count * out_dim;
    int block_count = std::min(BLOCK_COUNT, (len - 1 + TPB) / TPB);

    NumberPointerArray in_loss_arr, loss_arr;
    in_loss_arr.init((dtype**)in_losses.data(), in_losses.size(), stream);
    loss_arr.init((dtype**)losses.data(), losses.size(), stream);
    IntArray in_dim_arr;
    in_dim_arr.init((int*)in_dims.data(), in_dims.size(), stream);

    KernelConcatBackward<<<block_count, TPB, 0, *stream>>>((dtype *const *)in_loss_arr.value,
            in_dim_arr.value, loss_arr.value, count, in_count, out_dim);
    CheckCudaError();
}

__global__ void KernelScalarConcatForward(const dtype *const *ins, int count,
        const int *dims,
        int max_dim,
        dtype *const *results) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < max_dim * count; i += step) {
        int count_i = i / max_dim;
        int dim_i = i % max_dim;
        if (dim_i < dims[count_i]) {
            results[count_i][dim_i] = ins[count_i * max_dim + dim_i][0];
        }
    }
}

void ScalarConcatForward(const vector<dtype *> &ins, int count, const vector<int> &dims,
        int max_dim,
        const vector<dtype *> &results,
        cudaStream_t *stream) {
    NumberPointerArray result_arr;
    result_arr.init((dtype**)results.data(), results.size(), stream);
    NumberPointerArray in_arr;
    in_arr.init((dtype**)ins.data(), ins.size(), stream);
    IntArray dim_arr;
    dim_arr.init((int *)dims.data(), dims.size(), stream);

    int block_count = DefaultBlockCount(count * max_dim);
    KernelScalarConcatForward<<<block_count, TPB, 0, *stream>>>(in_arr.value, count, dim_arr.value,
            max_dim, (dtype *const *)result_arr.value);
    CheckCudaError();
}

__global__ void KernelScalarConcatBackward(const dtype *const *losses, int count, const int *dims,
        int max_dim,
        dtype *const *input_losses) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < max_dim * count; i += step) {
        int count_i = i / max_dim;
        int dim_i = i % max_dim;
        if (dim_i < dims[count_i]) {
            DeviceAtomicAdd(input_losses[count_i * max_dim + dim_i], losses[count_i][dim_i]);
        }
    }
}

void ScalarConcatBackward(const vector<dtype *> &losses, int count, const vector<int> &dims,
        int max_dim,
        const vector<dtype *> in_losses,
        cudaStream_t *stream) {
    NumberPointerArray loss_arr, in_loss_arr;
    loss_arr.init((dtype**)losses.data(), losses.size(), stream);
    in_loss_arr.init((dtype **)in_losses.data(), in_losses.size(), stream);
    IntArray dim_arr;
    dim_arr.init((int *)dims.data(), dims.size(), stream);
    int block_count = DefaultBlockCount(count * max_dim);
    KernelScalarConcatBackward<<<block_count, TPB, 0, *stream>>>(loss_arr.value, count,
            dim_arr.value, max_dim, (dtype *const *)in_loss_arr.value);
    CheckCudaError();
}

__global__ void KernelMemset(dtype *p, int len, dtype value) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < len; i+= step) {
        p[i] = value;
    }
}

void Memset(dtype *p, int len, dtype value, cudaStream_t *stream) {
    int block_count = std::min(BLOCK_COUNT, (len - 1 + TPB) / TPB);
    if (stream == nullptr) {
        KernelMemset<<<block_count, TPB, 0, 0>>>(p, len, value);
    } else {
        KernelMemset<<<block_count, TPB, 0, *stream>>>(p, len, value);
    }
    CheckCudaError();
}

__global__ void KernelMemset(bool *p, int len, bool value) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < len; i+= step) {
        p[i] = value;
    }
}

void Memset(bool *p, int len, bool value, cudaStream_t *stream) {
    int block_count = std::min(BLOCK_COUNT, (len - 1 + TPB) / TPB);
    KernelMemset<<<block_count, TPB>>>(p, len, value);
    CheckCudaError();
}

void *Malloc(int size) {
    void *p;
    CallCuda(cudaMalloc(&p, size));
    return p;
}

__global__ void KernelBatchMemset(dtype *const *p, int count, int *dims, int max_dim,
        dtype value) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < max_dim * count ; i += step) {
        int count_i = i / max_dim;
        int dim_i = i % max_dim;
        if (dim_i < dims[count_i]) {
            p[count_i][dim_i] = value;
        }
    }
}

void BatchMemset(const std::vector<dtype*> &vec, int count, const vector<int> &dims, dtype value,
        cudaStream_t *stream) {
    int max_dim = *max_element(dims.begin(), dims.end());
    int block_count = (count * max_dim -1 + TPB) / TPB;
    block_count = std::min(block_count, BLOCK_COUNT);
    NumberPointerArray vec_arr;
    vec_arr.init((dtype**)vec.data(), vec.size(), stream);
    IntArray dim_arr;
    dim_arr.init(dims.data(), dims.size(), stream);
    KernelBatchMemset<<<block_count, TPB, 0, *stream>>>((dtype *const *)vec_arr.value, count,
            dim_arr.value, max_dim, value);
    CheckCudaError();
}

__global__ void KernelLookupForward(const int *xids, const dtype *vocabulary,
        int count,
        int dim,
        dtype **vals) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count * dim; i += step) {
        int count_i = i / dim;
        int dim_i = i % dim;
        int xid = xids[count_i];
        if (xid >= 0) {
            int voc_i = xid * dim + dim_i;
            dtype v = vocabulary[voc_i];
            vals[count_i][dim_i] = v;
        } else {
            vals[count_i][dim_i] = 0.0f;
        }
    }
}

void LookupForward(const std::vector<int> &xids, const dtype *vocabulary,
        int count,
        int dim,
        std::vector<dtype*> &vals,
        cudaStream_t *stream) {
    int block_count = std::min(BLOCK_COUNT, (count * dim - 1 + TPB) / TPB);
    IntArray xid_arr;
    xid_arr.init((int*)xids.data(), xids.size(), stream);
    NumberPointerArray val_arr;
    val_arr.init((dtype**)vals.data(), vals.size(), stream);
    KernelLookupForward<<<block_count, TPB, 0, *stream>>>(xid_arr.value, vocabulary,
            count, dim, const_cast<dtype**>(val_arr.value));
    CheckCudaError();
}

__global__ void KernelLookupBackward(const int *xids, int unknown_id,
        bool fine_tune,
        const dtype** losses,
        int count,
        int dim,
        dtype *grad,
        bool *indexers) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count * dim; i += step) {
        int count_i = i / dim;
        int dim_i = i % dim;
        int xid = xids[count_i];
        if (xid == unknown_id || fine_tune) {
            assert(xid >= 0);
            if (dim_i == 0) {
                indexers[xid] = true;
            }
            DeviceAtomicAdd(grad + xid * dim + dim_i, losses[count_i][dim_i]);
        }
    }
}

void LookupBackward(const std::vector<int> &xids, int unknown_id,
        bool fine_tune,
        const std::vector<dtype*> &losses,
        int count,
        int dim,
        dtype *grad,
        bool *indexers,
        cudaStream_t *stream) {
    int block_count = std::min((count * dim - 1 + TPB) / TPB, BLOCK_COUNT);
    IntArray pl_arr;
    pl_arr.init((int*)xids.data(), xids.size(), stream);
    IntArray xid_arr;
    xid_arr.init((int*)pl_arr.value, xids.size(), stream);
    NumberPointerArray loss_arr;
    loss_arr.init((dtype**)losses.data(), losses.size(), stream);
    KernelLookupBackward<<<block_count, TPB, 0, *stream>>>(
            const_cast<const int *>(xid_arr.value),
            unknown_id,
            fine_tune,
            const_cast<const dtype**>(loss_arr.value),
            count,
            dim,
            grad,
            indexers);
    CheckCudaError();
}

__global__ void KernelLookupBackward(const int *xids, int unknown_id,
        bool fine_tune,
        const dtype** losses,
        int count,
        int dim,
        dtype *grad) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count * dim; i += step) {
        int count_i = i / dim;
        int dim_i = i % dim;
        int xid = xids[count_i];
        if (xid == unknown_id || fine_tune) {
            assert(xid >= 0);
            DeviceAtomicAdd(grad + xid * dim + dim_i, losses[count_i][dim_i]);
        }
    }
}

void LookupBackward(const std::vector<int> &xids, int unknown_id,
        bool fine_tune,
        const std::vector<dtype*> &losses,
        int count,
        int dim,
        dtype *grad,
        cudaStream_t *stream) {
    int block_count = std::min((count * dim - 1 + TPB) / TPB, BLOCK_COUNT);
    IntArray pl_arr;
    pl_arr.init((int*)xids.data(), xids.size(), stream);
    IntArray xid_arr;
    xid_arr.init((int*)pl_arr.value, xids.size(), stream);
    NumberPointerArray loss_arr;
    loss_arr.init((dtype**)losses.data(), losses.size(), stream);
    KernelLookupBackward<<<block_count, TPB, 0, *stream>>>(
            const_cast<const int *>(xid_arr.value),
            unknown_id,
            fine_tune,
            const_cast<const dtype**>(loss_arr.value),
            count,
            dim,
            grad);
    CheckCudaError();
}

__global__ void KernelParamRowForward(const dtype *param, int row_index, int param_row_count,
        int count,
        int dim,
        dtype *const *vals) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < dim * count; i += step) {
        int count_i = i / dim;
        int dim_i = i % dim;
        int param_offset = dim_i * param_row_count + row_index;
        vals[count_i][dim_i] = param[param_offset];
    }
}

void ParamRowForward(const dtype *param, int row_index, int param_row_count, int count, int dim,
        vector<dtype*> &vals,
        cudaStream_t *stream) {
    NumberPointerArray val_arr;
    val_arr.init((dtype**)vals.data(), vals.size(), stream);
    int block_count = DefaultBlockCount(count * dim);
    KernelParamRowForward<<<block_count, TPB, 0, *stream>>>(param, row_index, param_row_count,
            count, dim, (dtype *const *)val_arr.value);
    CheckCudaError();
}

__global__ void KernelPoolForward(PoolingEnum pooling, const dtype *const *ins,
        int *in_counts, int max_in_count, dtype *const *outs, int count, int dim,
        int* hit_inputs) {
    __shared__ volatile extern dtype pool_shared_arr[];
    volatile dtype* shared_indexers = pool_shared_arr + blockDim.x;
    int batch_i = blockIdx.y;
    int in_count = in_counts[batch_i];
    int in_count_i = threadIdx.x;
    int dim_i = blockIdx.x;
    if (in_count_i < in_count) {
        pool_shared_arr[threadIdx.x] = ins[batch_i * max_in_count +
            in_count_i][dim_i];
    } else {
        pool_shared_arr[threadIdx.x] = pooling == PoolingEnum::MAX ?
            -1e10 : 1e10;
    }
    shared_indexers[threadIdx.x] = threadIdx.x;
    __syncthreads();

    for (int i = (blockDim.x >> 1); i > 0;i >>=1) {
        if (threadIdx.x < i) {
            int plus_i = threadIdx.x + i;
            if (pooling == PoolingEnum::MAX) {
                if (pool_shared_arr[threadIdx.x] < pool_shared_arr[plus_i]) {
                    pool_shared_arr[threadIdx.x] = pool_shared_arr[plus_i];
                    shared_indexers[threadIdx.x] = shared_indexers[plus_i];
                }
            } else {
                if (pool_shared_arr[threadIdx.x] > pool_shared_arr[plus_i]) {
                    pool_shared_arr[threadIdx.x] = pool_shared_arr[plus_i];
                    shared_indexers[threadIdx.x] = shared_indexers[plus_i];
                }
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        hit_inputs[batch_i * dim + dim_i] = shared_indexers[0];
        outs[batch_i][dim_i] = pool_shared_arr[0];
    }
}

void PoolForward(PoolingEnum pooling, const std::vector<dtype*> &in_vals,
        std::vector<dtype*> &vals,
        int count,
        const std::vector<int> &in_counts,
        int dim,
        int *hit_inputs,
        cudaStream_t *stream) {
    int max_in_count = *std::max_element(in_counts.begin(), in_counts.end());
    int thread_count = 8;
    while (max_in_count > thread_count) {
        thread_count <<= 1;
    }
    dim3 block_dim(dim, count, 1);

    NumberPointerArray in_val_arr;
    in_val_arr.init((dtype**)in_vals.data(), in_vals.size(), stream);
    NumberPointerArray val_arr;
    val_arr.init((dtype**)vals.data(), vals.size(), stream);
    IntArray in_count_arr;
    in_count_arr.init((int*)in_counts.data(), in_counts.size(), stream);

    KernelPoolForward<<<block_dim, thread_count, thread_count * 2 * sizeof(dtype), *stream>>>(
            pooling, in_val_arr.value, in_count_arr.value, max_in_count,
            (dtype *const *)val_arr.value, count, dim, hit_inputs);
    CheckCudaError();
}

__global__ void KernelPoolBackward(const dtype *const * losses,
        const int *hit_inputs,
        int max_in_count,
        int count,
        int dim,
        dtype *const *in_losses) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < dim * count; i += step) {
        int count_i = i / dim;
        int dim_i = i % dim;
        int input_i = hit_inputs[i];
        dtype loss = losses[count_i][dim_i];
        DeviceAtomicAdd(in_losses[count_i * max_in_count + input_i] + dim_i,
                loss);
    }
}

void PoolBackward(const std::vector<dtype*> &losses,
        std::vector<dtype*> &in_losses,
        const std::vector<int> &in_counts,
        const int *hit_inputs,
        int count,
        int dim,
        cudaStream_t *stream) {
    NumberPointerArray loss_arr, in_loss_arr;
    loss_arr.init((dtype**)losses.data(), losses.size(), stream);
    in_loss_arr.init((dtype**)in_losses.data(), in_losses.size(), stream);
    int max_in_count = *std::max_element(in_counts.begin(), in_counts.end());
    int block_count = (count * dim - 1 + TPB) / TPB;
    block_count = std::min(block_count, BLOCK_COUNT);
    KernelPoolBackward<<<block_count, TPB, 0, *stream>>>((const dtype**)loss_arr.value,
            hit_inputs,
            max_in_count,
            count,
            dim,
            (dtype *const *)in_loss_arr.value);
    CheckCudaError();
}

__global__ void KernelSumPoolForward(PoolingEnum pooling,
        const dtype *const *in_vals,
        int count,
        int dim,
        const int *in_counts,
        int max_in_count,
        dtype *const *vals) {
    __shared__ volatile extern dtype pool_shared_arr[];
    int batch_i = blockIdx.y;
    int in_count = in_counts[batch_i];
    int in_count_i = threadIdx.x;
    int dim_i = blockIdx.x;
    if (in_count_i < in_count) {
        pool_shared_arr[threadIdx.x] = in_vals[batch_i * max_in_count +
            in_count_i][dim_i];
    } else {
        pool_shared_arr[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    for (int i = (blockDim.x >> 1); i > 0;i >>=1) {
        if (threadIdx.x < i) {
            int plus_i = threadIdx.x + i;
            pool_shared_arr[threadIdx.x] += pool_shared_arr[plus_i];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        vals[batch_i][dim_i] = pooling == PoolingEnum::SUM ?
            pool_shared_arr[0] : pool_shared_arr[0] / in_counts[batch_i];
    }
}

void SumPoolForward(PoolingEnum pooling, const std::vector<dtype*> &in_vals,
        int count,
        int dim,
        const std::vector<int> &in_counts,
        std::vector<dtype*> &vals,
        cudaStream_t *stream) {
    int max_in_count = *std::max_element(in_counts.begin(), in_counts.end());
    int thread_count = 8;
    while (max_in_count > thread_count) {
        thread_count <<= 1;
    }
    dim3 block_dim(dim, count, 1);
    NumberPointerArray in_val_arr;
    in_val_arr.init((dtype**)in_vals.data(), in_vals.size(), stream);
    IntArray in_count_arr;
    in_count_arr.init((int*)in_counts.data(), in_counts.size(), stream);
    NumberPointerArray val_arr;
    val_arr.init((dtype**)vals.data(), vals.size(), stream);

    KernelSumPoolForward<<<block_dim, thread_count, thread_count * sizeof(dtype), *stream>>>(
            pooling, in_val_arr.value, count, dim, in_count_arr.value, max_in_count,
            (dtype *const *)val_arr.value);
    CheckCudaError();
}

__global__ void KernelSumBackward(PoolingEnum pooling, const dtype *const *losses,
        const int *in_counts,
        int max_in_count,
        int count,
        int dim,
        dtype *const *in_losses) {
    int global_in_count_i = blockIdx.x * max_in_count + blockIdx.y;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        if (blockIdx.y < in_counts[blockIdx.x]) {
            DeviceAtomicAdd(in_losses[global_in_count_i] + i, pooling == PoolingEnum::SUM ?
                    losses[blockIdx.x][i] : losses[blockIdx.x][i] / in_counts[blockIdx.x]);
        }
    }
}

void SumPoolBackward(PoolingEnum pooling, const std::vector<dtype*> &losses,
        const std::vector<int> &in_counts,
        int count,
        int dim,
        std::vector<dtype*> &in_losses,
        cudaStream_t *stream) {
    int thread_count = 8;
    while (thread_count < dim) {
        thread_count <<= 1;
    }
    thread_count = std::min(TPB, thread_count);

    int max_in_count = *std::max_element(in_counts.begin(), in_counts.end());
    dim3 block_dim(count, max_in_count, 1);
    NumberPointerArray loss_arr;
    loss_arr.init((dtype**)losses.data(), losses.size(), stream);
    IntArray in_count_arr;
    in_count_arr.init((int*)in_counts.data(), in_counts.size(), stream);
    NumberPointerArray in_loss_arr;
    in_loss_arr.init((dtype**)in_losses.data(), in_losses.size(), stream);
    KernelSumBackward<<<block_dim, thread_count, 0, *stream>>>(pooling, loss_arr.value,
            (const int*)in_count_arr.value, max_in_count, count, dim,
            (dtype *const *)in_loss_arr.value);
    CheckCudaError();
}

__global__ void KernelPMultiForward(const dtype **ins1, const dtype **ins2,
        int count,
        int dim,
        dtype *const *vals) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count * dim; i += step) {
        int count_i = i / dim;
        int dim_i = i % dim;
        vals[count_i][dim_i] = ins1[count_i][dim_i] * ins2[count_i][dim_i];
    }
}

void PMultiForward(const std::vector<dtype*> &ins1,
        const std::vector<dtype*> &ins2,
        int count,
        int dim,
        std::vector<dtype*> &vals,
        cudaStream_t *stream) {
    int block_count = DefaultBlockCount(count * dim);
    NumberPointerArray ins1_arr, ins2_arr, vals_arr;
    ins1_arr.init((dtype**)ins1.data(), count, stream);
    ins2_arr.init((dtype**)ins2.data(), count, stream);
    vals_arr.init((dtype**)vals.data(), count, stream);
    KernelPMultiForward<<<block_count, TPB, 0, *stream>>>(ins1_arr.value, ins2_arr.value, count,
            dim, (dtype *const *)vals_arr.value);
    CheckCudaError();
}

__global__ void KernelDivForward(const dtype *const *numerators, const dtype *const *denominators,
        int count,
        int *dims,
        int max_dim,
        dtype *const *results) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count * max_dim; i += step) {
        int count_i = i / max_dim;
        int dim_i = i % max_dim;
        if (dim_i < dims[count_i]) {
            results[count_i][dim_i] = numerators[count_i][dim_i] / denominators[count_i][0];
        }
    }
}

void DivForward(const vector<const dtype*> numerators, const vector<const dtype*> denominators,
        int count,
        const vector<int> &dims,
        vector<dtype*> &results,
        cudaStream_t *stream) {
    int max_dim = *max_element(dims.begin(), dims.end());
    int block_count = DefaultBlockCount(count * max_dim);
    NumberPointerArray numerator_arr, denominator_arr, result_arr;
    numerator_arr.init((dtype**)numerators.data(), count, stream);
    denominator_arr.init((dtype**)denominators.data(), count, stream);
    result_arr.init((dtype**)results.data(), count, stream);
    IntArray dim_arr;
    dim_arr.init(dims.data(), dims.size(), stream);
    KernelDivForward<<<block_count, TPB, 0, *stream>>>(numerator_arr.value, denominator_arr.value,
            count, dim_arr.value, max_dim, (dtype *const *)result_arr.value);
    CheckCudaError();
}

__global__ void KernelDivNumeratorBackward(const dtype *const *losses,
        const dtype *const *denominator_vals,
        int count,
        int *dims,
        int max_dim,
        dtype *const *numerator_losses) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();

    for (int i = index; i < count * max_dim; i += step) {
        int count_i = i / max_dim;
        int dim_i = i % max_dim;
        if (dim_i < dims[count_i]) {
            DeviceAtomicAdd(numerator_losses[count_i] + dim_i, losses[count_i][dim_i] /
                    denominator_vals[count_i][0]);
        }
    }
}

__global__ void KernelDivDenominatorBackward(const dtype *const *losses,
        const dtype *const *numerator_vals,
        const dtype *const *denominator_vals,
        int count,
        int *dims,
        volatile dtype *block_sums,
        int *block_counters,
        dtype *const *denominator_losses) {
    __shared__ volatile dtype shared_sum[TPB];
    __shared__ volatile bool is_last_block;
    __shared__ volatile dtype square;
    if (threadIdx.x == 0 && blockIdx.y == 0) {
        block_counters[blockIdx.x] = 0;
    }
    int count_i = blockIdx.x;
    if (threadIdx.x == 0) {
        is_last_block = false;
        square = denominator_vals[count_i][0] * denominator_vals[count_i][0];
    }
    __syncthreads();

    int offset = blockIdx.y * blockDim.x + threadIdx.x;

    shared_sum[threadIdx.x] = offset < dims[count_i] ? losses[count_i][offset] *
        numerator_vals[count_i][offset] / square : 0.0f;
    __syncthreads();

    for (int i = (blockDim.x >> 1); i > 0; i >>= 1) {
        if (threadIdx.x < i) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + i];
        }
        __syncthreads();
    }

    int block_sums_offset = blockIdx.x * gridDim.y + blockIdx.y;
    if (threadIdx.x == 0) {
        block_sums[block_sums_offset] = shared_sum[0];
        if (atomicAdd(block_counters + blockIdx.x, 1) == gridDim.y - 1) {
            is_last_block = true;
        }
    }
    __syncthreads();

    if (is_last_block) {
        dtype sum = 0.0f;
        for (int i = threadIdx.x; i < gridDim.y; i += blockDim.x) {
            int offset = blockIdx.x * gridDim.y + i;
            sum += block_sums[offset];
        }

        shared_sum[threadIdx.x] = sum;
        __syncthreads();

        for (int i = (blockDim.x >> 1); i > 0; i >>= 1) {
            if (threadIdx.x < i) {
                shared_sum[threadIdx.x] += shared_sum[threadIdx.x + i];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            DeviceAtomicAdd(denominator_losses[count_i], -shared_sum[0]);
        }
    }
}

void DivBackward(const vector<const dtype*> &losses, const vector<const dtype*> &denominator_vals,
        const vector<const dtype*> &numerator_vals,
        int count,
        const vector<int> &dims,
        vector<dtype*> &numerator_losses,
        vector<dtype*> &denominator_losses,
        cudaStream_t *stream) {
    int max_dim = *max_element(dims.begin(), dims.end());
    NumberPointerArray loss_arr, denominator_val_arr, numerator_val_arr, numerator_loss_arr,
        denominator_loss_arr;
    loss_arr.init((dtype**)losses.data(), losses.size(), stream);
    denominator_val_arr.init((dtype**)denominator_vals.data(), denominator_vals.size(), stream);
    numerator_val_arr.init((dtype**)numerator_vals.data(), numerator_vals.size(), stream);
    numerator_loss_arr.init((dtype**)numerator_losses.data(), numerator_losses.size(), stream);
    denominator_loss_arr.init((dtype**)denominator_losses.data(), denominator_losses.size(), stream);
    IntArray dim_arr;
    dim_arr.init(dims.data(), dims.size(), stream);

    int block_count = DefaultBlockCount(count * max_dim);
    KernelDivNumeratorBackward<<<block_count, TPB, 0, *stream>>>(loss_arr.value,
            denominator_val_arr.value, count, dim_arr.value, max_dim,
            (dtype *const *)numerator_loss_arr.value);
    CheckCudaError();

    int thread_count = min(NextTwoIntegerPowerNumber(max_dim), TPB);
    int block_y_count = (max_dim - 1 + thread_count) / thread_count;
    dim3 block_dim(count, block_y_count, 1);

    NumberArray block_sums;
    block_sums.init(block_y_count * count);
    IntArray block_counters;
    block_counters.init(count);

    KernelDivDenominatorBackward<<<block_dim , thread_count>>>(loss_arr.value,
            numerator_val_arr.value, denominator_val_arr.value, count, dim_arr.value,
            block_sums.value, block_counters.value, (dtype *const *)denominator_loss_arr.value);
    CheckCudaError();
}

__global__ void KernelSplitForward(const dtype *const *inputs, const int *offsets, int count,
        int dim,
        dtype *const *results) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();

    for (int i = index; i < count * dim; i += step) {
        int count_i = i / dim;
        int dim_i = i % dim;
        int offset = offsets[count_i];
        results[count_i][dim_i] = inputs[count_i][offset + dim_i];
    }
}

void SplitForward(const vector<const dtype*> &inputs, const vector<int> &offsets, int count,
        int dim,
        vector<dtype*> &results,
        cudaStream_t *stream) {
    NumberPointerArray input_arr, result_arr;
    input_arr.init((dtype**)inputs.data(), inputs.size(), stream);
    result_arr.init((dtype**)results.data(), results.size(), stream);
    IntArray offset_arr;
    offset_arr.init((int*)offsets.data(), offsets.size(), stream);

    int block_count = DefaultBlockCount(count * dim);
    KernelSplitForward<<<block_count, TPB, 0, *stream>>>(input_arr.value, offset_arr.value, count,
            dim, (dtype *const *)result_arr.value);
    CheckCudaError();
}

__global__ void KernelSplitBackward(const dtype *const *losses, const int *offsets, int count,
        int dim,
        dtype *const *input_losses) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();

    for (int i = index; i < count * dim; i += step) {
        int count_i = i / dim;
        int dim_i = i % dim;
        int offset = offsets[count_i];
        DeviceAtomicAdd(input_losses[count_i] + offset + dim_i, losses[count_i][dim_i]);
    }
}

void SplitBackward(const vector<const dtype*> &losses, const vector<int> offsets, int count,
        int dim,
        const vector<dtype*> &input_losses,
        cudaStream_t *stream) {
    NumberPointerArray loss_arr, input_loss_arr;
    loss_arr.init((dtype**)losses.data(), losses.size(), stream);
    input_loss_arr.init((dtype**)input_losses.data(), input_losses.size(), stream);
    IntArray offset_arr;
    offset_arr.init((int*)offsets.data(), offsets.size(), stream);
    int block_count = DefaultBlockCount(count * dim);
    KernelSplitBackward<<<block_count, TPB, 0, *stream>>>(loss_arr.value, offset_arr.value, count,
            dim, (dtype *const *)input_loss_arr.value);
    CheckCudaError();
}

__global__ void KernelSubForward(const dtype *const *minuend, const dtype *const *subtrahend,
        int count,
        int *dims,
        int max_dim,
        dtype *const *results) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();

    for (int i = index; i < count * max_dim; i += step) {
        int count_i = i / max_dim;
        int dim_i = i % max_dim;
        if (dim_i < dims[count_i]) {
            results[count_i][dim_i] = minuend[count_i][dim_i] - subtrahend[count_i][dim_i];
        }
    }
}

void SubForward(const std::vector<const dtype*> &minuend,
        const std::vector<const dtype*> &subtrahend,
        int count,
        const vector<int> &dims,
        std::vector<dtype*> &results,
        cudaStream_t *stream) {
    int max_dim = *max_element(dims.begin(), dims.end());
    int block_count = DefaultBlockCount(count * max_dim);
    NumberPointerArray minuend_arr, subtrahend_arr, result_arr;
    minuend_arr.init((dtype**)minuend.data(), count, stream);
    subtrahend_arr.init((dtype**)subtrahend.data(), count, stream);
    result_arr.init((dtype**)results.data(), count, stream);
    IntArray dim_arr;
    dim_arr.init(dims.data(), dims.size(), stream);
    KernelSubForward<<<block_count, TPB, 0, *stream>>>((const dtype* const*)minuend_arr.value,
            (const dtype *const *)subtrahend_arr.value, count, dim_arr.value, max_dim,
            (dtype *const *)result_arr.value);
    CheckCudaError();
}

__global__ void KernelSubBackward(const dtype *const *losses, int count, int *dims, int max_dim,
        dtype *const *minuend_losses,
        dtype *const *subtrahend_losses) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count * max_dim; i += step) {
        int count_i = i / max_dim;
        int dim_i = i % max_dim;
        if (dim_i < dims[count_i]) {
            DeviceAtomicAdd(minuend_losses[count_i] + dim_i, losses[count_i][dim_i]);
            DeviceAtomicAdd(subtrahend_losses[count_i] + dim_i, -losses[count_i][dim_i]);
        }
    }
}

void SubBackward(const std::vector<const dtype*> &losses, int count, const vector<int> &dims,
        std::vector<dtype*> &minuend_losses,
        std::vector<dtype*> &subtrahend_losses,
        cudaStream_t *stream) {
    int max_dim = *max_element(dims.begin(), dims.end());
    int block_count = DefaultBlockCount(count * max_dim);
    NumberPointerArray loss_arr, minuend_loss_arr, subtrahend_loss_arr;
    loss_arr.init((dtype**)losses.data(), losses.size(), stream);
    minuend_loss_arr.init((dtype**)minuend_losses.data(), minuend_losses.size(), stream);
    subtrahend_loss_arr.init((dtype**)subtrahend_losses.data(), subtrahend_losses.size(), stream);

    IntArray dim_arr;
    dim_arr.init(dims.data(), dims.size(), stream);

    KernelSubBackward<<<block_count, TPB, 0, *stream>>>((const dtype *const *)loss_arr.value, count,
            dim_arr.value, max_dim, (dtype *const *)minuend_loss_arr.value,
            (dtype *const *)subtrahend_loss_arr.value);
    CheckCudaError();
}

__global__ void KernelPMultiBackward(const dtype **losses,
        const dtype *const *in_vals1,
        const dtype *const *in_vals2,
        int count,
        int dim,
        dtype *const *in_losses1,
        dtype *const *in_losses2) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count * dim; i += step) {
        int count_i = i / dim;
        int dim_i = i % dim;
        DeviceAtomicAdd(in_losses1[count_i] + dim_i,
                losses[count_i][dim_i] * in_vals2[count_i][dim_i]);
        DeviceAtomicAdd(in_losses2[count_i] + dim_i,
                losses[count_i][dim_i] * in_vals1[count_i][dim_i]);
    }
}

void PMultiBackward(const std::vector<dtype*> &losses,
        const std::vector<dtype*> &in_vals1,
        const std::vector<dtype*> &in_vals2,
        int count,
        int dim,
        std::vector<dtype*> &in_losses1,
        std::vector<dtype*> &in_losses2,
        cudaStream_t *stream) {
    int block_count = DefaultBlockCount(count * dim);
    NumberPointerArray losses_arr, in_vals1_arr, in_vals2_arr, in_losses1_arr,
                       in_losses2_arr;
    losses_arr.init((dtype**)losses.data(), losses.size(), stream);
    in_vals1_arr.init((dtype**)in_vals1.data(), in_vals1.size(), stream);
    in_vals2_arr.init((dtype**)in_vals2.data(), in_vals2.size(), stream);
    in_losses1_arr.init((dtype**)in_losses1.data(), in_losses1.size(), stream);
    in_losses2_arr.init((dtype**)in_losses2.data(), in_losses2.size(), stream);
    KernelPMultiBackward<<<block_count, TPB, 0, *stream>>>(losses_arr.value, in_vals1_arr.value,
            in_vals2_arr.value, count, dim, (dtype *const *)in_losses1_arr.value,
            (dtype *const *)in_losses2_arr.value);
    CheckCudaError();
}

__global__ void KernelPAddForward(const dtype *const *const *ins, int count, int dim,
        int in_count,
        dtype *const *vals) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count * dim; i+= step) {
        int count_i = i / dim;
        int dim_i = i % dim;
        dtype sum = ins[0][count_i][dim_i];
        for (int j = 1; j < in_count; ++j) {
            sum += ins[j][count_i][dim_i];
        }
        vals[count_i][dim_i] = sum;
    }
}

__global__ void KernelPDotForward(const dtype *const *in_vals1, const dtype *const *in_vals2,
        int count,
        int dim,
        dtype *const *vals) {
    volatile __shared__ extern dtype shared_val[];
    if (threadIdx.x < dim) {
        shared_val[threadIdx.x] = in_vals1[blockIdx.x][threadIdx.x] *
            in_vals2[blockIdx.x][threadIdx.x];
    } else {
        shared_val[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    for (int i = (blockDim.x >> 1); i > 0; i >>= 1) {
        if (threadIdx.x < i) {
            shared_val[threadIdx.x] += shared_val[threadIdx.x + i];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        vals[blockIdx.x][0] = shared_val[0];
    }
}

void PDotForward(const std::vector<dtype*> &ins1, const std::vector<dtype*> &ins2, int count,
        int dim,
        std::vector<dtype*> &vals,
        cudaStream_t *stream) {
    NumberPointerArray in1_arr, in2_arr, val_arr;
    in1_arr.init((dtype**)ins1.data(), ins1.size(), stream);
    in2_arr.init((dtype**)ins2.data(), ins2.size(), stream);
    val_arr.init((dtype**)vals.data(), vals.size(), stream);
    int thread_count = NextTwoIntegerPowerNumber(dim);
    KernelPDotForward<<<count, thread_count, thread_count * sizeof(dtype), *stream>>>(
            in1_arr.value, in2_arr.value, count, dim, (dtype *const *)val_arr.value);
    CheckCudaError();
}

__global__ void KernelPDotBackward(const dtype *const *losses,
        const dtype *const *in_vals1,
        const dtype *const *in_vals2,
        int count,
        int dim,
        dtype *const *in_losses1,
        dtype *const *in_losses2) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();

    for (int i = index; i < count * dim; i += step) {
        int count_i = i / dim;
        int dim_i = i % dim;
        DeviceAtomicAdd(in_losses1[count_i] + dim_i,
                losses[count_i][0] * in_vals2[count_i][dim_i]);
        DeviceAtomicAdd(in_losses2[count_i] + dim_i,
                losses[count_i][0] * in_vals1[count_i][dim_i]);
    }
}

void PDotBackward(const std::vector<dtype*> &losses,
        const std::vector<dtype*> &in_vals1,
        const std::vector<dtype*> &in_vals2,
        int count,
        int dim,
        std::vector<dtype*> &in_losses1,
        std::vector<dtype*> &in_losses2,
        cudaStream_t *stream) {
    NumberPointerArray in1_loss_arr, in2_loss_arr, loss_arr, in_val1_arr,
    in_val2_arr;
    in1_loss_arr.init((dtype**)in_losses1.data(), in_losses1.size(), stream);
    in2_loss_arr.init((dtype**)in_losses2.data(), in_losses2.size(), stream);
    loss_arr.init((dtype**)losses.data(), losses.size(), stream);
    in_val1_arr.init((dtype**)in_vals1.data(), in_vals1.size(), stream);
    in_val2_arr.init((dtype**)in_vals2.data(), in_vals2.size(), stream);
    int block_count = DefaultBlockCount(count * dim);
    KernelPDotBackward<<<block_count, TPB, 0, *stream>>>(loss_arr.value, in_val1_arr.value,
            in_val2_arr.value, count, dim, (dtype *const *)in1_loss_arr.value,
            (dtype *const *)in2_loss_arr.value);
    CheckCudaError();
}

void PAddForward(const std::vector<std::vector<dtype*>> &ins, int count, int dim, int in_count,
        std::vector<dtype*> &vals,
        cudaStream_t *stream) {
    std::vector<std::shared_ptr<NumberPointerArray>> gpu_addr;
    gpu_addr.reserve(ins.size());
    for (const std::vector<dtype*> &x : ins) {
        std::shared_ptr<NumberPointerArray> arr =
            std::make_shared<NumberPointerArray>();
        arr->init((dtype**)x.data(), x.size(), stream);
        gpu_addr.push_back(arr);
    }
    std::vector<dtype**> ins_gpu;
    ins_gpu.reserve(ins.size());
    for (auto &ptr : gpu_addr) {
        ins_gpu.push_back((dtype**)ptr->value);
    }

    NumberPointerPointerArray in_arr;
    in_arr.init(ins_gpu.data(), ins_gpu.size(), stream);
    NumberPointerArray out_arr;
    out_arr.init(vals.data(), vals.size(), stream);

    int block_count = DefaultBlockCount(count * dim);
    KernelPAddForward<<<block_count, TPB, 0, *stream>>>(in_arr.value, count, dim, in_count,
            (dtype *const *)out_arr.value);
    CheckCudaError();
}

__global__ void KernelPAddBackward(const dtype **losses, int count, int dim,
        int in_count,
        dtype *const *const *in_losses) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    int dim_mul_count = dim * count;
    for (int i = index; i < dim_mul_count * in_count; i += step) {
        int in_count_i = i / dim_mul_count;
        int dim_mul_count_i = i % dim_mul_count;
        int count_i = dim_mul_count_i / dim;
        int dim_i = dim_mul_count_i % dim;
        DeviceAtomicAdd(in_losses[in_count_i][count_i] + dim_i, losses[count_i][dim_i]);
    }
}

void PAddBackward(const std::vector<dtype*> &losses, int count, int dim,
        int in_count,
        std::vector<std::vector<dtype*>> &in_losses,
        cudaStream_t *stream) {
    std::vector<std::shared_ptr<NumberPointerArray>> gpu_addr;
    gpu_addr.reserve(in_losses.size());
    for (const std::vector<dtype*> &x : in_losses) {
        std::shared_ptr<NumberPointerArray> arr =
            std::make_shared<NumberPointerArray>();
        arr->init((dtype**)x.data(), x.size(), stream);
        gpu_addr.push_back(arr);
    }
    std::vector<dtype**> in_losses_gpu;
    in_losses_gpu.reserve(in_losses.size());
    for (auto &ptr : gpu_addr) {
        in_losses_gpu.push_back((dtype **)ptr->value);
    }

    NumberPointerPointerArray in_loss_arr;
    in_loss_arr.init(in_losses_gpu.data(), in_losses_gpu.size(), stream);
    NumberPointerArray out_loss_arr;
    out_loss_arr.init((dtype**)losses.data(), losses.size(), stream);

    int block_count = DefaultBlockCount(in_count * count * dim);
    KernelPAddBackward<<<block_count, TPB, 0, *stream>>>(out_loss_arr.value,
            count, dim, in_count, (dtype *const *const *)in_loss_arr.value);
    CheckCudaError();
}

//__global__ void KernelSoftMaxLoss(const dtype **vals, dtype **losses,
//        int *correct_count, int *answers, int batchsize, int count, int dim) {
//    volatile __shared__ int opt_label;
//    volatile __shared__ dtype shared_val[TPB];
//    volatile __shared__ int64_t max_indexes[TPB];
//    volatile __shared__ dtype scores_sum[TPB];
//    volatile __shared__ dtype scores[TPB];
//    int dim_i = threadIdx.x;
//    int count_i = blockIdx.x;
//    if (count_i == 0 && dim_i == 0) {
//        *correct_count = 0;
//    }
//    shared_val[dim_i] = dim_i < dim ? vals[count_i][dim_i] : -1e10;
//    max_indexes[dim_i] = dim_i;
//    __syncthreads();
//
//    for (int i = (blockDim.x >> 1); i > 0; i >>= 1) {
//        if (shared_val[threadIdx.x + i] > shared_val[threadIdx.x]) { // race
//            shared_val[threadIdx.x] = shared_val[threadIdx.x + i]; // race
//            max_indexes[threadIdx.x] = max_indexes[threadIdx.x + i]; // race
//        }
//        __syncthreads();
//    }
//
//    if (threadIdx.x == 0) {
//        opt_label = max_indexes[0];
//        if (answers[count_i] == opt_label) {
//            atomicAdd(correct_count, 1);
//        }
//    }
//    __syncthreads();
//
//    dtype max_score = vals[count_i][opt_label];
//    dtype score = dim_i < dim ? cuda_exp(vals[count_i][dim_i] - max_score) :
//        0.0f;
//    scores[dim_i] = score;
//    scores_sum[dim_i] = score;
//
//    for (int i = (blockDim.x >> 1); i > 0; i >>= 1) {
//        scores_sum[threadIdx.x] = scores_sum[threadIdx.x] +
//            scores_sum[threadIdx.x + i]; // race
//        __syncthreads();
//    }
//
//    if (dim_i < dim) {
//        losses[count_i][dim_i] = (scores[dim_i] / scores_sum[0] -
//                (dim_i == answers[count_i] ? 1 : 0)) / batchsize;
//    }
//}
//
//void SoftMaxLoss(const std::vector<dtype*> &vals, std::vector<dtype*> &losses,
//        int *correct_count,
//        const std::vector<int> &answers,
//        int batchsize,
//        int count,
//        int dim) {
//    if (dim > TPB) {
//        abort();
//    }
//    int thread_count = NextTwoIntegerPowerNumber(dim);
//    NumberPointerArray val_arr;
//    val_arr.init((dtype**)vals.data(), vals.size());
//    NumberPointerArray loss_arr;
//    loss_arr.init((dtype**)losses.data(), losses.size());
//    IntArray answer_arr;
//    answer_arr.init((int*)answers.data(), answers.size());
//    KernelSoftMaxLoss<<<count, thread_count>>>(
//            const_cast<const dtype **>(val_arr.value),
//            const_cast<dtype **>(loss_arr.value),
//            correct_count,
//            answer_arr.value,
//            batchsize,
//            count,
//            dim);
//    CheckCudaError();
//}

__global__ void KernelCrossEntropyLoss(const dtype *const *vals, const int *answers, int count,
        dtype factor,
        dtype *const *losses) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count; i += step) {
        int answer = answers[i];
        DeviceAtomicAdd(losses[i] + answer, - 1 / vals[i][answer] * factor);
    }
}

__global__ void KernelCrossEntropgyLossValue(const dtype *const *vals, const int *answers,
        int count,
        volatile dtype *global_sum,
        int *block_counter,
        dtype *result) {
    __shared__ volatile dtype shared_sum[TPB];
    __shared__ volatile bool is_last_block;
    int index = DeviceDefaultIndex();
    if (index == 0) {
        *block_counter = 0;
    }
    if (threadIdx.x == 0) {
        is_last_block = false;
    }
    shared_sum[threadIdx.x] = 0.0f;
    for (int i = index; i < count; i += blockDim.x * gridDim.x) {
        int answer_offset = answers[i];
        shared_sum[threadIdx.x] -= cuda_log(vals[i][answer_offset]);
    }

    __syncthreads();
    for (int i = (blockDim.x >> 1); i > 0; i >>= 1) {
        if (threadIdx.x < i) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + i];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        global_sum[blockIdx.x] = shared_sum[0];
        if (atomicAdd(block_counter, 1) == gridDim.x - 1) {
            is_last_block = true;
        }
    }
    __syncthreads();

    if (is_last_block) {
        dtype sum = 0.0f;
        for (int i = threadIdx.x; i < gridDim.x; i += blockDim.x) {
            sum += global_sum[i];
        }

        shared_sum[threadIdx.x] = sum;
        __syncthreads();

        for (int i = (blockDim.x >> 1); i > 0; i >>= 1) {
            if (threadIdx.x < i) {
                shared_sum[threadIdx.x] += shared_sum[threadIdx.x + i];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            *result = shared_sum[0];
        }
    }
}

dtype CrossEntropyLoss(const vector<dtype *> &vals, const vector<int> &answers, int count,
        dtype factor,
        vector<dtype *> &losses,
        cudaStream_t *stream) {
    NumberPointerArray val_arr, loss_arr;
    val_arr.init((dtype**)vals.data(), vals.size(), stream);
    loss_arr.init((dtype**)losses.data(), losses.size(), stream);
    IntArray answer_arr;
    answer_arr.init((int*)answers.data(), answers.size(), stream);

    KernelCrossEntropyLoss<<<DefaultBlockCount(count), TPB>>>(val_arr.value, answer_arr.value,
            count, factor, (dtype *const *)loss_arr.value);
    CheckCudaError();

    int block_count = DefaultBlockCount(count);
    NumberArray global_sum;
    global_sum.init(block_count);
    DeviceInt block_counter;
    block_counter.init();
    DeviceNumber result;
    result.init();
    KernelCrossEntropgyLossValue<<<block_count, TPB, 0, *stream>>>(val_arr.value,
            answer_arr.value, count, global_sum.value, block_counter.value, result.value);
    CheckCudaError();
    result.copyFromDeviceToHost(stream);
    return result.v * factor;
}

__global__ void KernelMultiCrossEntropyLoss(const dtype *const *vals, const int *const *answers,
        int count,
        int dim,
        dtype factor,
        dtype *const *losses) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count * dim; i += step) {
        int count_i = i / dim;
        int dim_i = i % dim;
        dtype val = vals[count_i][dim_i];
        dtype grad = (answers[count_i][dim_i] ? (-1 / val) : (1 / (1 - val))) * factor;
        DeviceAtomicAdd(losses[count_i] + dim_i, grad);
    }
}

__global__ void KernelMultiCrossEntropyLossVector(const dtype *const *in_vals,
        const int *const *answers,
        int count,
        int dim,
        dtype *const *result) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count * dim; i += step) {
        int count_i = i / dim;
        int dim_i = i % dim;
        dtype in_val = in_vals[count_i][dim_i];
        dtype v = answers[count_i][dim_i] ? -cuda_log(in_val) : -cuda_log(1 - in_val);
        result[count_i][dim_i] = v;
    }
}

template<typename T>
vector<T *> GPUArrayVectors(vector<shared_ptr<GPUArray<T>>> &ptrs, int count, int dim) {
    vector<T *> result;
    for (int i = 0; i < count; ++i) {
        shared_ptr<GPUArray<T>> e(new GPUArray<T>);
        e->init(dim);
        ptrs.push_back(e);
        result.push_back((T *)e->value);
    }
    return result;
}

dtype MultiCrossEntropyLoss(const vector<dtype*> &vals, const vector<vector<int>> &answers,
        int count,
        int dim,
        dtype factor,
        const vector<dtype*> &losses,
        cudaStream_t *stream) {
    int block_count = DefaultBlockCount(count * dim);
    NumberPointerArray val_arr, loss_arr;
    val_arr.init((dtype**)vals.data(), count, stream);
    loss_arr.init((dtype**)losses.data(), count, stream);

    vector<shared_ptr<IntArray>> answer_gpus;
    vector<int *> answer_gpu_pointers;
    for (auto &answer : answers) {
        shared_ptr<IntArray> answer_gpu(new IntArray);
        answer_gpu->init(answer.data(), answer.size(), stream);
        answer_gpus.push_back(answer_gpu);
        answer_gpu_pointers.push_back(answer_gpu->value);
    }

    IntPointerArray answer_arr;
    answer_arr.init((int**)answer_gpu_pointers.data(), count, stream);
    KernelMultiCrossEntropyLoss<<<block_count, TPB, 0, *stream>>>(val_arr.value, answer_arr.value,
            count, dim, factor, (dtype *const *)loss_arr.value);
    CheckCudaError();

    vector<shared_ptr<NumberArray>> nums;
    vector<dtype *> logged_vec = GPUArrayVectors(nums, count, dim);

    NumberPointerArray logged_arr;
    logged_arr.init(logged_vec.data(), count, stream);

    KernelMultiCrossEntropyLossVector<<<block_count, TPB, 0, *stream>>>(val_arr.value,
            answer_arr.value, count, dim, (dtype *const *)logged_arr.value);
    CheckCudaError();

    vector<shared_ptr<NumberArray>> ce_loss_arrs;
    vector<dtype *> ce_losses = GPUArrayVectors(ce_loss_arrs, count, 1);
    for (auto &ptr : ce_loss_arrs) {
        vector<dtype> vec = ptr->toCpu(stream);
    }
    vector<const dtype *> const_logged_arr;
    auto return_const = [](dtype *v) -> const dtype* {
        return const_cast<const dtype*>(v);
    };
    transform(logged_vec.begin(), logged_vec.end(), back_inserter(const_logged_arr), return_const);

    vector<int> dims;
    for (int i = 0; i < count; ++i) {
        dims.push_back(dim);
    }
    VectorSumForward(const_logged_arr, count, dims, ce_losses, stream);

    dtype ce_loss_sum = 0.0f;

    for (auto &ptr : ce_loss_arrs) {
        vector<dtype> vec = ptr->toCpu(stream);
        if (vec.size() != 1) {
            cerr << "vec size is not 1" << endl;
            abort();
        }
        dtype l = vec.front() * factor;
        ce_loss_sum += l;
    }

    return ce_loss_sum;
}

__global__ void KernelKLCrossEntropyLoss(const dtype *const *vals, const dtype *const *answers,
        int count,
        int dim,
        dtype factor,
        dtype *const *losses) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count * dim; i += step) {
        int count_i = i / dim;
        int dim_i = i % dim;
        dtype val = vals[count_i][dim_i];
        dtype grad = -answers[count_i][dim_i] / val * factor;
        DeviceAtomicAdd(losses[count_i] + dim_i, grad);
    }
}

__global__ void KernelKLCrossEntropyLossVector(const dtype *const *in_vals,
        const dtype *const *answers,
        int count,
        int dim,
        dtype *const *result) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count * dim; i += step) {
        int count_i = i / dim;
        int dim_i = i % dim;
        dtype in_val = in_vals[count_i][dim_i];
        dtype v = -answers[count_i][dim_i] * cuda_log(in_val);
        result[count_i][dim_i] = v;
    }
}

dtype KLCrossEntropyLoss(const vector<dtype*> &vals,
        const vector<shared_ptr<vector<dtype>>> &answers,
        int count,
        int dim,
        dtype factor,
        const vector<dtype*> &losses,
        cudaStream_t *stream) {
    int block_count = DefaultBlockCount(count * dim);
    NumberPointerArray val_arr, loss_arr;
    val_arr.init((dtype**)vals.data(), count, stream);
    loss_arr.init((dtype**)losses.data(), count, stream);

    vector<shared_ptr<NumberArray>> answer_gpus;
    vector<dtype *> answer_gpu_pointers;
    for (auto &answer : answers) {
        shared_ptr<NumberArray> answer_gpu(new NumberArray);
        answer_gpu->init(answer->data(), answer->size(), stream);
        answer_gpus.push_back(answer_gpu);
        answer_gpu_pointers.push_back(answer_gpu->value);
    }

    NumberPointerArray answer_arr;
    answer_arr.init((dtype**)answer_gpu_pointers.data(), count, stream);
    KernelKLCrossEntropyLoss<<<block_count, TPB, 0, *stream>>>(val_arr.value, answer_arr.value,
            count, dim, factor, (dtype *const *)loss_arr.value);
    CheckCudaError();

    vector<shared_ptr<NumberArray>> nums;
    vector<dtype *> logged_vec = GPUArrayVectors(nums, count, dim);

    NumberPointerArray logged_arr;
    logged_arr.init(logged_vec.data(), count, stream);

    KernelKLCrossEntropyLossVector<<<block_count, TPB, 0, *stream>>>(val_arr.value,
            answer_arr.value, count, dim, (dtype *const *)logged_arr.value);
    CheckCudaError();

    vector<shared_ptr<NumberArray>> ce_loss_arrs;
    vector<dtype *> ce_losses = GPUArrayVectors(ce_loss_arrs, count, 1);
    for (auto &ptr : ce_loss_arrs) {
        vector<dtype> vec = ptr->toCpu(stream);
    }
    vector<const dtype *> const_logged_arr;
    auto return_const = [](dtype *v) -> const dtype* {
        return const_cast<const dtype*>(v);
    };
    transform(logged_vec.begin(), logged_vec.end(), back_inserter(const_logged_arr), return_const);

    vector<int> dims;
    for (int i = 0; i < count; ++i) {
        dims.push_back(dim);
    }
    VectorSumForward(const_logged_arr, count, dims, ce_losses, stream);

    dtype ce_loss_sum = 0.0f;

    for (auto &ptr : ce_loss_arrs) {
        vector<dtype> vec = ptr->toCpu(stream);
        if (vec.size() != 1) {
            cerr << "vec size is not 1" << endl;
            abort();
        }
        dtype l = vec.front() * factor;
        ce_loss_sum += l;
    }

    return ce_loss_sum;
}

__global__ void KernelMax(const dtype *const *v, int count, int dim, volatile dtype *block_maxes,
        volatile int *block_max_is,
        int *block_counters,
        int *max_indexes,
        dtype *max_vals) {
    __shared__ volatile dtype shared_max[TPB];
    __shared__ volatile int shared_max_i[TPB];
    __shared__ volatile bool is_last_block;
    if (threadIdx.x == 0 && blockIdx.y == 0) {
        block_counters[blockIdx.x] = 0;
    }
    if (threadIdx.x == 0) {
        is_last_block = false;
    }

    int count_i = blockIdx.x;
    int offset = blockIdx.y * blockDim.x + threadIdx.x;
    shared_max[threadIdx.x] = offset < dim ? v[count_i][offset] : -1e10;
    shared_max_i[threadIdx.x] = offset;
    __syncthreads();

    for (int i = (blockDim.x >> 1); i > 0; i >>= 1) {
        if (threadIdx.x < i && shared_max[threadIdx.x] < shared_max[threadIdx.x + i]) {
            shared_max[threadIdx.x] = shared_max[threadIdx.x + i];
            shared_max_i[threadIdx.x] = shared_max_i[threadIdx.x + i];
        }
        __syncthreads();
    }

    int block_maxes_offset = blockIdx.x * gridDim.y + blockIdx.y;
    if (threadIdx.x == 0) {
        block_maxes[block_maxes_offset] = shared_max[0];
        block_max_is[block_maxes_offset] = shared_max_i[0];
        if (atomicAdd(block_counters + blockIdx.x, 1) == gridDim.y - 1) {
            is_last_block = true;
        }
    }
    __syncthreads();

    if (is_last_block) {
        dtype max = -1e10;
        int max_i = 100000;
        for (int i = threadIdx.x; i < gridDim.y; i += blockDim.x) {
            int offset = blockIdx.x * gridDim.y + i;
            if (block_maxes[offset] > max) {
                max = block_maxes[offset];
                max_i = block_max_is[offset];
            }
        }

        shared_max[threadIdx.x] = max;
        shared_max_i[threadIdx.x] = max_i;
        __syncthreads();

        for (int i = (blockDim.x >> 1); i > 0; i >>= 1) {
            if (threadIdx.x < i && shared_max[threadIdx.x + i] > shared_max[threadIdx.x]) {
                shared_max[threadIdx.x] = shared_max[threadIdx.x + i];
                shared_max_i[threadIdx.x] = shared_max_i[threadIdx.x + i];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            max_vals[count_i] = shared_max[0];
            max_indexes[count_i] = shared_max_i[0];
        }
    }
}

__global__ void KernelSingleMax(const dtype *const *v, int count, int dim,
        int *max_indexes,
        dtype *max_vals) {
    for (int count_i = 0; count_i < count; ++count_i) {
        dtype max_val = -1e10;
        int max_i;
        for (int dim_i = 0; dim_i < dim; ++ dim_i) {
            if (v[count_i][dim_i] > max_val) {
                max_val = v[count_i][dim_i];
                max_i = dim_i;
            }
        }

        max_indexes[count_i] = max_i;
        max_vals[count_i] = max_val;
    }
}

void Max(const dtype *const *v, int count, int dim, int *max_indexes, dtype *max_vals,
        cudaStream_t *stream) {
    int thread_count = min(NextTwoIntegerPowerNumber(dim), TPB);
    int block_y_count = (dim - 1 + thread_count) / thread_count;
    dim3 block_dim(count, block_y_count, 1);

    NumberArray block_maxes;
    block_maxes.init(block_y_count * count);
    IntArray block_max_is, block_counters;
    block_max_is.init(block_y_count * count);
    block_counters.init(count);

    KernelMax<<<block_dim, thread_count, 0, *stream>>>(v, count, dim, block_maxes.value,
            block_max_is.value, block_counters.value, max_indexes, max_vals);
    CheckCudaError();
#if TEST_CUDA
    NumberArray max_val_arr;
    IntArray max_indexer_arr;
    max_val_arr.init(count);
    max_indexer_arr.init(count);
    KernelSingleMax<<<1, 1>>>(v, count, dim, max_indexer_arr.value, max_val_arr.value);
    CheckCudaError();
    vector<int> max_indexer_target(count), max_indexer_gold(count);
    MyCudaMemcpy(max_indexer_target.data(), max_indexes, count * sizeof(int),
            cudaMemcpyDeviceToHost, nullptr);
    MyCudaMemcpy(max_indexer_gold.data(), max_indexer_arr.value, count * sizeof(int),
            cudaMemcpyDeviceToHost, nullptr);
    for (int i = 0; i < count; ++i) {
        if (max_indexer_target.at(i) != max_indexer_gold.at(i)) {
            cerr << format("max_indexer_target:%1% max_indexer_gold:%2%") % max_indexer_target.at(i)
                % max_indexer_gold.at(i) << endl;
            PrintNums(v, i, dim);
            abort();
        }
    }
#endif

    CheckCudaError();
}

vector<int> Predict(const vector<dtype*> &vals, int count, int dim, cudaStream_t *stream) {
    NumberPointerArray val_arr;
    val_arr.init((dtype**)vals.data(), vals.size(), stream);
    IntArray max_index_arr;
    max_index_arr.init(vals.size());
    NumberArray max_val_arr;
    max_val_arr.init(vals.size());
    Max(val_arr.value, count, dim, max_index_arr.value, max_val_arr.value, stream);
    return max_index_arr.toCpu(stream);
}


//__global__ void KernelSum(const dtype *const *v, int count, int dim, volatile dtype *block_sums,
//        int *block_counters,
//        dtype *sum_vals) {
//    __shared__ volatile dtype shared_sum[TPB];
//    __shared__ volatile bool is_last_block;
//    if (threadIdx.x == 0 && blockIdx.y == 0) {
//        block_counters[blockIdx.x] = 0;
//    }
//    if (threadIdx.x == 0) {
//        is_last_block = false;
//    }
//
//    int count_i = blockIdx.x;
//    int offset = blockIdx.y * blockDim.x + threadIdx.x;
//    shared_sum[threadIdx.x] = offset < dim ? v[count_i][offset] : 0.0f;
//    __syncthreads();
//
//    for (int i = (blockDim.x >> 1); i > 0; i >>= 1) {
//        if (threadIdx.x < i) {
//            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + i];
//        }
//        __syncthreads();
//    }
//
//    int block_sums_offset = blockIdx.x * gridDim.y + blockIdx.y;
//    if (threadIdx.x == 0) {
//        block_sums[block_sums_offset] = shared_sum[0];
//        if (atomicAdd(block_counters + blockIdx.x, 1) == gridDim.y - 1) {
//            is_last_block = true;
//        }
//    }
//    __syncthreads();
//
//    if (is_last_block) {
//        dtype sum = 0.0f;
//        for (int i = threadIdx.x; i < gridDim.y; i += blockDim.x) {
//            int offset = blockIdx.x * gridDim.y + i;
//            sum += block_sums[offset];
//        }
//
//        shared_sum[threadIdx.x] = sum;
//        __syncthreads();
//
//        for (int i = (blockDim.x >> 1); i > 0; i >>= 1) {
//            if (threadIdx.x < i) {
//                shared_sum[threadIdx.x] += shared_sum[threadIdx.x + i];
//            }
//            __syncthreads();
//        }
//
//        if (threadIdx.x == 0) {
//            sum_vals[count_i] = shared_sum[0];
//        }
//    }
//}
//
//void Sum(const dtype *const *v, int count, int dim, dtype *sum_vals) {
//    int thread_count = min(NextTwoIntegerPowerNumber(dim), TPB);
//    int block_y_count = (dim - 1 + thread_count) / thread_count;
//    dim3 block_dim(count, block_y_count, 1);
//
//    NumberArray block_sums;
//    block_sums.init(block_y_count * count);
//    IntArray block_counters;
//    block_counters.init(count);
//
//    KernelSum<<<block_dim, thread_count>>>(v, count, dim, block_sums.value, block_counters.value,
//            sum_vals);
//    CheckCudaError();
//}

__global__ void KernelMaxScalarForward(const dtype *const *v, int count, int* dims, int max_dim,
        volatile dtype *block_maxes,
        volatile int *block_max_is,
        int *block_counters,
        int *max_indexes,
        dtype *const *max_vals) {
    __shared__ volatile dtype shared_max[TPB];
    __shared__ volatile int shared_max_i[TPB];
    __shared__ volatile bool is_last_block;
    if (threadIdx.x == 0 && blockIdx.y == 0) {
        block_counters[blockIdx.x] = 0;
    }
    if (threadIdx.x == 0) {
        is_last_block = false;
    }

    int count_i = blockIdx.x;
    int offset = blockIdx.y * blockDim.x + threadIdx.x;
    shared_max[threadIdx.x] = offset < dims[count_i] ? v[count_i][offset] : -1e10;
    shared_max_i[threadIdx.x] = offset;
    __syncthreads();

    for (int i = (blockDim.x >> 1); i > 0; i >>= 1) {
        if (threadIdx.x < i && shared_max[threadIdx.x] < shared_max[threadIdx.x + i]) {
            shared_max[threadIdx.x] = shared_max[threadIdx.x + i];
            shared_max_i[threadIdx.x] = shared_max_i[threadIdx.x + i];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        int block_maxes_offset = blockIdx.x * gridDim.y + blockIdx.y;
        int max_ii = shared_max_i[0];
        if (max_ii < 0 || max_ii >= max_dim) {
            printf("threadIdx.x == 0 after first reduce max_ii:%d v:%f\n", max_ii, shared_max[0]);
            for (int i = 0; i < TPB; ++i) {
                printf("shared_max[%d]:%f shared_max_i[%d]:%d\n", i, shared_max[i], i,
                        shared_max_i[i]);
            }
            assert(false);
        }
        block_maxes[block_maxes_offset] = shared_max[0];
        block_max_is[block_maxes_offset] = shared_max_i[0];
        if (atomicAdd(block_counters + blockIdx.x, 1) == gridDim.y - 1) {
            is_last_block = true;
        }
    }
    __syncthreads();

    if (is_last_block) {
        dtype max = -1e10;
        int max_i = 100000;
        for (int i = threadIdx.x; i < gridDim.y; i += blockDim.x) {
            int offset = blockIdx.x * gridDim.y + i;
            int max_ii = block_max_is[offset];
            if (max_ii < 0 || max_ii >= max_dim) {
                printf("offset:%d is_last_block block_maxes[offset]:%f block_max_is[offset]:%d\n",
                        offset, block_maxes[offset], block_max_is[offset]);
                assert(false);
            }
            if (block_maxes[offset] > max) {
                max = block_maxes[offset];
                max_i = block_max_is[offset];
            }
        }

        shared_max[threadIdx.x] = max;
        shared_max_i[threadIdx.x] = max_i;
//        printf("max:%f max_i:%d\n", max, max_i);
        __syncthreads();

        for (int i = (blockDim.x >> 1); i > 0; i >>= 1) {
            if (threadIdx.x < i && shared_max[threadIdx.x + i] > shared_max[threadIdx.x]) {
                shared_max[threadIdx.x] = shared_max[threadIdx.x + i];
                shared_max_i[threadIdx.x] = shared_max_i[threadIdx.x + i];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            max_vals[count_i][0] = shared_max[0];
            max_indexes[count_i] = shared_max_i[0];
            int max_ii = max_indexes[count_i];
            if (max_ii < 0 || max_ii >= max_dim) {
                printf("threadIdx.x == 0 max_i:%d count_i:%d max_val:%f\n", max_indexes[count_i],
                        count_i, max_vals[count_i][0]);
                assert(false);
            }
        }
    }
}

void MaxScalarForward(const vector<const dtype*> &inputs, int count, const vector<int> &dims,
        vector<dtype*> &results,
        vector<int> &max_indexes,
        cudaStream_t *stream) {
    int max_dim = *max_element(dims.begin(), dims.end());
    int thread_count = min(NextTwoIntegerPowerNumber(max_dim), TPB);
    int block_y_count = (max_dim - 1 + thread_count) / thread_count;
    dim3 block_dim(count, block_y_count, 1);

    NumberArray block_maxes;
    block_maxes.init(block_y_count * count);
    IntArray block_max_is, block_counters;
    block_max_is.init(block_y_count * count);
    block_counters.init(count);

    NumberPointerArray input_arr;
    input_arr.init((dtype**)inputs.data(), inputs.size(), stream);
    NumberPointerArray result_arr;
    result_arr.init((dtype**)results.data(), results.size(), stream);
    IntArray max_index_arr;
    max_index_arr.init(max_indexes.size());

    IntArray dim_arr;
    dim_arr.init(dims.data(), dims.size(), stream);

    KernelMaxScalarForward<<<block_dim, thread_count, 0, *stream>>>(
            (const dtype *const *)input_arr.value,
            count, dim_arr.value, max_dim, block_maxes.value, block_max_is.value,
            block_counters.value,
            max_index_arr.value, (dtype *const *)result_arr.value);
    CheckCudaError();
    MyCudaMemcpy(max_indexes.data(), max_index_arr.value, count * sizeof(int),
            cudaMemcpyDeviceToHost, stream);
}

__global__ void KernelMaxScalarBackward(const dtype *const *losses, const int *indexes, int count,
        dtype *const *input_losses) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count; i += step) {
        DeviceAtomicAdd(input_losses[i] + indexes[i], losses[i][0]);
    }
}

void MaxScalarBackward(const vector<const dtype *> &losses, const vector<int> &indexes, int count,
        const vector<dtype*> &input_losses,
        cudaStream_t *stream) {
    int block_count = DefaultBlockCount(count);
    NumberPointerArray loss_arr, input_loss_arr;
    loss_arr.init((dtype**)losses.data(), losses.size(), stream);
    input_loss_arr.init((dtype**)input_losses.data(), input_losses.size(), stream);
    IntArray index_arr;
    index_arr.init((int*)indexes.data(), indexes.size(), stream);
    KernelMaxScalarBackward<<<block_count, TPB, 0, *stream>>>((const dtype *const *)loss_arr.value,
            index_arr.value, count, (dtype *const *)input_loss_arr.value);
    CheckCudaError();
}

__global__ void KernelVectorSumForward(const dtype *const *v, int count, int *dims,
        volatile dtype *block_sums,
        int *block_counters,
        dtype *const *results) {
    __shared__ volatile dtype shared_sum[TPB];
    __shared__ volatile bool is_last_block;
    if (threadIdx.x == 0 && blockIdx.y == 0) {
        block_counters[blockIdx.x] = 0;
    }
    if (threadIdx.x == 0) {
        is_last_block = false;
    }

    int count_i = blockIdx.x;
    int offset = blockIdx.y * blockDim.x + threadIdx.x;
    shared_sum[threadIdx.x] = offset < dims[count_i] ? v[count_i][offset] : 0.0f;
    __syncthreads();

    for (int i = (blockDim.x >> 1); i > 0; i >>= 1) {
        if (threadIdx.x < i) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + i];
        }
        __syncthreads();
    }

    int block_sums_offset = blockIdx.x * gridDim.y + blockIdx.y;
    if (threadIdx.x == 0) {
        block_sums[block_sums_offset] = shared_sum[0];
        if (atomicAdd(block_counters + blockIdx.x, 1) == gridDim.y - 1) {
            is_last_block = true;
        }
    }
    __syncthreads();

    if (is_last_block) {
        dtype sum = 0.0f;
        for (int i = threadIdx.x; i < gridDim.y; i += blockDim.x) {
            int offset = blockIdx.x * gridDim.y + i;
            sum += block_sums[offset];
        }

        shared_sum[threadIdx.x] = sum;
        __syncthreads();

        for (int i = (blockDim.x >> 1); i > 0; i >>= 1) {
            if (threadIdx.x < i) {
                shared_sum[threadIdx.x] += shared_sum[threadIdx.x + i];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            results[count_i][0] = shared_sum[0];
        }
    }
}


void VectorSumForward(const vector<const dtype *> &inputs, int count, const vector<int> &dims,
        vector<dtype*> &results, cudaStream_t *stream) {
    int max_dim = *max_element(dims.begin(), dims.end());
    int thread_count = min(NextTwoIntegerPowerNumber(max_dim), TPB);
    int block_y_count = (max_dim - 1 + thread_count) / thread_count;
    dim3 block_dim(count, block_y_count, 1);

    NumberArray block_sums;
    block_sums.init(block_y_count * count);
    IntArray block_counters;
    block_counters.init(count);

    NumberPointerArray input_arr;
    input_arr.init((dtype**)inputs.data(), inputs.size(), stream);
    NumberPointerArray result_arr;
    result_arr.init((dtype**)results.data(), results.size(), stream);

    IntArray dim_arr;
    dim_arr.init(dims.data(), dims.size(), stream);

    KernelVectorSumForward<<<block_dim, thread_count, 0, *stream>>>(
            (const dtype *const *)input_arr.value, count, dim_arr.value, block_sums.value,
            block_counters.value, (dtype *const *)result_arr.value);
    CheckCudaError();
}

__global__ void KernelVectorSumBackward(const dtype *const *losses, int count, int *dims,
        int max_dim,
        dtype *const *input_losses) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count * max_dim; i += step) {
        int count_i = i / max_dim;
        int dim_i = i % max_dim;

        if (dim_i < dims[count_i]) {
            DeviceAtomicAdd(input_losses[count_i] + dim_i, losses[count_i][0]);
        }
    }
}

void VectorSumBackward(const vector<const dtype*> &losses, int count, const vector<int> &dims,
        vector<dtype*> &input_losses,
        cudaStream_t *stream) {
    int max_dim = *max_element(dims.begin(), dims.end());
    int block_count = DefaultBlockCount(count * max_dim);
    NumberPointerArray loss_arr, input_loss_arr;
    loss_arr.init((dtype**)losses.data(), losses.size(), stream);
    input_loss_arr.init((dtype**)input_losses.data(), input_losses.size(), stream);
    IntArray dim_arr;
    dim_arr.init(dims.data(), dims.size(), stream);
    KernelVectorSumBackward<<<block_count, TPB, 0, *stream>>>(
            (const dtype *const *)loss_arr.value, count, dim_arr.value, max_dim,
            (dtype *const *)input_loss_arr.value);
    CheckCudaError();
}

__global__ void KernelScalarToVectorForward(const dtype* const* inputs, int count, int *dims,
        int max_dim,
        dtype *const *results) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count * max_dim; i += step) {
        int count_i = i / max_dim;
        int dim_i = i % max_dim;
        if (dim_i < dims[count_i]) {
            results[count_i][dim_i] = inputs[count_i][0];
        }
    }
}

void ScalarToVectorForward(const vector<const dtype*> &inputs, int count, const vector<int> &dims,
        vector<dtype*> &results,
        cudaStream_t *stream) {
    int max_dim = *max_element(dims.begin(), dims.end());
    int block_count = DefaultBlockCount(max_dim * count);
    NumberPointerArray input_arr;
    input_arr.init((dtype**)inputs.data(), inputs.size(), stream);
    NumberPointerArray result_arr;
    result_arr.init((dtype**)results.data(), inputs.size(), stream);
    IntArray dim_arr;
    dim_arr.init(dims.data(), dims.size(), stream);

    KernelScalarToVectorForward<<<block_count, TPB, 0, *stream>>>(
            (const dtype* const *)input_arr.value, count, dim_arr.value, max_dim,
            (dtype *const *)result_arr.value);
    CheckCudaError();
}

__global__ void KernelScalarToVectorBackward(const dtype *const *losses, int count, int *dims,
        volatile dtype *block_sums,
        int *block_counters,
        dtype *const *input_losses) {
    __shared__ volatile dtype shared_sum[TPB];
    __shared__ volatile bool is_last_block;
    if (threadIdx.x == 0 && blockIdx.y == 0) {
        block_counters[blockIdx.x] = 0;
    }
    if (threadIdx.x == 0) {
        is_last_block = false;
    }

    int count_i = blockIdx.x;
    int offset = blockIdx.y * blockDim.x + threadIdx.x;
    shared_sum[threadIdx.x] = offset < dims[count_i] ? losses[count_i][offset] : 0.0f;
    __syncthreads();

    for (int i = (blockDim.x >> 1); i > 0; i >>= 1) {
        if (threadIdx.x < i) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + i];
        }
        __syncthreads();
    }

    int block_sums_offset = blockIdx.x * gridDim.y + blockIdx.y;
    if (threadIdx.x == 0) {
        block_sums[block_sums_offset] = shared_sum[0];
        if (atomicAdd(block_counters + blockIdx.x, 1) == gridDim.y - 1) {
            is_last_block = true;
        }
    }
    __syncthreads();

    if (is_last_block) {
        dtype sum = 0.0f;
        for (int i = threadIdx.x; i < gridDim.y; i += blockDim.x) {
            int offset = blockIdx.x * gridDim.y + i;
            sum += block_sums[offset];
        }

        shared_sum[threadIdx.x] = sum;
        __syncthreads();

        for (int i = (blockDim.x >> 1); i > 0; i >>= 1) {
            if (threadIdx.x < i) {
                shared_sum[threadIdx.x] += shared_sum[threadIdx.x + i];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            DeviceAtomicAdd(input_losses[count_i], shared_sum[0]);
        }
    }
}

void ScalarToVectorBackward(const vector<const dtype*> &losses, int count, const vector<int> &dims,
        vector<dtype*> &input_losses,
        cudaStream_t *stream) {
    int max_dim = *max_element(dims.begin(), dims.end());

    int thread_count = min(NextTwoIntegerPowerNumber(max_dim), TPB);
    int block_y_count = (max_dim - 1 + thread_count) / thread_count;
    dim3 block_dim(count, block_y_count, 1);

    NumberArray block_sums;
    block_sums.init(block_y_count * count);
    IntArray block_counters;
    block_counters.init(count);

    NumberPointerArray loss_arr;
    loss_arr.init((dtype**)losses.data(), losses.size(), stream);
    NumberPointerArray input_loss_arr;
    input_loss_arr.init((dtype**)input_losses.data(), input_losses.size(), stream);

    IntArray dim_arr;
    dim_arr.init(dims.data(), dims.size(), stream);

    KernelScalarToVectorBackward<<<block_dim, thread_count, 0, *stream>>>(
            (const dtype *const *)loss_arr.value,
            count, dim_arr.value, block_sums.value, block_counters.value,
            (dtype *const *)input_loss_arr.value);
    CheckCudaError();
}

__global__ void KernelBiasForward(const dtype *const *in_vals, const dtype *bias, int count,
        int dim,
        dtype *const *vals) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count * dim; i += step) {
        int count_i = i / dim;
        int dim_i = i % dim;
        vals[count_i][dim_i] = in_vals[count_i][dim_i] + bias[dim_i];
    }
}

void BiasForward(const vector<dtype*> &in_vals, const dtype *bias, int count, int dim,
        const vector<dtype *> &vals, cudaStream_t *stream) {
    int block_count = DefaultBlockCount(count * dim);
    NumberPointerArray in_arr, val_arr;
    in_arr.init(in_vals.data(), in_vals.size(), stream);
    val_arr.init(vals.data(), vals.size(), stream);
    KernelBiasForward<<<block_count, TPB, 0, *stream>>>(in_arr.value, bias, count, dim,
            (dtype *const *)val_arr.value);
}

__global__ void KernelBiasBackward(const dtype *const *losses, int count, int dim,
        dtype *bias_losses,
        dtype *const *in_losses) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count * dim; i += step) {
        int count_i = i / dim;
        int dim_i = i % dim;
        DeviceAtomicAdd(bias_losses + dim_i, losses[count_i][dim_i]);
        DeviceAtomicAdd(in_losses[count_i] + dim_i, losses[count_i][dim_i]);
    }
}

void BiasBackward(const vector<dtype *> &losses, int count, int dim, dtype *bias_loss,
        const vector<dtype *> input_losses,
        cudaStream_t *stream) {
    int block_count = DefaultBlockCount(count * dim);
    NumberPointerArray loss_arr, input_loss_arr;
    loss_arr.init(losses.data(), losses.size(), stream);
    input_loss_arr.init(input_losses.data(), input_losses.size(), stream);
    KernelBiasBackward<<<block_count, TPB, 0, *stream>>>(loss_arr.value, count, dim, bias_loss,
            (dtype *const *)input_loss_arr.value);
}

__global__ void KernelSquareSum(const dtype *v, int len, volatile dtype *global_sum,
        int *block_counter, dtype *result) {
    __shared__ volatile dtype shared_sum[TPB];
    __shared__ volatile bool is_last_block;
    int index = DeviceDefaultIndex();
    if (index == 0) {
        *block_counter = 0;
    }
    if (threadIdx.x == 0) {
        is_last_block = false;
    }
    shared_sum[threadIdx.x] = 0.0f;
    for (int i = index; i < len; i += blockDim.x * gridDim.x) {
        shared_sum[threadIdx.x] += v[i] * v[i];
    }

    __syncthreads();
    for (int i = (blockDim.x >> 1); i > 0; i >>= 1) {
        if (threadIdx.x < i) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + i];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        global_sum[blockIdx.x] = shared_sum[0];
        if (atomicAdd(block_counter, 1) == gridDim.x - 1) {
            is_last_block = true;
        }
    }
    __syncthreads();

    if (is_last_block) {
        dtype sum = 0.0f;
        for (int i = threadIdx.x; i < gridDim.x; i += blockDim.x) {
            sum += global_sum[i];
        }

        shared_sum[threadIdx.x] = sum;
        __syncthreads();

        for (int i = (blockDim.x >> 1); i > 0; i >>= 1) {
            if (threadIdx.x < i) {
                shared_sum[threadIdx.x] += shared_sum[threadIdx.x + i];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            *result = shared_sum[0];
        }
    }
}

dtype SquareSum(const dtype *v, int len, cudaStream_t *stream) {
    int block_count = DefaultBlockCount(len);
    NumberArray global_sum;
    global_sum.init(block_count);
    DeviceInt block_counter;
    block_counter.init();
    DeviceNumber result;
    result.init();
    KernelSquareSum<<<block_count, TPB, 0, *stream>>>(v, len,
            global_sum.value, block_counter.value, result.value);
    CheckCudaError();
    result.copyFromDeviceToHost(stream);
    return result.v;
}

__global__ void KernelSquareSum(const dtype *v, const bool *indexers,
        int count,
        int dim,
        volatile dtype *global_sum,
        int *block_counter,
        dtype *result) {
    __shared__ volatile dtype shared_sum[TPB];
    __shared__ volatile bool is_last_block;
    int index = DeviceDefaultIndex();
    if (index == 0) {
        *block_counter = 0;
    }
    if (threadIdx.x == 0) {
        global_sum[blockIdx.x] = 0.0f;
        is_last_block = false;
    }
    int count_i = index / dim;
    if (index < count * dim && indexers[count_i]) {
        shared_sum[threadIdx.x] = v[index] * v[index];
    } else {
        shared_sum[threadIdx.x] = 0.0f;
    }
    __syncthreads();
    for (int i = (blockDim.x >> 1); i > 0; i >>= 1) {
        if (threadIdx.x < i) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + i];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        global_sum[blockIdx.x] = shared_sum[0];
        if (atomicAdd(block_counter, 1) == gridDim.x - 1) {
            is_last_block = true;
        }
    }
    __syncthreads();

    if (is_last_block) {
        float sum = 0.0f;
        for (int i = threadIdx.x; i < gridDim.x; i += blockDim.x) {
            sum += global_sum[i];
        }

        shared_sum[threadIdx.x] = sum;
        __syncthreads();

        for (int i = (blockDim.x >> 1); i > 0; i >>= 1) {
            if (threadIdx.x < i) {
                shared_sum[threadIdx.x] += shared_sum[threadIdx.x + i];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            *result = shared_sum[0];
        }
    }
}

dtype SquareSum(const dtype *v, const bool *indexers, int count, int dim, cudaStream_t *stream) {
    int block_count = DefaultBlockCountWithoutLimit(count * dim);
    NumberArray global_sum;
    global_sum.init(block_count);
    DeviceInt block_counter;
    block_counter.init();
    DeviceNumber result;
    result.init();
    KernelSquareSum<<<block_count, TPB, 0, *stream>>>(v, indexers,
            count, dim, global_sum.value, block_counter.value, result.value);
    CheckCudaError();
    result.copyFromDeviceToHost(stream);
    return result.v;
}

__global__ void KernelRescale(dtype *v, int len, dtype scale) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < len; i += step) {
        v[i] *= scale;
    }
}

void Rescale(dtype *v, int len, dtype scale, cudaStream_t *stream) {
    int block_count = DefaultBlockCount(len);
    KernelRescale<<<block_count, TPB, 0, *stream>>>(v, len, scale);
    CheckCudaError();
}

__global__ void KernelUpdateAdam(dtype *val, dtype *grad, int row, int col, bool is_bias,
        dtype *aux_mean,
        dtype *aux_square,
        int iter,
        dtype belta1,
        dtype belta2,
        dtype alpha,
        dtype reg,
        dtype eps,
        dtype x) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    int len = row * col;
    for (int i = index; i < len; i += step) {
        if (!is_bias) {
            grad[i] += val[i] * reg;
        }
        aux_mean[i] = belta1 * aux_mean[i] + (1 - belta1) * grad[i];
        aux_square[i] = belta2 * aux_square[i] + (1 - belta2) * grad[i] *
            grad[i];
        dtype lr_t = alpha * cuda_sqrt(1 - cuda_pow(belta2, iter + 1)) * x;
        dtype square_plus_eps = aux_square[i] + eps;
        val[i] = val[i] - aux_mean[i] * lr_t / cuda_sqrt(square_plus_eps);
    }
}

void UpdateAdam(dtype *val, dtype *grad, int row, int col, bool is_bias, dtype *aux_mean,
        dtype *aux_square,
        int iter,
        dtype belta1,
        dtype belta2,
        dtype alpha,
        dtype reg,
        dtype eps,
        cudaStream_t *stream) {
    int block_count = DefaultBlockCount(row * col);
    dtype x = 1.0f / (1 - pow(belta1, iter + 1));
    KernelUpdateAdam<<<block_count, TPB, 0, *stream>>>(val, grad, row, col, is_bias, aux_mean,
            aux_square,
            iter,
            belta1,
            belta2,
            alpha,
            reg,
            eps,
            x);
    CheckCudaError();
}

__global__ void KernelUpdateAdamW(dtype *val, dtype *grad, int row, int col, bool is_bias,
        dtype *aux_mean,
        dtype *aux_square,
        int iter,
        dtype belta1,
        dtype belta2,
        dtype alpha,
        dtype reg,
        dtype eps,
        dtype x) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    int len = row * col;
    for (int i = index; i < len; i += step) {
        aux_mean[i] = belta1 * aux_mean[i] + (1 - belta1) * grad[i];
        aux_square[i] = belta2 * aux_square[i] + (1 - belta2) * grad[i] *
            grad[i];
        dtype lr_t = alpha * cuda_sqrt(1 - cuda_pow(belta2, iter + 1)) * x;
        dtype square_plus_eps = aux_square[i] + eps;
        val[i] = (1 - (is_bias? 0.0f : reg)) * val[i] - aux_mean[i] * lr_t /
            cuda_sqrt(square_plus_eps);
    }
}

__global__ void KernelSelfPlusIters(const bool *indexers, int *iters,
        int count) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count; i += step) {
        if (indexers[i]) {
            ++iters[i];
        }
    }
}

void UpdateAdamW(dtype *val, dtype *grad, int row, int col, bool is_bias, dtype *aux_mean,
        dtype *aux_square,
        int iter,
        dtype belta1,
        dtype belta2,
        dtype alpha,
        dtype reg,
        dtype eps,
        cudaStream_t *stream) {
    int block_count = DefaultBlockCount(row * col);
    dtype x = 1.0f / (1 - pow(belta1, iter + 1));
    KernelUpdateAdamW<<<block_count, TPB, 0, *stream>>>(val, grad, row, col, is_bias, aux_mean,
            aux_square,
            iter,
            belta1,
            belta2,
            alpha,
            reg,
            eps,
            x);
    CheckCudaError();
}

__global__ void KernelUpdateAdam(dtype *val, dtype *grad, int row, int col,
        dtype *aux_mean,
        dtype *aux_square,
        const bool *indexers,
        int *iters,
        dtype belta1,
        dtype belta2,
        dtype alpha,
        dtype reg,
        dtype eps) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    int len = row * col;
    for (int i = index; i < len; i += step) {
        int count_i = i / row;
        if (indexers[count_i]) {
            if (row > 1 && col > 1) {
                grad[i] += val[i] * reg;
            }
            aux_mean[i] = belta1 * aux_mean[i] + (1 - belta1) * grad[i];
            aux_square[i] = belta2 * aux_square[i] + (1 - belta2) * grad[i] *
                grad[i];
            dtype lr_t = alpha * cuda_sqrt(1 - cuda_pow(belta2,
                        iters[count_i] + 1)) / (1 - cuda_pow(belta1,
                            iters[count_i] + 1));
            dtype square_plus_eps = aux_square[i] + eps;
            val[i] = val[i] - aux_mean[i] * lr_t / cuda_sqrt(square_plus_eps);
        }
    }
}

void UpdateAdam(dtype *val, dtype *grad, int row, int col, dtype *aux_mean,
        dtype *aux_square,
        const bool *indexers,
        int *iters,
        dtype belta1,
        dtype belta2,
        dtype alpha,
        dtype reg,
        dtype eps,
        cudaStream_t *stream) {
    int block_count = DefaultBlockCount(row * col);
    KernelUpdateAdam<<<block_count, TPB, 0, *stream>>>(val, grad, row, col, aux_mean,
            aux_square, indexers, iters, belta1, belta2, alpha, reg, eps);
    CheckCudaError();
    block_count = DefaultBlockCount(col);
    KernelSelfPlusIters<<<block_count, TPB, 0, *stream>>>(indexers, iters, col);
    CheckCudaError();
}

__global__ void KernelUpdateAdagrad(dtype *val, dtype *grad, int row, int col,
        dtype *aux_square,
        dtype alpha,
        dtype reg,
        dtype eps) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    int len = row * col;
    for (int i = index; i < len; i += step) {
        if (row > 1 && col > 1) {
            grad[i] += val[i] * reg;
        }
        aux_square[i] = aux_square[i] + grad[i] * grad[i];
        val[i] = val[i] - grad[i] * alpha / cuda_sqrt(aux_square[i] + eps);
    }
}

void UpdateAdagrad(dtype *val, dtype *grad, int row, int col,
        dtype *aux_square,
        dtype alpha,
        dtype reg,
        dtype eps,
        cudaStream_t *stream) {
    int block_count = DefaultBlockCount(row * col);
    KernelUpdateAdagrad<<<block_count, TPB, 0, *stream>>>(val, grad, row, col, aux_square,
            alpha, reg, eps);
    CheckCudaError();
}

__global__ void KernelUpdateAdagrad(dtype *val, dtype *grad, int row, int col,
        dtype *aux_square,
        const bool *indexers,
        dtype alpha,
        dtype reg,
        dtype eps) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    int len = row * col;
    for (int i = index; i < len; i += step) {
        int count_i = i / col;
        if (indexers[count_i]) {
            if (row > 1 && col > 1) {
                grad[i] += val[i] * reg;
            }
            aux_square[i] = aux_square[i] + grad[i] * grad[i];
            val[i] = val[i] - grad[i] * alpha / cuda_sqrt(aux_square[i] + eps);
        }
    }
}

void UpdateAdagrad(dtype *val, dtype *grad, int row, int col,
        dtype *aux_square,
        const bool *indexers,
        dtype alpha,
        dtype reg,
        dtype eps,
        cudaStream_t *stream) {
    int block_count = DefaultBlockCount(row * col);
    KernelUpdateAdagrad<<<block_count, TPB, 0, *stream>>>(val, grad, row, col, aux_square,
            indexers, alpha, reg, eps);
    CheckCudaError();
}

}
