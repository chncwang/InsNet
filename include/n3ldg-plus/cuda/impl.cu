#include "n3ldg_plus_cuda.h"
#include <array>
#include <cstdlib>
#include <cstddef>
#include <functional>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h>
#include <string>
#include <utility>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <unordered_map>
#include <thread>
#include <numeric>
#include <memory>
#include "fmt/core.h"
#include "n3ldg-plus/cuda/helper_cuda.h"
#include "n3ldg-plus/cuda/print.cu"
#include "n3ldg-plus/cuda/print.cuh"
#include "n3ldg-plus/cuda/memory_pool.h"
#include "n3ldg-plus/base/tensor.h"
#include "n3ldg-plus/base/memory.h"
#include "n3ldg-plus/util/profiler.h"

using n3ldg_plus::cuda::MemoryPool;
using std::vector;
using std::string;
using std::map;
using std::unordered_map;
using std::function;
using std::cerr;
using std::cout;
using std::endl;
using std::make_pair;
using std::pair;
using std::shared_ptr;
using fmt::format;

namespace n3ldg_plus {
namespace cuda {

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
constexpr int TPB_SQRT = 8;
constexpr int BLOCK_COUNT = 56;

template<typename T>
void printVector(const vector<T> &v) {
    int i = 0;
    for (const T &e : v) {
        cout << i++ << ":" << e << endl;
    }
}

void CallCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        cerr << "cuda error:" << cudaGetErrorString(status) << endl;
        abort();
    }
}

void CheckCudaError() {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        cerr << "cuda error:" << cudaGetErrorName(error) << endl;
        cerr << "cuda error:" << cudaGetErrorString(error) << endl;
        abort();
    }
}

void CallCublas(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        cerr << "cublasstatus:" << status << endl;
        abort();
    }
}

void CallCurand(curandStatus status) {
    assert(status == CURAND_STATUS_SUCCESS);
}

cublasHandle_t& GetCublasHandle() {
    static cublasHandle_t handle;
    static bool init;
    if (!init) {
        init = true;
        CallCublas(cublasCreate(&handle));
    }
    return handle;
}

void MyCudaMemcpy(void *dest, void *src, size_t count, MyCudaMemcpyKind kind) {
    cudaMemcpyKind k;
    if (kind == MyCudaMemcpyKind::DEVICE_TO_HOST) {
        k = cudaMemcpyDeviceToHost;
    } else if (kind == MyCudaMemcpyKind::HOST_TO_DEVICE) {
        k = cudaMemcpyHostToDevice;
    } else if (kind == MyCudaMemcpyKind::DEVICE_TO_DEVICE) {
        k = cudaMemcpyDeviceToDevice;
    } else {
        cerr << format("MyCudaMemcpy - invalid kind:{}\n", kind);
        abort();
    }
    cudaError_t e;
    e = cudaMemcpyAsync(dest, src, count, k);
    CallCuda(e);
}

int NextTwoIntegerPowerNumber(int number) {
    int result = 1;
    while (number > result) {
        result <<= 1;
    }
    return result;
}

template <>
vector<bool> GPUArray<bool>::toCpu() const {
    bool *cpu_arr = new bool[len];
    MyCudaMemcpy(cpu_arr, value, sizeof(bool) * len, MyCudaMemcpyKind::DEVICE_TO_HOST);
    vector<bool> result;
    result.resize(len);
    for (int i = 0; i < len; ++i) {
        result.at(i) = cpu_arr[i];
    }
    delete[] cpu_arr;
    return result;
}

void DeviceInt::init() {
    if (value != nullptr) {
        MemoryPool::Ins().Free(value);
        value = nullptr;
    }
    MemoryPool::Ins().Malloc((void**)&value, sizeof(int));
}

void DeviceInt::copyFromDeviceToHost() {
    MyCudaMemcpy(&v, value, sizeof(int), MyCudaMemcpyKind::DEVICE_TO_HOST);
}

void DeviceInt::copyFromHostToDevice() {
    MyCudaMemcpy(value, &v, sizeof(int), MyCudaMemcpyKind::HOST_TO_DEVICE);
}

DeviceInt::~DeviceInt() {
    if (value != nullptr) {
        MemoryPool::Ins().Free(value);
    }
}

void DeviceNumber::init() {
    if (value != nullptr) {
        MemoryPool::Ins().Free(value);
        value = nullptr;
    }
    MemoryPool::Ins().Malloc((void**)&value, sizeof(int));
}

void DeviceNumber::copyFromDeviceToHost() {
    MyCudaMemcpy(&v, value, sizeof(dtype), MyCudaMemcpyKind::DEVICE_TO_HOST);
}

DeviceNumber::~DeviceNumber() {
    if (value != nullptr) {
        MemoryPool::Ins().Free(value);
    }
}

void Tensor1D::init(int dim) {
    initOnDevice(dim);
#if TEST_CUDA
    v = new dtype[dim];
    zero();
#endif
}

void Tensor1D::init(int dim, const shared_ptr<MemoryContainer> &container) {
    value = static_cast<dtype*>(container->allocate(dim * sizeof(dtype)));
    memory_container_ = container;
    this->dim = dim;
#if TEST_CUDA
    v = new dtype[dim];
    zero();
#endif
}

void Tensor1D::initOnMemoryAndDevice(int dim) {
    initOnDevice(dim);
    if (v != nullptr) {
        cerr << "Tensor1D::initOnMemoryAndDevice v is not nullptr" << endl;
        abort();
    }
    v = new dtype[dim];
    zero();
}

void Tensor1D::initOnDevice(int dim) {
    MemoryPool::Ins().Malloc((void**)&value, dim * sizeof(dtype));
    this->dim = dim;
}

void Tensor1D::initOnMemory(int len) {
    if (v != nullptr) {
        cerr << "Tensor1D::initOnMemory v is not nullptr" << endl;
        abort();
    }
    v = new dtype[dim];
    zero();
}

Tensor1D::Tensor1D(const Tensor1D &t) {
    dim = t.dim;
    memcpy(v, t.v, dim *sizeof(dtype));
    MyCudaMemcpy(value, t.value, dim * sizeof(dtype), MyCudaMemcpyKind::DEVICE_TO_DEVICE);
}

std::vector<dtype> Tensor1D::toCpu() const {
    std::vector<dtype> result;
    result.resize(dim);
    MyCudaMemcpy(result.data(), value, dim * sizeof(dtype), MyCudaMemcpyKind::DEVICE_TO_HOST);
    return result;
}

Tensor1D::~Tensor1D() {
    releaseMemory();
}

void Tensor1D::releaseMemory() {
    if (v != nullptr) {
        delete[] v;
    }
    v = nullptr;
    ref_count_ = 0;

    if (value != nullptr) {
        if (memory_container_ == nullptr) {
            MemoryPool::Ins().Free(value);
        } else {
            memory_container_ = nullptr;
        }
        value = nullptr;
    }
}

void Tensor1D::print() const {
    cout << "dim:" << dim << endl;
    PrintNums(value, dim);
}

void Tensor1D::copyFromHostToDevice() {
    assert(v != nullptr);
    assert(value != nullptr);
    MyCudaMemcpy(value, v, dim * sizeof(dtype), MyCudaMemcpyKind::HOST_TO_DEVICE);
}

void Tensor1D::copyFromDeviceToHost() {
    if (v == nullptr) {
        initOnMemory(dim);
    }
    MyCudaMemcpy(v, value, dim * sizeof(dtype), MyCudaMemcpyKind::DEVICE_TO_HOST);
}

__device__ int DeviceDefaultIndex();
__device__ int DeviceDefaultStep();
int DefaultBlockCount(int len);

__global__ void KernelCheckIsNumber(dtype *v, int dim, int *error) {
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

void CheckIsNumber(dtype *v, int dim) {
    int block_count = DefaultBlockCount(dim);
    DeviceInt error;
    error.init();
    KernelCheckIsNumber<<<block_count, TPB>>>(v, dim, error.value);
    CheckCudaError();
    error.copyFromDeviceToHost();
    if (error.v != 0) {
        cerr << "nan checked!" << endl;
        abort();
    }
}

void Tensor1D::checkIsNumber() const {
    CheckIsNumber(value, dim);
}

void Tensor2D::initOnMemoryAndDevice(int row, int col) {
    initOnDevice(row, col);
    if (v != nullptr) {
        cerr << "Tensor2D::initOnMemoryAndDevice v is not nullptr" << endl;
        abort();
    }
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
    MemoryPool::Ins().Malloc((void**)&value, row * col * sizeof(dtype));
    this->row = row;
    this->col = col;
    this->size = row * col;
}

Tensor2D::Tensor2D(const Tensor2D &t) {
    row = t.row;
    col = t.col;
    memcpy(v, t.v, sizeof(dtype) * row * col);
    MyCudaMemcpy(value, t.value, sizeof(dtype) * row * col, MyCudaMemcpyKind::DEVICE_TO_DEVICE);
}

Tensor2D::~Tensor2D() {
    if (value != nullptr) {
        MemoryPool::Ins().Free(value);
    }
    if (v != nullptr) {
        delete[] v;
    }
}

void Tensor2D::print() const {
    cout << "row:" << row << " col:" << col << endl;
    PrintNums(value, size);
}

void Tensor2D::copyFromHostToDevice() {
    MyCudaMemcpy(value, v, size * sizeof(dtype), MyCudaMemcpyKind::HOST_TO_DEVICE);
}

void Tensor2D::copyFromDeviceToHost() {
    MyCudaMemcpy(v, value, size * sizeof(dtype), MyCudaMemcpyKind::DEVICE_TO_HOST);
}

void Assert(bool v, const string &message, const function<void(void)> &call) {
#if TEST_CUDA
    if (!v) {
        cerr << message << endl;
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

__device__ dtype cuda_dsqrt(dtype y) {
    return 0.5 / y;
}

__device__ dtype SELU_LAMBDA = 1.0507009873554804934193349852946;
__device__ dtype SELU_ALPHA = 1.6732632423543772848170429916717;

__device__ dtype cuda_selu(dtype x) {
    return x <= 0.0f ? SELU_LAMBDA * SELU_ALPHA * (cuda_exp(x) - 1.0f) :
        SELU_LAMBDA * x;
}

__device__ dtype cuda_dselu(dtype x, dtype y) {
    return x <= 0.0f ? SELU_LAMBDA * SELU_ALPHA + y : SELU_LAMBDA;
}

void Random(dtype *v, int len, dtype bound) {
    dtype *mem = (dtype*)malloc(len * sizeof(dtype));
    assert(mem != nullptr);
    dtype min = -bound, max = bound;
    for (int i = 0; i < len; i++) {
        mem[i] =  (dtype(rand()) / RAND_MAX) * (max - min) + min;
    }

    MyCudaMemcpy(v, mem, len * sizeof(dtype), MyCudaMemcpyKind::HOST_TO_DEVICE);

    free(mem);
}

__device__ int DeviceDefaultIndex() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ int DeviceDefaultStep() {
    return gridDim.x * blockDim.x;
}

__device__ dtype DeviceAbs(dtype d) {
    return d >= 0 ? d : -d;
}

int DefaultBlockCount(int len) {
    int block_count = (len - 1 + TPB) / TPB;
    return min(block_count, BLOCK_COUNT);
}

int DefaultBlockCountWithoutLimit(int len) {
    return (len - 1 + TPB) / TPB;
}

__global__ void KernelZero(dtype *v, int len) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= len) {
        return;
    }
    v[index] = 0;
}

void Zero(dtype *v, int len) {
    int block_count = (len - 1 + TPB) /
        TPB;
    KernelZero<<<block_count, TPB>>>(v, len);
    CheckCudaError();
}

__global__ void PrintPointers(void **p, int len) {
    for (int i = 0; i < len; ++i) {
        printf("%p\n", p[i]);
    }
}

__global__ void KernelPrintNums(dtype* p, int len) {
    for (int i = 0; i < len; ++i) {
        printf("%d %f\n", i, p[i]);
    }
}

void PrintNums(dtype* p, int len) {
    KernelPrintNums<<<1, 1>>>(p, len);
    cudaDeviceSynchronize();
    CheckCudaError();
}

__global__ void KernelPrintNums(dtype **p, int index, int len) {
    for (int i = 0; i < len; ++i) {
        printf("%d %f\n", i, p[index][i]);
    }
}

void PrintNums(dtype **p, int count_i, int len) {
    KernelPrintNums<<<1, 1>>>(p, count_i, len);
    cudaDeviceSynchronize();
    CheckCudaError();
}

__global__ void KernelPrintInts(int* p, int len) {
    for (int i = 0; i < len; ++i) {
        printf("%d\n", p[i]);
    }
}

void PrintInts(int* p, int len) {
    KernelPrintInts<<<1, 1>>>(p, len);
    cudaDeviceSynchronize();
    CheckCudaError();
}

void initCuda(int device_id, float memory_in_gb) {
    cout << "device_id:" << device_id << endl;
    CallCuda(cudaSetDeviceFlags(cudaDeviceMapHost));

    CallCuda(cudaSetDevice(device_id));
    CallCuda(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
    CallCuda(cudaPrintfInit());
    MemoryPool::Ins().Init(memory_in_gb);
}

__global__ void KernelCopyFromOneVectorToMultiVectors(dtype *src,
        dtype **dest, int count, int len) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count * len; i += step) {
        int count_i = i / len;
        int len_i = i % len;
        dest[count_i][len_i] = src[i];
    }
}

void CopyFromOneVectorToMultiVals(dtype *src, vector<dtype*> &vals,
        int count,
        int len) {
    NumberPointerArray val_arr;
    val_arr.init((dtype**)vals.data(), vals.size());
    int block_count = (len * count - 1 + TPB) / TPB;
    block_count = min(block_count, BLOCK_COUNT);
    KernelCopyFromOneVectorToMultiVectors<<<block_count, TPB>>>(src,
            (dtype **)val_arr.value, count, len);
    CheckCudaError();
}

void CopyFromHostToDevice(vector<dtype*> &src,
        vector<dtype*> &dest, int count, int dim) {
    dtype *long_src = (dtype*)malloc(count * dim * sizeof(dtype));
    if (long_src == nullptr) {
        cerr << "out of memory!" << endl;
        abort();
    }
    for (int i = 0; i < count; ++i) {
        memcpy(long_src + i * dim, src.at(i), dim * sizeof(dtype));
    }
    dtype *long_dest = nullptr;
    MemoryPool::Ins().Malloc((void**)&long_dest, count * dim * sizeof(dtype*));
    CallCuda(cudaMemcpy(long_dest, long_src, count * dim * sizeof(dtype*),
                cudaMemcpyHostToDevice));
    CopyFromOneVectorToMultiVals(long_dest, dest, count, dim);
    free(long_src);
    MemoryPool::Ins().Free(long_dest);
}

__global__ void KernelCopyFromMultiVectorsToOneVector(dtype **src, dtype *dest, int count,
        int len) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count * len; i += step) {
        int count_i = i / len;
        int len_i = i % len;
        dest[i] = src[count_i][len_i];
    }
}

void CopyFromMultiVectorsToOneVector(vector<dtype*> &src,
        dtype *dest,
        int count,
        int len) {
    NumberPointerArray src_arr;
    src_arr.init((dtype**)src.data(), src.size());
    int block_count = DefaultBlockCount(len * count);
    KernelCopyFromMultiVectorsToOneVector<<<block_count, TPB>>>(
            (dtype**)src_arr.value, dest, count, len);
    CheckCudaError();
}

void CopyFromDeviceToHost(vector<dtype*> &src, vector<dtype*> &dest, int count, int dim) {
    dtype *long_src = nullptr;
    MemoryPool::Ins().Malloc((void**)&long_src, count * dim * sizeof(dtype*));
    CopyFromMultiVectorsToOneVector(src, long_src, count, dim);
    dtype *long_dest = (dtype*)malloc(count * dim * sizeof(dtype));
    if (long_dest == nullptr) {
        cerr << "out of memory!" << endl;
        abort();
    }
    CallCuda(cudaMemcpy(long_dest, long_src, count * dim * sizeof(dtype),
                cudaMemcpyDeviceToHost));
    for (int i = 0; i < count; ++i) {
        memcpy(dest.at(i), long_dest + i * dim, dim * sizeof(dtype));
    }
    MemoryPool::Ins().Free(long_src);
    free(long_dest);
}

__global__ void KernelActivationForward(ActivatedEnum activated, dtype **xs,
        int count,
        int *dims,
        int max_dim,
        dtype **ys) {
    int i = DeviceDefaultIndex();
    int count_i = i / max_dim;
    if (count_i < count) {
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
            } else if (activated == ActivatedEnum::SQRT) {
                ys[count_i][dim_i] = cuda_sqrt(xs[count_i][dim_i]);
            } else {
                printf("KernelActivationForward - error enum\n");
                assert(false);
            }
        }
    }
}

void ActivationForward(ActivatedEnum activated, vector<dtype*> &xs, int count, vector<int> &dims,
        vector<dtype*> &ys) {
    int max_dim = *max_element(dims.begin(), dims.end());
    NumberPointerArray x_arr, y_arr;
    x_arr.init((dtype**)xs.data(), xs.size());
    y_arr.init((dtype**)ys.data(), ys.size());
    int block_count = DefaultBlockCountWithoutLimit(count * max_dim);

    IntArray dim_arr;
    dim_arr.init(dims.data(), dims.size());

    KernelActivationForward<<<block_count, TPB>>>(activated, (dtype* *)x_arr.value,
            count, dim_arr.value, max_dim, (dtype **)y_arr.value);
    CheckCudaError();
}

__global__ void KernelActivationBackward(ActivatedEnum activated,
        dtype **grads,
        dtype **vals,
        int count,
        int *dims,
        int max_dim,
        dtype** in_grads) {
    int i = DeviceDefaultIndex();
    if (i < max_dim * count) {
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
            } else if (activated == ActivatedEnum::SQRT) {
                l = cuda_dsqrt(vals[count_i][dim_i]);
            } else {
                printf("KernelActivationBackward - error enum\n");
                assert(false);
            }
            dtype v = l * grads[count_i][dim_i];
            DeviceAtomicAdd(in_grads[count_i] + dim_i, v);
        }
    }
}

void ActivationBackward(ActivatedEnum activated, vector<dtype*> &grads,
        vector<dtype*> &vals,
        int count,
        vector<int> &dims,
        vector<dtype*> &in_grads) {
    int max_dim = *max_element(dims.begin(), dims.end());
    NumberPointerArray loss_arr, val_arr, in_loss_arr;
    loss_arr.init((dtype**)grads.data(), grads.size());
    val_arr.init((dtype**)vals.data(), vals.size());
    in_loss_arr.init((dtype**)in_grads.data(), in_grads.size());
    int block_count = DefaultBlockCountWithoutLimit(count * max_dim);
    IntArray dim_arr;
    dim_arr.init(dims.data(), dims.size());
    KernelActivationBackward<<<block_count, TPB>>>(activated, loss_arr.value,
            val_arr.value, count, dim_arr.value, max_dim, (dtype **)in_loss_arr.value);
    CheckCudaError();
}

__global__ void KernelDropoutForward(dtype **xs, int count, int *dims, int max_dim, int *offsets,
        bool is_training,
        dtype* drop_mask,
        dtype drop_factor,
        dtype **ys) {
    int i = DeviceDefaultIndex();
    if (i < count * max_dim) {
        int count_i = i / max_dim;
        int dim_i = i % max_dim;
        int dim = dims[count_i];
        if (dim_i < dim) {
            if (is_training) {
                int offset = offsets[count_i];
                if (drop_mask[offset + dim_i] < drop_factor) {
                    ys[count_i][dim_i] = 0.0f;
                } else {
                    ys[count_i][dim_i] = xs[count_i][dim_i];
                }
            } else {
                ys[count_i][dim_i] = (1 - drop_factor) * xs[count_i][dim_i];
            }
        }
    }
}

void DropoutForward(vector<dtype*> &xs, int count, vector<int> &dims, int max_dim,
        vector<int> &offsets,
        bool is_training,
        dtype *drop_mask,
        dtype drop_factor,
        vector<dtype*> &ys) {
    if (drop_factor < 0 || drop_factor >= 1.0f) {
        cerr << "drop value is " << drop_factor << endl;
        abort();
    }
    NumberPointerArray x_arr, y_arr;
    x_arr.init(xs.data(), count);
    y_arr.init(ys.data(), count);
    IntArray dim_arr, offset_arr;
    dim_arr.init(dims.data(), count);
    offset_arr.init(offsets.data(), count);
    int block_count = DefaultBlockCountWithoutLimit(count * max_dim);
    KernelDropoutForward<<<block_count, TPB>>>(x_arr.value, count, dim_arr.value, max_dim,
            offset_arr.value, is_training, drop_mask, drop_factor, (dtype **)y_arr.value);
    CheckCudaError();
}

__global__ void KernelDropoutBackward(dtype **grads, int count, int *dims, int max_dim,
        int *offsets,
        bool is_training,
        dtype* drop_mask,
        dtype drop_factor,
        dtype **in_grads) {
    int i = DeviceDefaultIndex();
    if (i < count * max_dim) {
        int count_i = i / max_dim;
        int dim_i = i % max_dim;
        int dim = dims[count_i];
        if (dim_i < dim) {
            if (is_training) {
                int offset = offsets[count_i];
                if (drop_mask[offset + dim_i] >= drop_factor) {
                    DeviceAtomicAdd(in_grads[count_i] + dim_i, grads[count_i][dim_i]);
                }
            } else {
                DeviceAtomicAdd(in_grads[count_i] + dim_i,
                        (1 - drop_factor) * grads[count_i][dim_i]);
            }
        }
    }
}

void DropoutBackward(vector<dtype*> &grads, int count, vector<int> &dims, int max_dim,
        vector<int> &offsets,
        bool is_training,
        dtype *drop_mask,
        dtype drop_factor,
        vector<dtype*> &in_grads) {
    if (drop_factor < 0 || drop_factor >= 1) {
        cerr << "drop value is " << drop_factor << endl;
        abort();
    }
    NumberPointerArray grad_arr, in_grad_arr;
    grad_arr.init((dtype**)grads.data(), grads.size());
    in_grad_arr.init((dtype**)in_grads.data(), in_grads.size());
    IntArray dim_arr, offset_arr;
    dim_arr.init(dims.data(), count);
    offset_arr.init(offsets.data(), count);
    int block_count = DefaultBlockCountWithoutLimit(count * max_dim);
    KernelDropoutBackward<<<block_count, TPB>>>(grad_arr.value, count, dim_arr.value, max_dim,
            offset_arr.value, is_training, drop_mask, drop_factor, (dtype **)in_grad_arr.value);
    CheckCudaError();
}

__global__ void KernelBucketForward(dtype *input, int count, int dim, dtype **ys) {
    int index = DeviceDefaultIndex();
    for (int i = index; i < count * dim; i+= DeviceDefaultStep()) {
        int count_i = i / dim;
        int dim_i = i % dim;
        ys[count_i][dim_i] = input[count_i * dim + dim_i];
    }
}

void BucketForward(vector<dtype> input, int count, int dim, vector<dtype*> &ys) {
    NumberArray input_arr;
    NumberPointerArray ys_arr;
    input_arr.init((dtype*)input.data(), input.size());
    ys_arr.init((dtype**)ys.data(), ys.size());
    int block_count = DefaultBlockCount(count * dim);
    KernelBucketForward<<<block_count, TPB>>>((dtype*)input_arr.value, count, dim,
            (dtype **)ys_arr.value);
    CheckCudaError();
}

__global__ void KernelConcat(dtype **src, int count, int *dims, int max_dim, int *offsets,
        dtype *dest) {
    int i = DeviceDefaultIndex();
    int count_i = i / max_dim;
    if (count_i < count) {
        int dim_i = i % max_dim;
        int dim = dims[count_i];
        if (dim_i < dim) {
            int offset = offsets[count_i];
            dest[offset + dim_i] = src[count_i][dim_i];
        }
    }
}

__global__ void KernelCopy(dtype *src, int count, int *dims, int max_dim, int *offsets,
        int row,
        dtype *y,
        dtype **dest) {
    int i = DeviceDefaultIndex();
    int count_i = i / max_dim;
    if (count_i < count) {
        int dim_i = i % max_dim;
        int dim = dims[count_i];
        if (dim_i < dim) {
            int offset = offsets[count_i];
            int row_i = i % row;
            dtype a = src == nullptr ? 0 : src[row_i];
            dtype b = y[offset + dim_i];
            dest[count_i][dim_i] = a + b;
        }
    }
}

void MatrixMultiplyMatrix(dtype *W, dtype *x, dtype *y, int row, int col, int count, bool useb,
        bool should_x_transpose,
        bool should_W_transpose) {
    cublasHandle_t &handle = GetCublasHandle();
    dtype alpha = 1;
    dtype beta = useb? 1 : 0;
    cublasOperation_t x_op = should_x_transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
    int ldx = should_x_transpose ? count : col;
    cublasOperation_t W_op = should_W_transpose ? CUBLAS_OP_T : CUBLAS_OP_N;
    int ldw = should_W_transpose ? col : row;
#if USE_FLOAT
    CallCublas(cublasSgemm(handle, W_op, x_op, row, count, col,
                &alpha, W, ldw, x, ldx, &beta, y, row));
#else
    CallCublas(cublasDgemm(handle, W_op, x_op, row, count, col,
                &alpha, W, ldw, x, ldx, &beta, y, row));
#endif
}

void LinearForward(dtype **in_val_arr, int count, vector<int> &cols, int in_row,
        int out_row,
        dtype *W,
        dtype *bias,
        vector<dtype *> &vals) {
    int col_sum = 0;
    vector<int> in_offsets, out_offsets, in_dims, out_dims;
    in_offsets.reserve(count);
    out_offsets.reserve(count);
    in_dims.reserve(count);
    out_dims.reserve(count);
    for (int col : cols) {
        in_offsets.push_back(col_sum * in_row);
        out_offsets.push_back(col_sum * out_row);
        col_sum += col;
        in_dims.push_back(in_row * col);
        out_dims.push_back(out_row * col);
    }

    NumberArray y;
    NumberArray concated_in_val;
    concated_in_val.init(col_sum * in_row);
    y.init(col_sum * out_row);

    NumberPointerArray val_arr;
    val_arr.init(vals.data(), count);

    IntArray in_dim_arr, in_offset_arr, out_dim_arr, out_offset_arr;
    in_dim_arr.init(in_dims.data(), count);
    in_offset_arr.init(in_offsets.data(), count);
    out_dim_arr.init(out_dims.data(), count);
    out_offset_arr.init(out_offsets.data(), count);

    int max_col = *max_element(cols.begin(), cols.end());
    int max_in_dim = max_col * in_row;
    int block_count = DefaultBlockCountWithoutLimit(count * max_col * in_row);

    KernelConcat<<<block_count, TPB>>>(in_val_arr, count, in_dim_arr.value, max_in_dim,
            in_offset_arr.value, concated_in_val.value);
    CheckCudaError();

    MatrixMultiplyMatrix(W, concated_in_val.value, y.value, out_row, in_row, col_sum, false,
            false, true);

    int max_out_dim = max_col * out_row;
    block_count = DefaultBlockCountWithoutLimit(count * max_out_dim);
    KernelCopy<<<block_count, TPB>>>(bias, count, out_dim_arr.value, max_out_dim,
            out_offset_arr.value, out_row, y.value, val_arr.value);
    CheckCudaError();
}

__global__ void KernelAtomicAdd(dtype *src, int count, int *dims, int max_dim, int *offsets,
        dtype **dest) {
    int i = DeviceDefaultIndex();
    int count_i = i / max_dim;
    if (count_i < count) {
        int dim_i = i % max_dim;
        int dim = dims[count_i];
        if (dim_i < dim) {
            int offset = offsets[count_i];
            dtype b = src[offset + dim_i];
            DeviceAtomicAdd(dest[count_i] + dim_i, b);
        }
    }
}

__global__ void KernelLinearBackwardForBias(dtype *grad, int col, int row, dtype *bias_grad,
        int *block_counters,
        volatile dtype *block_sums) {
    __shared__ volatile extern dtype shared_sum[];
    __shared__ volatile bool is_last_block;
    if (threadIdx.x == 0 && blockIdx.y == 0) {
        block_counters[blockIdx.x] = 0;
    }
    if (threadIdx.x == 0) {
        is_last_block = false;
    }

    int row_i = blockIdx.x;
    int col_i = blockIdx.y * blockDim.x + threadIdx.x;
    shared_sum[threadIdx.x] = col_i < col ?  grad[row * col_i + row_i] : 0.0f;
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
            dtype x = shared_sum[0];
            DeviceAtomicAdd(bias_grad + row_i, x);
        }
    }
}

void LinearBackward(vector<dtype *> &grads, int count, vector<int> &cols, int in_row, int out_row,
        dtype *W_val,
        dtype **in_val_arr,
        dtype *bias_grad,
        vector<dtype *> &in_grads,
        dtype *W_grad) {
    int col_sum = 0;
    vector<int> in_offsets, out_offsets, in_dims, out_dims;
    in_offsets.reserve(count);
    out_offsets.reserve(count);
    in_dims.reserve(count);
    out_dims.reserve(count);
    for (int col : cols) {
        in_offsets.push_back(col_sum * in_row);
        out_offsets.push_back(col_sum * out_row);
        col_sum += col;
        in_dims.push_back(in_row * col);
        out_dims.push_back(out_row * col);
    }

    NumberArray concated_grad, concated_in_grad, concated_in_val;
    concated_grad.init(col_sum * out_row);
    concated_in_grad.init(col_sum * in_row);
    concated_in_val.init(col_sum * in_row);

    NumberPointerArray in_grad_arr, grad_arr;
    in_grad_arr.init(in_grads.data(), count);
    grad_arr.init(grads.data(), count);

    IntArray in_dim_arr, in_offset_arr, out_dim_arr, out_offset_arr;
    in_dim_arr.init(in_dims.data(), count);
    in_offset_arr.init(in_offsets.data(), count);
    out_dim_arr.init(out_dims.data(), count);
    out_offset_arr.init(out_offsets.data(), count);

    int max_col = *max_element(cols.begin(), cols.end());
    int max_out_dim = max_col * out_row;
    int block_count = DefaultBlockCountWithoutLimit(count * max_col * out_row);

    KernelConcat<<<block_count, TPB>>>(grad_arr.value, count, out_dim_arr.value, max_out_dim,
            out_offset_arr.value, concated_grad.value);
    CheckCudaError();

    int max_in_dim = max_col * in_row;
    block_count = DefaultBlockCountWithoutLimit(count * max_col * in_row);

    KernelConcat<<<block_count, TPB>>>(in_val_arr, count, in_dim_arr.value, max_in_dim,
            in_offset_arr.value, concated_in_val.value);
    CheckCudaError();
    KernelConcat<<<block_count, TPB>>>(in_grad_arr.value, count, in_dim_arr.value, max_in_dim,
            in_offset_arr.value, concated_in_grad.value);
    CheckCudaError();

    MatrixMultiplyMatrix(concated_in_val.value, concated_grad.value, W_grad, in_row, col_sum,
            out_row, true, true, false);
    MatrixMultiplyMatrix(W_val, concated_grad.value, concated_in_grad.value, in_row, out_row,
            col_sum, false, false, false);

    KernelAtomicAdd<<<block_count, TPB>>>(concated_in_grad.value, count, in_dim_arr.value,
            max_in_dim, in_offset_arr.value, in_grad_arr.value);
    CheckCudaError();

    if (bias_grad != nullptr) {
        int thread_count = min(NextTwoIntegerPowerNumber(col_sum), TPB);
        int block_y_count = (col_sum - 1 + thread_count) / thread_count;
        dim3 block_dim(out_row, block_y_count, 1);
        NumberArray block_sums;
        block_sums.init(block_y_count * out_row);
        IntArray block_counters;
        block_counters.init(out_row);
        KernelLinearBackwardForBias<<<block_dim, thread_count, thread_count * sizeof(dtype)>>>(
                concated_grad.value, col_sum, out_row, bias_grad, block_counters.value,
                block_sums.value);
        CheckCudaError();
    }
}

__device__ dtype maxIgnoringINF(dtype a, dtype b) {
    return a > b && a < 1e10 ? a : b;
}

__global__ void KernelAbsMax(dtype *v, int len, volatile dtype *global_sum,
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
    shared_sum[threadIdx.x] = 0;
    for (int i = index; i < len; i += blockDim.x * gridDim.x) {
        dtype abs_v = DeviceAbs(v[i]);
        shared_sum[threadIdx.x] = maxIgnoringINF(shared_sum[threadIdx.x], abs_v);
    }

    __syncthreads();
    for (int i = (blockDim.x >> 1); i > 0; i >>= 1) {
        if (threadIdx.x < i) {
            shared_sum[threadIdx.x] = maxIgnoringINF(shared_sum[threadIdx.x],
                    shared_sum[threadIdx.x + i]);
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
            sum = maxIgnoringINF(sum, global_sum[i]);
        }

        shared_sum[threadIdx.x] = sum;
        __syncthreads();

        for (int i = (blockDim.x >> 1); i > 0; i >>= 1) {
            if (threadIdx.x < i) {
                shared_sum[threadIdx.x] = maxIgnoringINF(shared_sum[threadIdx.x],
                        shared_sum[threadIdx.x + i]);
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            *result = shared_sum[0];
        }
    }
}

dtype AbsMax(dtype *v, int len) {
    int block_count = DefaultBlockCount(len);
    NumberArray global_sum;
    global_sum.init(block_count);
    DeviceInt block_counter;
    block_counter.init();
    DeviceNumber result;
    result.init();
    KernelAbsMax<<<block_count, TPB>>>(v, len, global_sum.value, block_counter.value, result.value);
    CheckCudaError();
    result.copyFromDeviceToHost();
    if (result.v > 1e10) {
        cerr << "AbsMax return inf" << endl;
        abort();
    }
    return result.v;
}

__global__ void KernelVerify(dtype *host, dtype *device, int len, dtype abs_avg,
        char *message, bool *success) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < len; i += step) {
        dtype loss = host[i] - device[i];
        if (DeviceAbs(host[i]) > 1e10 && DeviceAbs(device[i]) > 1e10 && host[i] * device[i] > 0) {
            continue; // Ignore INF values that are both postitive or negtative.
        }
        if (DeviceAbs(loss) > 1e-3 * abs_avg && DeviceAbs(loss) > 1e-2 * DeviceAbs(host[i]) &&
                ((DeviceAbs(host[i]) > 1e-6) || (DeviceAbs(device[i]) > 1e-6))) {
            *success = false;
            KernelPrintLine("KernelVerify: host:%.9f device:%.9f abs(loss):%.9f", host[i],
                    device[i], DeviceAbs(loss));
        }
    }
}

bool Verify(dtype *host, dtype *device, int len, const char* message) {
    NumberArray arr;
    arr.init(host, len);
    int block_count = DefaultBlockCount(len);
    char *m = nullptr;
    MemoryPool::Ins().Malloc((void**)&m, (strlen(message) + 1) * sizeof(char));
    MyCudaMemcpy(m, (void *)message, (strlen(message) + 1) * sizeof(char),
            MyCudaMemcpyKind::HOST_TO_DEVICE);
    bool success = true;
    bool *dev_success = nullptr;
    MemoryPool::Ins().Malloc((void**)&dev_success, 8 * sizeof(bool));
    MyCudaMemcpy(dev_success, &success, sizeof(bool), MyCudaMemcpyKind::HOST_TO_DEVICE);
    dtype abs_max = AbsMax(arr.value, len);
    KernelVerify<<<block_count, TPB>>>(arr.value, device, len, abs_max, m, dev_success);
    CheckCudaError();
    MyCudaMemcpy(&success, dev_success, sizeof(bool), MyCudaMemcpyKind::DEVICE_TO_HOST);
    MemoryPool::Ins().Free(dev_success);
    MemoryPool::Ins().Free(m);
    cudaDeviceSynchronize();
    cudaPrintfDisplay(stdout, true);
    if (!success) {
        fprintf(stderr, "abs max:%.9f\n%s", abs_max, message);
    }
    return success;
}

__global__ void KernelVerify(bool *host, bool *device, int len,
        char *message, bool *success) {
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
    arr.init(host, len);
    int block_count = (len + TPB - 1) / TPB;
    char *m = nullptr;
    MemoryPool::Ins().Malloc((void**)&m, (strlen(message) + 1) * sizeof(char));
    MyCudaMemcpy(m, (void *)message, (strlen(message) + 1) * sizeof(char),
            MyCudaMemcpyKind::HOST_TO_DEVICE);
    bool success = true;
    bool *dev_success = nullptr;
    MemoryPool::Ins().Malloc((void**)&dev_success, 8 * sizeof(bool));
    MyCudaMemcpy(dev_success, &success, sizeof(bool), MyCudaMemcpyKind::HOST_TO_DEVICE);
    KernelVerify<<<block_count, TPB>>>(arr.value, device, len, m, dev_success);
    CheckCudaError();
    MyCudaMemcpy(&success, dev_success, sizeof(bool), MyCudaMemcpyKind::DEVICE_TO_HOST);
    MemoryPool::Ins().Free(dev_success);
    MemoryPool::Ins().Free(m);
    cudaDeviceSynchronize();
    cudaPrintfDisplay(stdout, true);
    return success;
}

__global__ void KernelVerify(int *host, int *device, int len,
        char *message, bool *success) {
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
    arr.init(host, len);
    int block_count = (len + TPB - 1) / TPB;
    char *m = nullptr;
    MemoryPool::Ins().Malloc((void**)&m, (strlen(message) + 1) * sizeof(char));
    MyCudaMemcpy(m, (void*)message, (strlen(message) + 1) * sizeof(char),
            MyCudaMemcpyKind::HOST_TO_DEVICE);
    bool success = true;
    bool *dev_success = nullptr;
    MemoryPool::Ins().Malloc((void**)&dev_success, sizeof(bool));
    MyCudaMemcpy(dev_success, &success, sizeof(bool), MyCudaMemcpyKind::HOST_TO_DEVICE);
    KernelVerify<<<block_count, TPB>>>(arr.value, device, len, m, dev_success);
    CheckCudaError();
    MyCudaMemcpy(&success, dev_success, sizeof(bool), MyCudaMemcpyKind::DEVICE_TO_HOST);
    MemoryPool::Ins().Free(dev_success);
    MemoryPool::Ins().Free(m);
    cudaDeviceSynchronize();
    cudaPrintfDisplay(stdout, true);
    return success;
}

constexpr int MAX_BLOCK_POWER = 50;

MemoryPool& MemoryPool::Ins() {
    static MemoryPool *p;
    if (p == nullptr) {
        p = new MemoryPool;
        p->free_blocks_.resize(MAX_BLOCK_POWER + 1);
        p->busy_blocks_.reserve(10000);
    }
    return *p;
}

void appendFreeBlock(MemoryBlock &memory_block,
        vector<map<void*, MemoryBlock>> &free_blocks,
        int i,
        unordered_map<void*, MemoryBlock> &busy_blocks) {
    if (memory_block.size != (1 << i)) {
        cerr << format("incorrect block size {}, but i is {}\n", memory_block.size, i);
        abort();
    }
    free_blocks.at(i).insert(make_pair(memory_block.p, memory_block));
}

void MemoryPool::Malloc(void **p, int size) {
    if (*p != nullptr) {
        cerr << "MemoryPool Malloc p is not nullptr.\n";
        abort();
    }
#if DEVICE_MEMORY
    cudaError_t r = cudaMalloc(p, size);
    if (r != cudaSuccess) {
        cerr << format("MemoryPool Malloc cudaMalloc failed status:{}\n", r);
        abort();
    }
#else
    int fit_size = 1;
    int n = 0;
    while (fit_size < size) {
        fit_size <<= 1;
        ++n;
    }
    Profiler &profiler = Profiler::Ins();
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
                    if (status != cudaSuccess) {
                        cerr << fmt::format("Malloc cudaMalloc failed - status:{} fit_size:{} size:{}",
                                status, fit_size, size) << endl;
                        abort();
                    }
                }
                CallCuda(status);
                MemoryBlock block(*p, fit_size);
                busy_blocks_.insert(make_pair(*p, block));
            } else {
                profiler.BeginEvent("split_memory_block");
                while (higher_power > n) {
                    auto &v = free_blocks_.at(higher_power);
                    MemoryBlock &to_split = v.rbegin()->second;
                    int half_size = to_split.size >> 1;
                    void *half_address = static_cast<void*>(static_cast<char*>(to_split.p) +
                            half_size);
                    MemoryBlock low_block(to_split.p, half_size, to_split.buddy),
                                high_block(half_address, half_size, to_split.p);
                    v.erase(v.rbegin()->first);

                    --higher_power;
                    appendFreeBlock(low_block, free_blocks_, higher_power, busy_blocks_);
                    appendFreeBlock(high_block, free_blocks_, higher_power, busy_blocks_);
                }
                profiler.EndEvent();
            }
        } else {
            status = cudaSuccess;
            int this_size = free_blocks_.at(n).size();
            MemoryBlock &block = free_blocks_.at(n).rbegin()->second;
            *p = block.p;
            busy_blocks_.insert(make_pair(block.p, block));
            free_blocks_.at(n).erase(free_blocks_.at(n).rbegin()->first);
        }
    }

    if (status != cudaSuccess) {
        cerr << format("MemoryPool Malloc cudaMalloc failed status:{}\n", status);
        abort();
    }
#endif
}

pair<MemoryBlock *, MemoryBlock *> lowerAndhigherBlocks(MemoryBlock &a,
        MemoryBlock &b) {
    if (a.size != b.size) {
        cerr << "a.size is not equal to b.size" << endl;
        abort();
    }
    int distance = static_cast<char*>(a.p) - static_cast<char*>(b.p);
    if (distance == 0) {
        cerr << "block a and b has the same address" << endl;
        abort();
    }
    MemoryBlock &low = distance > 0 ? b : a;
    MemoryBlock &high = distance > 0 ? a : b;
    return make_pair(&low, &high);
}

bool isBuddies(MemoryBlock &a, MemoryBlock &b) {
    if (a.size != b.size) {
        return false;
    }
    auto pair = lowerAndhigherBlocks(a, b);
    return pair.second->buddy == pair.first->p &&
        ((char*)pair.second->p - (char*)pair.first->p) == a.size;
}

MemoryBlock mergeBlocks(MemoryBlock &a, MemoryBlock &b) {
    if (a.size != b.size) {
        cerr << "sizes of memory blocks to merge not equal" << endl;
        abort();
    }

    auto pair = lowerAndhigherBlocks(a, b);
    if ((char*)pair.second->p - (char*)pair.first->p != a.size ||
            (a.p != b.buddy && a.buddy != b.p)) {
        cerr << "a and b are not buddies" << endl;
        cerr << format("a:{}\nb:{}\n", a.toString(), b.toString());
        abort();
    }
    MemoryBlock block(pair.first->p, pair.first->size << 1, pair.first->buddy);
    return block;
}

void returnFreeBlock(MemoryBlock &block, vector<map<void*, MemoryBlock>> &free_blocks,
        int power,
        unordered_map<void*, MemoryBlock> &busy_blocks) {
    Profiler &profiler = Profiler::Ins();
    profiler.BeginEvent("return_free_block");
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

void MemoryPool::Free(void *p) {
#if DEVICE_MEMORY
    cudaError_t r = cudaFree(p);
    if (r != cudaSuccess) {
        cerr << format("MemoryPool::Free error status:{}\n", r);
        abort();
    }
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
        cerr << format("size:{} n:{}\n", it->second.size, n);
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

    if (busy_blocks_.find(p) != busy_blocks_.end()) {
        cerr << "MemoryPool Free - find freed p in busy blocks" << endl;
        abort();
    }
#endif
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
    if (states == nullptr) {
        MemoryPool &pool = MemoryPool::Ins();
        pool.Malloc((void**)&states, sizeof(curandState_t) * MAX_BATCH_COUNT);
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

void CalculateDropoutMask(dtype drop_factor, int dim, dtype* mask) {
    curandGenerator_t &gen = GetGenerator();
    CallCurand(curandGenerateUniform(gen, mask, dim));
}

__global__ void KernelConcatForward(dtype **ins, int *in_rows, dtype **outs, int count,
        int in_count,
        int out_row,
        int *cols,
        int max_col) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    int n = max_col * out_row;

    for (int i = index; i < out_row * count * max_col; i += step) {
        int offset = i % n;
        int count_i = i / n;
        int col = cols[count_i];
        int col_i = offset / out_row;
        if (col_i < col) {
            int out_row_i = offset % out_row;
            int in_row_sum = 0;
            int last_in_row_sum;
            int offset_j = 0;
            for (int j = 0; j < in_count; ++j) {
                last_in_row_sum = in_row_sum;
                in_row_sum += in_rows[j];
                offset_j = j;
                if (out_row_i < in_row_sum) {
                    break;
                }
            }
            int in_row_i = out_row_i - last_in_row_sum;
            dtype v = ins[count_i * in_count + offset_j][col_i * in_rows[offset_j] + in_row_i];
            outs[count_i][col_i * out_row + out_row_i] = v;
        }
    }
}

void ConcatForward(vector<dtype*> &in_vals, vector<int> &in_rows, vector<dtype*> &vals, int count,
        int in_count,
        int out_row,
        vector<int> &cols) {
    int max_col = *max_element(cols.begin(), cols.end());
    int len = count * out_row * max_col;
    int block_count = min(BLOCK_COUNT, (len - 1 + TPB) / TPB);

    NumberPointerArray in_val_arr, val_arr;
    in_val_arr.init((dtype**)in_vals.data(), in_vals.size());
    val_arr.init((dtype**)vals.data(), vals.size());
    IntArray in_row_arr, col_arr;
    in_row_arr.init(in_rows.data(), in_rows.size());
    col_arr.init(cols.data(), count);

    KernelConcatForward<<<block_count, TPB>>>(in_val_arr.value, in_row_arr.value, val_arr.value,
            count, in_count, out_row, col_arr.value, max_col);
    CheckCudaError();
}

__global__ void KernelConcatBackward(dtype **in_grads, int *in_rows, dtype **out_grads, int count,
        int in_count,
        int out_row,
        int *cols,
        int max_col) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    int n = max_col * out_row;

    for (int i = index; i < out_row * count * max_col; i += step) {
        int offset = i % n;
        int count_i = i / n;
        int col = cols[count_i];
        int col_i = offset / out_row;
        if (col_i < col) {
            int out_row_i = offset % out_row;
            int in_row_sum = 0;
            int last_in_row_sum;
            int offset_j = 0;
            for (int j = 0; j < in_count; ++j) {
                last_in_row_sum = in_row_sum;
                in_row_sum += in_rows[j];
                offset_j = j;
                if (out_row_i < in_row_sum) {
                    break;
                }
            }
            int in_row_i = out_row_i - last_in_row_sum;
            DeviceAtomicAdd(in_grads[count_i * in_count + offset_j] +
                    in_row_i + col_i * in_rows[offset_j],
                    out_grads[count_i][col_i * out_row + out_row_i]);
        }
    }
}

void ConcatBackward(vector<dtype*> &in_grads, vector<int> &in_rows, vector<dtype*> &grads,
        int count,
        int in_count,
        int out_row,
        vector<int> &cols) {
    int max_col = *max_element(cols.begin(), cols.end());
    int len = count * out_row * max_col;
    int block_count = min(BLOCK_COUNT, (len - 1 + TPB) / TPB);

    NumberPointerArray in_loss_arr, loss_arr;
    in_loss_arr.init(in_grads.data(), in_grads.size());
    loss_arr.init((dtype**)grads.data(), grads.size());
    IntArray in_row_arr, col_arr;
    in_row_arr.init(in_rows.data(), in_rows.size());
    col_arr.init(cols.data(), cols.size());

    KernelConcatBackward<<<block_count, TPB>>>((dtype **)in_loss_arr.value, in_row_arr.value,
            loss_arr.value, count, in_count, out_row, col_arr.value, max_col);
    CheckCudaError();
}

__global__ void KernelScalarConcatForward(dtype **ins, int count,
        int *dims,
        int max_dim,
        dtype **results) {
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

void ScalarConcatForward(vector<dtype *> &ins, int count, vector<int> &dims,
        int max_dim,
        vector<dtype *> &results) {
    NumberPointerArray result_arr;
    result_arr.init((dtype**)results.data(), results.size());
    NumberPointerArray in_arr;
    in_arr.init((dtype**)ins.data(), ins.size());
    IntArray dim_arr;
    dim_arr.init((int *)dims.data(), dims.size());

    int block_count = DefaultBlockCount(count * max_dim);
    KernelScalarConcatForward<<<block_count, TPB>>>(in_arr.value, count, dim_arr.value,
            max_dim, (dtype **)result_arr.value);
    CheckCudaError();
}

__global__ void KernelScalarConcatBackward(dtype **grads, int count, int *dims,
        int max_dim,
        dtype **input_grads) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < max_dim * count; i += step) {
        int count_i = i / max_dim;
        int dim_i = i % max_dim;
        if (dim_i < dims[count_i]) {
            DeviceAtomicAdd(input_grads[count_i * max_dim + dim_i], grads[count_i][dim_i]);
        }
    }
}

void ScalarConcatBackward(vector<dtype *> &grads, int count, vector<int> &dims,
        int max_dim,
        vector<dtype *> in_grads) {
    NumberPointerArray loss_arr, in_loss_arr;
    loss_arr.init((dtype**)grads.data(), grads.size());
    in_loss_arr.init((dtype **)in_grads.data(), in_grads.size());
    IntArray dim_arr;
    dim_arr.init((int *)dims.data(), dims.size());
    int block_count = DefaultBlockCount(count * max_dim);
    KernelScalarConcatBackward<<<block_count, TPB>>>(loss_arr.value, count, dim_arr.value,
            max_dim, (dtype **)in_loss_arr.value);
    CheckCudaError();
}

__global__ void KernelMemset(dtype *p, int len, dtype value) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < len; i+= step) {
        p[i] = value;
    }
}

void Memset(dtype *p, int len, dtype value) {
    int block_count = min(BLOCK_COUNT, (len - 1 + TPB) / TPB);
    KernelMemset<<<block_count, TPB>>>(p, len, value);
    CheckCudaError();
}

__global__ void KernelMemset(bool *p, int len, bool value) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < len; i+= step) {
        p[i] = value;
    }
}

void Memset(bool *p, int len, bool value) {
    int block_count = min(BLOCK_COUNT, (len - 1 + TPB) / TPB);
    KernelMemset<<<block_count, TPB>>>(p, len, value);
    CheckCudaError();
}

void *Malloc(int size) {
    void *p;
    CallCuda(cudaMalloc(&p, size));
    return p;
}

__global__ void KernelBatchMemset(dtype **p, int count, int *dims, int max_dim,
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

void BatchMemset(vector<dtype*> &vec, int count, const vector<int> &dims, dtype value) {
    if (count == 0) {
        return;
    }
    int max_dim = *max_element(dims.begin(), dims.end());
    int block_count = (count * max_dim -1 + TPB) / TPB;
    block_count = min(block_count, BLOCK_COUNT);
    NumberPointerArray vec_arr;
    vec_arr.init((dtype**)vec.data(), vec.size());
    IntArray dim_arr;
    dim_arr.init((int *)dims.data(), dims.size());
    KernelBatchMemset<<<block_count, TPB>>>((dtype **)vec_arr.value, count, dim_arr.value,
            max_dim, value);
    CheckCudaError();
}

__global__ void KernelLookupForward(int *ids, dtype *vocabulary, int count, int row, int *cols,
        int max_col,
        dtype **vals) {
    int i = DeviceDefaultIndex();
    int n = row * max_col;
    int count_i = i / n;
    if (count_i < count) {
        int ni = i % n;
        int col_i = ni / row;
        int col = cols[count_i];
        if (col_i < col) {
            int row_i = ni % row;
            int id = ids[count_i * max_col + col_i];
            int voc_i = id * row + row_i;
            vals[count_i][ni] = vocabulary[voc_i];
        }
    }
}

void LookupForward(int *ids, dtype *vocabulary, int count, int row, int *cols, int max_col,
        vector<dtype*> &vals) {
    int block_count = DefaultBlockCountWithoutLimit(count * row * max_col);
    NumberPointerArray val_arr;
    val_arr.init((dtype**)vals.data(), vals.size());
    KernelLookupForward<<<block_count, TPB>>>(ids, vocabulary, count, row, cols, max_col,
            val_arr.value);
    CheckCudaError();
}

__global__ void KernelLookupBackward(int *ids, dtype** grads, int count, int row, int *cols,
        int max_col,
        dtype *param_grad,
        bool *indexers) {
    int i = DeviceDefaultIndex();
    int n = row * max_col;
    int count_i = i / n;
    if (count_i < count) {
        int ni = i % n;
        int col_i = ni / row;
        int col = cols[count_i];
        if (col_i < col) {
            int row_i = ni % row;
            int id = ids[count_i * max_col + col_i];
            if (indexers != nullptr && row_i == 0 && col_i == 0) {
                indexers[id] = true;
            }
            int voc_i = id * row + row_i;
            DeviceAtomicAdd(param_grad + voc_i, grads[count_i][ni]);
        }
    }
}

void LookupBackward(int *ids, vector<dtype*> &grads, int count, int row, int *cols, int max_col,
        dtype *param_grad,
        bool *indexers) {
    int block_count = DefaultBlockCountWithoutLimit(count * row * max_col);
    NumberPointerArray grad_arr;
    grad_arr.init((dtype**)grads.data(), grads.size());
    KernelLookupBackward<<<block_count, TPB>>>(ids, grad_arr.value, count, row, cols, max_col,
            param_grad, indexers);
    CheckCudaError();
}

__global__ void KernelParamRowForward(dtype *param, int row_index, int param_row_count,
        int count,
        int dim,
        dtype **vals) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < dim * count; i += step) {
        int count_i = i / dim;
        int dim_i = i % dim;
        int param_offset = dim_i * param_row_count + row_index;
        vals[count_i][dim_i] = param[param_offset];
    }
}

void ParamRowForward(dtype *param, int row_index, int param_row_count, int count, int dim,
        vector<dtype*> &vals) {
    NumberPointerArray val_arr;
    val_arr.init((dtype**)vals.data(), vals.size());
    int block_count = DefaultBlockCount(count * dim);
    KernelParamRowForward<<<block_count, TPB>>>(param, row_index, param_row_count, count, dim,
            (dtype **)val_arr.value);
    CheckCudaError();
}

__global__ void KernelPoolForward(PoolingEnum pooling, dtype **ins, int *in_counts,
        int max_in_count,
        dtype **outs,
        int count,
        int dim,
        int* hit_inputs) {
    __shared__ volatile extern dtype pool_shared_arr[];
    volatile dtype* shared_indexers = pool_shared_arr + blockDim.x;
    int batch_i = blockIdx.y;
    int dim_i = blockIdx.x;
    int in_count_i = threadIdx.x;
    int in_count = in_counts[batch_i];
    if (in_count_i < in_count) {
        pool_shared_arr[threadIdx.x] = ins[batch_i * max_in_count + in_count_i][dim_i];
    } else {
        pool_shared_arr[threadIdx.x] = pooling == PoolingEnum::MAX ?  -1e10 : 1e10;
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

void PoolForward(PoolingEnum pooling, vector<dtype*> &in_vals, vector<dtype*> &vals,
        int count,
        vector<int> &in_counts,
        int dim,
        int *hit_inputs) {
    int max_in_count = *max_element(in_counts.begin(), in_counts.end());
    int thread_count = 8;
    while (max_in_count > thread_count) {
        thread_count <<= 1;
    }
    dim3 block_dim(dim, count, 1);

    NumberPointerArray in_val_arr;
    in_val_arr.init((dtype**)in_vals.data(), in_vals.size());
    NumberPointerArray val_arr;
    val_arr.init((dtype**)vals.data(), vals.size());
    IntArray in_count_arr;
    in_count_arr.init((int*)in_counts.data(), in_counts.size());

    KernelPoolForward<<<block_dim, thread_count, thread_count * 2 *
        sizeof(dtype)>>>(pooling, in_val_arr.value, in_count_arr.value,
                max_in_count, (dtype **)val_arr.value, count, dim, hit_inputs);
    CheckCudaError();
}

__global__ void KernelPoolBackward(dtype ** grads, int *hit_inputs,
        int max_in_count,
        int count,
        int dim,
        dtype **in_grads) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < dim * count; i += step) {
        int count_i = i / dim;
        int dim_i = i % dim;
        int input_i = hit_inputs[i];
        dtype loss = grads[count_i][dim_i];
        DeviceAtomicAdd(in_grads[count_i * max_in_count + input_i] + dim_i,
                loss);
    }
}

void PoolBackward(vector<dtype*> &grads, vector<dtype*> &in_grads,
        vector<int> &in_counts,
        int *hit_inputs,
        int count,
        int dim) {
    NumberPointerArray loss_arr, in_loss_arr;
    loss_arr.init((dtype**)grads.data(), grads.size());
    in_loss_arr.init((dtype**)in_grads.data(), in_grads.size());
    int max_in_count = *max_element(in_counts.begin(), in_counts.end());
    int block_count = (count * dim - 1 + TPB) / TPB;
    block_count = min(block_count, BLOCK_COUNT);
    KernelPoolBackward<<<block_count, TPB>>>((dtype**)loss_arr.value,
            hit_inputs,
            max_in_count,
            count,
            dim,
            (dtype **)in_loss_arr.value);
    CheckCudaError();
}

__global__ void KernelSumPoolForward(PoolingEnum pooling, dtype **in_vals, int count,
        int dim,
        int *in_counts,
        int max_in_count,
        dtype **vals) {
    __shared__ volatile extern dtype pool_shared_arr[];
    int batch_i = blockIdx.y;
    int in_count = in_counts[batch_i];
    int in_count_i = threadIdx.x;
    int dim_i = blockIdx.x;
    if (in_count_i < in_count) {
        pool_shared_arr[threadIdx.x] = in_vals[batch_i * max_in_count + in_count_i][dim_i];
    } else {
        pool_shared_arr[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    for (int i = (blockDim.x >> 1); i > 0;i >>= 1) {
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

void SumPoolForward(PoolingEnum pooling, vector<dtype*> &in_vals, int count, int dim,
        vector<int> &in_counts,
        vector<dtype*> &vals) {
    int max_in_count = *max_element(in_counts.begin(), in_counts.end());
    int thread_count = 8;
    while (max_in_count > thread_count) {
        thread_count <<= 1;
    }
    dim3 block_dim(dim, count, 1);
    NumberPointerArray in_val_arr;
    in_val_arr.init((dtype**)in_vals.data(), in_vals.size());
    IntArray in_count_arr;
    in_count_arr.init((int*)in_counts.data(), in_counts.size());
    NumberPointerArray val_arr;
    val_arr.init((dtype**)vals.data(), vals.size());

    KernelSumPoolForward<<<block_dim, thread_count,
        thread_count * sizeof(dtype)>>>(pooling, in_val_arr.value, count, dim, in_count_arr.value,
                max_in_count, (dtype **)val_arr.value);
    CheckCudaError();
}

__global__ void KernelSumBackward(PoolingEnum pooling, dtype **grads,
        int *in_counts,
        int max_in_count,
        int count,
        int dim,
        dtype **in_grads) {
    int global_in_count_i = blockIdx.x * max_in_count + blockIdx.y;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        if (blockIdx.y < in_counts[blockIdx.x]) {
            DeviceAtomicAdd(in_grads[global_in_count_i] + i, pooling == PoolingEnum::SUM ?
                    grads[blockIdx.x][i] : grads[blockIdx.x][i] / in_counts[blockIdx.x]);
        }
    }
}

void SumPoolBackward(PoolingEnum pooling, vector<dtype*> &grads,
        vector<int> &in_counts,
        int count,
        int dim,
        vector<dtype*> &in_grads) {
    int thread_count = 8;
    while (thread_count < dim) {
        thread_count <<= 1;
    }
    thread_count = min(TPB, thread_count);

    int max_in_count = *max_element(in_counts.begin(), in_counts.end());
    dim3 block_dim(count, max_in_count, 1);
    NumberPointerArray loss_arr;
    loss_arr.init((dtype**)grads.data(), grads.size());
    IntArray in_count_arr;
    in_count_arr.init((int*)in_counts.data(), in_counts.size());
    NumberPointerArray in_loss_arr;
    in_loss_arr.init((dtype**)in_grads.data(), in_grads.size());
    KernelSumBackward<<<block_dim, thread_count>>>(pooling, loss_arr.value,
            (int*)in_count_arr.value, max_in_count, count, dim,
            (dtype **)in_loss_arr.value);
    CheckCudaError();
}

__global__ void KernelMatrixConcatForward(dtype **in_vals, int count, int in_dim, int *in_counts,
        int max_in_count,
        dtype **vals) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count * max_in_count * in_dim; i += step) {
        int max_in_dim_sum = max_in_count * in_dim;
        int count_i = i / max_in_dim_sum;
        int x = i % max_in_dim_sum;
        int in_count_i = x / in_dim;
        if (in_count_i < in_counts[count_i]) {
            int dim_i = x % in_dim;
            vals[count_i][x] = in_vals[count_i * max_in_count + in_count_i][dim_i];
        }
    }
}

void MatrixConcatForward(vector<dtype*> &in_vals, int count, int in_dim, vector<int> &in_counts,
        vector<dtype*> &vals) {
    int max_in_count = *max_element(in_counts.begin(), in_counts.end());
    int len = count * max_in_count * in_dim;
    int block_count = DefaultBlockCount(len);
    NumberPointerArray in_val_arr, val_arr;
    in_val_arr.init((dtype **)in_vals.data(), in_vals.size());
    val_arr.init((dtype **)vals.data(), vals.size());
    IntArray in_count_arr;
    in_count_arr.init((int*)in_counts.data(), in_counts.size());

    KernelMatrixConcatForward<<<block_count, TPB>>>((dtype**)in_val_arr.value, count, in_dim,
            in_count_arr.value, max_in_count, (dtype**)val_arr.value);
    CheckCudaError();
}

__global__ void KernelMatrixConcatBackward(dtype **grads, int count, int in_dim, int *in_counts,
        int max_in_count,
        dtype **in_grads) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count * max_in_count * in_dim; i += step) {
        int max_in_dim_sum = max_in_count * in_dim;
        int count_i = i / max_in_dim_sum;
        int x = i % max_in_dim_sum;
        int in_count_i = x / in_dim;
        if (in_count_i < in_counts[count_i]) {
            int dim_i = x % in_dim;
            DeviceAtomicAdd(in_grads[count_i * max_in_count + in_count_i] + dim_i,
                    grads[count_i][x]);
        }
    }
}

void MatrixConcatBackward(vector<dtype *> &grads, int count, int in_dim, vector<int> &in_counts,
        vector<dtype *> &in_grads) {
    int max_in_count = *max_element(in_counts.begin(), in_counts.end());
    int len = count * max_in_count * in_dim;
    int block_count = DefaultBlockCount(len);
    NumberPointerArray grad_arr, in_grad_arr;
    grad_arr.init((dtype**)grads.data(), grads.size());
    in_grad_arr.init((dtype **)in_grads.data(), in_grads.size());
    IntArray in_count_arr;
    in_count_arr.init((int*)in_counts.data(), in_counts.size());
    KernelMatrixConcatBackward<<<block_count, TPB>>>((dtype **)grad_arr.value, count, in_dim,
            in_count_arr.value, max_in_count, (dtype**)in_grad_arr.value);
}

__global__ void KernelMatMul(dtype **a, bool transpose_a, dtype **b, bool transpose_b, int *a_rows,
        int *b_cols,
        int *ks,
        dtype **vals,
        bool acc,
        bool use_lower_triangle_mask = false) {
    int count_i = blockIdx.x;
    int b_col = b_cols[count_i];
    int b_col_i = blockDim.x * blockIdx.z + threadIdx.x;
    if (b_col_i >= b_col) {
        return;
    }
    int a_row = a_rows[count_i];
    int a_row_i = blockDim.y * blockIdx.y + threadIdx.y;
    if (a_row_i >= a_row) {
        return;
    }

    if (use_lower_triangle_mask && a_row_i > b_col_i) {
        int v_offset = a_row * b_col_i + a_row_i;
        vals[count_i][v_offset] = -INF;
        return;
    }

    int k = ks[count_i];

    dtype sum = 0;
    for (int i = 0; i < k; ++i) {
        int a_offset = transpose_a ? k * a_row_i + i : a_row * i + a_row_i;
        dtype av = a[count_i][a_offset];

        int b_offset = transpose_b ? b_col * i + b_col_i : k * b_col_i + i;
        dtype bv = b[count_i][b_offset];

        sum += av * bv;
    }

    int v_offset = a_row * b_col_i + a_row_i;
    if (acc) {
        DeviceAtomicAdd(vals[count_i] + v_offset, sum);
    } else {
        vals[count_i][v_offset] = sum;
    }
}


void TranMatrixMulMatrixForward(vector<dtype *> &input_a_vals, vector <dtype *> &input_b_vals,
        int count,
        vector<int> &a_cols,
        vector<int> &b_cols,
        int row,
        bool use_lower_triangle_mask,
        vector<dtype *> &vals) {
    NumberPointerArray a_val_arr, b_val_arr, val_arr;
    a_val_arr.init(input_a_vals.data(), count);
    b_val_arr.init(input_b_vals.data(), count);
    val_arr.init(vals.data(), count);
    IntArray a_col_arr, b_col_arr;
    a_col_arr.init(a_cols.data(), count);
    b_col_arr.init(b_cols.data(), count);
    int max_a_col = *max_element(a_cols.begin(), a_cols.end());
    int max_b_col = *max_element(b_cols.begin(), b_cols.end());

    dim3 thread_dim(TPB_SQRT, TPB_SQRT, 1);
    int block_y = (max_a_col + TPB_SQRT - 1) / TPB_SQRT;
    int block_z = (max_b_col + TPB_SQRT - 1) / TPB_SQRT;
    dim3 block_dim(count, block_y, block_z);

    vector<int> rows;
    rows.reserve(count);
    for (int i = 0; i < count; ++i) {
        rows.push_back(row);
    }
    IntArray row_arr;
    row_arr.init(rows.data(), count);

    KernelMatMul<<<block_dim, thread_dim>>>(a_val_arr.value, true, b_val_arr.value, false,
            a_col_arr.value, b_col_arr.value, row_arr.value, val_arr.value, false,
            use_lower_triangle_mask);

    CheckCudaError();
}

void TranMatrixMulMatrixBackward(vector<dtype *> &grads, vector<dtype *> &a_vals,
        vector<dtype *> &b_vals,
        int count,
        vector<int> &a_cols,
        vector<int> &b_cols,
        int row,
        vector<dtype *> &a_grads,
        vector<dtype *> &b_grads) {
    NumberPointerArray grad_arr, a_val_arr, b_val_arr, a_grad_arr, b_grad_arr;
    grad_arr.init(grads.data(), count);
    a_val_arr.init(a_vals.data(), count);
    b_val_arr.init(b_vals.data(), count);
    a_grad_arr.init(a_grads.data(), count);
    b_grad_arr.init(b_grads.data(), count);
    IntArray a_col_arr, b_col_arr, row_arr;
    a_col_arr.init(a_cols.data(), count);
    b_col_arr.init(b_cols.data(), count);

    vector<int> rows;
    rows.reserve(count);
    for (int i = 0; i < count; ++i) {
        rows.push_back(row);
    }
    row_arr.init(rows.data(), count);

    int max_a_col = *max_element(a_cols.begin(), a_cols.end());
    int max_b_col = *max_element(b_cols.begin(), b_cols.end());

    dim3 thread_dim(TPB_SQRT, TPB_SQRT, 1);
    int block_y = (row + TPB_SQRT -1) / TPB_SQRT;

    {
        int block_z = (max_a_col + TPB_SQRT - 1) / TPB_SQRT;
        dim3 block_dim(count, block_y, block_z);

        KernelMatMul<<<block_dim, thread_dim>>>(b_val_arr.value, false, grad_arr.value, true,
                row_arr.value, a_col_arr.value, b_col_arr.value, a_grad_arr.value, true);
        CheckCudaError();
    }
    {
        int block_z = (max_b_col + TPB_SQRT - 1) / TPB_SQRT;
        dim3 block_dim(count, block_y, block_z);
        KernelMatMul<<<block_dim, thread_dim>>>(a_val_arr.value, false, grad_arr.value, false,
                row_arr.value, b_col_arr.value, a_col_arr.value, b_grad_arr.value, true);

        CheckCudaError();
    }
}

void MatrixMulMatrixForward(vector<dtype *> &a, vector<dtype *> &b, int count, vector<int> &ks,
        vector<int> &b_cols,
        int row,
        vector<dtype *> &vals) {
    NumberPointerArray a_arr, b_arr, val_arr;
    a_arr.init(a.data(), count);
    b_arr.init(b.data(), count);
    val_arr.init(vals.data(), count);
    IntArray k_arr, b_col_arr;
    k_arr.init(ks.data(), count);
    b_col_arr.init(b_cols.data(), count);
    int max_b_col = *max_element(b_cols.begin(), b_cols.end());
    int max_k = *max_element(ks.begin(), ks.end());
    dim3 thread_dim(TPB_SQRT, TPB_SQRT, 1);
    int block_y = (row + TPB_SQRT -1) / TPB_SQRT;
    int block_z = (max_b_col + TPB_SQRT - 1) / TPB_SQRT;
    dim3 block_dim(count, block_y, block_z);

    vector<int> rows;
    rows.reserve(count);
    for (int i = 0; i < count; ++i) {
        rows.push_back(row);
    }
    IntArray row_arr;
    row_arr.init(rows.data(), count);

    KernelMatMul<<<block_dim, thread_dim>>>(a_arr.value, false, b_arr.value, false, row_arr.value,
            b_col_arr.value, k_arr.value, val_arr.value, false);
    CheckCudaError();
}

void MatrixMulMatrixBackward(vector<dtype *> &grads, vector<dtype *> &a_vals,
        vector<dtype *> &b_vals,
        int count,
        vector<int> &ks,
        vector<int> &b_cols,
        int row,
        vector<dtype *> &a_grads,
        vector<dtype *> &b_grads) {
    NumberPointerArray grad_arr, a_val_arr, b_val_arr, a_grad_arr, b_grad_arr;
    grad_arr.init(grads.data(), count);
    a_val_arr.init(a_vals.data(), count);
    b_val_arr.init(b_vals.data(), count);
    a_grad_arr.init(a_grads.data(), count);
    b_grad_arr.init(b_grads.data(), count);

    IntArray k_arr, b_col_arr;
    k_arr.init(ks.data(), count);
    b_col_arr.init(b_cols.data(), count);

    vector<int> rows;
    rows.reserve(count);
    for (int i = 0; i < count; ++i) {
        rows.push_back(row);
    }
    IntArray row_arr;
    row_arr.init(rows.data(), count);

    int max_k = *max_element(ks.begin(), ks.end());
    int max_b_col = *max_element(b_cols.begin(), b_cols.end());
    dim3 thread_dim(TPB_SQRT, TPB_SQRT, 1);

    {
        int block_y = (row + TPB_SQRT - 1) / TPB_SQRT;
        int block_z = (max_k + TPB_SQRT - 1) / TPB_SQRT;
        dim3 block_dim(count, block_y, block_z);
        KernelMatMul<<<block_dim, thread_dim>>>(grad_arr.value, false, b_val_arr.value, true,
                row_arr.value, k_arr.value, b_col_arr.value, a_grad_arr.value, true);

        CheckCudaError();
    } {
        int block_y = (max_k + TPB_SQRT - 1) / TPB_SQRT;
        int block_z = (max_b_col + TPB_SQRT - 1) / TPB_SQRT;
        dim3 block_dim(count, block_y, block_z);
        KernelMatMul<<<block_dim, thread_dim>>>(a_val_arr.value, true, grad_arr.value, false,
                k_arr.value, b_col_arr.value, row_arr.value, b_grad_arr.value, true);

        CheckCudaError();
    }
}

__global__ void KernelPMultiForward(dtype **ins1, dtype **ins2, int count, int dim,
        dtype **vals) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count * dim; i += step) {
        int count_i = i / dim;
        int dim_i = i % dim;
        vals[count_i][dim_i] = ins1[count_i][dim_i] * ins2[count_i][dim_i];
    }
}

void PMultiForward(vector<dtype*> &ins1, vector<dtype*> &ins2, int count, int dim,
        vector<dtype*> &vals) {
    int block_count = DefaultBlockCount(count * dim);
    NumberPointerArray ins1_arr, ins2_arr, val_arr;
    ins1_arr.init((dtype**)ins1.data(), count);
    ins2_arr.init((dtype**)ins2.data(), count);
    val_arr.init((dtype**)vals.data(), count);
    KernelPMultiForward<<<block_count, TPB>>>(ins1_arr.value, ins2_arr.value, count, dim,
            (dtype **)val_arr.value);
    CheckCudaError();
}

__global__ void KernelFullDivForward(dtype **numerators,
        dtype **denominators,
        int count,
        int *dims,
        int max_dim,
        dtype **results) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count * max_dim; i += step) {
        int count_i = i / max_dim;
        int dim_i = i % max_dim;
        int dim = dims[count_i];
        if (dim_i < dim) {
            results[count_i][dim_i] = numerators[count_i][dim_i] / denominators[count_i][dim_i];
        }
    }
}

void FullDivForward(vector<dtype*> &numerators,
        vector<dtype*> &denominators,
        int count,
        vector<int> &dims,
        vector<dtype*> &results) {
    int max_dim = *max_element(dims.begin(), dims.end());
    int block_count = DefaultBlockCount(count * max_dim);
    NumberPointerArray numerator_arr, denominator_arr, result_arr;
    numerator_arr.init((dtype**)numerators.data(), count);
    denominator_arr.init((dtype**)denominators.data(), count);
    result_arr.init((dtype**)results.data(), count);
    IntArray dim_arr;
    dim_arr.init(dims.data(), dims.size());
    KernelFullDivForward<<<block_count, TPB>>>(numerator_arr.value, denominator_arr.value, count,
            dim_arr.value, max_dim, (dtype **)result_arr.value);
    CheckCudaError();
}

__global__ void KernelFullDivBackward(dtype **grads,
        dtype **numerator_vals,
        dtype **denominator_vals,
        int count,
        int *dims,
        int max_dim,
        dtype **numerator_grads,
        dtype **denominator_grads) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();

    for (int i = index; i < count * max_dim; i += step) {
        int count_i = i / max_dim;
        int dim_i = i % max_dim;
        if (dim_i < dims[count_i]) {
            DeviceAtomicAdd(numerator_grads[count_i] + dim_i, grads[count_i][dim_i] /
                    denominator_vals[count_i][dim_i]);
            DeviceAtomicAdd(denominator_grads[count_i] + dim_i, -grads[count_i][dim_i] *
                    numerator_vals[count_i][dim_i] / (denominator_vals[count_i][dim_i] *
                        denominator_vals[count_i][dim_i]));
        }
    }
}

void FullDivBackward(vector<dtype*> &grads,
        vector<dtype*> &denominator_vals,
        vector<dtype*> &numerator_vals,
        int count,
        vector<int> &dims,
        vector<dtype*> &numerator_grads,
        vector<dtype*> &denominator_grads) {
    int max_dim = *max_element(dims.begin(), dims.end());
    NumberPointerArray loss_arr, denominator_val_arr, numerator_val_arr, numerator_loss_arr,
        denominator_loss_arr;
    loss_arr.init((dtype**)grads.data(), grads.size());
    denominator_val_arr.init((dtype**)denominator_vals.data(), denominator_vals.size());
    numerator_val_arr.init((dtype**)numerator_vals.data(), numerator_vals.size());
    numerator_loss_arr.init((dtype**)numerator_grads.data(), numerator_grads.size());
    denominator_loss_arr.init((dtype**)denominator_grads.data(), denominator_grads.size());
    IntArray dim_arr;
    dim_arr.init(dims.data(), dims.size());

    int block_count = DefaultBlockCount(count * max_dim);
    KernelFullDivBackward<<<block_count, TPB>>>(loss_arr.value, numerator_val_arr.value,
            denominator_val_arr.value, count, dim_arr.value, max_dim,
            (dtype **)numerator_loss_arr.value, (dtype **)denominator_loss_arr.value);
    CheckCudaError();
}

__global__ void KernelSplitForward(dtype **inputs, int *offsets, int count, int *rows, int max_row,
        int *in_rows,
        int *cols,
        int max_col,
        dtype **results) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    int n = max_col * max_row;

    for (int i = index; i < count * max_row * max_col; i += step) {
        int count_i = i / n;
        int row = rows[count_i];
        int col = cols[count_i];
        int result_offset = i % n;
        int col_i = result_offset / row;
        if (col_i < col) {
            int row_i = result_offset % row;
            int in_row = in_rows[count_i];
            int offset = offsets[count_i];
            results[count_i][col_i * row + row_i] =
                inputs[count_i][in_row * col_i + offset + row_i];
        }
    }
}

void SplitForward(vector<dtype*> &inputs, vector<int> &offsets, int count, vector<int> &rows,
        vector<int> &in_rows,
        vector<int> &cols,
        vector<dtype*> &results) {
    NumberPointerArray input_arr, result_arr;
    input_arr.init(inputs.data(), inputs.size());
    result_arr.init(results.data(), results.size());
    IntArray offset_arr, row_arr, in_row_arr, col_arr;
    offset_arr.init((int*)offsets.data(), offsets.size());
    row_arr.init(rows.data(), rows.size());
    in_row_arr.init(in_rows.data(), in_rows.size());
    col_arr.init(cols.data(), cols.size());
    int max_row = *max_element(rows.begin(), rows.end());
    int max_col = *max_element(cols.begin(), cols.end());

    int block_count = DefaultBlockCount(count * max_row * max_col);
    KernelSplitForward<<<block_count, TPB>>>(input_arr.value, offset_arr.value, count,
            row_arr.value, max_row, in_row_arr.value, col_arr.value, max_col, result_arr.value);
    CheckCudaError();
}

__global__ void KernelSplitBackward(dtype **grads, int *offsets, int count, int *rows, int max_row,
        int *in_rows,
        int *cols,
        int max_col,
        dtype **input_grads) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    int n = max_col * max_row;

    for (int i = index; i < count * max_row * max_col; i += step) {
        int count_i = i / n;
        int row = rows[count_i];
        int col = cols[count_i];
        int result_offset = i % n;
        int col_i = result_offset / row;
        if (col_i < col) {
            int row_i = result_offset % row;
            int in_row = in_rows[count_i];
            int offset = offsets[count_i];
            DeviceAtomicAdd(input_grads[count_i] + in_row * col_i + offset + row_i,
                    grads[count_i][col_i * row + row_i]);
        }
    }
}

void SplitBackward(vector<dtype*> &grads, vector<int> offsets, int count, vector<int> &rows,
        vector<int> &in_rows,
        vector<int> &cols,
        vector<dtype*> &input_grads) {
    NumberPointerArray grad_arr, input_grad_arr;
    grad_arr.init((dtype**)grads.data(), grads.size());
    input_grad_arr.init((dtype**)input_grads.data(), input_grads.size());
    IntArray offset_arr, row_arr, in_row_arr, col_arr;
    offset_arr.init(offsets.data(), offsets.size());
    row_arr.init(rows.data(), rows.size());
    in_row_arr.init(in_rows.data(), in_rows.size());
    col_arr.init(cols.data(), cols.size());
    int max_row = *max_element(rows.begin(), rows.end());
    int max_col = *max_element(cols.begin(), cols.end());
    int block_count = DefaultBlockCount(count * max_row * max_col);
    KernelSplitBackward<<<block_count, TPB>>>(grad_arr.value, offset_arr.value, count,
            row_arr.value, max_row, in_row_arr.value, col_arr.value, max_col,
            input_grad_arr.value);
    CheckCudaError();
}

__global__ void KernelSubForward(dtype **minuend, dtype **subtrahend,
        int count,
        int *dims,
        int max_dim,
        dtype **results) {
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

void SubForward(vector<dtype*> &minuend,
        vector<dtype*> &subtrahend,
        int count,
        vector<int> &dims,
        vector<dtype*> &results) {
    int max_dim = *max_element(dims.begin(), dims.end());
    int block_count = DefaultBlockCount(count * max_dim);
    NumberPointerArray minuend_arr, subtrahend_arr, result_arr;
    minuend_arr.init((dtype**)minuend.data(), count);
    subtrahend_arr.init((dtype**)subtrahend.data(), count);
    result_arr.init((dtype**)results.data(), count);
    IntArray dim_arr;
    dim_arr.init(dims.data(), dims.size());
    KernelSubForward<<<block_count, TPB>>>((dtype**)minuend_arr.value,
            (dtype **)subtrahend_arr.value, count, dim_arr.value, max_dim,
            (dtype **)result_arr.value);
    CheckCudaError();
}

__global__ void KernelSubBackward(dtype **grads, int count, int *dims, int max_dim,
        dtype **minuend_grads,
        dtype **subtrahend_grads) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count * max_dim; i += step) {
        int count_i = i / max_dim;
        int dim_i = i % max_dim;
        if (dim_i < dims[count_i]) {
            DeviceAtomicAdd(minuend_grads[count_i] + dim_i, grads[count_i][dim_i]);
            DeviceAtomicAdd(subtrahend_grads[count_i] + dim_i, -grads[count_i][dim_i]);
        }
    }
}

void SubBackward(vector<dtype*> &grads, int count, vector<int> &dims,
        vector<dtype*> &minuend_grads,
        vector<dtype*> &subtrahend_grads) {
    int max_dim = *max_element(dims.begin(), dims.end());
    int block_count = DefaultBlockCount(count * max_dim);
    NumberPointerArray loss_arr, minuend_loss_arr, subtrahend_loss_arr;
    loss_arr.init((dtype**)grads.data(), grads.size());
    minuend_loss_arr.init((dtype**)minuend_grads.data(), minuend_grads.size());
    subtrahend_loss_arr.init((dtype**)subtrahend_grads.data(), subtrahend_grads.size());

    IntArray dim_arr;
    dim_arr.init(dims.data(), dims.size());

    KernelSubBackward<<<block_count, TPB>>>((dtype **)loss_arr.value, count,
            dim_arr.value, max_dim, (dtype **)minuend_loss_arr.value,
            (dtype **)subtrahend_loss_arr.value);
    CheckCudaError();
}

__global__ void KernelPMultiBackward(dtype **grads,
        dtype **in_vals1,
        dtype **in_vals2,
        int count,
        int dim,
        dtype **in_grads1,
        dtype **in_grads2) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count * dim; i += step) {
        int count_i = i / dim;
        int dim_i = i % dim;
        DeviceAtomicAdd(in_grads1[count_i] + dim_i,
                grads[count_i][dim_i] * in_vals2[count_i][dim_i]);
        DeviceAtomicAdd(in_grads2[count_i] + dim_i,
                grads[count_i][dim_i] * in_vals1[count_i][dim_i]);
    }
}

void PMultiBackward(vector<dtype*> &grads,
        vector<dtype*> &in_vals1,
        vector<dtype*> &in_vals2,
        int count,
        int dim,
        vector<dtype*> &in_grads1,
        vector<dtype*> &in_grads2) {
    int block_count = DefaultBlockCount(count * dim);
    NumberPointerArray grad_arr, in_vals1_arr, in_vals2_arr, in_grads1_arr,
                       in_grads2_arr;
    grad_arr.init((dtype**)grads.data(), grads.size());
    in_vals1_arr.init((dtype**)in_vals1.data(), in_vals1.size());
    in_vals2_arr.init((dtype**)in_vals2.data(), in_vals2.size());
    in_grads1_arr.init((dtype**)in_grads1.data(), in_grads1.size());
    in_grads2_arr.init((dtype**)in_grads2.data(), in_grads2.size());
    KernelPMultiBackward<<<block_count, TPB>>>(grad_arr.value, in_vals1_arr.value,
            in_vals2_arr.value, count, dim, (dtype **)in_grads1_arr.value,
            (dtype **)in_grads2_arr.value);
    CheckCudaError();
}

__global__ void KernelPAddForward(dtype **ins, int count, int *dims, int max_dim, int in_count,
        dtype **vals) {
    int i = DeviceDefaultIndex();
    if (i < count * max_dim) {
        int count_i = i / max_dim;
        int dim_i = i % max_dim;
        int dim = dims[count_i];
        if (dim_i < dim) {
            dtype sum = 0;
            for (int j = 0; j < in_count; ++j) {
                sum += ins[count_i * in_count + j][dim_i];
            }
            vals[count_i][dim_i] = sum;
        }
    }
}

void PAddForward(vector<dtype*> &ins, int count, vector<int> &dims, int max_dim, int in_count,
        vector<dtype*> &vals,
        IntArray &dim_arr) {
    NumberPointerArray out_arr, in_arr;
    in_arr.init(ins.data(), ins.size());
    out_arr.init(vals.data(), vals.size());
    dim_arr.init(dims.data(), dims.size());
    int block_count = DefaultBlockCountWithoutLimit(count * max_dim);
    KernelPAddForward<<<block_count, TPB>>>(in_arr.value, count, dim_arr.value, max_dim, in_count,
            out_arr.value);
    CheckCudaError();
}

__global__ void KernelPAddBackward(dtype **grads, int count, int *dims, int max_dim, int in_count,
        dtype **in_grads) {
    int i = DeviceDefaultIndex();
    int n = in_count * max_dim;
    if (i < count * in_count * max_dim) {
        int count_i = i / n;
        int ni = i % n;
        int dim_i = ni % max_dim;
        int in_count_i = ni / max_dim;
        int dim = dims[count_i];
        if (dim_i < dim) {
            DeviceAtomicAdd(in_grads[count_i * in_count + in_count_i] + dim_i,
                    grads[count_i][dim_i]);
        }
    }
}

void PAddBackward(vector<dtype*> &grads, int count, int max_dim, int in_count,
        vector<dtype*> &in_grads,
        IntArray &dim_arr) {
    NumberPointerArray in_grad_arr, out_grad_arr;
    in_grad_arr.init(in_grads.data(), in_grads.size());
    out_grad_arr.init(grads.data(), grads.size());

    int block_count = DefaultBlockCountWithoutLimit(in_count * count * max_dim);
    KernelPAddBackward<<<block_count, TPB>>>(out_grad_arr.value, count, dim_arr.value, max_dim,
            in_count, in_grad_arr.value);
    CheckCudaError();
}

__global__ void KernelSoftMaxLoss(dtype **vals, dtype **grads,
        int *correct_count, int *answers, int batchsize, int count, int dim) {
    volatile __shared__ int opt_label;
    volatile __shared__ dtype shared_val[TPB];
    volatile __shared__ int64_t max_indexes[TPB];
    volatile __shared__ dtype scores_sum[TPB];
    volatile __shared__ dtype scores[TPB];
    int dim_i = threadIdx.x;
    int count_i = blockIdx.x;
    if (count_i == 0 && dim_i == 0) {
        *correct_count = 0;
    }
    shared_val[dim_i] = dim_i < dim ? vals[count_i][dim_i] : -1e10;
    max_indexes[dim_i] = dim_i;
    __syncthreads();

    for (int i = (blockDim.x >> 1); i > 0; i >>= 1) {
        if (shared_val[threadIdx.x + i] > shared_val[threadIdx.x]) { // race
            shared_val[threadIdx.x] = shared_val[threadIdx.x + i]; // race
            max_indexes[threadIdx.x] = max_indexes[threadIdx.x + i]; // race
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        opt_label = max_indexes[0];
        if (answers[count_i] == opt_label) {
            atomicAdd(correct_count, 1);
        }
    }
    __syncthreads();

    dtype max_score = vals[count_i][opt_label];
    dtype score = dim_i < dim ? cuda_exp(vals[count_i][dim_i] - max_score) :
        0.0f;
    scores[dim_i] = score;
    scores_sum[dim_i] = score;

    for (int i = (blockDim.x >> 1); i > 0; i >>= 1) {
        scores_sum[threadIdx.x] = scores_sum[threadIdx.x] +
            scores_sum[threadIdx.x + i]; // race
        __syncthreads();
    }

    if (dim_i < dim) {
        grads[count_i][dim_i] = (scores[dim_i] / scores_sum[0] -
                (dim_i == answers[count_i] ? 1 : 0)) / batchsize;
    }
}

void SoftMaxLoss(vector<dtype*> &vals, vector<dtype*> &losses, int *correct_count,
        vector<int> &answers,
        int batchsize,
        int count,
        int dim) {
    if (dim > TPB) {
        abort();
    }
    int thread_count = NextTwoIntegerPowerNumber(dim);
    NumberPointerArray val_arr;
    val_arr.init((dtype**)vals.data(), vals.size());
    NumberPointerArray loss_arr;
    loss_arr.init((dtype**)losses.data(), losses.size());
    IntArray answer_arr;
    answer_arr.init((int*)answers.data(), answers.size());
    KernelSoftMaxLoss<<<count, thread_count>>>(
            const_cast<dtype **>(val_arr.value),
            const_cast<dtype **>(loss_arr.value),
            correct_count,
            answer_arr.value,
            batchsize,
            count,
            dim);
    CheckCudaError();
}

__global__ void KernelCrossEntropyLoss(dtype **vals, int *answers, int count, int *cols,
        int max_col,
        int row,
        dtype factor,
        dtype **grads) {
    int i = DeviceDefaultIndex();
    if (i < count * max_col) {
        int count_i = i / max_col;
        int col_i = i % max_col;
        int col = cols[count_i];
        if (col_i < col) {
            int answer = answers[count_i * max_col + col_i];
            DeviceAtomicAdd(grads[count_i] + col_i * row + answer,
                    -1 / vals[count_i][col_i * row + answer] * factor);
        }
    }
}

__global__ void KernelCrossEntropgyLossValue(dtype **vals, int *answers, int count, int *cols,
        int max_col,
        int row,
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
    int count_i = index / max_col;
    if (count_i < count) {
        int col = cols[count_i];
        int col_i = index % max_col;
        if (col_i < col) {
            int answer_id = answers[index];
            shared_sum[threadIdx.x] = -cuda_log(vals[count_i][row * col_i + answer_id]);
        } else {
            shared_sum[threadIdx.x] = 0.0f;
        }
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

dtype CrossEntropyLoss(vector<dtype *> &vals, const vector<vector<int>> &answers, int count,
        int row,
        dtype factor,
        vector<dtype *> &grads) {
    NumberPointerArray val_arr, grad_arr;
    val_arr.init((dtype**)vals.data(), vals.size());
    grad_arr.init((dtype**)grads.data(), grads.size());
    IntArray answer_arr, col_arr;

    vector<int> cols;
    cols.reserve(count);
    for (const auto &it : answers) {
        cols.push_back(it.size());
    }
    col_arr.init(cols.data(), cols.size());

    int max_col = *max_element(cols.begin(), cols.end());
    vector<int> answers_1d;
    answers_1d.reserve(count * max_col);
    for (int i = 0; i < count; ++i) {
        for (int answer : answers.at(i)) {
            answers_1d.push_back(answer);
        }
        for (int j = 0; j < max_col - answers.at(i).size(); ++j) {
            answers_1d.push_back(0);
        }
    }
    answer_arr.init(answers_1d.data(), answers_1d.size());
    KernelCrossEntropyLoss<<<DefaultBlockCountWithoutLimit(count * max_col), TPB>>>(val_arr.value,
            answer_arr.value, count, col_arr.value, max_col, row, factor, grad_arr.value);
    CheckCudaError();

    int block_count = DefaultBlockCountWithoutLimit(count * max_col);
    NumberArray global_sum;
    global_sum.init(block_count);
    DeviceInt block_counter;
    block_counter.init();
    DeviceNumber result;
    result.init();
    KernelCrossEntropgyLossValue<<<block_count, TPB>>>(val_arr.value, answer_arr.value, count,
            col_arr.value, max_col, row, global_sum.value, block_counter.value, result.value);
    CheckCudaError();
    result.copyFromDeviceToHost();
    return result.v * factor;
}

__global__ void KernelMultiCrossEntropyLoss(dtype **vals, int **answers,
        int count,
        int dim,
        dtype factor,
        dtype **losses) {
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

__global__ void KernelMultiCrossEntropyLossVector(dtype **in_vals,
        int **answers,
        int count,
        int dim,
        dtype **result) {
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

dtype MultiCrossEntropyLoss(vector<dtype*> &vals, vector<vector<int>> &answers,
        int count,
        int dim,
        dtype factor,
        vector<dtype*> &losses) {
    int block_count = DefaultBlockCount(count * dim);
    NumberPointerArray val_arr, loss_arr;
    val_arr.init((dtype**)vals.data(), count);
    loss_arr.init((dtype**)losses.data(), count);

    vector<shared_ptr<IntArray>> answer_gpus;
    vector<int *> answer_gpu_pointers;
    for (auto &answer : answers) {
        shared_ptr<IntArray> answer_gpu(new IntArray);
        answer_gpu->init(answer.data(), answer.size());
        answer_gpus.push_back(answer_gpu);
        answer_gpu_pointers.push_back(answer_gpu->value);
    }

    IntPointerArray answer_arr;
    answer_arr.init((int**)answer_gpu_pointers.data(), count);
    KernelMultiCrossEntropyLoss<<<block_count, TPB>>>(val_arr.value, answer_arr.value, count, dim,
            factor, (dtype **)loss_arr.value);
    CheckCudaError();

    vector<shared_ptr<NumberArray>> nums;
    vector<dtype *> logged_vec = GPUArrayVectors(nums, count, dim);

    NumberPointerArray logged_arr;
    logged_arr.init(logged_vec.data(), count);

    KernelMultiCrossEntropyLossVector<<<block_count, TPB>>>(val_arr.value, answer_arr.value, count,
            dim, (dtype **)logged_arr.value);
    CheckCudaError();

    vector<shared_ptr<NumberArray>> ce_loss_arrs;
    vector<dtype *> ce_losses = GPUArrayVectors(ce_loss_arrs, count, 1);
    for (auto &ptr : ce_loss_arrs) {
        vector<dtype> vec = ptr->toCpu();
    }
    vector<dtype *> const_logged_arr;
    auto return_const = [](dtype *v) -> dtype* {
        return const_cast<dtype*>(v);
    };
    transform(logged_vec.begin(), logged_vec.end(), back_inserter(const_logged_arr), return_const);

    vector<int> dims;
    for (int i = 0; i < count; ++i) {
        dims.push_back(dim);
    }
    VectorSumForward(const_logged_arr, count, 1, dims, ce_losses);

    dtype ce_loss_sum = 0.0f;

    for (auto &ptr : ce_loss_arrs) {
        vector<dtype> vec = ptr->toCpu();
        if (vec.size() != 1) {
            cerr << "vec size is not 1" << endl;
            abort();
        }
        dtype l = vec.front() * factor;
        ce_loss_sum += l;
    }

    return ce_loss_sum;
}

__global__ void KernelKLCrossEntropyLoss(dtype **vals, dtype **answers,
        int count,
        int dim,
        dtype factor,
        dtype **losses) {
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

__global__ void KernelKLCrossEntropyLossVector(dtype **in_vals,
        dtype **answers,
        int count,
        int dim,
        dtype **result) {
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

dtype KLCrossEntropyLoss(vector<dtype*> &vals,
        vector<shared_ptr<vector<dtype>>> &answers,
        int count,
        int dim,
        dtype factor,
        vector<dtype*> &losses) {
    int block_count = DefaultBlockCount(count * dim);
    NumberPointerArray val_arr, loss_arr;
    val_arr.init((dtype**)vals.data(), count);
    loss_arr.init((dtype**)losses.data(), count);

    vector<shared_ptr<NumberArray>> answer_gpus;
    vector<dtype *> answer_gpu_pointers;
    for (auto &answer : answers) {
        shared_ptr<NumberArray> answer_gpu(new NumberArray);
        answer_gpu->init(answer->data(), answer->size());
        answer_gpus.push_back(answer_gpu);
        answer_gpu_pointers.push_back(answer_gpu->value);
    }

    NumberPointerArray answer_arr;
    answer_arr.init((dtype**)answer_gpu_pointers.data(), count);
    KernelKLCrossEntropyLoss<<<block_count, TPB>>>(val_arr.value, answer_arr.value, count, dim,
            factor, (dtype **)loss_arr.value);
    CheckCudaError();

    vector<shared_ptr<NumberArray>> nums;
    vector<dtype *> logged_vec = GPUArrayVectors(nums, count, dim);

    NumberPointerArray logged_arr;
    logged_arr.init(logged_vec.data(), count);

    KernelKLCrossEntropyLossVector<<<block_count, TPB>>>(val_arr.value, answer_arr.value, count,
            dim, (dtype **)logged_arr.value);
    CheckCudaError();

    vector<shared_ptr<NumberArray>> ce_loss_arrs;
    vector<dtype *> ce_losses = GPUArrayVectors(ce_loss_arrs, count, 1);
    for (auto &ptr : ce_loss_arrs) {
        vector<dtype> vec = ptr->toCpu();
    }
    vector<dtype *> const_logged_arr;
    auto return_const = [](dtype *v) -> dtype* {
        return const_cast<dtype*>(v);
    };
    transform(logged_vec.begin(), logged_vec.end(), back_inserter(const_logged_arr), return_const);

    vector<int> dims;
    for (int i = 0; i < count; ++i) {
        dims.push_back(dim);
    }
    VectorSumForward(const_logged_arr, count, 1, dims, ce_losses);

    dtype ce_loss_sum = 0.0f;

    for (auto &ptr : ce_loss_arrs) {
        vector<dtype> vec = ptr->toCpu();
        if (vec.size() != 1) {
            cerr << "vec size is not 1" << endl;
            abort();
        }
        dtype l = vec.front() * factor;
        ce_loss_sum += l;
    }

    return ce_loss_sum;
}

__global__ void KernelMax(dtype **v, int count, int dim, volatile dtype *block_maxes,
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
        int max_i = 100000000;
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

__global__ void KernelSingleMax(dtype **v, int count, int dim,
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

void Max(dtype **v, int count, int dim, int *max_indexes, dtype *max_vals) {
    int thread_count = min(NextTwoIntegerPowerNumber(dim), TPB);
    int block_y_count = (dim - 1 + thread_count) / thread_count;
    dim3 block_dim(count, block_y_count, 1);

    NumberArray block_maxes;
    block_maxes.init(block_y_count * count);
    IntArray block_max_is, block_counters;
    block_max_is.init(block_y_count * count);
    block_counters.init(count);

    KernelMax<<<block_dim, thread_count>>>(v, count, dim, block_maxes.value, block_max_is.value,
            block_counters.value, max_indexes, max_vals);
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
            MyCudaMemcpyKind::DEVICE_TO_HOST);
    MyCudaMemcpy(max_indexer_gold.data(), max_indexer_arr.value, count * sizeof(int),
            MyCudaMemcpyKind::DEVICE_TO_HOST);
    for (int i = 0; i < count; ++i) {
        if (max_indexer_target.at(i) != max_indexer_gold.at(i)) {
            cerr << fmt::format("max_indexer_target:{} max_indexer_gold:{}\n",
                    max_indexer_target.at(i), max_indexer_gold.at(i));
            PrintNums(v, i, dim);
            abort();
        }
    }
#endif

    CheckCudaError();
}

vector<vector<int>> Predict(vector<dtype*> &vals, int count, vector<int> &cols, int row) {
    int col_sum = accumulate(cols.begin(), cols.end(), 0);
    vector<dtype *> split_vals;
    split_vals.reserve(col_sum);
    int loop_i = 0;
    for (dtype *addr : vals) {
        int col = cols.at(loop_i++);
        for (int i = 0; i < col; ++i) {
            split_vals.push_back(addr + row * i);
        }
    }

    NumberPointerArray val_arr;
    val_arr.init(split_vals.data(), col_sum);
    IntArray max_index_arr;
    max_index_arr.init(col_sum);
    NumberArray max_val_arr;
    max_val_arr.init(col_sum);
    Max(val_arr.value, col_sum, row, max_index_arr.value, max_val_arr.value);
    vector<int> merged_indexes = max_index_arr.toCpu();
    int merged_indexes_i = 0;

    vector<vector<int>> result_indexes;
    result_indexes.reserve(count);
    for (int i = 0; i < count; ++i) {
        int col = cols.at(i);
        vector<int> indexes;
        indexes.reserve(col);
        for (int j = 0; j < col; ++j) {
            indexes.push_back(merged_indexes.at(merged_indexes_i++));
        }
        result_indexes.push_back(move(indexes));
    }
    return result_indexes;
}


__global__ void KernelSoftMaxLossByExp(dtype **exps, int count, int dim,
        dtype **vals,
        dtype *sums,
        dtype *max_vals,
        int *answers,
        dtype reverse_batchsize,
        dtype **grads,
        dtype *losses) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < dim * count; i += step) {
        int count_i = i / dim;
        int dim_i = i % dim;

        dtype loss = exps[count_i][dim_i] / sums[count_i];
        if (dim_i == answers[count_i]) {
            loss -= 1.0f;
        }
        grads[count_i][dim_i] = loss * reverse_batchsize;
        losses[count_i] = (cuda_log(sums[count_i]) - vals[count_i][answers[count_i]] + max_vals[count_i])
            * reverse_batchsize;
    }
}

void SoftMaxLossByExp(dtype **exps, int count, int dim, dtype **vals,
        dtype *sums,
        dtype *max_vals,
        int *answers,
        dtype reverse_batchsize,
        dtype **grads,
        dtype *losses) {
    int block_count = DefaultBlockCount(dim * count);
    KernelSoftMaxLossByExp<<<block_count, TPB>>>(exps, count, dim, vals, sums, max_vals, answers,
            reverse_batchsize, (dtype **)grads, losses);
    CheckCudaError();
}

__global__ void KernelMaxScalarForward(dtype **v, int count, int head_count, int* head_dims,
        int max_dim,
        volatile dtype *block_maxes,
        volatile int *block_max_is,
        int *block_counters,
        int *max_indexes,
        dtype **max_vals) {
    __shared__ volatile dtype shared_max[TPB];
    __shared__ volatile int shared_max_i[TPB];
    __shared__ volatile bool is_last_block;
    if (threadIdx.x == 0 && blockIdx.y == 0) {
        block_counters[blockIdx.x] = 0;
    }
    if (threadIdx.x == 0) {
        is_last_block = false;
    }

    int head_count_i = blockIdx.x % head_count;
    int head_offset = blockIdx.y * blockDim.x + threadIdx.x;
    int count_i = blockIdx.x / head_count;
    int offset = head_offset + head_dims[count_i] * head_count_i;
    shared_max[threadIdx.x] = head_offset < head_dims[count_i] ? v[count_i][offset] : -1e10;
    shared_max_i[threadIdx.x] = head_offset;
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
        __syncthreads();

        for (int i = (blockDim.x >> 1); i > 0; i >>= 1) {
            if (threadIdx.x < i && shared_max[threadIdx.x + i] > shared_max[threadIdx.x]) {
                shared_max[threadIdx.x] = shared_max[threadIdx.x + i];
                shared_max_i[threadIdx.x] = shared_max_i[threadIdx.x + i];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            max_vals[count_i][head_count_i] = shared_max[0];
            int max_ii = shared_max_i[0];
            max_indexes[blockIdx.x] = max_ii;
            if (max_ii < 0 || max_ii >= max_dim) {
                printf("threadIdx.x == 0 max_i:%d head_count_i:%d max_val:%f\n",
                        max_indexes[blockIdx.x], head_count_i, max_vals[count_i][head_count_i]);
                assert(false);
            }
        }
    }
}

void MaxScalarForward(vector<dtype*> &inputs, int count, int head_count, vector<int> &head_dims,
        vector<dtype*> &results,
        vector<int> *max_indexes) {
    int max_dim = *max_element(head_dims.begin(), head_dims.end());
    int thread_count = min(NextTwoIntegerPowerNumber(max_dim), TPB);
    int block_y_count = (max_dim - 1 + thread_count) / thread_count;
    dim3 block_dim(count * head_count, block_y_count, 1);

    NumberArray block_maxes;
    block_maxes.init(block_y_count * count * head_count);
    IntArray block_max_is, block_counters;
    block_max_is.init(block_y_count * count * head_count);
    block_counters.init(count * head_count);

    NumberPointerArray input_arr;
    input_arr.init((dtype**)inputs.data(), inputs.size());
    NumberPointerArray result_arr;
    result_arr.init((dtype**)results.data(), results.size());
    IntArray max_index_arr;
    max_index_arr.init(max_indexes->size());

    IntArray head_dim_arr;
    head_dim_arr.init(head_dims.data(), head_dims.size());

    KernelMaxScalarForward<<<block_dim, thread_count>>>((dtype **)input_arr.value,
            count, head_count, head_dim_arr.value, max_dim, block_maxes.value, block_max_is.value,
            block_counters.value,
            max_index_arr.value, (dtype **)result_arr.value);
    CheckCudaError();
    if (max_indexes != nullptr) {
        MyCudaMemcpy(max_indexes->data(), max_index_arr.value, count * sizeof(int),
                MyCudaMemcpyKind::DEVICE_TO_HOST);
    }
}

__global__ void KernelMaxScalarBackward(dtype **grads, int *indexes, int count,
        dtype **input_grads) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count; i += step) {
        DeviceAtomicAdd(input_grads[i] + indexes[i], grads[i][0]);
    }
}

void MaxScalarBackward(vector<dtype *> &grads, vector<int> &indexes, int count,
        vector<dtype*> &input_grads) {
    int block_count = DefaultBlockCount(count);
    NumberPointerArray grad_arr, input_grad_arr;
    grad_arr.init((dtype**)grads.data(), grads.size());
    input_grad_arr.init((dtype**)input_grads.data(), input_grads.size());
    IntArray index_arr;
    index_arr.init((int*)indexes.data(), indexes.size());
    KernelMaxScalarBackward<<<block_count, TPB>>>((dtype **)grad_arr.value,
            index_arr.value, count,
            (dtype **)input_grad_arr.value);
    CheckCudaError();
}

__global__ void KernelVectorSumForward(dtype **v, int count, int col, int *rows,
        volatile dtype *block_sums,
        int *block_counters,
        dtype **results) {
    __shared__ volatile extern dtype shared_sum[];
    __shared__ volatile bool is_last_block;
    if (threadIdx.x == 0 && blockIdx.y == 0) {
        block_counters[blockIdx.x] = 0;
    }
    if (threadIdx.x == 0) {
        is_last_block = false;
    }

    int count_i = blockIdx.x / col;
    int col_i = blockIdx.x % col;
    int row_i = blockIdx.y * blockDim.x + threadIdx.x;
    int row = rows[count_i];
    int offset = row * col_i + row_i;

    shared_sum[threadIdx.x] = row_i < row ? v[count_i][offset] : 0.0f;
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
            results[count_i][col_i] = shared_sum[0];
        }
    }
}

void VectorSumForward(vector<dtype *> &inputs, int count, int col, vector<int> &head_dims,
        vector<dtype*> &results) {
    int max_row = *max_element(head_dims.begin(), head_dims.end());
    int thread_count = min(NextTwoIntegerPowerNumber(max_row), TPB);
    int block_y_count = (max_row - 1 + thread_count) / thread_count;
    dim3 block_dim(count * col, block_y_count, 1);

    NumberArray block_sums;
    block_sums.init(block_y_count * count * col);
    IntArray block_counters;
    block_counters.init(count * col);

    NumberPointerArray input_arr;
    input_arr.init((dtype**)inputs.data(), inputs.size());
    NumberPointerArray result_arr;
    result_arr.init((dtype**)results.data(), results.size());

    IntArray head_dim_arr;
    head_dim_arr.init(head_dims.data(), head_dims.size());

    KernelVectorSumForward<<<block_dim, thread_count, thread_count * sizeof(dtype)>>>(
            (dtype **)input_arr.value, count, col, head_dim_arr.value, block_sums.value,
            block_counters.value, (dtype **)result_arr.value);
    CheckCudaError();
}

__global__ void KernelVectorSumBackward(dtype **grads, int count, int col, int *rows, int max_row,
        dtype **input_grads) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    int n = col * max_row;
    for (int i = index; i < count * max_row * col; i += step) {
        int count_i = i / n;
        int row = rows[count_i];
        int x = i % n;
        int row_i = x % max_row;
        int col_i = x / max_row;

        if (row_i < row) {
            DeviceAtomicAdd(input_grads[count_i] + col_i * row + row_i, grads[count_i][col_i]);
        }
    }
}

void VectorSumBackward(vector<dtype*> &grads, int count, int col, vector<int> &rows,
        vector<dtype*> &input_grads) {
    int max_row = *max_element(rows.begin(), rows.end());
    int block_count = DefaultBlockCount(count * max_row);
    NumberPointerArray grad_arr, input_grad_arr;
    grad_arr.init((dtype**)grads.data(), grads.size());
    input_grad_arr.init((dtype**)input_grads.data(), input_grads.size());
    IntArray row_arr;
    row_arr.init(rows.data(), rows.size());
    KernelVectorSumBackward<<<block_count, TPB>>>((dtype **)grad_arr.value, count, col,
            row_arr.value, max_row, (dtype **)input_grad_arr.value);
    CheckCudaError();
}

__global__ void KernelMaxScalar(dtype **v, int count, int *rows, int *cols,
        volatile dtype *block_maxes,
        int *block_counters,
        dtype *max_vals) {
    __shared__ volatile extern dtype shared_max[];
    __shared__ volatile bool is_last_block;

    int count_i = blockIdx.x;
    int col = cols[count_i];
    int col_i = blockIdx.z;
    if (col_i >= col) {
        return;
    }
    if (threadIdx.x == 0 && blockIdx.y == 0) {
        block_counters[blockIdx.x * gridDim.z + blockIdx.z] = 0;
    }
    if (threadIdx.x == 0) {
        is_last_block = false;
    }

    int row = rows[count_i];
    int row_i = blockIdx.y * blockDim.x + threadIdx.x;
    int offset = col_i * row + row_i;
    shared_max[threadIdx.x] = row_i < row ? v[count_i][offset] : -1e10;
    __syncthreads();

    for (int i = (blockDim.x >> 1); i > 0; i >>= 1) {
        if (threadIdx.x < i && shared_max[threadIdx.x] < shared_max[threadIdx.x + i]) {
            shared_max[threadIdx.x] = shared_max[threadIdx.x + i];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        int block_maxes_offset = blockIdx.x * gridDim.y * gridDim.z + blockIdx.z * gridDim.y +
            blockIdx.y;
        block_maxes[block_maxes_offset] = shared_max[0];
        if (atomicAdd(block_counters + blockIdx.x * gridDim.z + blockIdx.z, 1) == gridDim.y - 1) {
            is_last_block = true;
        }
    }
    __syncthreads();

    if (is_last_block) {
        dtype max = -1e10;
        for (int i = threadIdx.x; i < gridDim.y; i += blockDim.x) {
            int offset = blockIdx.x * gridDim.y * gridDim.z + blockIdx.z * gridDim.y + i;
            if (block_maxes[offset] > max) {
                max = block_maxes[offset];
            }
        }

        shared_max[threadIdx.x] = max;
        __syncthreads();

        for (int i = (blockDim.x >> 1); i > 0; i >>= 1) {
            if (threadIdx.x < i && shared_max[threadIdx.x + i] > shared_max[threadIdx.x]) {
                shared_max[threadIdx.x] = shared_max[threadIdx.x + i];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            max_vals[count_i * gridDim.z + blockIdx.z] = shared_max[0];
        }
    }
}

__global__ void KernelExp(dtype **in_vals, dtype *subtrahends, int count, int *rows, int max_row,
        int *cols,
        int max_col,
        dtype **vals) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    int n = max_row * max_col;
    for (int i = index; i < count * max_row * max_col; i += step) {
        int count_i = i / n;
        int offset = i % n;
        int col = cols[count_i];
        int row = rows[count_i];
        int col_i = offset / row;
        if (col_i < col) {
            int row_i = offset % row;
            if (row_i < row) {
                vals[count_i][offset] = cuda_exp(in_vals[count_i][offset] -
                        subtrahends[count_i * max_col + col_i]);
            }
        }
    }
}

__global__ void KernelDiv(dtype **numerators, dtype *denominators, int count, int *rows,
        int max_row,
        int *cols,
        int max_col,
        dtype **results) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    int n = max_row * max_col;
    for (int i = index; i < count * max_row * max_col; i += step) {
        int count_i = i / n;
        int offset = i % n;
        int col = cols[count_i];
        int row = rows[count_i];
        int col_i = offset / row;
        if (col_i < col) {
            int row_i = offset % row;
            if (row_i < row) {
                results[count_i][offset] = numerators[count_i][offset] /
                    denominators[count_i * max_col + col_i];
            }
        }
    }
}

__global__ void KernelVectorSum(dtype **v, int count, int *rows, int *cols,
        volatile dtype *block_sums,
        int *block_counters,
        dtype *results) {
    __shared__ volatile extern dtype shared_sum[];
    __shared__ bool is_last_block;

    int count_i = blockIdx.x;
    int col = cols[count_i];
    int col_i = blockIdx.z;
    if (col_i >= col) {
        return;
    }

    if (threadIdx.x == 0 && blockIdx.y == 0) {
        block_counters[blockIdx.x * gridDim.z + blockIdx.z] = 0;
    }
    if (threadIdx.x == 0) {
        is_last_block = false;
    }

    int row = rows[count_i];
    int row_i = blockIdx.y * blockDim.x + threadIdx.x;
    int offset = col_i * row + row_i;
    shared_sum[threadIdx.x] = row_i < row ? v[count_i][offset] : 0;
    __syncthreads();

    for (int i = (blockDim.x >> 1); i > 0; i >>= 1) {
        if (threadIdx.x < i) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + i];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        int block_sums_offset = blockIdx.x * gridDim.y * gridDim.z + blockIdx.z * gridDim.y +
            blockIdx.y;
        block_sums[block_sums_offset] = shared_sum[0];
        if (atomicAdd(block_counters + blockIdx.x * gridDim.z + blockIdx.z, 1) == gridDim.y - 1) {
            is_last_block = true;
        }
    }
    __syncthreads();

    if (is_last_block) {
        dtype sum = 0.0f;
        for (int i = threadIdx.x; i < gridDim.y; i += blockDim.x) {
            int offset = blockIdx.x * gridDim.y * gridDim.z + blockIdx.z * gridDim.y + i;
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
            results[count_i * gridDim.z + blockIdx.z] = shared_sum[0];
        }
    }
}

void SoftmaxForward(vector<dtype *> &in_vals, int count, int *rows, int max_row, int *cols,
        int max_col,
        dtype **vals) {
    int thread_count = min(NextTwoIntegerPowerNumber(max_row), TPB);
    int block_y_count = (max_row - 1 + thread_count) / thread_count;
    dim3 block_dim(count, block_y_count, max_col);
    NumberArray block_maxes;
    block_maxes.init(block_y_count * count * max_col);
    IntArray block_counters;
    block_counters.init(count * max_col);
    NumberPointerArray input_arr;
    input_arr.init((dtype**)in_vals.data(), in_vals.size());
    NumberArray max_val_arr;
    max_val_arr.init(count * max_col);
    KernelMaxScalar<<<block_dim, thread_count, thread_count * sizeof(dtype)>>>(input_arr.value,
            count, rows, cols, block_maxes.value, block_counters.value, max_val_arr.value);
    CheckCudaError();

    int block_count = DefaultBlockCount(count * max_row * max_col);
    KernelExp<<<block_count, TPB>>>(input_arr.value, max_val_arr.value, count, rows, max_row,
            cols, max_col, vals);
    CheckCudaError();

    KernelVectorSum<<<block_dim, thread_count, thread_count * sizeof(dtype)>>>(vals, count, rows,
            cols, block_maxes.value, block_counters.value, max_val_arr.value);
    CheckCudaError();

    KernelDiv<<<block_count, TPB>>>(vals, max_val_arr.value, count, rows, max_row, cols, max_col,
            vals);
    CheckCudaError();
}

__global__ void KernelPointwiseMul(dtype **a, dtype **b, int count, int *rows, int max_row,
        int *cols,
        int max_col,
        int *val_offsets,
        dtype *vals) {
    int i = DeviceDefaultIndex();
    int n = max_row * max_col;
    if (i < count * max_row * max_col) {
        int count_i = i / n;
        int offset = i % n;
        int col = cols[count_i];
        int row = rows[count_i];
        int col_i = offset / row;
        if (col_i < col) {
            int row_i = offset % row;
            if (row_i < row) {
                int val_offset = val_offsets[count_i];
                vals[val_offset + offset] = a[count_i][offset] * b[count_i][offset];
            }
        }
    }
}

__global__ void KernelVectorSum(dtype *v, int count, int *rows, int max_row, int *cols,
        int *v_offsets,
        volatile dtype *block_sums,
        int *block_counters,
        dtype *results) {
    __shared__ volatile extern dtype shared_sum[];
    __shared__ bool is_last_block;

    int count_i = blockIdx.x;
    int col = cols[count_i];
    int col_i = blockIdx.z;
    if (col_i >= col) {
        return;
    }

    if (threadIdx.x == 0 && blockIdx.y == 0) {
        block_counters[blockIdx.x * gridDim.z + blockIdx.z] = 0;
    }
    if (threadIdx.x == 0) {
        is_last_block = false;
    }

    int row = rows[count_i];
    int row_i = blockIdx.y * blockDim.x + threadIdx.x;
    int offset = col_i * row + row_i;
    int v_offset = v_offsets[count_i];
    shared_sum[threadIdx.x] = row_i < row ? v[v_offset + offset] : 0.0f;
    __syncthreads();

    for (int i = (blockDim.x >> 1); i > 0; i >>= 1) {
        if (threadIdx.x < i) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + i];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        int block_sums_offset = blockIdx.x * gridDim.y * gridDim.z + blockIdx.z * gridDim.y +
            blockIdx.y;
        block_sums[block_sums_offset] = shared_sum[0];
        if (atomicAdd(block_counters + blockIdx.x * gridDim.z + blockIdx.z, 1) == gridDim.y - 1) {
            is_last_block = true;
        }
    }
    __syncthreads();

    if (is_last_block) {
        dtype sum = 0.0f;
        for (int i = threadIdx.x; i < gridDim.y; i += blockDim.x) {
            int offset = blockIdx.x * gridDim.y * gridDim.z + blockIdx.z * gridDim.y + i;
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
            results[count_i * gridDim.z + blockIdx.z] = shared_sum[0];
        }
    }
}

__global__ void KernelSoftmaxBackward(dtype **vals, dtype **grads, int count, int *rows,
        int max_row,
        int *cols,
        int max_col,
        dtype *z,
        dtype *a,
        int *a_offsets,
        dtype **input_grads) {
    int i = DeviceDefaultIndex();
    int n = max_row * max_col;
    if (i < count * max_row * max_col) {
        int count_i = i / n;
        int offset = i % n;
        int col = cols[count_i];
        int row = rows[count_i];
        int col_i = offset / row;
        if (col_i < col) {
            int row_i = offset % row;
            if (row_i < row) {
                int a_offset = a_offsets[count_i];
                dtype b = z[count_i * max_col + col_i] - a[a_offset + offset];
                dtype x = vals[count_i][offset] *
                    ((1 - vals[count_i][offset]) * grads[count_i][offset] - b);
                DeviceAtomicAdd(input_grads[count_i] + offset, x);
            }
        }
    }
}

void SoftmaxBackward(vector<dtype *> &grads, dtype **vals, int count, int *rows, int max_row,
        int *cols,
        int max_col,
        int *offsets,
        int dim_sum,
        vector<dtype *> &in_grads) {
    NumberArray a;
    a.init(dim_sum);
    NumberPointerArray grad_arr;
    grad_arr.init(grads.data(), count);
    int block_count = DefaultBlockCountWithoutLimit(max_col * max_row * count);
    KernelPointwiseMul<<<block_count, TPB>>>(grad_arr.value, vals, count, rows, max_row, cols,
            max_col, offsets, a.value);
    CheckCudaError();

    int thread_count = min(NextTwoIntegerPowerNumber(max_row), TPB);
    int block_y_count = (max_row - 1 + thread_count) / thread_count;
    dim3 block_dim(count, block_y_count, max_col);
    NumberArray block_vals;
    block_vals.init(block_y_count * count * max_col);
    IntArray block_counters;
    block_counters.init(count * max_col);
    NumberArray z;
    z.init(count * max_col);
    KernelVectorSum<<<block_dim, thread_count, thread_count * sizeof(dtype)>>>(a.value, count,
            rows, max_row, cols, offsets, block_vals.value, block_counters.value, z.value);
    CheckCudaError();

    NumberPointerArray in_grad_arr;
    in_grad_arr.init(in_grads.data(), count);
    KernelSoftmaxBackward<<<block_count, TPB>>>(vals, grad_arr.value, count, rows, max_row, cols,
            max_col, z.value, a.value, offsets, in_grad_arr.value);
}

__global__ void KernelScaledForward(dtype **in_vals, int count, int *dims, int max_dim,
        dtype *factors,
        dtype **vals) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();

    for (int i = index; i < count * max_dim; i += step) {
        int count_i = i / max_dim;
        int dim_i = i % max_dim;

        if (dim_i < dims[count_i]) {
            vals[count_i][dim_i] = in_vals[count_i][dim_i] * factors[count_i];
        }
    }
}

void ScaledForward(vector<dtype *> &in_vals, int count, vector<int> &dims, vector<dtype> &factors,
        vector<dtype *> &vals) {
    NumberPointerArray in_val_arr, val_arr;
    in_val_arr.init(in_vals.data(), in_vals.size());
    val_arr.init(vals.data(), vals.size());
    NumberArray factor_arr;
    factor_arr.init(factors.data(), count);
    IntArray dim_arr;
    dim_arr.init(dims.data(), dims.size());
    int max_dim = *max_element(dims.begin(), dims.end());
    int block_count = DefaultBlockCount(count * max_dim);
    KernelScaledForward<<<block_count, TPB>>>(in_val_arr.value, count, dim_arr.value, max_dim,
            factor_arr.value, val_arr.value);
    CheckCudaError();
}

__global__ void KernelScaledBackward(dtype **grads, int count, int *dims, int max_dim,
        dtype *factors,
        dtype **in_grads) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();

    for (int i = index; i < count * max_dim; i += step) {
        int count_i = i / max_dim;
        int dim_i = i % max_dim;

        if (dim_i < dims[count_i]) {
            DeviceAtomicAdd(in_grads[count_i] + dim_i, grads[count_i][dim_i] * factors[count_i]);
        }
    }
}

void ScaledBackward(vector<dtype *> &grads, int count, vector<int> &dims, vector<dtype> &factors,
        vector<dtype *> &in_grads) {
    NumberPointerArray grad_arr, in_grad_arr;
    grad_arr.init(grads.data(), count);
    in_grad_arr.init(in_grads.data(), count);

    NumberArray factor_arr;
    factor_arr.init(factors.data(), count);

    IntArray dim_arr;
    dim_arr.init(dims.data(), count);

    int max_dim = *max_element(dims.begin(), dims.end());
    int block_count = DefaultBlockCount(count * max_dim);
    KernelScaledBackward<<<block_count, TPB>>>(grad_arr.value, count, dim_arr.value, max_dim,
            factor_arr.value, in_grad_arr.value);
    CheckCudaError();
}

__global__ void KernelScalarToVectorForward(dtype **inputs, int count, int input_col,
        int *rows,
        int max_row,
        dtype **results) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    int n = max_row * input_col;
    for (int i = index; i < count * max_row * input_col; i += step) {
        int count_i = i / n;
        int dim_i = i % n;
        int col_i = dim_i / max_row;
        int row_i = dim_i % max_row;
        int row = rows[count_i];
        if (row_i < row) {
            results[count_i][col_i * row + row_i] = inputs[count_i][col_i];
        }
    }
}

void ScalarToVectorForward(vector<dtype*> &inputs, int count, int input_col,
        vector<int> &rows,
        vector<dtype*> &results) {
    int max_row = *max_element(rows.begin(), rows.end());
    int block_count = DefaultBlockCount(max_row * input_col * count);
    NumberPointerArray input_arr;
    input_arr.init((dtype**)inputs.data(), inputs.size());
    NumberPointerArray result_arr;
    result_arr.init((dtype**)results.data(), inputs.size());
    IntArray row_arr;
    row_arr.init(rows.data(), rows.size());

    KernelScalarToVectorForward<<<block_count, TPB>>>((dtype **)input_arr.value,
            count, input_col, row_arr.value, max_row, (dtype **)result_arr.value);
    CheckCudaError();
}

__global__ void KernelScalarToVectorBackward(dtype **grads, int count, int input_col, int *rows,
        volatile dtype *block_sums,
        int *block_counters,
        dtype **input_grads) {
    __shared__ volatile extern dtype shared_sum[];
    __shared__ volatile bool is_last_block;
    if (threadIdx.x == 0 && blockIdx.y == 0) {
        block_counters[blockIdx.x] = 0;
    }
    if (threadIdx.x == 0) {
        is_last_block = false;
    }

    int count_i = blockIdx.x / input_col;
    int col_i = blockIdx.x % input_col;
    int row_i = blockIdx.y * blockDim.x + threadIdx.x;
    int row = rows[count_i];
    shared_sum[threadIdx.x] = row_i < rows[count_i] ? grads[count_i][row * col_i + row_i] : 0.0f;
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
            DeviceAtomicAdd(input_grads[count_i] + col_i, shared_sum[0]);
        }
    }
}

void ScalarToVectorBackward(vector<dtype*> &grads, int count, int input_col, vector<int> &rows,
        vector<dtype*> &input_grads) {
    int max_dim = *max_element(rows.begin(), rows.end());

    int thread_count = min(NextTwoIntegerPowerNumber(max_dim), TPB);
    int block_y_count = (max_dim - 1 + thread_count) / thread_count;
    dim3 block_dim(count * input_col, block_y_count, 1);

    NumberArray block_sums;
    block_sums.init(block_y_count * count * input_col);
    IntArray block_counters;
    block_counters.init(count * input_col);

    NumberPointerArray grad_arr;
    grad_arr.init((dtype**)grads.data(), grads.size());
    NumberPointerArray input_grad_arr;
    input_grad_arr.init((dtype**)input_grads.data(), input_grads.size());

    IntArray row_arr;
    row_arr.init(rows.data(), rows.size());

    KernelScalarToVectorBackward<<<block_dim, thread_count, thread_count * sizeof(dtype)>>>(
            (dtype **)grad_arr.value, count, input_col, row_arr.value, block_sums.value,
            block_counters.value, (dtype **)input_grad_arr.value);
    CheckCudaError();
}

__global__ void KernelBiasForward(dtype **in_vals, dtype *bias, int count,
        int dim,
        dtype **vals) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count * dim; i += step) {
        int count_i = i / dim;
        int dim_i = i % dim;
        vals[count_i][dim_i] = in_vals[count_i][dim_i] + bias[dim_i];
    }
}

void BiasForward(vector<dtype*> &in_vals, dtype *bias, int count, int dim,
        vector<dtype *> &vals) {
    int block_count = DefaultBlockCount(count * dim);
    NumberPointerArray in_arr, val_arr;
    in_arr.init(in_vals.data(), in_vals.size());
    val_arr.init(vals.data(), vals.size());
    KernelBiasForward<<<block_count, TPB>>>(in_arr.value, bias, count, dim,
            (dtype **)val_arr.value);
}

__global__ void KernelBiasBackward(dtype **grads, int count, int dim,
        dtype *bias_grads,
        dtype **in_grads) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count * dim; i += step) {
        int count_i = i / dim;
        int dim_i = i % dim;
        DeviceAtomicAdd(bias_grads + dim_i, grads[count_i][dim_i]);
        DeviceAtomicAdd(in_grads[count_i] + dim_i, grads[count_i][dim_i]);
    }
}

void BiasBackward(vector<dtype *> &grads, int count, int dim, dtype *bias_grad,
        vector<dtype *> input_grads) {
    int block_count = DefaultBlockCount(count * dim);
    NumberPointerArray grad_arr, input_grad_arr;
    grad_arr.init(grads.data(), grads.size());
    input_grad_arr.init(input_grads.data(), input_grads.size());
    KernelBiasBackward<<<block_count, TPB>>>(grad_arr.value, count, dim, bias_grad,
            (dtype **)input_grad_arr.value);
}

__global__ void KernelSum(dtype **v, int count, int row, int *cols, volatile dtype *block_sums,
        int *block_counters,
        bool cal_mean,
        bool cal_sqrt,
        dtype *sum_vals) {
    __shared__ volatile extern dtype shared_sum[];
    __shared__ bool is_last_block;

    int col = cols[blockIdx.x];
    if (blockIdx.z >= col) {
        return;
    }

    if (threadIdx.x == 0 && blockIdx.y == 0) {
        block_counters[blockIdx.x * gridDim.z + blockIdx.z] = 0;
    }
    if (threadIdx.x == 0) {
        is_last_block = false;
    }

    int count_i = blockIdx.x;
    int row_i = blockIdx.y * blockDim.x + threadIdx.x;
    int offset = row * blockIdx.z + row_i;
    shared_sum[threadIdx.x] = row_i < row ? v[count_i][offset] : 0.0f;
    __syncthreads();

    for (int i = (blockDim.x >> 1); i > 0; i >>= 1) {
        if (threadIdx.x < i) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + i];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        int block_sums_offset = blockIdx.x * gridDim.y * gridDim.z + gridDim.y * blockIdx.z +
            blockIdx.y;
        block_sums[block_sums_offset] = shared_sum[0];
        if (atomicAdd(block_counters + blockIdx.x * gridDim.z + blockIdx.z, 1) == gridDim.y - 1) {
            is_last_block = true;
        }
    }
    __syncthreads();

    if (is_last_block) {
        dtype sum = 0.0f;
        for (int i = threadIdx.x; i < gridDim.y; i += blockDim.x) {
            int offset = blockIdx.x * gridDim.y * gridDim.z + gridDim.y * blockIdx.z + i;
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
            dtype x = cal_mean ? shared_sum[0] / row :  shared_sum[0];
            x = cal_sqrt ? cuda_sqrt(x) : x;
            sum_vals[count_i * gridDim.z + blockIdx.z] = x;
        }
    }
}

void Sum(dtype **v, int count, int row, int *cols, int max_col, dtype *sum_vals,
        bool cal_mean = false,
        bool cal_sqrt = false) {
    int thread_count = min(NextTwoIntegerPowerNumber(row), TPB);
    int block_y_count = (row - 1 + thread_count) / thread_count;
    dim3 block_dim(count, block_y_count, max_col);

    NumberArray block_sums;
    block_sums.init(block_y_count * count * max_col);
    IntArray block_counters;
    block_counters.init(count * max_col);

    KernelSum<<<block_dim, thread_count, thread_count * sizeof(dtype)>>>(v, count, row, cols,
            block_sums.value, block_counters.value, cal_mean, cal_sqrt, sum_vals);
    CheckCudaError();
}

__global__ void KernelSquare(dtype **in_vals, dtype *subtrahends, int count, int row, int *cols,
        int max_col,
        dtype **vals) {
    int i = DeviceDefaultIndex();
    int n = max_col * row;
    if (i < count * n) {
        int count_i = i / n;
        int offset = i % n;
        int col_i = offset / row;
        int col = cols[count_i];
        if (col_i < col) {
            dtype x = in_vals[count_i][offset] - subtrahends[count_i * max_col + col_i];
            vals[count_i][offset] = x * x;
        }
    }
}

__global__ void KernelDiv(dtype **in_vals, dtype *subtrahends, dtype *denominators, int count,
        int row,
        int *cols,
        int max_col,
        dtype **vals) {
    int i = DeviceDefaultIndex();
    int n = max_col * row;
    if (i < count * n) {
        int count_i = i / n;
        int offset = i % n;
        int col_i = offset / row;
        int col = cols[count_i];
        if (col_i < col) {
            vals[count_i][offset] = (in_vals[count_i][offset] -
                    subtrahends[count_i * max_col + col_i]) /
                denominators[count_i * max_col + col_i];
        }
    }
}

void StandardLayerNormForward(dtype **in_vals, int count, int row, int *cols, int max_col,
        dtype **vals,
        dtype *sds) {
    NumberArray mean_arr;
    mean_arr.init(count * max_col);
    Sum(in_vals, count, row, cols, max_col, mean_arr.value, true);
    int block_count = DefaultBlockCountWithoutLimit(count * row * max_col);
    KernelSquare<<<block_count, TPB>>>(in_vals, mean_arr.value, count, row, cols, max_col, vals);
    CheckCudaError();
    Sum(vals, count, row, cols, max_col, sds, true, true);
    KernelDiv<<<block_count, TPB>>>(in_vals, mean_arr.value, sds, count, row, cols, max_col, vals);
    CheckCudaError();
}

__global__ void KernelPointwiseMul(dtype **a, dtype **b, int count, int *dims, int max_dim,
        int *offsets,
        dtype *vals) {
    int i = DeviceDefaultIndex();
    if (i < count * max_dim) {
        int count_i = i / max_dim;
        int dim_i = i % max_dim;
        int dim = dims[count_i];
        if (dim_i < dim) {
            int offset = offsets[count_i];
            vals[offset + dim_i] = a[count_i][dim_i] * b[count_i][dim_i];
        }
    }
}

__global__ void KernelSum(dtype *v, int count, int row, int *cols, int *col_offsets,
        volatile dtype *block_sums,
        int *block_counters,
        dtype *sum_vals) {
    __shared__ volatile extern dtype shared_sum[];
    __shared__ volatile bool is_last_block;

    int col = cols[blockIdx.x];
    if (blockIdx.z >= col) {
        return;
    }

    if (threadIdx.x == 0 && blockIdx.y == 0) {
        block_counters[blockIdx.x * gridDim.z + blockIdx.z] = 0;
    }
    if (threadIdx.x == 0) {
        is_last_block = false;
    }

    int count_i = blockIdx.x;
    int row_i = blockIdx.y * blockDim.x + threadIdx.x;
    int offset = row * blockIdx.z + row_i;
    int col_offset = col_offsets[count_i];
    shared_sum[threadIdx.x] = row_i < row ? v[col_offset * row + offset] : 0.0f;
    __syncthreads();

    for (int i = (blockDim.x >> 1); i > 0; i >>= 1) {
        if (threadIdx.x < i) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + i];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        int block_sums_offset = blockIdx.x * gridDim.y * gridDim.z + gridDim.y * blockIdx.z +
            blockIdx.y;
        block_sums[block_sums_offset] = shared_sum[0];
        if (atomicAdd(block_counters + blockIdx.x * gridDim.z + blockIdx.z, 1) == gridDim.y - 1) {
            is_last_block = true;
        }
    }
    __syncthreads();

    if (is_last_block) {
        dtype sum = 0.0f;
        for (int i = threadIdx.x; i < gridDim.y; i += blockDim.x) {
            int offset = blockIdx.x * gridDim.y * gridDim.z + gridDim.y * blockIdx.z + i;
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
            dtype x = shared_sum[0];
            sum_vals[count_i * gridDim.z + blockIdx.z] = x;
        }
    }
}

__global__ void KernelStandardLayerNormBackward(dtype **grads, dtype **vals, int count, int row,
        int *cols,
        int max_col,
        int *col_offsets,
        dtype *sds,
        dtype *m,
        dtype *m_sum,
        dtype *grad_sum,
        dtype **in_grads) {
    int i = DeviceDefaultIndex();
    int n = max_col * row;
    if (i < count * n) {
        int count_i = i / n;
        int dim_i = i % n;
        int col = cols[count_i];
        int col_i = dim_i / row;
        if (col_i < col) {
            dtype y = vals[count_i][dim_i];
            int col_offset = col_offsets[count_i];
            dtype x = 1.0 / (row * sds[count_i * max_col + col_i]) * ((row - 1 - y * y) *
                    grads[count_i][dim_i] - ((m_sum[count_i * max_col + col_i] -
                        m[col_offset * row + dim_i]) * vals[count_i][dim_i] +
                        grad_sum[count_i * max_col + col_i] - grads[count_i][dim_i]));
            DeviceAtomicAdd(in_grads[count_i] + dim_i, x);
        }
    }
}

void StandardLayerNormBackward(dtype **grads, int count, int row, int *cols, int col_sum,
        int max_col,
        int *col_offsets,
        int *dims,
        int *dim_offsets,
        dtype **vals,
        dtype *sds,
        dtype **in_grads) {
    NumberArray m, m_sum, grad_sum;
    m.init(col_sum * row);
    m_sum.init(count * max_col);
    grad_sum.init(count * max_col);
    int block_count = DefaultBlockCountWithoutLimit(count * max_col * row);
    int max_dim = max_col * row;
    KernelPointwiseMul<<<block_count, TPB>>>(grads, vals, count, dims, max_dim, dim_offsets,
            m.value);
    CheckCudaError();

    int thread_count = min(NextTwoIntegerPowerNumber(row), TPB);
    int block_y_count = (row - 1 + thread_count) / thread_count;
    dim3 block_dim(count, block_y_count, max_col);

    NumberArray block_sums;
    block_sums.init(block_y_count * count * max_col);
    IntArray block_counters;
    block_counters.init(count * max_col);

    KernelSum<<<block_dim, thread_count, thread_count * sizeof(dtype)>>>(m.value, count, row, cols,
            col_offsets, block_sums.value, block_counters.value, m_sum.value);
    CheckCudaError();

    KernelSum<<<block_dim, thread_count, thread_count * sizeof(dtype)>>>(grads, count, row, cols,
            block_sums.value, block_counters.value, false, false, grad_sum.value);
    CheckCudaError();

    KernelStandardLayerNormBackward<<<block_count, TPB>>>(grads, vals, count, row, cols, max_col,
            col_offsets, sds, m.value, m_sum.value, grad_sum.value, in_grads);
    CheckCudaError();
}

__global__ void KernelPointwiseLinearForward(dtype **in_vals, int count, int row, int *cols,
        int max_col,
        dtype *g,
        dtype *b,
        dtype **vals) {
    int i = DeviceDefaultIndex();
    int n = max_col * row;
    if (i < count * n) {
        int count_i = i / n;
        int dim_i = i % n;
        int col_i = dim_i / row;
        int col = cols[count_i];
        if (col_i < col) {
            int row_i = dim_i % row;
            vals[count_i][dim_i] = g[row_i] * in_vals[count_i][dim_i] + b[row_i];
        }
    }
}

void PointwiseLinearForward(dtype **in_vals, int count, int row, int *cols, int max_col, dtype *g,
        dtype *b,
        dtype **vals) {
    int block_count = DefaultBlockCountWithoutLimit(count * row * max_col);
    KernelPointwiseLinearForward<<<block_count, TPB>>>(in_vals, count, row, cols, max_col, g, b,
            vals);
    CheckCudaError();
}
__global__ void KernelPointwiseLinearBackwardForInput(dtype **grads, dtype *g_vals, int count,
        int row,
        int *cols,
        int max_col,
        dtype **in_grads) {
    int i = DeviceDefaultIndex();
    int n = max_col * row;
    if (i < count * n) {
        int count_i = i / n;
        int dim_i = i % n;
        int col_i = dim_i / row;
        int col = cols[count_i];
        if (col_i < col) {
            int row_i = dim_i % row;
            DeviceAtomicAdd(in_grads[count_i] + dim_i, grads[count_i][dim_i] * g_vals[row_i]);
        }
    }
}

void PointwiseLinearBackward(dtype **grads, dtype **in_vals, dtype *g_vals, int count, int row,
        int *cols,
        int max_col,
        int col_sum,
        int *dims,
        int *dim_offsets,
        dtype **in_grads,
        dtype *g_grads,
        dtype *bias_grads) {
    int block_count = DefaultBlockCountWithoutLimit(count * max_col * row);
    int max_dim = max_col * row;
    NumberArray a;
    a.init(col_sum * row);
    KernelPointwiseMul<<<block_count, TPB>>>(grads, in_vals, count, dims, max_dim, dim_offsets,
            a.value);

    int thread_count = min(NextTwoIntegerPowerNumber(col_sum), TPB);
    int block_y_count = (col_sum - 1 + thread_count) / thread_count;
    dim3 block_dim(row, block_y_count, 1);
    NumberArray block_sums;
    block_sums.init(block_y_count * row);
    IntArray block_counters;
    block_counters.init(row);

    KernelLinearBackwardForBias<<<block_dim, thread_count, thread_count * sizeof(dtype)>>>(
            a.value, col_sum, row, g_grads, block_counters.value, block_sums.value);
    CheckCudaError();

    KernelConcat<<<block_count, TPB>>>(grads, count, dims, max_dim, dim_offsets, a.value);

    KernelLinearBackwardForBias<<<block_dim, thread_count, thread_count * sizeof(dtype)>>>(
            a.value, col_sum, row, bias_grads, block_counters.value, block_sums.value);
    CheckCudaError();

    KernelPointwiseLinearBackwardForInput<<<block_count, TPB>>>(grads, g_vals, count, row, cols,
            max_col, in_grads);
    CheckCudaError();
}

__global__ void KernelBroadcastForward(dtype **in_vals, int count, int in_dim, int *ns, int max_n,
        dtype **vals) {
    int i = DeviceDefaultIndex();
    int m = max_n * in_dim;
    int count_i = i / m;
    if (count_i < count) {
        int x = i % m;
        int n = ns[count_i];
        int in_dim_i = x / n;
        if (in_dim_i < in_dim) {
            int n_i = x % n;
            vals[count_i][n_i * in_dim + in_dim_i] = in_vals[count_i][in_dim_i];
        }
    }
}

void BroadcastForward(dtype **in_vals, int count, int in_dim, int *ns, int max_n, dtype **vals) {
    int block_count = DefaultBlockCountWithoutLimit(count * max_n * in_dim);
    KernelBroadcastForward<<<block_count, TPB>>>(in_vals, count, in_dim, ns, max_n, vals);
    CheckCudaError();
}

__global__ void KernelBroadcastBackward(dtype **grads, int count, int in_dim, int *ns, int max_n,
        dtype **in_grads) {
    __shared__ volatile extern dtype accumulated_shared_arr[];
    int count_i = blockIdx.y;
    int n = ns[count_i];
    int n_i = threadIdx.x;
    int dim_i = blockIdx.x;
    if (n_i < n) {
        accumulated_shared_arr[threadIdx.x] = grads[count_i][in_dim * n_i + dim_i];
    } else {
        accumulated_shared_arr[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    for (int i = (blockDim.x >> 1); i > 0; i >>= 1) {
        if (threadIdx.x < i) {
            int j = threadIdx.x + i;
            accumulated_shared_arr[threadIdx.x] += accumulated_shared_arr[j];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        dtype *in_grad = in_grads[count_i];
        DeviceAtomicAdd(in_grad + dim_i, accumulated_shared_arr[0]);
    }
}

void BroadcastBackward(dtype **grads, int count, int in_dim, int *ns, int max_n,
    dtype **in_grads) {
    int thread_count = 8;
    while (max_n > thread_count) {
        thread_count <<= 1;
    }
    dim3 block_dim(in_dim, count, 1);
    KernelBroadcastBackward<<<block_dim, thread_count, thread_count * sizeof(dtype)>>>(grads,
            count, in_dim, ns, max_n, in_grads);
    CheckCudaError();
}

__global__ void KernelSquareSum(dtype *v, int len, volatile dtype *global_sum,
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

dtype SquareSum(dtype *v, int len) {
    int block_count = DefaultBlockCount(len);
    NumberArray global_sum;
    global_sum.init(block_count);
    DeviceInt block_counter;
    block_counter.init();
    DeviceNumber result;
    result.init();
    KernelSquareSum<<<block_count, TPB>>>(v, len,
            global_sum.value, block_counter.value, result.value);
    CheckCudaError();
    result.copyFromDeviceToHost();
    return result.v;
}

__global__ void KernelSquareSum(dtype *v, bool *indexers,
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

dtype SquareSum(dtype *v, bool *indexers, int count, int dim) {
    int block_count = DefaultBlockCountWithoutLimit(count * dim);
    NumberArray global_sum;
    global_sum.init(block_count);
    DeviceInt block_counter;
    block_counter.init();
    DeviceNumber result;
    result.init();
    KernelSquareSum<<<block_count, TPB>>>(v, indexers,
            count, dim, global_sum.value, block_counter.value, result.value);
    CheckCudaError();
    result.copyFromDeviceToHost();
    return result.v;
}

__global__ void KernelRescale(dtype *v, int len, dtype scale) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < len; i += step) {
        v[i] *= scale;
    }
}

void Rescale(dtype *v, int len, dtype scale) {
    int block_count = DefaultBlockCount(len);
    KernelRescale<<<block_count, TPB>>>(v, len, scale);
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
        dtype eps) {
    int block_count = DefaultBlockCount(row * col);
    dtype x = 1.0f / (1 - pow(belta1, iter + 1));
    KernelUpdateAdam<<<block_count, TPB>>>(val, grad, row, col, is_bias, aux_mean,
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

void UpdateAdamW(dtype *val, dtype *grad, int row, int col, bool is_bias, dtype *aux_mean,
        dtype *aux_square,
        int iter,
        dtype belta1,
        dtype belta2,
        dtype alpha,
        dtype reg,
        dtype eps) {
    int block_count = DefaultBlockCount(row * col);
    dtype x = 1.0f / (1 - pow(belta1, iter + 1));
    KernelUpdateAdamW<<<block_count, TPB>>>(val, grad, row, col, is_bias, aux_mean,
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
        bool *indexers,
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

__global__ void KernelSelfPlusIters(bool *indexers, int *iters,
        int count) {
    int index = DeviceDefaultIndex();
    int step = DeviceDefaultStep();
    for (int i = index; i < count; i += step) {
        if (indexers[i]) {
            ++iters[i];
        }
    }
}

void UpdateAdam(dtype *val, dtype *grad, int row, int col, dtype *aux_mean,
        dtype *aux_square,
        bool *indexers,
        int *iters,
        dtype belta1,
        dtype belta2,
        dtype alpha,
        dtype reg,
        dtype eps) {
    int block_count = DefaultBlockCount(row * col);
    KernelUpdateAdam<<<block_count, TPB>>>(val, grad, row, col, aux_mean,
            aux_square, indexers, iters, belta1, belta2, alpha, reg, eps);
    CheckCudaError();
    block_count = DefaultBlockCount(col);
    KernelSelfPlusIters<<<block_count, TPB>>>(indexers, iters, col);
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
        dtype eps) {
    int block_count = DefaultBlockCount(row * col);
    KernelUpdateAdagrad<<<block_count, TPB>>>(val, grad, row, col, aux_square,
            alpha, reg, eps);
    CheckCudaError();
}

__global__ void KernelUpdateAdagrad(dtype *val, dtype *grad, int row, int col,
        dtype *aux_square,
        bool *indexers,
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
        bool *indexers,
        dtype alpha,
        dtype reg,
        dtype eps) {
    int block_count = DefaultBlockCount(row * col);
    KernelUpdateAdagrad<<<block_count, TPB>>>(val, grad, row, col, aux_square,
            indexers, alpha, reg, eps);
    CheckCudaError();
}

void *GraphHostAlloc() {
    void *m;
    CallCuda(cudaHostAlloc(&m, 10000000, cudaHostAllocWriteCombined));
    if (m == nullptr) {
        abort();
    }
    return m;
}

}
}
