#ifndef BasicTensor_DEF
#define BasicTensor_DEF

#include <vector>
#include <cmath>
#include <fstream>
#include <map>
#include <memory>
#include "Def.h"
#include <iostream>
#include <iostream>
#include "cereal/cereal.hpp"
#include "cereal/archives/binary.hpp"
#include "cereal/types/unordered_map.hpp"
#include "cereal/types/string.hpp"
#include "cereal/types/vector.hpp"
#include "json/json.h"
#include <boost/format.hpp>

using namespace Eigen;

class N3LDGSerializable {
public:
    virtual Json::Value toJson() const = 0;
    virtual void fromJson(const Json::Value &) = 0;

    std::string toString() const {
        Json::StreamWriterBuilder builder;
        builder["commentStyle"] = "None";
        builder["indentation"] = "";
        return Json::writeString(builder, toJson());
    }

    void fromString(const std::string &str) {
        Json::CharReaderBuilder builder;
        auto reader = std::unique_ptr<Json::CharReader>(builder.newCharReader());
        Json::Value root;
        std::string error;
        if (!reader->parse(str.c_str(), str.c_str() + str.size(), &root, &error)) {
            std::cerr << boost::format("parse json error:%1%") % error << std::endl;
            abort();
        }
    }
};

namespace n3ldg_cpu {

struct Tensor1D : public N3LDGSerializable {
    dtype *v = nullptr;
    int dim = 0;

    Tensor1D() = default;

    virtual ~Tensor1D();

    virtual void init(int ndim);

    void zero();

    std::string toString() const;

    const Mat mat() const;

    Mat mat();

    const Mat tmat() const;

    Mat tmat();

    const Vec vec() const;

    Vec vec();

    dtype& operator[](const int i);

    const dtype& operator[](const int i) const;

    Tensor1D& operator=(const dtype &a);

    Tensor1D& operator=(const std::vector<dtype> &a);

    Tensor1D& operator=(const Tensor1D &a);

    virtual void random(dtype bound);

    virtual Json::Value toJson() const;

    virtual void fromJson(const Json::Value &json);

    template<typename Archive>
    void serialize(Archive &ar);

    virtual void print() const;

    virtual std::vector<dtype> toCpu() const;

    virtual void checkIsNumber() const;
};

struct Tensor2D : public N3LDGSerializable {
    dtype *v;
    int col, row, size;

    Tensor2D();

    virtual ~Tensor2D();

    virtual void init(int nrow, int ncol);

    virtual void print() const;

    std::string toString() const;

    void zero();

    const Mat mat() const;

    Mat mat();

    const Vec vec() const;

    Vec vec();


    dtype* operator[](const int icol);

    const dtype* operator[](const int icol) const;

    virtual void assignAll(dtype a);

    Tensor2D& operator=(const std::vector<dtype> &a);

    Tensor2D& operator=(const std::vector<std::vector<dtype> > &a);

    Tensor2D& operator=(const Tensor2D &a);

    virtual void random(dtype bound);

    virtual void randomNorm(dtype sd);

    void norm2one(dtype norm = 1.0);

    virtual Json::Value toJson() const;

    virtual void fromJson(const Json::Value &json);

    template<typename Archive>
    void serialize(Archive &ar);
};

}

#if USE_GPU

namespace n3ldg_cuda {

class Transferable {
public:
    virtual void copyFromHostToDevice() = 0;
    virtual void copyFromDeviceToHost() = 0;
};

bool Verify(dtype *host, dtype* device, int len, const char* message);

struct Tensor1D : public n3ldg_cpu::Tensor1D, public Transferable {
    dtype *value = NULL;

    Tensor1D();
    Tensor1D(const Tensor1D &);
    Tensor1D(Tensor1D &&);
    void init(int len) override;
    void initOnMemoryAndDevice(int len);
    void initOnMemory(int len);
    ~Tensor1D();

    virtual std::string name() const;

    Tensor1D& operator=(const Tensor1D &tensor);

    Tensor1D& operator=(dtype v);

    void print() const override;

    virtual std::vector<dtype> toCpu() const override;

    void random(dtype bound) override;

    bool verify(const char *message) const;

    void copyFromHostToDevice() override;
    void copyFromDeviceToHost() override;
    void checkIsNumber() const override;

private:
    void initOnDevice(int len);
};

struct Tensor2D : public n3ldg_cpu::Tensor2D, public Transferable {
    dtype *value = NULL;

    Tensor2D();
    Tensor2D(const Tensor2D &);
    Tensor2D(Tensor2D &&);
    ~Tensor2D();

    void init(int row, int col) override;

    virtual std::string name() const;

    void print() const override;

    void zero();

    void random(dtype bound) override;

    void randomNorm(dtype sd) override;

    bool verify(const char* message);

    void assignAll(dtype a) override;

    void initOnMemoryAndDevice(int row, int col);

    void copyFromHostToDevice() override;
    void copyFromDeviceToHost() override;
private:
    void initOnDevice(int row, int col);
};


}
#endif

#endif
