#ifndef N3LDG_INCLUDE_SERIALIZABLE_H
#define N3LDG_INCLUDE_SERIALIZABLE_H

#include <iostream>
#include <json/json.h>
#include <boost/format.hpp>

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

#endif
