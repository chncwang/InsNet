#ifndef N3LDG_PLUS_PROFILER_H
#define N3LDG_PLUS_PROFILER_H

#include <string>
#include <map>
#include <chrono>
#include <stack>

namespace n3ldg_plus {

struct Event {
    std::string name;
    int count;
    float total_time_in_nanoseconds;

    Event(const std::string &name, int count, float total_time_in_nanoseconds) {
        this->name = name;
        this->count = count;
        this->total_time_in_nanoseconds = total_time_in_nanoseconds;
    }

    Event() = default;
    Event(const Event &event) = default;
};

struct Elapsed {
    typedef std::chrono::time_point<std::chrono::high_resolution_clock> Timestamp;
    Timestamp begin;
    Timestamp end;
    std::string name;
};

enum ProfilerMode {
    ANALYSIS = 0,
    METRIC = 1
};

class Profiler;

static Profiler*& ProfilerPtr() {
    static Profiler* p;
    return p;
}

class Profiler {
public:
    static Profiler &Ins();

    static void Reset();

    void BeginEvent(const std::string &name);

    void EndEvent();

    void EndCudaEvent();

    void Print();

    void SetEnabled(bool enabled) {
        enabled_ = enabled;
    }

private:
    Profiler() = default;
    std::map<std::string, Event> event_map_;
    std::stack<Elapsed> running_events_;
    Event *root_ = nullptr;
    bool enabled_ = false;
};

}

#endif
