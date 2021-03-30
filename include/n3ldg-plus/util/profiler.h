#ifndef N3LDG_CUDA_PROFILER_H
#define N3LDG_CUDA_PROFILER_H

#include <string>
#include <map>
#include <chrono>
#include <utility>
#include <iostream>
#include <stack>
#include <vector>
#include <algorithm>

namespace n3ldg_cuda {

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
    static Profiler &Ins() {
        Profiler *&p = ProfilerPtr();
        if (p == nullptr) {
            p = new Profiler;
        }
        return *p;
    }

    static void Reset() {
        Profiler *&p = ProfilerPtr();
        if (p != nullptr) {
            delete p;
            p = nullptr;
        }
        Ins();
    }

    void BeginEvent(const std::string &name) {
        if (!enabled_) return;
        Elapsed elapsed;
        elapsed.name = name;
        running_events_.push(elapsed);
        running_events_.top().begin =
            std::chrono::high_resolution_clock::now();
    }

    void EndEvent() {
        if (!enabled_) return;
        if (running_events_.empty()) {
            std::cout << "running_events_ empty" << std::endl;
            abort();
        }
        Elapsed &top = running_events_.top();
        top.end = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                top.end - top.begin).count();
        auto it = event_map_.find(top.name);
        if (it == event_map_.end()) {
            Event event(top.name, 1, time);
            event_map_.insert(std::make_pair(event.name, event));
        } else {
            Event &event = it->second;
            event.count++;
            event.total_time_in_nanoseconds += time;
        }
        if (running_events_.size() == 1) {
            root_ = &event_map_.at(top.name);
        }
        running_events_.pop();
    }

#if USE_GPU
    void EndCudaEvent();
#else
    void EndCudaEvent() {
        EndEvent();
    }
#endif

    void Print() {
        while (!running_events_.empty()) {
            std::cout << running_events_.top().name << std::endl;
            running_events_.pop();
        }
        if (!running_events_.empty()) {
            std::cerr << "events empty" << std::endl;
            abort();
        }
        std::vector<Event> events;
        for (auto &it : event_map_) {
            Event &event = it.second;
            events.push_back(event);
        }

        std::sort(events.begin(), events.end(), [](const Event &a,
                    const Event &b) {return a.total_time_in_nanoseconds >
                b.total_time_in_nanoseconds;});
        std::cout << "events count" << events.size() << std::endl;

        for (Event &event : events) {
            std::cout << "name:" << event.name << " count:" << event.count <<
                " total time:" << event.total_time_in_nanoseconds / 1000000000.0
                << " avg:" << event.total_time_in_nanoseconds / event.count /
                1000000 << " ratio:" << event.total_time_in_nanoseconds /
                root_->total_time_in_nanoseconds << std::endl;
        }
    }

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
