#include <utility>
#include <iostream>
#include <vector>
#include <algorithm>
#include "fmt/core.h"
#include "profiler.h"
#if USE_GPU
#include <cuda_runtime.h>
#endif

namespace insnet {

Profiler &Profiler::Ins() {
    Profiler *&p = ProfilerPtr();
    if (p == nullptr) {
        p = new Profiler;
    }
    return *p;
}

void Profiler::Reset() {
    Profiler *&p = ProfilerPtr();
    if (p != nullptr) {
        delete p;
        p = nullptr;
    }
    Ins();
}

void Profiler::BeginEvent(const std::string &name) {
    if (!enabled_) return;
    Elapsed elapsed;
    elapsed.name = name;
    running_events_.push(elapsed);
    running_events_.top().begin = std::chrono::high_resolution_clock::now();
}

void Profiler::EndEvent() {
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

void Profiler::EndCudaEvent() {
    if (!enabled_) return;
#if USE_GPU
    cudaDeviceSynchronize();
#endif
    EndEvent();
}

void Profiler::Print() {
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

}
