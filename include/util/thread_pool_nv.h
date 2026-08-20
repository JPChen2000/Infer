/* Copyright (c) 2022 NVIDIA CORPORATION AND AFFILIATES.  All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted
provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this list of
      conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of
      conditions and the following disclaimer in the documentation and/or other materials
      provided with the distribution.
    * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
      to endorse or promote products derived from this software without specific prior written
      permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. */
#ifndef FEATHER_UTIL_THREAD_POOL_NV_H
#define FEATHER_UTIL_THREAD_POOL_NV_H

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

// copy from Nvidia

class ThreadPoolNv {
   private:
    class Batch {
       public:
        template <class F>
        Batch(size_t task_count, size_t target_workers, F&& function)
            : function_(std::forward<F>(function)),
              next_task_index_(0),
              task_count_(task_count),
              target_workers_(target_workers),
              remaining_(task_count) {}

        void RunTasks() {
            for (;;) {
                const size_t task_index = next_task_index_.fetch_add(1, std::memory_order_relaxed);
                if (task_index >= task_count_) {
                    return;
                }
                RunTask(task_index);
            }
        }

        bool TryRegisterWorker() {
            if (registered_workers_ >= target_workers_) {
                return false;
            }
            ++registered_workers_;
            return true;
        }

        void WaitForTasks() {
            std::unique_lock<std::mutex> lock(mutex_);
            completed_.wait(lock, [this] { return remaining_.load(std::memory_order_acquire) == 0; });
        }

        void RethrowIfFailed() {
            std::lock_guard<std::mutex> lock(mutex_);
            if (exception_ != nullptr) {
                std::rethrow_exception(exception_);
            }
        }

       private:
        void RunTask(size_t task_index) {
            try {
                function_(static_cast<int>(task_index));
            } catch (...) {
                std::lock_guard<std::mutex> lock(mutex_);
                if (exception_ == nullptr) {
                    exception_ = std::current_exception();
                }
            }

            if (remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                std::lock_guard<std::mutex> lock(mutex_);
                completed_.notify_one();
            }
        }

        std::function<void(int)> function_;
        std::atomic<size_t> next_task_index_;
        size_t task_count_;
        size_t target_workers_;
        size_t registered_workers_{0};
        std::atomic<size_t> remaining_;
        std::mutex mutex_;
        std::condition_variable completed_;
        std::exception_ptr exception_;
    };

   public:
    ThreadPoolNv(size_t);
    template <class F, class... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<typename std::result_of<F(int, Args...)>::type> {
        // using return_type = typename std::result_of<F(Args...)>::type;

        auto task = std::make_shared<std::packaged_task<int(int)>>(
            std::bind(std::forward<F>(f), std::placeholders::_1, std::forward<Args>(args)...));

        std::future<int> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex);

            // don't allow enqueueing after stopping the pool
            if (stop) throw std::runtime_error("enqueue on stopped ThreadPool");

            tasks.emplace([task](int tid) { (*task)(tid); });
        }
        condition.notify_one();
        return res;
    }

    template <class F>
    void RunBatch(size_t task_count, F&& function) {
        if (task_count == 0) {
            return;
        }

        std::unique_lock<std::mutex> batch_lock(batch_mutex);
        const size_t target_workers = std::min(task_count, workers.size());
        if (target_workers == 0) {
            for (size_t task_index = 0; task_index < task_count; ++task_index) {
                function(static_cast<int>(task_index));
            }
            return;
        }
        const auto batch = std::make_shared<Batch>(task_count, target_workers, std::forward<F>(function));
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (stop) throw std::runtime_error("enqueue on stopped ThreadPool");
            active_batch_ = batch;
        }
        for (size_t worker = 0; worker < target_workers; ++worker) {
            condition.notify_one();
        }
        batch->WaitForTasks();
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            active_batch_.reset();
            completed.notify_all();
        }
        batch->RethrowIfFailed();
    }

    void wait() {
        std::unique_lock<std::mutex> lock(this->queue_mutex);
        completed.wait(lock, [this] { return this->in_flight == 0 && this->tasks.empty() && active_batch_ == nullptr; });
    }
    ~ThreadPoolNv() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread& worker : workers) worker.join();
    }

   private:
    // need to keep track of threads so we can join them
    std::vector<std::thread> workers;
    // the task queue
    std::queue<std::function<void(int)>> tasks;

    // synchronization
    std::mutex queue_mutex;
    std::condition_variable condition;
    std::condition_variable completed;
    int in_flight;
    bool stop;
    std::mutex batch_mutex;
    std::shared_ptr<Batch> active_batch_;
};

// the constructor just launches some amount of workers
inline ThreadPoolNv::ThreadPoolNv(size_t threads) : workers(threads), in_flight(0), stop(false) {
    for (size_t i = 0; i < threads; ++i)
        workers[i] = std::thread([this, i] {
            for (;;) {
                std::function<void(int)> task;
                std::shared_ptr<Batch> batch;

                {
                    std::unique_lock<std::mutex> lock(this->queue_mutex);
                    this->condition.wait(lock, [this] {
                        return this->stop || !this->tasks.empty() || this->active_batch_ != nullptr;
                    });
                    if (this->stop && this->tasks.empty() && this->active_batch_ == nullptr) return;
                    if (this->active_batch_ != nullptr) {
                        batch = this->active_batch_;
                        if (!batch->TryRegisterWorker()) {
                            // Only the batch's target workers may execute it. Keep
                            // surplus workers asleep until the batch is retired;
                            // there is no ordinary task to run in this branch.
                            this->condition.wait(lock, [this, batch] {
                                return this->stop || this->active_batch_ != batch;
                            });
                            continue;
                        }
                    } else {
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                        in_flight++;
                    }
                }
                if (batch != nullptr) {
                    batch->RunTasks();
                    std::unique_lock<std::mutex> lock(this->queue_mutex);
                    this->condition.wait(lock, [this, batch] { return this->stop || this->active_batch_ != batch; });
                    continue;
                }
                task(i);
                std::unique_lock<std::mutex> lock(this->queue_mutex);
                in_flight--;
                if ((this->in_flight == 0) && this->tasks.empty()) completed.notify_one();
            }
        });
}
#endif
