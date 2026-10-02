#include "pokitlms/model/tensor_linear.hpp"

#include "pokitlms/ops/linear.hpp"
#include "pokitlms/ops/quantized_linear.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <exception>
#include <functional>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace pokitlms::model {
namespace {

class TensorRowExecutor {
public:
    TensorRowExecutor() {
        const auto hardware = std::thread::hardware_concurrency();
#if defined(__ANDROID__)
        const auto count = std::min(4U, std::max(2U, hardware > 1 ? hardware - 1 : 2U));
#else
        const auto count = std::min(6U, std::max(2U, hardware > 1 ? hardware - 1 : 2U));
#endif
        workers_.reserve(count);
        try {
            for (unsigned i = 0; i < count; ++i) workers_.emplace_back([this] { run(); });
        } catch (...) {
            {
                std::lock_guard lock(mutex_);
                stopping_ = true;
            }
            ready_.notify_all();
            for (auto& worker : workers_) if (worker.joinable()) worker.join();
            throw;
        }
    }

    ~TensorRowExecutor() {
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
        }
        ready_.notify_all();
        for (auto& worker : workers_) worker.join();
    }

    TensorRowExecutor(const TensorRowExecutor&) = delete;
    TensorRowExecutor& operator=(const TensorRowExecutor&) = delete;

    void parallel_for(std::size_t count, std::size_t grain,
                      std::function<void(std::size_t, std::size_t)> work) {
        std::lock_guard serialize(serial_);
        {
            std::lock_guard lock(mutex_);
            work_ = std::move(work);
            total_ = count;
            grain_ = grain;
            next_.store(0, std::memory_order_relaxed);
            workers_remaining_ = workers_.size();
            error_ = nullptr;
            ++generation_;
        }
        ready_.notify_all();
        std::unique_lock lock(mutex_);
        finished_.wait(lock, [this] { return workers_remaining_ == 0; });
        work_ = {};
        if (error_) std::rethrow_exception(error_);
    }

private:
    void run() {
        std::size_t observed_generation = 0;
        for (;;) {
            std::function<void(std::size_t, std::size_t)> work;
            std::size_t total = 0;
            std::size_t grain = 0;
            {
                std::unique_lock lock(mutex_);
                ready_.wait(lock, [this, observed_generation] {
                    return stopping_ || generation_ != observed_generation;
                });
                if (stopping_) return;
                observed_generation = generation_;
                total = total_;
                grain = grain_;
                work = work_;
            }
            try {
                for (;;) {
                    const auto first = next_.fetch_add(grain, std::memory_order_relaxed);
                    if (first >= total) break;
                    work(first, std::min(grain, total - first));
                }
            } catch (...) {
                std::lock_guard lock(mutex_);
                if (!error_) error_ = std::current_exception();
            }
            {
                std::lock_guard lock(mutex_);
                if (--workers_remaining_ == 0) finished_.notify_one();
            }
        }
    }

    std::mutex serial_;
    std::mutex mutex_;
    std::condition_variable ready_;
    std::condition_variable finished_;
    std::vector<std::thread> workers_;
    std::function<void(std::size_t, std::size_t)> work_;
    std::atomic<std::size_t> next_{};
    std::size_t total_{};
    std::size_t grain_{};
    std::size_t workers_remaining_{};
    std::size_t generation_{};
    std::exception_ptr error_;
    bool stopping_{};
};

TensorRowExecutor& row_executor() {
    static TensorRowExecutor executor;
    return executor;
}

void linear_rows(const TensorReader& weights, std::span<const float> input,
                 std::span<float> output, std::size_t first_row,
                 std::size_t row_batch, bool floating, TensorLinearScratch& buffers) {
    const auto& tensor = weights.tensor();
    const auto feature_count = input.size();
    if (row_batch == 0) {
        constexpr std::size_t target_working_set_bytes = 256U * 1024U;
        std::size_t working_set_row_bytes = weights.row_bytes();
        if (floating) {
            if (feature_count > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
                throw std::length_error("GGUF tensor row size overflows");
            }
            const auto converted_bytes = feature_count * sizeof(float);
            if (working_set_row_bytes > std::numeric_limits<std::size_t>::max() - converted_bytes) {
                throw std::length_error("GGUF tensor row size overflows");
            }
            working_set_row_bytes += converted_bytes;
        }
        if (working_set_row_bytes == 0) throw std::length_error("GGUF tensor row size overflows");
        row_batch = std::max<std::size_t>(1, target_working_set_bytes / working_set_row_bytes);
    }
    const auto chunk_limit = std::min(row_batch, output.size());
    for (std::size_t processed = 0; processed < output.size();) {
        const auto rows = std::min(chunk_limit, output.size() - processed);
        auto destination = output.subspan(processed, rows);
        if (weights.row_bytes() > std::numeric_limits<std::size_t>::max() / rows) {
            throw std::length_error("GGUF matrix batch byte size overflows");
        }
        const auto encoded_bytes = weights.row_bytes() * rows;
        if (feature_count > std::numeric_limits<std::size_t>::max() / rows) {
            throw std::length_error("GGUF matrix batch dimensions overflow");
        }
        if (floating) {
            const auto value_count = feature_count * rows;
            buffers.floating.resize(value_count);
            if (tensor.type != 0) buffers.encoded.resize(encoded_bytes);
            weights.read_float_rows_into(first_row + processed, rows,
                std::span<float>(buffers.floating.data(), value_count),
                tensor.type == 0 ? std::span<std::byte>{}
                                 : std::span<std::byte>(buffers.encoded.data(), encoded_bytes));
            linear(input, std::span<const float>(buffers.floating.data(), value_count), {}, destination);
        } else {
            buffers.encoded.resize(encoded_bytes);
            weights.read_rows_into(first_row + processed, rows,
                std::span<std::byte>(buffers.encoded.data(), encoded_bytes));
            const std::span<const std::byte> encoded(buffers.encoded.data(), encoded_bytes);
            switch (tensor.type) {
                case 2: linear_q4_0(input, encoded, rows, {}, destination); break;
                case 3: linear_q4_1(input, encoded, rows, {}, destination); break;
                case 6: linear_q5_0(input, encoded, rows, {}, destination); break;
                case 7: linear_q5_1(input, encoded, rows, {}, destination); break;
                case 8: linear_q8_0(input, encoded, rows, {}, destination); break;
                case 10: linear_q2_k(input, encoded, rows, {}, destination); break;
                case 11: linear_q3_k(input, encoded, rows, {}, destination); break;
                case 12: linear_q4_k(input, encoded, rows, {}, destination); break;
                case 13: linear_q5_k(input, encoded, rows, {}, destination); break;
                case 14: linear_q6_k(input, encoded, rows, {}, destination); break;
                default: throw std::logic_error("unreachable GGML type dispatch");
            }
        }
        processed += rows;
    }
}

}  // namespace

void tensor_linear(const TensorReader& weights, std::span<const float> input,
                   std::span<float> output, std::size_t row_batch,
                   TensorLinearScratch* scratch) {
    const auto& tensor = weights.tensor();
    if (tensor.dimensions.empty() || input.empty() || output.empty() ||
        tensor.dimensions.front() != input.size() || weights.row_count() != output.size()) {
        throw std::invalid_argument("GGUF matrix dimensions do not match linear input/output");
    }

    const bool floating = tensor.type == 0 || tensor.type == 1 || tensor.type == 30;
    const bool quantized = tensor.type == 2 || tensor.type == 3 || tensor.type == 6 ||
        tensor.type == 7 || tensor.type == 8 || tensor.type == 10 ||
        tensor.type == 11 || tensor.type == 12 || tensor.type == 13 || tensor.type == 14;
    if (!floating && !quantized) {
        throw std::invalid_argument("unsupported GGUF matrix type for tensor_linear");
    }

    if (output.size() >= 256) {
        row_executor().parallel_for(output.size(), 128,
            [&weights, input, output, row_batch, floating](std::size_t first, std::size_t rows) {
                thread_local TensorLinearScratch worker_scratch;
                linear_rows(weights, input, output.subspan(first, rows), first,
                            row_batch, floating, worker_scratch);
            });
        return;
    }
    TensorLinearScratch local_scratch;
    linear_rows(weights, input, output, 0, row_batch, floating,
                scratch ? *scratch : local_scratch);
}

}  // namespace pokitlms::model
