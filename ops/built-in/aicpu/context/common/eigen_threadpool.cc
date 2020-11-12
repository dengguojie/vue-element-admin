/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * Description: implement of eigen thread pool
 */

#include "eigen_threadpool.h"

#include <unistd.h>
#include <sys/sysinfo.h>

#include "log.h"

namespace {
const uint32_t TASK_SIZE = 40000;
const uint32_t MAX_OVERSHARDING_FACTOR = 4;
const uint32_t TOTAL_COST_FACTOR = 210000;
constexpr uint32_t MAX_TASK_SIZE = TASK_SIZE * MAX_OVERSHARDING_FACTOR;
}

namespace aicpu {
std::mutex EigenThreadPool::mutex_;
bool EigenThreadPool::initFlag_(false);
int32_t EigenThreadPool::coreNum_(0);
std::unique_ptr<Eigen::ThreadPool> EigenThreadPool::eigenThreadpool_(nullptr);
std::unique_ptr<Eigen::ThreadPoolDevice> EigenThreadPool::threadpoolDevice_(nullptr);

EigenThreadPool *EigenThreadPool::GetInstance()
{
    KERNEL_LOG_INFO("EigenThreadPool GetInstance begin");
    {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!initFlag_) {
            coreNum_ = get_nprocs(); // obtains the number of CPU cores that can be used by users.
            if (coreNum_ <= 0) {
                KERNEL_LOG_INFO("Get the number of CPU cores that can be used failed, core number:%d", coreNum_);
                return nullptr;
            }
            eigenThreadpool_.reset(new Eigen::ThreadPool(coreNum_));
            threadpoolDevice_.reset(new Eigen::ThreadPoolDevice(eigenThreadpool_.get(), coreNum_));
            initFlag_ = true;
            KERNEL_LOG_INFO("EigenThreadPool init success, core number:%d", coreNum_);
        }
    }

    static EigenThreadPool instance;
    KERNEL_LOG_INFO("EigenThreadPool GetInstance success");
    return &instance;
}

void EigenThreadPool::ParallelFor(int64_t total, int64_t perUnitSize, const SharderWork &work)
{
    KERNEL_LOG_INFO("eigen threadpool parallel for begin, total:%lld, perUnitSize: %lld", total, perUnitSize);
    if ((total <= 0) || (work == nullptr) || (perUnitSize <= 0)) {
        KERNEL_LOG_ERROR("invalid param: total:%lld <= 0 or perUnitSize:%lld <= 0 or work is nullptr", total,
            perUnitSize);
        return;
    }

    int64_t totalCheck = static_cast<int64_t>(static_cast<Eigen::Index>(total));
    if (totalCheck != total) {
        KERNEL_LOG_ERROR("invalid param: total:%lld, value:%lld after eigen conversion", total, totalCheck);
        return;
    }

    double perUnitCost = 1.0;
    if (perUnitSize >= total) {
        // use the current thread to process the task
        perUnitCost = 1.0 * TASK_SIZE / total;
    } else if ((perUnitSize) <= (total / coreNum_)) {
        // run tasks with the maximum number of threads, maximum = MAX_OVERSHARDING_FACTOR * coreNum_
        perUnitCost = (1.0 * MAX_TASK_SIZE * coreNum_ / total) > (1.0 * TOTAL_COST_FACTOR / total) ?
            (1.0 * MAX_TASK_SIZE * coreNum_ / total) :
            (1.0 * TOTAL_COST_FACTOR / total);
    } else {
        // the task is fragmented based on the number of data slices.
        perUnitCost = 1.0 * MAX_TASK_SIZE / perUnitSize;
    }

    KERNEL_LOG_INFO("eigen threadpool parallel for, perUnitCost:%.6f", perUnitCost);

    threadpoolDevice_->parallelFor(total, Eigen::TensorOpCost(0, 0, perUnitCost),
        [&work](Eigen::Index first, Eigen::Index last) { work(first, last); });
    KERNEL_LOG_INFO("eigen threadpool parallel for success");
}

/*
 * Get CPU number
 */
uint32_t EigenThreadPool::GetCPUNum()
{
    return static_cast<uint32_t>(coreNum_);
}
} // namespace aicpu
