/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * Description: api of eigen thread pool
 */

#ifndef CPU_KERNELS_EIGEN_THREAD_POOL_H
#define CPU_KERNELS_EIGEN_THREAD_POOL_H

#include <mutex>
#include <memory>
#include <functional>
#define EIGEN_USE_THREADS
#include <unsupported/Eigen/CXX11/Tensor>

namespace aicpu {
using SharderWork = std::function<void(int64_t, int64_t)>;

class EigenThreadPool {
public:
    static EigenThreadPool *GetInstance();

    /*
     * ParallelFor shards the "total" units of work.
     */
    void ParallelFor(int64_t total, int64_t perUnitSize, const SharderWork &work);

    /*
     * Get CPU number
     * @return CPU number
     */
    uint32_t GetCPUNum();

private:
    EigenThreadPool() = default;
    ~EigenThreadPool() = default;

    EigenThreadPool(const EigenThreadPool &) = delete;
    EigenThreadPool(EigenThreadPool &&) = delete;
    EigenThreadPool &operator = (const EigenThreadPool &) = delete;
    EigenThreadPool &operator = (EigenThreadPool &&) = delete;

private:
    static std::mutex mutex_; // protect initFlag_
    static bool initFlag_;    // true means initialized
    static int32_t coreNum_;  // the number of CPU cores that can be used by users
    static std::unique_ptr<Eigen::ThreadPool> eigenThreadpool_;
    static std::unique_ptr<Eigen::ThreadPoolDevice> threadpoolDevice_;
};
};     // namespace aicpu
#endif // CPU_KERNELS_EIGEN_THREAD_POOL_H
