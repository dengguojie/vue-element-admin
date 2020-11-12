/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * Description: implement of host sharder
 */

#include "host_sharder.h"

#include "log.h"
#include "eigen_threadpool.h"

namespace aicpu {
/*
 * ParallelFor shards the "total" units of work.
 */
void HostSharder::ParallelFor(int64_t total, int64_t perUnitSize,
    const std::function<void(int64_t, int64_t)> &work) const
{
    EigenThreadPool *threadpool = EigenThreadPool::GetInstance();
    if (threadpool == nullptr) {
        KERNEL_LOG_ERROR("get eigen thread pool failed");
        return;
    }

    threadpool->ParallelFor(total, perUnitSize, work);
}

/*
 * Get CPU number
 */
uint32_t HostSharder::GetCPUNum() const
{
    EigenThreadPool *threadpool = EigenThreadPool::GetInstance();
    if (threadpool == nullptr) {
        KERNEL_LOG_ERROR("get eigen thread pool failed");
        return 0;
    }

    return threadpool->GetCPUNum();
}
} // namespace aicpu
