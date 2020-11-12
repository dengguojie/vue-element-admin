/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * Description: implement of device sharder
 */

#include "device_sharder.h"

#include "aicpu_schedule/aicpu_sharder/aicpu_sharder.h"

namespace aicpu {
/*
 * ParallelFor shards the "total" units of work.
 */
void DeviceSharder::ParallelFor(int64_t total, int64_t perUnitSize,
    const std::function<void(int64_t, int64_t)> &work) const
{
    SharderNonBlock::GetInstance().ParallelFor(total, perUnitSize, work);
}

/*
 * Get CPU number
 */
uint32_t DeviceSharder::GetCPUNum() const
{
    return SharderNonBlock::GetInstance().GetCPUNum();
}
} // namespace aicpu
