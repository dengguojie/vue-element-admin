/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * Description: api of host sharder
 */

#ifndef CPU_KERNELS_HOST_SHARDER_H
#define CPU_KERNELS_HOST_SHARDER_H
#include "sharder.h"

namespace aicpu {
class HostSharder : public Sharder {
public:
    explicit HostSharder(DeviceType device) : Sharder(device) {};

    ~HostSharder() = default;

    /*
     * ParallelFor shards the "total" units of work.
     * @param total: size of total work
     * @param perUnitSize: expect size of per unit work
     * @param work: process of per unit work
     */
    void ParallelFor(int64_t total, int64_t perUnitSize,
        const std::function<void(int64_t, int64_t)> &work) const override;

    /*
     * Get CPU number
     * @return CPU number
     */
    uint32_t GetCPUNum() const override;
private:
    HostSharder(const HostSharder &) = delete;
    HostSharder(HostSharder &&) = delete;
    HostSharder &operator = (const HostSharder &) = delete;
    HostSharder &operator = (HostSharder &&) = delete;
};
} // namespace aicpu
#endif // CPU_KERNELS_HOST_SHARDER_H
