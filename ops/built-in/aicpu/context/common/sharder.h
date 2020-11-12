/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * Description: api of sharder
 */


#ifndef CPU_KERNELS_SHARDER_H
#define CPU_KERNELS_SHARDER_H
#include <functional>
#include "cpu_types.h"

namespace aicpu {
class Sharder {
public:
    explicit Sharder(DeviceType device) : device_(device) {}

    virtual ~Sharder() = default;

    /*
     * ParallelFor shards the "total" units of work.
     * @param total: size of total work
     * @param perUnitSize: expect size of per unit work
     * @param work: process of per unit work
     */
    virtual void ParallelFor(int64_t total, int64_t perUnitSize,
        const std::function<void(int64_t, int64_t)> &work) const = 0;

    /*
     * Get CPU number
     * @return CPU number
     */
    virtual uint32_t GetCPUNum() const = 0;

private:
    Sharder(const Sharder &) = delete;
    Sharder(Sharder &&) = delete;
    Sharder &operator = (const Sharder &) = delete;
    Sharder &operator = (Sharder &&) = delete;

private:
    DeviceType device_; // device type, HOST/DEVICE
};
} // namespace aicpu
#endif // CPU_KERNELS_SHARDER_H
