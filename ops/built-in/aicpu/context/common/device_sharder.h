/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * Description: api of device sharder
 */

#ifndef CPU_KERNELS_DEVICE_SHARDER_H
#define CPU_KERNELS_DEVICE_SHARDER_H
#include "sharder.h"

namespace aicpu {
class DeviceSharder : public Sharder {
public:
    explicit DeviceSharder(DeviceType device) : Sharder(device) {};

    ~DeviceSharder() = default;

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
    DeviceSharder(const DeviceSharder &) = delete;
    DeviceSharder(DeviceSharder &&) = delete;
    DeviceSharder &operator = (const DeviceSharder &) = delete;
    DeviceSharder &operator = (DeviceSharder &&) = delete;
};
} // namespace aicpu
#endif // CPU_KERNELS_DEVICE_SHARDER_H
