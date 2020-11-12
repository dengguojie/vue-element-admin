/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * Description: api of device
 */

#ifndef CPU_KERNELS_DEVICE_H
#define CPU_KERNELS_DEVICE_H

#include "sharder.h"

namespace aicpu {
class Device {
public:
    explicit Device(DeviceType device);

    ~Device();

    /*
     * get device type.
     * @return DeviceType: HOST/DEVICE
     */
    DeviceType GetDeviceType() const;

    /*
     * get sharder.
     * @return Sharder *: host or device sharder
     */
    const Sharder *GetSharder() const;

private:
    Device(const Device &) = delete;
    Device(Device &&) = delete;
    Device &operator = (const Device &) = delete;
    Device &operator = (Device &&) = delete;

    /*
     * init sharder.
     * param device: type of device
     * @return Sharder *: not null->success, null->success
     */
    Sharder *InitSharder(DeviceType device);

private:
    DeviceType device_; // type of device
    Sharder *sharder_;
};
} // namespace aicpu
#endif // CPU_KERNELS_DEVICE_H
