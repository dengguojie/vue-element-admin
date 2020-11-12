/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * Description: implement of device
 */

#include "device.h"

#include <new>
#include "device_sharder.h"
#include "host_sharder.h"

namespace aicpu {
Device::Device(DeviceType device)
{
    device_ = device;
    sharder_ = InitSharder(device);
}

Device::~Device()
{
    if (sharder_ != nullptr) {
        delete sharder_;
    }
}

/*
 * get device type.
 * @return DeviceType: HOST/DEVICE
 */
DeviceType Device::GetDeviceType() const
{
    return device_;
}

/*
 * get sharder.
 * @return Sharder *: host or device sharder
 */
const Sharder *Device::GetSharder() const
{
    if (sharder_ != nullptr) {
        return sharder_;
    }
    return nullptr;
}

/*
 * init sharder.
 * param device: type of device
 * @return Sharder *: not null->success, null->success
 */
Sharder *Device::InitSharder(DeviceType device_)
{
    if (device_ == DEVICE) {
        return new (std::nothrow) DeviceSharder(device_);
    } else {
        return new (std::nothrow) HostSharder(device_);
    }
}
} // namespace aicpu
