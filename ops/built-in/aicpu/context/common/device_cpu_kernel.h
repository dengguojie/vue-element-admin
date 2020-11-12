/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * Description: api of device cpu kernel
 */

#ifndef DEVICE_CPU_KERNEL
#define DEVICE_CPU_KERNEL
#include <cstdint>
extern "C" {
uint32_t RunCpuKernel(void *param);
}
#endif // DEVICE_CPU_KERNEL