/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * Description: status
 */

#ifndef KERNEL_STATUS_H
#define KERNEL_STATUS_H

#include <cstdint>

namespace aicpu {
/*
 * status code
 */
enum KernelStatus : uint32_t {
    KERNEL_STATUS_OK = 0,
    KERNEL_STATUS_PARAM_INVALID,
    KERNEL_STATUS_INNER_ERROR,
    KERNEL_STATUS_PROTOBUF_ERROR,
    KERNEL_STATUS_SHARDER_ERROR
};
} // namespace aicpu
#endif // KERNEL_STATUS_H
