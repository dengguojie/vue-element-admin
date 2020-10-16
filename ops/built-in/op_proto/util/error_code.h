/* Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use
 * this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#ifndef OP_PROTO_UTIL_ERROR_CODE_H
#define OP_PROTO_UTIL_ERROR_CODE_H

namespace ge {

// error code for report purpose.
// 30000~34999 for aicpu engine error
// and 35000~39999 for infershape error of aicpu op
enum ViewErrorCode{
    INVALID_INFER_SHAPE = 14001,
    INVALID_INPUT_SHAPE = 35000,
    INVALID_ATTR_VALUE = 35001,
    INVALID_ATTR_SIZE = 35002,
    OTHER_ERROR = 35003,
    INVALID_MISS_INPUT = 70001,
    INVALID_INPUT_FORMAT = 70002,
    INVALID_INPUT_DTYPE = 70003,
    INVALID_INPUT_TYPE = 70004,
    INVALID_GET_ATTR = 70005,
    INVALID_SET_ATTR = 70006,
    INVALID_OPS_ATTR_VALUE = 70007,
    FAILED_UPDATE_OP = 70008,
    INVALID_SHAPE = 70009,
    INVALID_SHAPE_SIZE = 70010,
    INVALID_SHAPE_DIM = 70011,
    INVALID_BROADCAST_SHAPE = 70012,
    INVALID_TWO_INPUT_DTYPE = 70013,
    INVALID_AIPP_ERROR = 70014,
    INVALID_ONE_INPUT_SHAPE = 70015,
    INVALID_TWO_INPUT_SHAPE = 70016,
    INVALID_ONE_OUTPUT_SHAPE = 70017,
    FAILED_GET_COMPILIE_PARAMS = 70018,
};



}  // namespace ge

#endif  // OP_PROTO_UTIL_ERROR_CODE_H
