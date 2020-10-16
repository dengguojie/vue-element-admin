/**
 * Copyright (C)  2019. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file get_data_ops.h
 *
 * @brief
 *
 * @version 1.0
 *
 */

#ifndef GE_OP_GET_DATA_OPS_H_
#define GE_OP_GET_DATA_OPS_H_

#include "graph/operator_reg.h"

namespace ge {

REG_OP(MakeIterator)
    .INPUT(x, TensorType::ALL())
    .INPUT(x1, TensorType::ALL())
    .ATTR(_kernel, String, "dp")
    .OP_END_FACTORY_REG(MakeIterator)

REG_OP(IteratorV2)
    .OUTPUT(y, TensorType::ALL())
    .ATTR(output_types, ListInt, {})
    .ATTR(output_shapes,ListListInt, {{}, {}})
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .OP_END_FACTORY_REG(IteratorV2)

REG_OP(IteratorGetNext)
    .INPUT(x, TensorType::ALL())
    .DYNAMIC_OUTPUT(y, TensorType::ALL())
    .ATTR(output_types, ListInt, {})
    .ATTR(output_shapes, ListListInt, {{},{}})
    .ATTR(output_num, Int, 1)
    .ATTR(_kernel, String, "dp")
    .OP_END_FACTORY_REG(IteratorGetNext)

REG_OP(DeviceQueueDataset)
    .OUTPUT(y, TensorType::ALL())
    .ATTR(output_types, ListInt, {})
    .ATTR(output_shapes, ListListInt, {{},{}})
    .ATTR(channel_name, String, "")
    .ATTR(_iterator_name, String, "IteratorV2")
    .OP_END_FACTORY_REG(DeviceQueueDataset)

} // namespace ge


#endif  // GE_OP_GET_DATA_OPS_H_
