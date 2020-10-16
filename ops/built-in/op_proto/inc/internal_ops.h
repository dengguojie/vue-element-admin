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
 * @file  internal_ops.h
 *
 * @brief
 *
 * @version 1.0
 *
 */
#ifndef GE_OP_INTERNAL_OPS_H_
#define GE_OP_INTERNAL_OPS_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {

/**
*@brief aicpu assit help op for auxiliary matrix generation.

*@par Inputs:
*The input is dynamic for attribute func_name \n

*@par Attributes:
*@li func_name:An required param, for example "topkv2". \n

*@par Outputs:
*The output is dynamic for attribute func_name.
*/
REG_OP(AssistHelp)
    .DYNAMIC_INPUT(x, TensorType({ DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16,
        DT_UINT8, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE }))
    .DYNAMIC_OUTPUT(y, TensorType({ DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16,
        DT_UINT8, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    . REQUIRED_ATTR (func_name, String)
    . OP_END_FACTORY_REG(AssistHelp)

/**
*@brief aicpu cache help for lhisi cache flush.

*@par Inputs:
*The input is dynamic for attribute func_name \n

*@par Outputs:
*The output is dynamic for attribute func_name.
*/
REG_OP(CacheUpdate)
    .INPUT(x, TensorType::BasicType())
    .OUTPUT(x, TensorType::BasicType())
    .OP_END_FACTORY_REG(CacheUpdate)

/**
*@brief transfer data from L1 buffer to DDR or DDR to L1.

*@par Inputs:
*The input is dynamic for attribute func_name \n

*@par Outputs:
*The output is dynamic for attribute func_name.
*/
REG_OP(InternalDataMove)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .REQUIRED_ATTR(src_buf, String)
    .REQUIRED_ATTR(dst_buf, String)
    .OP_END_FACTORY_REG(InternalDataMove)

}  // namespace ge

#endif  // GE_OP_INTERNAL_OPS_H_
