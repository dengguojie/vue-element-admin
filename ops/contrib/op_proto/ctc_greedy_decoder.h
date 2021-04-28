/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
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
 *
 * @file ctc_greedy_decoder.h
 *
 * @version 1.0
 */

#ifndef GE_OP_CTC_GREEDY_DECODER_H
#define GE_OP_CTC_GREEDY_DECODER_H

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Performs greedy decoding on the logits given in input
 *
 * @par Inputs:
 * 1 inputs, including:
 * @li  input : 3-D Tensor sized [batchsize, timestep, class_num].
 * required.
 * DataType: float16
 * Format: ND
 *
 * @par Outputs:
 * 1 outputs, including:
 * @li  output : 2-D Tensor sized [batchsize, sequence_length[i] + 1] \n
 * required, the data in [batchsize, -1] represents the \n
 * number of significant digits in every batch.
 * DataType: float16
 * Format: ND
 *
 * @par Attributes:
 * @li sequence_length: it containing sequence lengths, having size \n
 * [batch_size], required.
 * @li max_sequence_len: Int(0), max sequence, optional.
 * @li merge_repeated: Bool(True), If merge_repeated is True, merge \n
 * repeated classes in output. This means that if consecutive \n
 * logits' maximum indices are the same, only the first of these \n
 * is emitted, optional.
 * @li default_value: Int(0), Scalar value to set for indices not \n
 * specified in output, optional.
 *
 * @attention Constraints:
 * @li sequence_length must be list or tuple
 * @li sequence_length should equal to batchsize
 * @li only support input data size < 200K
 * @li support max class_num is 2048
 */

REG_OP(ctc_greedy_decoder)
    .INPUT(input, TensorType({ DT_FLOAT16 }))   /* "First operand." */
    .OUTPUT(output, TensorType({ DT_FLOAT16 })) /* "Result, has same element type as two inputs" */
    .REQUIRED_ATTR(sequence_length, ListInt)
    .ATTR(max_sequence_len, Int, 0)
    .ATTR(merge_repeated, Bool, true)
    .ATTR(default_value, Int, 0)
    .OP_END_FACTORY_REG(ctc_greedy_decoder)
}

#endif // GE_OP_CTC_GREEDY_DECODER_H
