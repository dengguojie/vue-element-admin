/**
 * Copyright 2019 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file parsing_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_PARSING_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_PARSING_OPS_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {

/**
*@brief Converts each string in the input Tensor to the specified numeric type . \n

*@par Inputs:
*Inputs include:
*x: A Tensor. Must be one of the following types: string . \n

*@par Attributes:
*out_type: The numeric type to interpret each string in string_tensor as . \n

*@par Outputs:
*y: A Tensor. Has the same type as x . \n

*@attention Constraints:
*The implementation for StringToNumber on Ascend uses AICPU, with bad performance. \n

*@par Third-party framework compatibility
*@li compatible with tensorflow StringToNumber operator.
*/
REG_OP(StringToNumber)
    .INPUT(x, TensorType({DT_STRING}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64}))
    .ATTR(out_type, Type, DT_FLOAT)
    .OP_END_FACTORY_REG(StringToNumber)

/**
*@brief Convert serialized tensorflow.TensorProto prototype to Tensor.
*@brief Parse an Example prototype. 
*@par Input:
*serialized: A Tensor of type string.
*dense_defaults:  DYNAMIC INPUT Tensor type as string, float, int64. \n

*@par Attributes:
*num_sparse: type int num of inputs sparse_indices , sparse_values, sparse_shapes
*out_type: output type
*sparse_keys: ListString
*sparse_types: types of sparse_values
*dense_keys: ListString
*dense_shapes: output of dense_defaults shape
*dense_types: output of dense_defaults type  \n

*@par Outputs:
*sparse_indices: A Tensor of type string. 
*sparse_values:  Has the same type as sparse_types.
*sparse_shapes: A Tensor of type int64
*dense_values:  Has the same type as dense_defaults.

*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
**/
REG_OP(ParseSingleExample)
    .INPUT(serialized, TensorType({DT_STRING}))
    .DYNAMIC_INPUT(dense_defaults, TensorType({DT_STRING,DT_FLOAT,DT_INT64}))
    .DYNAMIC_OUTPUT(sparse_indices, TensorType({DT_INT64}))
    .DYNAMIC_OUTPUT(sparse_values, TensorType({DT_STRING,DT_FLOAT,DT_INT64}))
    .DYNAMIC_OUTPUT(sparse_shapes, TensorType({DT_INT64}))
    .DYNAMIC_OUTPUT(dense_values, TensorType({DT_STRING,DT_FLOAT,DT_INT64}))
    .ATTR(num_sparse, Int, 0)
    .ATTR(sparse_keys, ListString, {})
    .ATTR(dense_keys, ListString, {})
    .ATTR(sparse_types, ListType, {})
    .ATTR(dense_types, ListType, {})
    .ATTR(dense_shapes, ListListInt, {})
    .OP_END_FACTORY_REG(ParseSingleExample)

/**
*@brief Decodes raw file into  tensor . \n
*@par Input:
*contents: A Tensor of type string.

*@par Attributes:
*little_endian: bool ture
*out_type: output type

*@par Outputs:
*Output: A Tensor
**/
REG_OP(DecodeRaw)
    .INPUT(bytes, TensorType({DT_STRING}))
    .OUTPUT(output, TensorType({DT_BOOL,DT_FLOAT16,DT_DOUBLE,DT_FLOAT,
                                    DT_INT64,DT_INT32,DT_INT8,DT_UINT8,DT_INT16,
                                    DT_UINT16,DT_COMPLEX64,DT_COMPLEX128}))
    .ATTR(out_type, Type, DT_FLOAT)
    .ATTR(little_endian, Bool, true)
    .OP_END_FACTORY_REG(DecodeRaw)

/**
*@brief Convert serialized tensorflow.TensorProto prototype to Tensor. \n

*@par Inputs:
*serialized: A Tensor of string type. Scalar string containing serialized
*TensorProto prototype. \n

*@par Attributes:
*out_type: The numeric type to interpret each string in string_tensor as . \n

*@par Outputs:
*y: A Tensor. Has the same type as serialized. \n

*@attention Constraints:
*The implementation for StringToNumber on Ascend uses AICPU,
*with badperformance. \n

*@par Third-party framework compatibility
*@li compatible with tensorflow ParseTensor operator.
*/
REG_OP(ParseTensor)
    .INPUT(serialized, TensorType({DT_STRING}))
    .OUTPUT(output, TensorType(DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16,
                          DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32,
                          DT_UINT64, DT_BOOL, DT_DOUBLE, DT_STRING,
                          DT_COMPLEX64, DT_COMPLEX128}))
    .ATTR(out_type, Type, DT_FLOAT)
    .OP_END_FACTORY_REG(ParseTensor)

/**
*@brief Converts each string in the input Tensor to the specified numeric
*type . \n

*@par Inputs:
*Inputs include:
*x: A Tensor. Must be one of the following types: string . \n

*@par Attributes:
*out_type: The numeric type to interpret each string in string_tensor as . \n

*@par Outputs:
*y: A Tensor. Has the same type as x . \n

*@attention Constraints:
*The implementation for StringToNumber on Ascend uses AICPU, with bad
*performance. \n

*@par Third-party framework compatibility
*@li compatible with tensorflow StringToNumber operator.
*/
REG_OP(DecodeCSV)
    .INPUT(records, TensorType({DT_STRING}))
    .DYNAMIC_INPUT(record_defaults, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT32,
                                        DT_INT64, DT_STRING, DT_RESOURCE}))
    .DYNAMIC_OUTPUT(output, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT32,
                                        DT_INT64, DT_STRING, DT_RESOURCE}))
    .ATTR(OUT_TYPE, ListType, {})
    .ATTR(field_delim, String, ",")
    .ATTR(use_quote_delim, Bool, true)
    .ATTR(na_value, String, ",")
    .ATTR(select_cols, ListInt, {})
    .OP_END_FACTORY_REG(DecodeCSV)

}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_PARSING_OPS_H_
