/**
 * Copyright 2020 Huawei Technologies Co., Ltd

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 * http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

/*!
 * \file acg_pool_v2_grad.h
 * \brief Performs average pooling on the input.
*/
#ifndef GE_OP_AVGPOOLV2GRAD_OPS_H
#define GE_OP_AVGPOOLV2GRAD_OPS_H

#include "graph/operator.h"
#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Computes AvgPoolV2Grad function

* @par Inputs:
* orig_input_shape: A tensor of type int32.
* input_grad: A tensor of type float16, float32, double.

* @par Attributes: 
* @li ksize: filter window size, 4-D list.
* @li strides: strides over h and w axis, 4-D list.
* @li padding_mode: a 'string' from `"same", "valid", "CALCULATED"'. 
* @li pads: 4-D list, the pads of pooling.
* @li data_format: support 'NHWC' or 'NCHW'.
* @li global_pooling: Whether to use the global pooling. If global_pooling=true,
*                     ksize and pads will be ignored. Default False.   
* @li ceil_mode: Whether to use the ceil function to calculate output height and
*                width. Default False.  
* @li exclusive: Whether to exclude padding points. default is true.
* @par Outputs:
* out_grad: A tensor of type as same as input_grad.
*/
REG_OP(AvgPoolV2Grad)
  .INPUT(orig_input_shape, TensorType({DT_INT32}))
  .INPUT(input_grad, TensorType({DT_FLOAT16, DT_FLOAT32, DT_DOUBLE}))
  .OUTPUT(out_grad, TensorType({DT_FLOAT16, DT_FLOAT32, DT_DOUBLE}))
  .REQUIRED_ATTR(ksize, ListInt)
  .REQUIRED_ATTR(strides, ListInt)
  .ATTR(padding_mode, String, "CALCULATED")
  .ATTR(pads, ListInt, {0, 0, 0, 0})
  .ATTR(data_format, String, "NCHW")
  .ATTR(global_pooling, Bool, false)
  .ATTR(ceil_mode, Bool, false)
  .ATTR(exclusive, Bool, true)
  .OP_END_FACTORY_REG(AvgPoolV2Grad)
/**
* @brief Computes AvgPoolV2GradD function

* @par Inputs:
* orig_input_shape: A tensor of type int32.
* input_grad: A tensor of type float16, float32, double.

* @par Attributes: 
* @li ksize: filter window size, 4-D list.
* @li strides: strides over h and w axis, 4-D list.
* @li padding_mode: a 'string' from `"same", "valid", "CALCULATED"'. 
* @li pads: 4-D list, the pads of pooling.
* @li data_format: support 'NHWC' or 'NCHW'.
* @li global_pooling: Whether to use the global pooling. If global_pooling=true,
*                     ksize and pads will be ignored. Default False.   
* @li ceil_mode: Whether to use the ceil function to calculate output height and
*                width. Default False.  
* @li exclusive: Whether to exclude padding points. default is true.
* @par Outputs:
* out_grad: A tensor of type as same as input_grad.
*/
REG_OP(AvgPoolV2GradD)
  .INPUT(input_grad, TensorType({DT_FLOAT16, DT_FLOAT32}))
  .OPTIONAL_INPUT(mean_matrix, TensorType({DT_FLOAT16, DT_FLOAT32, }))
  .OPTIONAL_INPUT(kernel_matrix, TensorType({DT_FLOAT16, DT_FLOAT32}))
  .OUTPUT(out_grad, TensorType({DT_FLOAT16, DT_FLOAT32}))
  .REQUIRED_ATTR(orig_input_shape, ListInt)
  .REQUIRED_ATTR(ksize, ListInt)
  .REQUIRED_ATTR(strides, ListInt)
  .ATTR(padding_mode, String, "CALCULATED")
  .ATTR(pads, ListInt, {0, 0, 0, 0})
  .ATTR(data_format, String, "NCHW")
  .ATTR(global_pooling, Bool, false)
  .ATTR(ceil_mode, Bool, false)
  .ATTR(exclusive, Bool, true)
  .OP_END_FACTORY_REG(AvgPoolV2GradD)

}  // namespace ge

#endif  // GE_OP_AVGPOOLV2GRAD_OPS_H
