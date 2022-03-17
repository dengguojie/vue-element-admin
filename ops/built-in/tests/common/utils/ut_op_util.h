/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
 * \file ut_op_util.h
 */
#include <graph/utils/type_utils.h>
#include "graph/tensor.h"
#include "op_proto_test_util.h"
#include "third_party/transformer/inc/transfer_shape_according_to_format.h"
#include "third_party/transformer/inc/transfer_range_according_to_format.h"

#define TENSOR_INPUT_WITH_SHAPE(paras, key, shape, dtype, foramt, range)                        \
  auto tensor_desc_##key = create_desc_shape_range(shape, dtype, foramt, shape, foramt, range); \
  auto data##key = op::Data(#key);                                                              \
  data##key.update_input_desc_x(tensor_desc_##key);                                             \
  data##key.update_output_desc_y(tensor_desc_##key);                                            \
  paras.set_input_##key(data##key);                                                             \
  paras.UpdateInputDesc(#key, tensor_desc_##key)

#define TENSOR_INPUT_WITH_ORI_SHAPE(paras, key, shape, dtype, foramt, ori_shape, ori_foramt, range)                        \
  auto tensor_desc_##key = create_desc_shape_and_origin_shape_range(shape, dtype, foramt, ori_shape, ori_foramt,range); \
  auto data##key = op::Data(#key);                                                              \
  data##key.update_input_desc_x(tensor_desc_##key);                                             \
  data##key.update_output_desc_y(tensor_desc_##key);                                            \
  paras.set_input_##key(data##key);                                                             \
  paras.UpdateInputDesc(#key, tensor_desc_##key)

#define TENSOR_DY_INPUT_WITH_SHAPE(paras, key, idx, shape, dtype, foramt, range)                \
  auto tensor_desc_##key = create_desc_shape_range(shape, dtype, foramt, shape, foramt, range); \
  auto data##key = op::Data(#key);                                                              \
  data##key.update_input_desc_x(tensor_desc_##key);                                             \
  data##key.update_output_desc_y(tensor_desc_##key);                                            \
  paras.UpdateDynamicInputDesc(#key, idx, tensor_desc_##key)

#define TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(paras, key, shape, dtype, foramt, const_value) \
  auto tensor_desc_##key = create_desc_shape_range(shape, dtype, foramt, shape, foramt, {});   \
  Tensor tensor_const_##key(tensor_desc_##key);                                                \
  tensor_desc_##key.SetName(#key);                                                             \
  SetValueToConstTensor(tensor_const_##key, const_value);                                      \
  auto const##key = op::Constant(#key).set_attr_value(tensor_const_##key);                     \
  paras.set_input_##key(const##key);                                                           \
  paras.UpdateInputDesc(#key, tensor_desc_##key)

#define TENSOR_OUTPUT_WITH_SHAPE(paras, key, shape, dtype, foramt, range)                           \
  auto tensor_out_desc_##key = create_desc_shape_range(shape, dtype, foramt, shape, foramt, range); \
  paras.UpdateOutputDesc(#key, tensor_out_desc_##key)

#define RUN_TILING_V3(op, tiling_iter, compile_info, run_info) \
  ge::AscendString op_compile_info_as = compile_info.c_str();  \
  ASSERT_TRUE(tiling_iter.tiling_func_v3_(op, tiling_iter.parse_func_v3_(op, op_compile_info_as), run_info));

#define RUN_TILING_V3_FALSE(op, tiling_iter, compile_info, run_info) \
  ge::AscendString op_compile_info_as = compile_info.c_str();        \
  ASSERT_FALSE(tiling_iter.tiling_func_v3_(op, tiling_iter.parse_func_v3_(op, op_compile_info_as), run_info));

#define RUN_TILING_V4(op, tiling_iter, compile_info, run_info) \
  ge::AscendString op_compile_info_as = compile_info.c_str();  \
  ASSERT_TRUE(tiling_iter.tiling_func_v4_(op, tiling_iter.parse_func_v4_(op, op_compile_info_as), run_info));

#define RUN_TILING_V4_FALSE(op, tiling_iter, compile_info, run_info) \
  ge::AscendString op_compile_info_as = compile_info.c_str();        \
  ASSERT_FALSE(tiling_iter.tiling_func_v4_(op, tiling_iter.parse_func_v4_(op, op_compile_info_as), run_info));

namespace ut_util {

template <typename T>
void SetValueToConstTensor(ge::Tensor const_tensor, std::vector<T> const_value) {
  T* cosnt_data = new T[const_value.size()];
  for (size_t dim = 0; dim < const_value.size(); dim++) {
    *(cosnt_data + dim) = const_value[dim];
  }
  const_tensor.SetData((uint8_t*)cosnt_data, const_value.size() * sizeof(T));
  delete[] cosnt_data;
}

/*
 * trans a tensor of op from oriformat to special format, NCHW -> NC1HWC0
 * this func will modify the shape/format/range base the orishape/oriformat/range
 * param[in] op: one Operator
 * param[in] input_name: which tensordesc will be update
 * param[in] storage_format: des format
 */
void TransformerOpBaseFormat(const Operator& op, const std::string& input_name, const Format storage_format);

/*
 * set a tensor oriformat to special format
 * this func will modify the shape/format/range base the orishape/oriformat/range
 * param[in] op: one Operator
 * param[in] input_name: which tensordesc will be update
 * param[in] storage_format: des format
 */
void SetGetOriginFormat(const Operator& op, const std::string& input_name, const Format storage_format);

/*
 * trans the string to GeDataType
 * param[in] dtype_string: string of type, ex: float32/float16
 * return DataType: the ge DataType of string data_type
 */
DataType StringToDtype(std::string dtype_string);

/*
 * trans the tiling_data to string for int32
 * param[in] tiling_data: stringstream
 * return string: string of tiling_data
 */
string to_string_int32(const std::stringstream& tiling_data);

/*
 * trans the tiling_data to string for int64
 * param[in] tiling_data: stringstream
 * return string: string of tiling_data
 */
string to_string_int64(const std::stringstream& tiling_data);

}  // namespace ut_util
