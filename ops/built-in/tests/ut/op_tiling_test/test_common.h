/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
 * \file test_common.h
 */

#define TENSOR_INPUT(paras, tensor, key)  \
  auto data##key = op::Data(#key);        \
  data##key.update_input_desc_x(tensor);  \
  data##key.update_output_desc_y(tensor); \
  paras.set_input_##key(data##key);       \
  tensor.SetName(#key);                   \
  paras.UpdateInputDesc(#key, tensor)

#define TENSOR_INPUT_CONST(paras, tensor, key, data, size)                 \
  Tensor tensor_const_##key(tensor);                                       \
  tensor_const_##key.SetData((uint8_t*)data, size);                        \
  auto const##key = op::Constant(#key).set_attr_value(tensor_const_##key); \
  paras.set_input_##key(const##key);                                       \
  TENSOR_INPUT(paras, tensor, key)

#define TENSOR_OUTPUT(paras, tensor, key)    \
  auto dataout##key = op::Data(#key);        \
  dataout##key.update_input_desc_x(tensor);  \
  dataout##key.update_output_desc_y(tensor); \
  paras.UpdateOutputDesc(#key, tensor)
