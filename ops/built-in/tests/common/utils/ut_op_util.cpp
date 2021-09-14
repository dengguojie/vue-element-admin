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
 * \file ut_op_util.cpp
 */
#include "ut_op_util.h"
#include "op_log.h"

namespace ut_util {

void TransformerOpBaseFormat(const Operator& op, const std::string& input_name, const Format storage_format) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto tensordesc_input = op_desc->MutableInputDesc(input_name);
  auto shape_shape = tensordesc_input->MutableShape();
  std::vector<int64_t> shape_dims = tensordesc_input->MutableShape().GetDims();
  std::vector<int64_t> new_shape_dims;
  auto format = tensordesc_input->GetOriginFormat();
  auto data_type = tensordesc_input->GetDataType();

  // transfer shape
  transformer::ShapeTransferAccordingToFormat shape_transfer;
  transformer::ShapeAndFormat shape_and_format_info{shape_dims,     new_shape_dims, format,
                                                    storage_format, data_type,      transformer::EN_IMPL_HW_TBE};
  (void)shape_transfer.GetShapeAccordingToFormat(shape_and_format_info);
  auto new_shape_shape = GeShape(new_shape_dims);
  OP_LOGI("TransformerOpBaseFormat",
          "Transform shape successfully, node:%s, name:%s, shape:%s, format:%s, new shape:%s, new format:%s.",
          op_desc->GetName().c_str(), input_name.c_str(), shape_shape.ToString().c_str(), to_string(format).c_str(),
          new_shape_shape.ToString().c_str(), to_string(storage_format).c_str());
  std::vector<std::pair<int64_t, int64_t>> origin_range;
  std::vector<std::pair<int64_t, int64_t>> output_range;
  tensordesc_input->GetOriginShapeRange(origin_range);
  // transfer range
  transformer::RangeTransferAccordingToFormat range_transfer;
  transformer::RangeAndFormat range_and_format_info{
      shape_shape, origin_range, output_range, format, storage_format, data_type, transformer::EN_IMPL_HW_TBE};
  (void)range_transfer.GetRangeAccordingToFormat(range_and_format_info);
  tensordesc_input->SetFormat(storage_format);
  tensordesc_input->SetShape(GeShape(new_shape_dims));
  tensordesc_input->SetShapeRange(output_range);
}

DataType StringToDtype(std::string dtype_string) {
  auto find_it = optiling::STR_TO_DATATYPE.find(dtype_string);
  if (find_it != optiling::STR_TO_DATATYPE.end()) {
    return find_it->second;
  }
  return ge::DT_FLOAT16;
}

string to_string_int32(const std::stringstream& tiling_data) {
  auto data = tiling_data.str();
  string result;
  int32_t tmp = 0;
  for (size_t i = 0; i < data.length(); i += sizeof(int32_t)) {
    memcpy(&tmp, data.c_str() + i, sizeof(tmp));
    result += std::to_string(tmp);
    result += " ";
  }
  return result;
}

string to_string_int64(const std::stringstream& tiling_data) {
  auto data = tiling_data.str();
  string result;
  int32_t tmp = 0;
  for (size_t i = 0; i < data.length(); i += sizeof(int64_t)) {
    memcpy(&tmp, data.c_str() + i, sizeof(tmp));
    result += std::to_string(tmp);
    result += " ";
  }
  return result;
}

}  // namespace ut_util
