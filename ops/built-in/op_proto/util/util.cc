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
 * \file util.cpp
 * \brief
 */
#include "util.h"
#include <string>
#include <vector>
#include <map>
#include <functional>
#include <algorithm>
#include "./error_util.h"
#include "op_common_util.h"
#include "graph/utils/type_utils.h"

namespace ge {

bool GetInputDataType(const ge::DataType& data_type, const std::vector<ge::DataType>& supportList) {
  std::vector<ge::DataType>::const_iterator supportIter = find(supportList.begin(), supportList.end(), data_type);
  if (supportIter == supportList.end()) {
    return false;
  }
  return true;
}

bool CheckInputDtypeAndShape(const Operator& op, const std::map<std::string, std::vector<DataType>>& inputTensorMap) {
  auto iter = inputTensorMap.begin();
  auto first_name = iter->first;
  auto first_shape_dims = op.GetInputDesc(iter->first).GetShape().GetDims();
  auto first_input_dtype = op.GetInputDesc(iter->first).GetDataType();
  for (; iter != inputTensorMap.end(); ++iter) {
    const TensorDesc input_desc = op.GetInputDesc(iter->first);
    // check input dtype
    auto input_type = input_desc.GetDataType();
    if (input_type != first_input_dtype) {
      OP_LOGE(op.GetName().c_str(), "the op type of param %s must equal with param %s", iter->first.c_str(),
              first_name.c_str());
      return false;
    }
    auto dims = input_desc.GetShape().GetDims();
    if (dims != first_shape_dims) {
      OP_LOGE(op.GetName().c_str(), "the op shape of param %s must equal with param %s", iter->first.c_str(),
              first_name.c_str());
      return false;
    }
  }
  return true;
}

bool CheckInputDataType(const Operator& op, const std::string& input_name,
                        const std::vector<ge::DataType>& support_list) {
  bool valid = false;
  DataType input_type = op.GetInputDesc(input_name).GetDataType();
  do {
    const auto& found_list = find(support_list.begin(), support_list.end(), input_type);

    if (found_list == support_list.end()) {
      break;
    }

    const auto& found_map = DTYPE_STR_MAP.find(input_type);
    if (found_map == DTYPE_STR_MAP.end()) {
      break;
    }

    valid = true;
  } while (0);

  if (!valid) {
    OpsInputDtypeErrReport(op.GetName(), input_name, DebugString(support_list), ConcatString(input_type));
    OP_LOGE(op.GetName().c_str(), "The op do not support the dtype %s",
            ge::TypeUtils::DataTypeToSerialString(input_type).c_str());
    return false;
  }

  return true;
}

bool CheckTwoInputDtypeSame(const Operator& op, const string& input_name1, const string& input_name2) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  DataType input_type_x1 = op_desc->MutableInputDesc(input_name1)->GetDataType();
  DataType input_type_x2 = op_desc->MutableInputDesc(input_name2)->GetDataType();
  if (input_type_x1 != input_type_x2) {
    OpsTwoInputDtypeErrReport(op.GetName(), input_name1, input_name2, ConcatString(input_type_x1),
                              ConcatString(input_type_x2));
    OP_LOGE(op.GetName().c_str(), "The %s op dtype is not same, type1:%s, type2:%s", op.GetName().c_str(),
            ge::TypeUtils::DataTypeToSerialString(input_type_x1).c_str(),
            ge::TypeUtils::DataTypeToSerialString(input_type_x2).c_str());
    return false;
  }

  return true;
}

bool CheckInputDtypeSame(const Operator& op, std::vector<std::string>& input_tensors) {
  auto first_name = input_tensors.begin();
  auto first_input_dtype = op.GetInputDesc(*first_name).GetDataType();
  for (const string& input_name : input_tensors) {
    const TensorDesc input_desc = op.GetInputDesc(input_name);
    auto input_dtype = input_desc.GetDataType();
    if (input_dtype != first_input_dtype) {
      OP_LOGE(op.GetName().c_str(), "the op type of param %s must equal with param %s", input_name.c_str(),
              (*first_name).c_str());
      return false;
    }
  }
  return true;
}

bool CheckInputsShapeDtypeSame(const Operator& op, const std::vector<std::string>& input_names) {
  auto first_input_name = input_names.begin();
  auto first_input_des = op.GetInputDesc(*first_input_name);
  auto input_name = first_input_name;
  for (++input_name; input_name != input_names.end(); ++input_name) {
    auto input_des = op.GetInputDesc(*first_input_name);

    if (input_des.GetDataType() != first_input_des.GetDataType() ||
        input_des.GetShape().GetDims() != first_input_des.GetShape().GetDims()) {
      OpsAttrValueErrReport(
          op.GetName(), ConcatString(input_name->c_str(), "'s dtype and shape"),
          ConcatString("same as", first_input_name->c_str(), "[", first_input_des.GetDataType(), "]", "[",
                       DebugString(first_input_des.GetShape().GetDims()), "]"),
          ConcatString("[", input_des.GetDataType(), "]", "[", DebugString(input_des.GetShape().GetDims()), "]"));
      OP_LOGE(op.GetName().c_str(), "the dtype and shape of param %s must be same as param %s",
              first_input_name->c_str(), input_name->c_str());
      return false;
    }
  }

  return true;
}

bool InferShapeAndTypeTwoInOneOutBroadcast(Operator& op, const string& input_name1, const string& input_name2,
                                           const string& output_name) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  DataType input_dtype = op_desc->MutableInputDesc(input_name1)->GetDataType();

  GeTensorDescPtr tensordesc_output = op_desc->MutableOutputDesc(output_name);

  ge::GeShape shapeX = op_desc->MutableInputDesc(input_name1)->GetShape();
  ge::GeShape shapeY = op_desc->MutableInputDesc(input_name2)->GetShape();
  OP_LOGI(op.GetName().c_str(), "shape %s: %s, shape %s: %s.", input_name1.c_str(), to_string(shapeX).c_str(),
          input_name2.c_str(), to_string(shapeY).c_str());
  std::vector<int64_t> dimsX = shapeX.GetDims();
  std::vector<int64_t> dimsY = shapeY.GetDims();
  // swap based on shape size
  if (dimsX.size() < dimsY.size()) {
    std::vector<int64_t> dimsTmp = dimsX;
    dimsX = dimsY;
    dimsY = dimsTmp;
  }
  // infer shape of empty tensor
  if (IsEmptyTensor(dimsX) && !IsEmptyTensor(dimsY)) {
    OpsInputShapeErrReport(op.GetName(), "two inputs should both be empty", input_name2, "not empty");
    OP_LOGE(op.GetName().c_str(), "One input is empty tensor and another is not.");
    return false;
  } else if (IsEmptyTensor(dimsY) && !IsEmptyTensor(dimsX)) {
    OpsInputShapeErrReport(op.GetName(), "two inputs should both be empty", input_name1, "not empty");
    OP_LOGE(op.GetName().c_str(), "One input is empty tensor and another is not.");
    return false;
  }

  std::vector<int64_t> dimVec;

  // unknown rank
  if (IsUnknownRankShape(dimsX) || IsUnknownRankShape(dimsY)) {
    dimVec.push_back(-2);
    tensordesc_output->SetShape(ge::GeShape(dimVec));
    tensordesc_output->SetDataType(input_dtype);
    OP_LOGI(op.GetName().c_str(), "output shape is: %s, output dtype is:%d.", to_string(ge::Shape(dimVec)).c_str(),
            input_dtype);
    return true;
  }

  // pad 1 for small shape
  if (dimsX.size() != dimsY.size()) {
    int dec = dimsX.size() - dimsY.size();
    for (int i = 0; i < dec; i++) {
      dimsY.insert(dimsY.begin(), (int64_t)1);
    }
  }

  for (size_t i = 0; i < dimsX.size(); i++) {
    if ((dimsX[i] != dimsY[i]) && (dimsX[i] != 1) && (dimsY[i] != 1) && (dimsX[i] != -1) && (dimsY[i] != -1)) {
      OpsInputShapeBroadcastErrReport(op.GetName(), input_name1, input_name2, ConcatString(dimsX[i]),
                                      ConcatString(dimsY[i]));
      OP_LOGE(op.GetName().c_str(), "The %s's dimensions does not match the broadcast rule(%lu %lu).",
              op.GetName().c_str(), dimsX[i], dimsY[i]);
      return false;
    }

    if ((dimsX[i] == -1) && (dimsY[i] != -1)) {
      if (dimsY[i] > 1) {
        int64_t dims = dimsX[i] > dimsY[i] ? dimsX[i] : dimsY[i];
        dimVec.push_back(dims);
      } else if (dimsY[i] == 1) {
        int64_t dims = dimsX[i] > dimsY[i] ? dimsX[i] : dimsY[i];
        dimVec.push_back(dims);
        dimVec[i] = -1;
      } else if ((dimsY[i] == 0) || (dimsX[i] == 0)) {
        dimVec.push_back(0);
      }
    } else if ((dimsX[i] != -1) && (dimsY[i] == -1)) {
      if (dimsX[i] > 1) {
        int64_t dims = dimsX[i] > dimsY[i] ? dimsX[i] : dimsY[i];
        dimVec.push_back(dims);
      } else if (dimsX[i] == 0) {
        dimVec.push_back(0);
      } else if (dimsX[i] == 1) {
        int64_t dims = dimsX[i] > dimsY[i] ? dimsX[i] : dimsY[i];
        dimVec.push_back(dims);
        dimVec[i] = -1;
      }
    } else {
      if ((dimsX[i] == -1) && (dimsY[i] == -1)) {
        int64_t dims = dimsX[i] > dimsY[i] ? dimsX[i] : dimsY[i];
        dimVec.push_back(dims);
        dimVec[i] = -1;
      } else {
        if (dimsY[i] == 0 || dimsX[i] == 0) {
          dimVec.push_back(0);
        } else {
          int64_t dims = dimsX[i] > dimsY[i] ? dimsX[i] : dimsY[i];
          dimVec.push_back(dims);
        }
      }
    }
  }
  ge::GeShape outputShape = ge::GeShape(dimVec);

  tensordesc_output->SetShape(outputShape);
  tensordesc_output->SetDataType(input_dtype);
  OP_LOGI(op.GetName().c_str(), "output shape is: %s, output dtype is:%s.", to_string(outputShape).c_str(),
          ge::TypeUtils::DataTypeToSerialString(input_dtype).c_str());

  return true;
}

bool InferShapeRangeTwoInOneOutBroadcase(Operator& op, const string& input_name1, const string& input_name2,
                                         const string& output_name) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ge::GeShape shape_x = op_desc->MutableInputDesc(input_name1)->GetShape();
  ge::GeShape shape_y = op_desc->MutableInputDesc(input_name2)->GetShape();

  std::vector<int64_t> dims_x = shape_x.GetDims();
  std::vector<int64_t> dims_y = shape_y.GetDims();

  std::vector<std::pair<int64_t, int64_t>> shape_range_x;
  op_desc->MutableInputDesc(input_name1)->GetShapeRange(shape_range_x);
  std::vector<std::pair<int64_t, int64_t>> shape_range_y;
  op_desc->MutableInputDesc(input_name2)->GetShapeRange(shape_range_y);

  MakeUpShapeRange(dims_x, shape_range_x);
  MakeUpShapeRange(dims_y, shape_range_y);

  ge::GeShape shape_out = op_desc->MutableOutputDesc(output_name)->GetShape();
  std::vector<int64_t> dims_out = shape_out.GetDims();
  size_t size_shape_out = dims_out.size();

  std::vector<std::pair<int64_t, int64_t>> out_range;

  if (!IsUnknownRankShape(dims_out)) {
    // shape switch by shape dim size
    if (dims_x.size() < dims_y.size()) {
      std::vector<int64_t> dims_tmp = dims_x;
      dims_x = dims_y;
      dims_y = dims_tmp;

      std::vector<std::pair<int64_t, int64_t>> range_temp = shape_range_x;
      shape_range_x = shape_range_y;
      shape_range_y = range_temp;
    }

    while (dims_x.size() > shape_range_y.size()) {
      shape_range_y.insert(shape_range_y.begin(), std::pair<int64_t, int64_t>(1, 1));
    }

    for (size_t i = 0; i < size_shape_out; i++) {
      if (dims_out[i] != -1) {
        out_range.push_back(std::pair<int64_t, int64_t>(dims_out[i], dims_out[i]));
        continue;
      }
      if (i < shape_range_x.size() && i < shape_range_y.size()) {
        if (shape_range_x[i].second == -1 && shape_range_y[i].second == 1) {
          out_range.push_back(std::pair<int64_t, int64_t>(1, -1));
        } else if (shape_range_x[i].second == 1 && shape_range_y[i].second == -1) {
          out_range.push_back(std::pair<int64_t, int64_t>(1, -1));
        } else {
          out_range.push_back(std::pair<int64_t, int64_t>(std::min(shape_range_x[i].first, shape_range_y[i].first),
                                                          std::max(shape_range_x[i].second, shape_range_y[i].second)));
        }
      }
    }
  } else {
    size_t max_size = std::max(shape_range_x.size(), shape_range_y.size());
    while (shape_range_x.size() < max_size) {
      shape_range_x.insert(shape_range_x.begin(), std::pair<int64_t, int64_t>(1, 1));
    }
    while (shape_range_y.size() < max_size) {
      shape_range_y.insert(shape_range_y.begin(), std::pair<int64_t, int64_t>(1, 1));
    }

    for (size_t i = 0; i < max_size; i++) {
      out_range.push_back(std::pair<int64_t, int64_t>(std::min(shape_range_x[i].first, shape_range_y[i].first),
                                                      std::max(shape_range_x[i].second, shape_range_y[i].second)));
    }
  }

  GeTensorDescPtr tensor_out = op_desc->MutableOutputDesc(output_name);
  tensor_out->SetShapeRange(out_range);

  return true;
}

bool GetInputDataType(const ge::DataType& dataType, const std::vector<ge::DataType>& supportList, std::string& dType) {
  std::vector<ge::DataType>::const_iterator supportIter = find(supportList.begin(), supportList.end(), dataType);
  if (supportIter == supportList.end()) {
    return false;
  }

  std::map<ge::DataType, std::string>::const_iterator totalIter = DTYPE_STR_MAP.find(dataType);
  if (totalIter == DTYPE_STR_MAP.end()) {
    return false;
  }

  dType = totalIter->second;
  return true;
}

bool CheckInputDataType(const Operator& op, std::string* data_type, const std::string& input_name,
                        const std::vector<ge::DataType>& supportList) {
  DataType input_type = op.GetInputDesc(input_name).GetDataType();
  if (false == GetInputDataType(input_type, supportList, *data_type)) {
    LOG_ERROR("[ERROR]op [%s] [%s] do not supported dtype [%s]!\n", op.GetName().c_str(), input_name.c_str(),
              data_type->c_str());
    return false;
  }
  return true;
}

bool GetConstValue(const ge::Operator& op, const std::string& key_name, float& attr_value) {
  if (ge::GRAPH_SUCCESS != op.GetAttr(key_name, attr_value)) {
    LOG_ERROR("[ERROR]op [%s] GetOpAttr [%s] failed!\n", op.GetName().c_str(), key_name.c_str());
    return false;
  }
  return true;
}

bool GetConstValue(const ge::Operator& op, const std::string& key_name, int64_t& attr_value) {
  if (ge::GRAPH_SUCCESS != op.GetAttr(key_name, attr_value)) {
    LOG_ERROR("[ERROR]op [%s] GetOpAttr [%s] failed!\n", op.GetName().c_str(), key_name.c_str());
    return false;
  }
  return true;
}

bool GetConstValue(const ge::Operator& op, const std::string& key_name, bool& attr_value) {
  if (ge::GRAPH_SUCCESS != op.GetAttr(key_name, attr_value)) {
    LOG_ERROR("[ERROR]op [%s] GetOpAttr [%s] failed!\n", op.GetName().c_str(), key_name.c_str());
    return false;
  }
  return true;
}

bool GetConstValue(const ge::Operator& op, const std::string& key_name, std::vector<int32_t>& attr_value) {
  if (ge::GRAPH_SUCCESS != op.GetAttr(key_name, attr_value)) {
    LOG_ERROR("[ERROR]op [%s] GetOpAttr [%s] failed!\n", op.GetName().c_str(), key_name.c_str());
    return false;
  }
  return true;
}

template <typename T>
static std::vector<int64_t> GetConstIntData(const uint8_t* const_data, size_t data_size) {
  size_t size = data_size / sizeof(T);
  std::vector<int64_t> result(size);
  T* data = (T*)const_data;
  for (size_t i = 0; i < size; i++) {
    result[i] = *(data + i);
  }

  return result;
}

bool GetConstIntData(const Tensor& data, DataType data_type, std::vector<int64_t>& const_values) {
  using namespace std::placeholders;
  const std::map<DataType, std::function<std::vector<int64_t>(const uint8_t*, size_t)>> type_call_map = {
      {DT_INT8, std::bind(GetConstIntData<int8_t>, _1, _2)},
      {DT_INT16, std::bind(GetConstIntData<int16_t>, _1, _2)},
      {DT_INT32, std::bind(GetConstIntData<int32_t>, _1, _2)},
      {DT_INT64, std::bind(GetConstIntData<int64_t>, _1, _2)},
  };

  auto found = type_call_map.find(data_type);
  if (found == type_call_map.end()) {
    USER_GE_LOGE("[ERROR]GetConstIntData is not support data_type[%s]!",
                 ge::TypeUtils::DataTypeToSerialString(data_type).c_str());
    return false;
  }

  const_values = found->second(data.GetData(), data.GetSize());

  return true;
}

bool GetConstValue(const Operator& op, const Tensor& const_tensor, const DataType& dtype,
                   std::vector<int64_t>& const_data) {
  size_t size = 0;
  if (dtype == ge::DT_INT32) {
    int32_t* const_data_ptr = (int32_t*)const_tensor.GetData();
    size = const_tensor.GetSize() / sizeof(int32_t);
    for (size_t i = 0; i < size; ++i) {
      const_data.push_back((int32_t)((*(const_data_ptr + i))));
      OP_LOGD(op.GetName().c_str(), "const data int32 fusion pass ====== %d", (int32_t)(*(const_data_ptr + i)));
    }
  } else if (dtype == ge::DT_INT64) {
    int64_t* const_data_ptr = (int64_t*)const_tensor.GetData();
    size = const_tensor.GetSize() / sizeof(int64_t);
    for (size_t i = 0; i < size; ++i) {
      const_data.push_back(((int64_t)(*(const_data_ptr + i))));
      OP_LOGD(op.GetName().c_str(), "const data int64 fusion pass ====== %d", (int64_t)(*(const_data_ptr + i)));
    }
  } else {
    OP_LOGE(op.GetName().c_str(), "not support this type");
    return false;
  }
  return true;
}

bool GetScalerValue(const Operator& op, const Tensor& const_tensor, const DataType& dtype, std::int64_t& const_data) {
  if (dtype == ge::DT_INT32) {
    int32_t* const_data_ptr = (int32_t*)const_tensor.GetData();
    const_data = (int32_t)(*const_data_ptr);
  } else if (dtype == ge::DT_INT64) {
    int64_t* const_data_ptr = (int64_t*)const_tensor.GetData();
    const_data = (int64_t)(*const_data_ptr);
  } else {
    OP_LOGE(op.GetName().c_str(), "not support this type");
    return false;
  }
  return true;
}

string to_string(const vector<int64_t>& shape) {
  return ops::to_string(shape);
}

std::string to_string(const ge::Shape& shape) {
  return to_string(shape.GetDims());
}

std::string to_string(const ge::GeShape& shape) {
  return to_string(shape.GetDims());
}

std::string to_string(const vector<pair<int64_t, int64_t>>& ranges) {
  return ops::to_string(ranges);
}

bool DynamicShapeInfer::CatchFormatAndShape() {
  inputs = op_desc->GetAllInputName();
  outputs = op_desc->GetAllOutputName();
  GeTensorDescPtr tensor_desc_input, tensor_desc_output;

  // get and save current input shape&format, and assign origin ones to them
  std::string input_name;
  for (map<std::string, uint32_t>::iterator it = inputs.begin(); it != inputs.end(); it++) {
    input_name = it->first;
    tensor_desc_input = op_desc->MutableInputDesc(input_name);
    if (tensor_desc_input == nullptr) {
      continue;
    }
    Format curr_format = tensor_desc_input->GetFormat();

    map_format.insert(std::pair<std::string, Format>(input_name, curr_format));
    map_dtype.insert(std::pair<std::string, DataType>(input_name, tensor_desc_input->GetDataType()));

    if (tensor_desc_input->GetOriginFormat() == curr_format) {
      continue;
    }
    tensor_desc_input->SetFormat(tensor_desc_input->GetOriginFormat());
    tensor_desc_input->SetShape(tensor_desc_input->GetOriginShape());
  }

  // get and save current output shape&format, and assign origin ones to them
  std::string output_name;
  for (map<std::string, uint32_t>::iterator it = outputs.begin(); it != outputs.end(); it++) {
    output_name = it->first;
    tensor_desc_output = op_desc->MutableOutputDesc(output_name);
    if (tensor_desc_output == nullptr) {
      continue;
    }
    Format curr_format = tensor_desc_output->GetFormat();

    map_format.insert(std::pair<std::string, Format>(output_name, curr_format));
    map_dtype.insert(std::pair<std::string, DataType>(output_name, tensor_desc_output->GetDataType()));

    if (tensor_desc_output->GetOriginFormat() == curr_format) {
      continue;
    }
    tensor_desc_output->SetFormat(tensor_desc_output->GetOriginFormat());
  }

  return true;
}

bool DynamicShapeInfer::UpdateFormatAndShape() {
  const int64_t opImplType = EN_IMPL_CUSTOM_TBE;
  GeTensorDescPtr tensor_desc_input, tensor_desc_output;
  // assign output's after infershape to origin shape
  for (map<std::string, uint32_t>::iterator it = outputs.begin(); it != outputs.end(); it++) {
    tensor_desc_output = op_desc->MutableOutputDesc(it->first);
    if (tensor_desc_output == nullptr) {
      continue;
    }
    tensor_desc_output->SetOriginShape(tensor_desc_output->GetShape());
  }

  // transfer input's origin shape to current shape
  Format ori_input_format, cur_input_format;
  GeShape ori_infer_shape, current_shape;
  std::string input_name;
  for (map<std::string, uint32_t>::iterator it = inputs.begin(); it != inputs.end(); it++) {
    input_name = it->first;
    tensor_desc_input = op_desc->MutableInputDesc(input_name);
    if (tensor_desc_input == nullptr) {
      continue;
    }
    ori_input_format = tensor_desc_input->GetFormat();
    ori_infer_shape = tensor_desc_input->GetShape();
    cur_input_format = map_format[input_name];

    // print some info
    OP_LOGI(op.GetName().c_str(), "origin input shape %s is %s", input_name.c_str(),
            to_string(ori_infer_shape).c_str());

    ShapeAndFormat shapeAndFormatInfoInput = {ori_infer_shape,  current_shape,         ori_input_format,
                                              cur_input_format, map_dtype[input_name], opImplType};
    if (ori_input_format == cur_input_format) {
      // no need to transfer shape
      continue;
    } else {
      ShapeTransferAccordingToFormat* global_object = new ShapeTransferAccordingToFormat();
      global_object->GetShapeAccordingToFormat(shapeAndFormatInfoInput);

      // print some info
      OP_LOGI(op.GetName().c_str(), "current input shape %s is %s", input_name.c_str(),
              to_string(current_shape).c_str());

      tensor_desc_input->SetFormat(cur_input_format);
      tensor_desc_input->SetShape(current_shape);
      delete global_object;
    }
  }

  // transfer output's origin shape to current shape
  Format ori_output_format, cur_output_format;
  GeShape ori_infer_out_shape, current_out_shape;
  std::string output_name;
  for (map<std::string, uint32_t>::iterator it = outputs.begin(); it != outputs.end(); it++) {
    output_name = it->first;
    tensor_desc_output = op_desc->MutableOutputDesc(output_name);
    if (tensor_desc_output == nullptr) {
      continue;
    }
    ori_output_format = tensor_desc_output->GetFormat();
    ori_infer_out_shape = tensor_desc_output->GetShape();
    cur_output_format = map_format[output_name];

    // print some info
    OP_LOGI(op.GetName().c_str(), "origin output shape %s is %s", output_name.c_str(),
            to_string(ori_infer_out_shape).c_str());

    ShapeAndFormat shapeAndFormatInfoOutput = {ori_infer_out_shape, current_out_shape,      ori_output_format,
                                               cur_output_format,   map_dtype[output_name], opImplType};
    if (ori_output_format == cur_output_format) {
      // no need to transfer shape
      continue;
    } else {
      ShapeTransferAccordingToFormat* global_object = new ShapeTransferAccordingToFormat();
      global_object->GetShapeAccordingToFormat(shapeAndFormatInfoOutput);

      // print some info
      OP_LOGI(op.GetName().c_str(), "current output shape %s is %s", output_name.c_str(),
              to_string(current_out_shape).c_str());

      tensor_desc_output->SetFormat(cur_output_format);
      tensor_desc_output->SetShape(current_out_shape);
      delete global_object;
    }
  }

  return true;
}

bool IsEmptyTensor(const std::vector<int64_t>& dims) {
  if (dims.size() == 1 && dims[0] == 0) {
    return true;
  } else {
    return false;
  }
}

bool IsUnknownRank(const Operator& op, const std::string& tensor_name, const std::string& types) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr tensor_desc;
  if (types == "input") {
    tensor_desc = op_desc->MutableInputDesc(tensor_name);
  } else if (types == "output") {
    tensor_desc = op_desc->MutableOutputDesc(tensor_name);
  } else {
    OP_LOGE(op.GetName().c_str(), "invalid params of types to judge.");
    return false;
  }

  std::vector<int64_t> shape_vec = tensor_desc->GetShape().GetDims();
  if (shape_vec.size() == 1 && shape_vec[0] == -2) {
    return true;
  }
  return false;
}

bool IsUnknownRankShape(const std::vector<int64_t>& shape_vec) {
  if (shape_vec.size() == 1 && shape_vec[0] == -2) {
    return true;
  }
  return false;
}

bool IsUnKnownShape(const std::vector<int64_t>& shape_vec) {
  auto found = find(shape_vec.begin(), shape_vec.end(), -1);
  return found != shape_vec.end();
}

bool IsUnknown(const std::vector<int64_t>& shape_vec) {
  return (IsUnKnownShape(shape_vec) || IsUnknownRankShape(shape_vec));
}

bool IsUnknownShape(const Operator& op, const std::string& tensor_name, const std::string& types) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDescPtr tensor_desc;
  if (types == "input") {
    tensor_desc = op_desc->MutableInputDesc(tensor_name);
  } else if (types == "output") {
    tensor_desc = op_desc->MutableOutputDesc(tensor_name);
  } else {
    OP_LOGE(op.GetName().c_str(), "invalid params of types to judge.");
    return false;
  }

  std::vector<int64_t> shape_vec = tensor_desc->GetShape().GetDims();
  std::vector<int64_t>::iterator it_shape;
  it_shape = find(shape_vec.begin(), shape_vec.end(), -1);
  if (it_shape == shape_vec.end()) {
    return false;
  } else {
    return true;
  }
}

bool IsUnknownVec(std::vector<int64_t>& shape_vec) {
  std::vector<int64_t>::iterator it_shape;
  it_shape = find(shape_vec.begin(), shape_vec.end(), -1);
  if (it_shape == shape_vec.end()) {
    return false;
  } else {
    return true;
  }
}

void MakeUpShapeRange(const std::vector<int64_t>& shape, std::vector<std::pair<int64_t, int64_t>>& range) {
  if (IsUnknownRankShape(shape)) {
    return;
  }

  if (range.empty()) {
    for (size_t i = 0; i < shape.size(); i++) {
      if (shape[i] == -1) {
        range.push_back(std::pair<int64_t, int64_t>(1, -1));
      } else {
        range.push_back(std::pair<int64_t, int64_t>(shape[i], shape[i]));
      }
    }
  }
}

std::string DataTypeToStringDesc(const ge::DataType& dataType) {
  std::map<ge::DataType, std::string>::const_iterator totalIter = DTYPE_STR_MAP.find(dataType);
  if (totalIter == DTYPE_STR_MAP.end()) {
    return "UNDEFINED";
  }
  return totalIter->second;
}

bool OneInOneOutDynamicInfer(const Operator& op,
                             const std::string& input_name,
                             const std::vector<std::string>& output_name_list) {
  // get input desc
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto input_desc = op_info->MutableInputDesc(input_name);
  vector<int64_t> input_shape = input_desc->MutableShape().GetDims();
  DataType input_dtype = input_desc->GetDataType();

  if (IsUnknown(input_shape)) {
    std::vector<std::pair<int64_t, int64_t>> input_range;
    input_desc->GetShapeRange(input_range);
    MakeUpShapeRange(input_shape, input_range);

    auto output_desc = op_info->MutableOutputDesc(0);
    for (const string& output_name : output_name_list) {
      output_desc = op_info->MutableOutputDesc(output_name);
      output_desc->SetShape(GeShape(input_shape));
      output_desc->SetOriginShape(GeShape(input_shape));
      output_desc->SetShapeRange(input_range);
      output_desc->SetDataType(input_dtype);
    }
  } else {
    auto output_desc = op_info->MutableOutputDesc(0);
    for (const string& output_name : output_name_list) {
      output_desc = op_info->MutableOutputDesc(output_name);
      output_desc->SetShape(GeShape(input_shape));
      output_desc->SetDataType(input_dtype);
    }
  }
  return true;
}

void FixShapeRangeWithDims(const std::vector<int64_t>& dims,
                           std::vector<int64_t>& shape_1,
                           std::vector<int64_t>& shape_2,
                           std::vector<std::pair<int64_t, int64_t>>& range_1,
                           std::vector<std::pair<int64_t, int64_t>>& range_2) {
  MakeUpShapeRange(shape_1, range_1);
  MakeUpShapeRange(shape_2, range_2);
  bool is_all_fix = dims.empty();

  if (shape_1 == UNKNOWN_RANK && shape_2 == UNKNOWN_RANK) {
    return;
  }
  if (shape_1 == UNKNOWN_RANK) {
    shape_1 = shape_2;
    range_1 = range_2;
    return;
  }
  if (shape_2 == UNKNOWN_RANK) {
    shape_2 = shape_1;
    range_2 = range_1;
    return;
  }
  if ((shape_1.size() != shape_2.size()) || (range_1.size() != range_2.size())) {
    return;
  }
  auto loop_size = is_all_fix ? shape_1.size() : dims.size();
  for (size_t i = 0; i < loop_size; i ++) {
    auto dim_num = is_all_fix ? i : dims[i];
    if (shape_1[dim_num] != -1) {
      shape_2[dim_num] = shape_1[dim_num];
      range_1[dim_num] = std::pair<int64_t, int64_t>(shape_1[dim_num], shape_1[dim_num]);
      range_2[dim_num] = std::pair<int64_t, int64_t>(shape_1[dim_num], shape_1[dim_num]);
      continue;
    }
    if (shape_2[dim_num] != -1) {
      shape_1[dim_num] = shape_2[dim_num];
      range_1[dim_num] = std::pair<int64_t, int64_t>(shape_2[dim_num], shape_2[dim_num]);
      range_2[dim_num] = std::pair<int64_t, int64_t>(shape_2[dim_num], shape_2[dim_num]);
      continue;
    }
    // both the dim in shape1 and shape2 are -1
    auto range_1_min = range_1[dim_num].first;
    auto range_2_min = range_2[dim_num].first;
    auto range_1_max = range_1[dim_num].second;
    auto range_2_max = range_2[dim_num].second;
    auto range_fisrt = range_1_min > range_2_min ? range_1_min : range_2_min;
    auto range_second_min = range_1_max > range_2_max ? range_2_max : range_1_max;
    auto range_second_max = range_1_max > range_2_max ? range_1_max : range_2_max;
    range_second_min = range_second_min == -1 ? range_second_max : range_second_min;
    range_1[dim_num] = std::pair<int64_t, int64_t>(range_fisrt, range_second_min);
    range_2[dim_num] = std::pair<int64_t, int64_t>(range_fisrt, range_second_min);
  }
}

bool TwoInOneOutDynamicInferNoBroadcast(Operator& op,
                                        const string& input1_name,
                                        const string& input2_name,
                                        const std::vector<string>& output_name_list) {
  // get input1 desc
  auto op_info = OpDescUtils::GetOpDescFromOperator(op);
  auto input1_desc = op_info->MutableInputDesc(input1_name);
  vector<int64_t> input1_shape = input1_desc->MutableShape().GetDims();
  DataType input_dtype = input1_desc->GetDataType();

  // get input2 desc
  auto input2_desc = op_info->MutableInputDesc(input2_name);
  vector<int64_t> input2_shape = input2_desc->MutableShape().GetDims();

  if (IsUnknown(input1_shape) || IsUnknown(input2_shape)) {
    std::vector<std::pair<int64_t, int64_t>> input1_range;
    input1_desc->GetShapeRange(input1_range);
    std::vector<std::pair<int64_t, int64_t>> input2_range;
    input2_desc->GetShapeRange(input2_range);

    vector<int64_t> dim_size = {};
    FixShapeRangeWithDims(dim_size, input1_shape, input2_shape, input1_range, input2_range);

    // update output desc
    auto output_desc = op_info->MutableOutputDesc(0);
    for (const string& output_name : output_name_list) {
      output_desc = op_info->MutableOutputDesc(output_name);
      output_desc->SetShape(GeShape(input1_shape));
      output_desc->SetOriginShape(GeShape(input1_shape));
      output_desc->SetShapeRange(input1_range);
      output_desc->SetDataType(input_dtype);
    }
  } else {
    auto output_desc = op_info->MutableOutputDesc(0);
    for (const string& output_name : output_name_list) {
      output_desc = op_info->MutableOutputDesc(output_name);
      output_desc->SetShape(GeShape(input1_shape));
      output_desc->SetDataType(input_dtype);
    }
  }
  return true;
}

}  // namespace ge
