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
 * \file util.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_UTIL_UTIL_H_
#define OPS_BUILT_IN_OP_PROTO_UTIL_UTIL_H_

#include <memory.h>
#include <string>
#include <vector>
#include <map>
#include <algorithm>

#include "framework/omg/omg_inner_types.h"
#include "graph/operator.h"
#include "graph/operator_reg.h"
#include "graph/operator_reg.h"
#include "transfer_shape_according_to_format.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/ge_tensor.h"

#include "op_log.h"

#define LOG_ERROR(format, args...) printf(format, ##args)
namespace ge {

// enum type and string type mapping
static const std::map<ge::DataType, std::string> DTYPE_STR_MAP{
    {ge::DT_FLOAT16, "float16"}, {ge::DT_FLOAT, "float32"}, {ge::DT_INT8, "int8"},   {ge::DT_INT16, "int16"},
    {ge::DT_INT32, "int32"},     {ge::DT_INT64, "int64"},   {ge::DT_UINT8, "uint8"}, {ge::DT_UINT16, "uint16"},
    {ge::DT_UINT32, "uint32"},   {ge::DT_UINT64, "uint64"}, {ge::DT_BOOL, "bool"}};

// define the input num of shape
const size_t INPUT_NUM0 = 0;
const size_t INPUT_NUM1 = 1;
const size_t INPUT_NUM2 = 2;
const size_t INPUT_NUM3 = 3;
const size_t INPUT_NUM4 = 4;
const size_t INPUT_NUM5 = 5;
const size_t INPUT_NUM6 = 6;
const size_t INPUT_NUM7 = 7;
const size_t INPUT_NUM8 = 8;
const size_t INPUT_NUM9 = 9;

// define the dims size of shape
const size_t DIM_SIZE0 = 0;
const size_t DIM_SIZE1 = 1;
const size_t DIM_SIZE2 = 2;
const size_t DIM_SIZE3 = 3;
const size_t DIM_SIZE4 = 4;
const size_t DIM_SIZE5 = 5;
const size_t DIM_SIZE6 = 6;
const size_t DIM_SIZE7 = 7;
const size_t DIM_SIZE8 = 8;

// define the index of shape dim
const size_t DIM_INDEX0 = 0;
const size_t DIM_INDEX1 = 1;
const size_t DIM_INDEX2 = 2;
const size_t DIM_INDEX3 = 3;
const size_t DIM_INDEX4 = 4;
const size_t DIM_INDEX5 = 5;
const size_t DIM_INDEX6 = 6;
const size_t DIM_INDEX7 = 7;
const size_t DIM_INDEX8 = 8;

/*
 * get the datatype of input
 * param[in] dataType  input datatype of enum value
 * param[in] supportList  the support range of op
 * return true :get type success
 *        false:get type failed
 */
bool GetInputDataType(const ge::DataType& data_type, const std::vector<ge::DataType>& supportList);

bool GetInputDataType(const ge::DataType& dataType, const std::vector<ge::DataType>& supportList, std::string& dType);

/* infer shape of two input and on output with broadcast
 * param[in] op  op desc supply by ge
 * param[in] inputName1  first input name
 * param[in] inputName2  second input name
 * param[in] outputName  output name
 * return SUCCESS:infer success
 *        FAILED:infer failed like unsupported broadcast input shape
 */
bool CheckInputDataType(const Operator& op, const std::string& input_name,
                        const std::vector<ge::DataType>& support_list);

/*
 * check the datatype and shape of input
 * param[in] op  the operator
 * param[in] inputTensorMap  the map of input name and support datatype
 * param[in] paramType  the mode of input param, tensor or scalar
 * return true
 *        false
 */
bool CheckInputDtypeAndShape(const Operator& op, const std::map<std::string, std::vector<DataType>>& inputTensorMap);

/*
 * infer shape of two input and on output with broadcast
 * param[in] op  op desc supply by ge
 * param[in] inputName1  first input name
 * param[in] inputName2  second input name
 * param[in] outputName  output name
 * return SUCCESS:infer success
 *        FAILED:infer failed like unsupported broadcast input shape
 */
bool InferShapeAndTypeTwoInOneOutBroadcast(Operator& op, const string& input_name1, const string& input_name2,
                                           const string& output_name);

/*
 * infer shape of two input and on output with broadcast
 * param[in] op  op desc supply by ge
 * param[in] inputName1  first input name
 * param[in] inputName2  second input name
 * param[in] outputName  output name
 * param[in] is_dynamic  whether the shape of output is dynamic shape
 * return SUCCESS:infer success
 *        FAILED:infer failed like unsupported broadcast input shape
 */

bool InferShapeAndTypeTwoInOneOutBroadcast(Operator& op, const string& input_name1, const string& input_name2,
                                           const string& output_name, bool& is_dynamic);

bool InferShapeRangeTwoInOneOutBroadcase(Operator& op, const string& input_name1, const string& input_name2,
                                         const string& output_name);

bool CheckInputDataType(const Operator& op, std::string* data_type, const std::string& input_name,
                        const std::vector<ge::DataType>& supportList);

std::vector<int64_t> TwoBroadcastShape(const std::vector<int64_t>& dimsX, const std::vector<int64_t>& dimsY);

std::vector<std::pair<int64_t, int64_t>> TwoShapeAndRangeBroadcast(
                        const std::vector<int64_t>& dims_out,
                        const std::vector<std::pair<int64_t, int64_t>>& shape_range_x,
                        std::vector<std::pair<int64_t, int64_t>>& shape_range_y);

/*
 * infer shape of two input and on output with broadcast,Change the original value dimVec and Vec_range.
 * param[in] op  op desc supply by ge
 * param[in] dimVec  Before a shape broadcast value
 * param[in] Vec_range   Before a range broadcast value
 * param[in] dims   The value of broadcast is currently required
 * param[in] range      The value of broadcast is currently required
 * param[in] inputName1  first input name
 * param[in] inputName2  second input name
 */
bool TwoShapeAndRangeBroadcastIntegration(Operator& op, std::vector<int64_t>& dimVec,
                                          std::vector<std::pair<int64_t, int64_t>>& Vec_range,
                                          std::vector<int64_t> dims,
                                          std::vector<std::pair<int64_t, int64_t>> range,
                                          const string& input_name1, const string& input_name2);

bool CheckTwoInputDtypeSame(const Operator& op, const string& input_name1, const string& input_name2);

bool CheckInputDtypeSame(const Operator& op, std::vector<std::string>& input_tensors);

bool CheckInputsShapeDtypeSame(const Operator& op, const std::vector<std::string>& input_names);

bool GetConstValue(const ge::Operator& op, const std::string& key_name, float& attr_value);

bool GetConstValue(const ge::Operator& op, const std::string& key_name, int64_t& attr_value);

bool GetConstValue(const ge::Operator& op, const std::string& key_name, bool& attr_value);

bool GetConstValue(const ge::Operator& op, const std::string& key_name, std::vector<int32_t>& attr_value);

/**
 * Get int type const value from tensor data
 * @param [in] data const tensor data
 * @param [in] data_type DT_INT8, DT_INT16, DT_INT32, DT_INT64
 * @param [out] const_values const int values
 * @return true:success, false:failed.
 */
bool GetConstIntData(const Tensor& data, DataType data_type, std::vector<int64_t>& const_values);

bool GetConstValue(const Operator& op, const Tensor& const_tensor, const DataType& dtype,
                   std::vector<int64_t>& const_data);
bool GetConstValue(const Operator& op, const Tensor& const_tensor, const DataType& dtype,
                   std::vector<uint64_t>& const_data);
bool GetConstValue(const Operator& op, const GeTensorPtr& const_tensor, const DataType& dtype,
                   std::vector<int64_t>& const_data);
bool GetScalerValue(const Operator& op, const Tensor& const_tensor, const DataType& dtype, std::int64_t& const_data);
bool InferShapeAndTypeTwoInOneOutBroadcast(Operator& op, const string& input_name1, const string& input_name2,
                                           const string& output_name);

std::vector<int64_t> GetNewAxis4NewFormat(std::size_t ori_shape_len, int64_t axis, const std::string& ori_format,
                                          const std::string& new_format, bool reduce_mode = false);
std::string ToFormatString(ge::Format format);

/*
 * Check input dtype and format is supported in supportList from inputNumBeg to inputNumEnd
 * param[in] op  op desc supply by ge
 * param[in] inputNumBeg input index begin, [0, N]
 * param[in] inputNumEnd input index end need to be checked
 * param[in] supportList, support type of ge::DataType and ge::Format
 * return true: check pass
 *        false: check failed
 */
template <typename T>
bool CheckSimilarInputDtypeAndFormat(const Operator& op, std::size_t inputNumBeg, std::size_t inputNumEnd,
                                     const std::vector<T>& supportList) {
  for (std::size_t i = inputNumBeg; i < inputNumEnd; i++) {
    if (std::is_same<typename std::decay<T>::type, ge::DataType>::value) {
      ge::DataType inType = op.GetInputDesc(i).GetDataType();
      const auto& findDtype = std::find(supportList.begin(), supportList.end(), inType);
      if (findDtype == supportList.end()) {
        return false;
      }
    } else if (std::is_same<typename std::decay<T>::type, ge::Format>::value) {
      ge::Format inType = op.GetInputDesc(i).GetFormat();
      const auto& findDtype = std::find(supportList.begin(), supportList.end(), inType);
      if (findDtype == supportList.end()) {
        return false;
      }
    }
  }
  return true;
}

/*
 * Check input dtype and format is supported in supportList from inputNumBeg to inputNumEnd
 * param[in] op  op desc supply by ge
 * param[in] indexNeedCheck input index need to be checked
 * param[in] supportList, support type of ge::DataType and ge::Format
 * return true: check pass
 *        false: check failed
 */
template <typename T>
bool CheckSimilarInputDtypeAndFormat(const Operator& op, const std::vector<std::size_t>& indexNeedCheck,
                                     const std::vector<T>& supportList) {
  for (auto i : indexNeedCheck) {
    if (std::is_same<typename std::decay<T>::type, ge::DataType>::value) {
      ge::DataType inType = op.GetInputDesc(i).GetDataType();
      const auto& findDtype = std::find(supportList.begin(), supportList.end(), inType);
      if (findDtype == supportList.end()) {
        return false;
      }
    } else if (std::is_same<typename std::decay<T>::type, ge::Format>::value) {
      ge::Format inType = op.GetInputDesc(i).GetFormat();
      const auto& findDtype = std::find(supportList.begin(), supportList.end(), inType);
      if (findDtype == supportList.end()) {
        return false;
      }
    }
  }
  return true;
}

/*
 * get const attr
 * param[in] op op desc supply by ge
 * param[in] attrName list need to be get
 * param[out] attr vector
 * return true: get success
 *        false: get failed
 */
template <typename T>
bool GetConstAttr(const Operator& op, const std::vector<std::string>& attrNameList, std::vector<T>& attrVec) {
  T value;
  for (auto name : attrNameList) {
    if (op.GetAttr(name, value) != ge::GRAPH_SUCCESS) {
      return false;
    }
    attrVec.push_back(value);
  }
  return true;
}

/*
 * get const attr list
 * param[in] op op desc supply by ge
 * param[in] attrName list need to be get
 * param[out] attr vector
 * return true: get success
 *        false: get failed
 */
template <typename T>
bool GetConstAttr(const Operator& op, const std::vector<std::string>& attrNameList,
                  std::vector<std::vector<T>>& attrListVec) {
  for (auto name : attrNameList) {
    std::vector<T> valueList;
    if (op.GetAttr(name, valueList) != ge::GRAPH_SUCCESS) {
      return false;
    }
    attrListVec.push_back(valueList);
  }
  return true;
}

std::string to_string(const vector<int64_t>& shape);
std::string to_string(const ge::Shape& shape);
std::string to_string(const ge::GeShape& shape);
std::string to_string(const vector<pair<int64_t, int64_t>>& ranges);

class DynamicShapeInfer {
 public:
  std::map<std::string, Format> map_format;
  std::map<std::string, DataType> map_dtype;
  std::map<std::string, uint32_t> inputs;
  std::map<std::string, uint32_t> outputs;
  Operator& op;
  OpDescPtr& op_desc;

  DynamicShapeInfer(Operator& op_v, OpDescPtr& opDesc_v) : op(op_v), op_desc(opDesc_v) {
  }
  bool CatchFormatAndShape();
  bool UpdateFormatAndShape();

  ~DynamicShapeInfer() {
    UpdateFormatAndShape();
  }
};

#define PREPARE_DYNAMIC_SHAPE(depends_names) auto op_desc = OpDescUtils::GetOpDescFromOperator(op);\
    do {                                                    \
      if (!depends_names.empty()) {                         \
        op_desc->SetOpInferDepends(depends_names);          \
      }                                                     \
    } while(0)


bool IsEmptyTensor(const std::vector<int64_t>& dims);

bool IsUnknownRank(const Operator& op, const std::string& tensor_name, const std::string& types = "input");

bool IsUnknownRankShape(const std::vector<int64_t>& shape_vec);

bool IsUnKnownShape(const std::vector<int64_t>& shape_vec);

bool IsUnknownShape(const Operator& op, const std::string& tensor_name, const std::string& types = "input");

bool IsUnknownVec(std::vector<int64_t>& shape_vec);

bool IsUnknown(const std::vector<int64_t>& shape_vec);

void MakeUpShapeRange(const std::vector<int64_t>& shape, std::vector<std::pair<int64_t, int64_t>>& range);

std::string DataTypeToStringDesc(const ge::DataType& dataType);

bool OneInOneOutDynamicInfer(const Operator& op,
                             const std::string& input_name,
                             const std::vector<std::string>& output_name_list);

bool TwoInOneOutDynamicInferNoBroadcast(Operator& op,
                                        const string& input1_name,
                                        const string& input2_name,
                                        const std::vector<string>& output_name_list);

void FixShapeRangeWithDims(const std::vector<int64_t>& dims,
                           std::vector<int64_t>& shape_1,
                           std::vector<int64_t>& shape_2,
                           std::vector<std::pair<int64_t, int64_t>>& range_1,
                           std::vector<std::pair<int64_t, int64_t>>& range_2);

bool SetScalarOutputDesc(const string& input, 
                         const string& output,
                         OpDescPtr op_desc, 
                         GeShape& output_shape);

bool IsEmptyTensor(GeTensorDescPtr tensor_desc);

bool IsEmptyTensor(const GeShape &ge_shape);

std::string RangeToString(const std::vector<std::pair<int64_t, int64_t>> &ranges);

namespace array_ops {
bool CheckInt64MulOverflow(int64_t a, int64_t b);
int64_t CalcMaxElementsCount(const Operator &op, const std::vector<std::pair<int64_t, int64_t>> &x_shape_range,
                             const GeShape &x_shape);
void GenerateWorstYShapeAndYShapeRange(int64_t y_rank, int64_t max_elements_count, 
                                      std::vector<std::pair<int64_t, int64_t>> &y_shape_range, GeShape &y_shape);
bool RepairAndCheckRange(const std::vector<std::pair<int64_t, int64_t>> &x_shape_range,
                         std::vector<std::pair<int64_t, int64_t>> &value_range);
void InferShapeRangeForEmptyTensor(int64_t y_rank, int64_t max_elements_count, 
                                   const std::vector<std::pair<int64_t, int64_t>> &value_range,
                                   std::vector<std::pair<int64_t, int64_t>> &y_shape_range, GeShape &y_shape);
void UpdateDimsAndShapeRange(const Operator &op, int64_t max_elements_count,
                             const std::vector<std::pair<int64_t, int64_t>> &value_range,
                             std::vector<int64_t> &y_dims,
                             std::vector<std::pair<int64_t, int64_t>> &y_shape_range);

void ReshapeRangeInferAllDims(const Operator &op, const std::vector<std::pair<int64_t, int64_t>> &x_shape_range,
                              const GeShape &x_shape,
                              const std::vector<std::pair<int64_t, int64_t>> &shape_value_range, int64_t y_rank,
                              std::vector<std::pair<int64_t, int64_t>> &y_shape_range, GeShape &y_shape);

void ReshapeRangeInfer(const Operator &op, const std::vector<std::pair<int64_t, int64_t>>& x_range, 
                       std::vector<std::pair<int64_t, int64_t>>& y_range, GeShape& output_shape);
}    
}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_UTIL_UTIL_H_
