/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef FUSION_ENGINE_INC_OPS_STORE_OP_KERNEL_INFO_H_
#define FUSION_ENGINE_INC_OPS_STORE_OP_KERNEL_INFO_H_
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "graph_optimizer/graph_optimize_register_error_codes.h"
#include "common/aicore_util_types.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "graph/op_desc.h"
#include "graph/types.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/type_utils.h"

namespace fe {
using AttrTypePair = std::pair<std::string, ge::GeAttrValue::ValueType>;
const std::string kReshapeTypeForbidden = "FORBIDDEN";
const std::string kReshapeTypeDefault = "DEFAULT";


class InputOrOutputInfo {
  /* Set OpsKernelInfoConstructor as a friend class of InputOrOutputInfo to
   * conveniently set the private attributes of InputOrOutputInfo.
   */
  friend class OpKernelInfoConstructor;

 public:
  explicit InputOrOutputInfo(std::string name);
  ~InputOrOutputInfo();

  const std::string GetName() const;
  const bool& GetIsInput();
  bool GetNeedCompile();
  uint32_t GetIndex() const;
  std::vector<ge::DataType> GetDataType();
  std::vector<ge::Format> GetFormat();
  std::vector<ge::DataType> GetUnknownShapeDataType();
  std::vector<ge::Format> GetUnknownShapeFormat();
  std::vector<ge::GeShape> GetShape();
  OpParamType GetParamType() const;
  OpConstValueDepend GetConstValueDepend() const;
  std::string GetReshapeType();
  void SetReshapeType(std::string reshape_type);
  const string GetUniqueName();

 private:
  bool is_input_;    /* whether it's input or output. true means input */
  std::string name_; /* name of input or output */
  std::string unique_name_;

  /* The index of input or output; Index is generated from json files,
   * like "input0", "input1" ....If the number after input is not consecutive or absent,
   * program will report error when loading op kernel information
   */
  uint32_t index_;

  /* Reshape type is a method which is used when we transfer this input into 4D if it is
   * is 1,2,3D or more than 6D. */
  std::string reshape_type_;

  /* Reshape type which needed to be transfered which is used when we transfer
   * this input into 4D if it is 1,2,3D or more than 6D. */
  std::string propagate_reshape_type_;

  OpParamType op_param_type_; /* Type of input or output, can be one of {dynamic, required, optional} */

  OpConstValueDepend op_const_value_depend_; /* Whether the node of other end of input or output is const or constant. */

  /* Data formats that are supported by this op. The size of supported_formats_
   * should strictly be the same as supported_dtypes_. If not, program will report error
   * when initialize this op.
   */
  std::vector<ge::Format> supported_formats_;

  /* Data types that are supported by this op. There not exists a flag
   * called "all_data_types_supported", because that does not make sense.
   */
  std::vector<ge::DataType> supported_dtypes_;

  /* Data formats that are supported by this op. The size of supported_formats_
   * should strictly be the same as supported_dtypes_. If not, program will report error
   * when initialize this op.
   */
  std::vector<ge::Format> supported_unknown_shape_formats_;

  /* Data types that are supported by this op. There not exists a flag
   * called "all_data_types_supported", because that does not make sense.
   */
  std::vector<ge::DataType> supported_unknown_shape_dtypes_;
};

class AttrInfo {
  /* Set OpsKernelInfoConstructor as a friend class of AttrInfo to
   * conveniently set the private attributes of AttrInfo.
   */
  friend class OpKernelInfoConstructor;

 public:
  explicit AttrInfo(std::string attr_name);
  ~AttrInfo();

  const std::string& GetAttrName() const;

  const ge::GeAttrValue::ValueType& GetAttrDType() const;

  const bool& GetSupportAllValue() const;

  bool GetIsRequired() const;

  const std::vector<ge::GeAttrValue>& GetSupportedAttrValueVector() const;

  const bool& GetDefaultValueDefinedFlag() const;

  const ge::GeAttrValue& GetDefaultValue() const;

 private:
  std::string attr_name_;
  /* Whether this attribute is necessary or not. If is_required_ = true,
   * this attribute is required. On the contrary, it is optional.
   * For optional attribute, if we failed to find this attribute is ops_desc,
   * we will loop it up from default_value_. If is_default_value_defined_ = false,
   * the program will report error when doing operation "CheckSupport".
   */
  bool is_required_;

  ge::GeAttrValue::ValueType dtype_;

  /* is_support_all_value_ tells Whether this attribute supports all kinds of value.
   * If is_support_all_value_ = false, all supported values are stored in vector supported_values_.
   */
  bool is_support_all_value_;

  std::vector<ge::GeAttrValue> supported_values_;

  bool is_default_value_defined_;

  /* If default value is not set, is_default_value_defined_ will be false.
   * In that situation, default_value_ will automatically calls its default constructor and will
   * be constructed as an unknown value. At the same time, it is not appropriate to assign a
   * meaningless default value to default_value_ because the meaningless one will be used int some
   * special cases. So we introduce a flag called "isDefaultValueDefined_" to explicitly elucidate
   * whether default_value_ is set by the user.
   */
  ge::GeAttrValue default_value_;
};

struct OpStoreInfo {
  bool precision_reduce;             // op precision policy
  PrecisionPolicy precision_policy;  // precision policy : white, black, gray
};

class OpKernelInfo;
using OpKernelInfoPtr = std::shared_ptr<OpKernelInfo>;
using InputOrOutputInfoPtr = std::shared_ptr<InputOrOutputInfo>;
using AttrInfoPtr = std::shared_ptr<AttrInfo>;

class OpKernelInfo {
  /* We separate the definition and initialization in two classes because
   * there exists more than one method to initialize op kernel.
   * Set OpsKernelInfoConstructor as a friend class of OpKernelInfo to
   * conveniently set the private attributes of OpKernelInfo.
   */
  friend class OpKernelInfoConstructor;
  friend class SubOpsStore;

 public:
  explicit OpKernelInfo(std::string op_type);
  ~OpKernelInfo();

  std::string GetOpType() const;

  /* Now all AttrInfo are stored in a vector, so we offer a find function(loop in vector)
   * for specific attribute with name attr_name to get its Attr Value Type or Attr
   * Value
   */
  const Status GetAttrTypeByAttrName(const std::string& attr_name, ge::GeAttrValue::ValueType& dtype) const;
  const Status GetAttrValueByAttrName(const std::string& attr_name, std::vector<ge::GeAttrValue>& attr_value) const;

  /* In other cases, we only provide an interface to get vector of whole attr information.
   * Outside callers can do loop in this vector and get information. */
  const std::vector<AttrInfoPtr> GetVecAttrInfo() const;

  ge::OpInfo GetOpInfo();
  OpStoreInfo GetOpStoreInfo();
  OpStoreInfo GetOpStoreInfoBf16();

  const std::vector<InputOrOutputInfoPtr>& GetAllInputInfo() const;
  const std::vector<InputOrOutputInfoPtr>& GetAllOutputInfo() const;
  /* The following function is an interface for operations
   * which is the same for input and output. */
  Status GetTensorInfoByName(const bool& isinput, const string& tensor_name,
                             InputOrOutputInfoPtr& tensor_info_ptr);

  Status GetInputInfoByName(const std::string& input_name, InputOrOutputInfoPtr& input_info_ptr) const;
  Status GetOutputInfoByName(const std::string& output_name, InputOrOutputInfoPtr& output_info_ptr) const;
  bool GetHeavyOpFlag() const;
  bool GetInputMemContinuesFlag() const;
  bool GetOutputMemContinuesFlag() const;
  OpImplType GetOpStoreImplType() const;
  OpPattern GetOpPattern() const;
  std::string GetOpImpPath() const;
  void SetOpImpPath(std::string op_imp_path);
  void SetImplType(const OpImplType impl_type);
  bool GetSupportDynamicShape() const;
  void SetSupportDynamicShape(const bool is_support_dynamic_shape);
  bool GetSupportDynamicRank() const;
  bool GetNeedCheckSupportFlag() const;
  bool GetFlagDynamicCompileStatic() const;
  std::string GetRangeLimitValue() const;
  SlicePattern GetOpSlicePattern() const;
  CoreType GetCoreType() const;

 private:
  bool init_flag_;
  std::string op_type_;
  ge::OpInfo op_info_;
  std::vector<InputOrOutputInfoPtr> input_infos_;
  std::vector<InputOrOutputInfoPtr> output_infos_;
  std::vector<AttrInfoPtr> attrs_info_;
  bool must_check_support_flag_; /* Indicate whether this op must to be Checked */
  bool need_check_support_;
  OpImplType impl_type_;
  bool is_heavy_o_p_;
  OpPattern op_pattern_ = OP_PATTERN_OP_KERNEL;
  OpStoreInfo op_store_info_;
  OpStoreInfo op_store_info_bf16_;
  bool input_mem_continues_ = false;
  bool output_mem_continues_ = false;
  std::string op_imp_path_;
  bool is_support_dynamic_shape_ = false;
  bool is_support_dynamic_rank_ = false;
  SlicePattern slice_pattern_;
  bool dynamic_compile_static_ = false;
  CoreType core_type_ = CoreType::VECTOR_CORE;
  std::string is_limited_range_;
};
}  // namespace fe
#endif  // FUSION_ENGINE_INC_OPS_STORE_OP_KERNEL_INFO_H_
