/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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

#ifndef FUSION_ENGINE_INC_OPS_STORE_OP_KERNEL_INFO_CONSTRUCTOR_H_
#define FUSION_ENGINE_INC_OPS_STORE_OP_KERNEL_INFO_CONSTRUCTOR_H_
#include <map>
#include <memory>
#include <string>
#include "graph_optimizer/graph_optimize_register_error_codes.h"
#include "ops_store/op_kernel_info.h"

namespace fe {
struct OpContent {
  // optype
  std::string op_type_;

  // opType content map
  std::map<std::string, std::map<std::string, std::string>> map_kernel_info_;
};

class OpKernelInfoConstructor {
 public:
  OpKernelInfoConstructor();
  ~OpKernelInfoConstructor();
  Status InitializeOpKernelInfo(std::string engine_name, const OpContent &op_content, OpKernelInfoPtr op_kernel_info);
  Status FinalizeOpKernelInfo(OpKernelInfoPtr op_kernel_info) const;

  Status GetStrFromOpContent(const OpContent &op_content, const std::string &key1,
                             const std::string &key2, std::string &value) const;

 private:
  Status ParseBasicParameter(const OpContent &op_content, OpKernelInfoPtr op_kernel_info) const;

  Status ParseInputAndOutputFromOpContent(const OpContent &op_content, OpKernelInfoPtr op_kernel_info);

  Status InitFormatAndDtypeForSingleInputAndOutput(OpPattern op_pattren, const std::map<string, string> &map_info,
                                                   const InputOrOutputInfoPtr &input_or_output_info,
                                                   OpKernelInfoPtr op_kernel_info,
                                                   uint32_t &dtype_and_format_size_of_first_input);

  Status InitializeInputAndOutput(OpPattern op_pattren, const std::string &op_type,
                                  const std::map<std::string, std::string> &map_info, uint32_t index,
                                  InputOrOutputInfoPtr input_or_output_info,
                                  uint32_t &dtype_and_format_size_of_first_input, OpKernelInfoPtr op_kernel_info);
  Status FinalizeInputAndOutput(InputOrOutputInfoPtr input_or_output_info) const;

  Status InitDtypeAndFormat(const std::map<string, string> &map_info, InputOrOutputInfoPtr input_or_output_info,
                            uint32_t &dtype_and_format_size_of_first_input);

  Status InitUnknownFormatAndDtype(const std::map<string, string> &map_info, InputOrOutputInfoPtr input_or_output_info,
                                   uint32_t &dtype_and_format_size_of_first_input);

  Status InitDtypeAndAllFormat(const std::map<string, string> &map_info, InputOrOutputInfoPtr input_or_output_info,
                               uint32_t &dtype_of_first_input, OpKernelInfoPtr op_kernel_info);

  Status InitDtype(const std::map<string, string> &map_info, InputOrOutputInfoPtr input_or_output_info,
                   uint32_t &dtype_and_format_size_of_first_input);

  Status GetStrFormMap(const std::map<std::string, std::string> &map_info, std::string key, std::string &value);

  /* Convert listed attribute value from a long string to a 2D-Vector */
  template <typename T>
  Status ConvertListAttrValue(const OpContent &op_content, std::string attr_name,
                              std::vector<std::vector<T>> &list_attr_vec, AttrInfoPtr attr_info) const;

  template <typename T>
  Status InitAttrTemplate(const OpContent &op_content, std::string attr, AttrInfoPtr attr_info) const;

  template <typename T>
  Status InitAttrListTemplate(const OpContent &op_content, std::string attr, AttrInfoPtr attr_info);

  Status InitAttrValue(const OpContent &op_content, OpKernelInfoPtr op_kernel_info);
  Status InitAttrValueSub(const OpContent &op_content, OpKernelInfoPtr op_kernel_info);

  Status InitOpInfo(std::string engine_name, const OpContent &op_content, OpKernelInfoPtr op_kernel_info);

  Status GetPrecisionPolicyFromOpContent(const OpContent &op_content, const OpKernelInfoPtr &op_kernel_info) const;

  Status InitShape(const string &op_type, const std::map<string, string> &map_info,
                   InputOrOutputInfoPtr input_or_output_info);

  Status CheckFormatAgnosticOp(OpKernelInfoPtr op_kernel_info) const;

  void SetUniqueName(InputOrOutputInfoPtr input_or_output_info_ptr) const;

  Status InitDtypeByPattern(const std::map<string, string> &map_info, InputOrOutputInfoPtr input_or_output_info,
                            uint32_t &dtype_and_format_size_of_first_input, const OpPattern &op_pattern);

  Status InitSlicePattern(const OpContent &op_content, OpKernelInfoPtr op_kernel_info) const;

  void ParserReferTensorNameVec(OpKernelInfoPtr op_kernel_info) const;
};

}  // namespace fe

#endif  // FUSION_ENGINE_INC_OPS_STORE_OP_KERNEL_INFO_CONSTRUCTOR_H_
