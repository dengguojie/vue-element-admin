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
 * \file tensorflow_fusion_op_parser_util.h
 * \brief
 */
#ifndef OPS_BUILT_IN_FRAMEWORK_TF_PLUGIN_TENSORFLOW_FUSION_OP_PARSER_UTIL_H_
#define OPS_BUILT_IN_FRAMEWORK_TF_PLUGIN_TENSORFLOW_FUSION_OP_PARSER_UTIL_H_

#include "register/register.h"
#include "proto/tensorflow/attr_value.pb.h"
#include "proto/tensorflow/node_def.pb.h"

namespace domi {
Status ParseParamFromConst(const domi::tensorflow::NodeDef* node_def, int32_t& param, int index = 0);
Status ParseParamFromConst(const domi::tensorflow::NodeDef* node_def, float& param, int index = 0);

class TensorflowFusionOpParserUtil {
 public:
  static bool FindAttrValue(const domi::tensorflow::NodeDef* nodeDef, const string& attr_name,
                            domi::tensorflow::AttrValue& attr_value);
  static Status CheckAttrHasType(const domi::tensorflow::AttrValue& attr_value, const string& type);
  static Status GetTensorFromNode(const domi::tensorflow::NodeDef* node_def, domi::tensorflow::TensorProto& tensor);
};
}  // namespace domi

#endif  // OPS_BUILT_IN_FRAMEWORK_TF_PLUGIN_TENSORFLOW_FUSION_OP_PARSER_UTIL_H_
