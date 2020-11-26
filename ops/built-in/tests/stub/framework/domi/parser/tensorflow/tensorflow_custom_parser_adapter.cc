/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "parser/tensorflow/tensorflow_custom_parser_adapter.h"
#include "common/util.h"
#include "framework/common/debug/ge_log.h"
#include "parser/common/op_parser_factory.h"
#include "register/op_registry.h"

using domi::ParseParamFunc;

namespace ge {
Status TensorFlowCustomParserAdapter::ParseParams(const Message *op_src, ge::OpDescPtr &op_dest) {
  GE_CHECK_NOTNULL(op_src);
  const NodeDef *node_src = DOMI_DYNAMIC_CAST<const NodeDef *>(op_src);
  GE_CHECK_NOTNULL(node_src);
  GELOGD("TF op node name = %s, op type= %s, parse params", node_src->name().c_str(), node_src->op().c_str());
  GE_CHECK_NOTNULL(op_dest);

  ParseParamFunc custom_op_parser = domi::OpRegistry::Instance()->GetParseParamFunc(op_dest->GetType(), node_src->op());
  GE_CHECK_NOTNULL(custom_op_parser);
  ge::Operator op = ge::OpDescUtils::CreateOperatorFromOpDesc(op_dest);
  GE_CHK_BOOL_RET_STATUS(custom_op_parser(op_src, op) == SUCCESS, FAILED, "Custom parser params failed");

  op.BreakConnect();

  return SUCCESS;
}
}  // namespace ge
