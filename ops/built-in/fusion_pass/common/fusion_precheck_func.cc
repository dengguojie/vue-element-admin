/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
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
 * \file fusion_precheck_func.cpp
 * \brief precheck functions.
 */
#include "fusion_precheck_func.h"
#include "op_log.h"

namespace fe {
Status ApplyRmsPropPreCheck(ge::NodePtr node) {
  OP_LOGI(node->GetType().c_str(), "Current Node name is :%s", node->GetName().c_str());
  ge::OpDescPtr apply_rms_prop_op = node->GetOpDesc();

  if (apply_rms_prop_op->GetInputDesc("var").GetDataType() != ge::DT_FLOAT) {
    return NOT_CHANGED;
  }

  return SUCCESS;
}

Status FusedMulApplyMomentumPreCheck(ge::NodePtr node) {
  OP_LOGI(node->GetType().c_str(), "Current Node name is :%s", node->GetName().c_str());
  ge::OpDescPtr fused_mul_apply_momentum_op = node->GetOpDesc();

  if (fused_mul_apply_momentum_op->GetInputDesc("var").GetDataType() != ge::DT_FLOAT &&
      fused_mul_apply_momentum_op->GetInputDesc("var").GetDataType() != ge::DT_FLOAT16) {
    return NOT_CHANGED;
  }
  return SUCCESS;
}

Status FusedMulApplyMomentumExternPreCheck(ge::NodePtr node) {
  OP_LOGI(node->GetType().c_str(), "Current Node name is :%s", node->GetName().c_str());
  ge::OpDescPtr fused_mul_apply_momentum_extern_op = node->GetOpDesc();

  if (fused_mul_apply_momentum_extern_op->GetInputDesc("var").GetDataType() != ge::DT_FLOAT) {
    return NOT_CHANGED;
  }
  return SUCCESS;
}

Status FusedMulApplyKerasMomentumPreCheck(ge::NodePtr node) {
  OP_LOGI(node->GetType().c_str(), "Current Node name is :%s", node->GetName().c_str());
  ge::OpDescPtr fused_mul_apply_keras_momentum_op = node->GetOpDesc();

  if (fused_mul_apply_keras_momentum_op->GetInputDesc("var").GetDataType() != ge::DT_FLOAT) {
    return NOT_CHANGED;
  }
  return SUCCESS;
}

Status SparseApplyRmsPropPreCheck(ge::NodePtr node) {
  OP_LOGI(node->GetType().c_str(), "Current Node name is :%s", node->GetName().c_str());
  ge::OpDescPtr apply_rms_prop_op = node->GetOpDesc();

  if (apply_rms_prop_op->GetInputDesc("var").GetDataType() != ge::DT_FLOAT) {
    return NOT_CHANGED;
  }

  return SUCCESS;
}

Status ApplyAdagradV2PreCheck(ge::NodePtr node) {
  OP_LOGI(node->GetType().c_str(), "Current Node name is :%s", node->GetName().c_str());
  ge::OpDescPtr apply_adagradv2_op = node->GetOpDesc();

  if (apply_adagradv2_op->GetInputDesc("var").GetDataType() != ge::DT_FLOAT &&
      apply_adagradv2_op->GetInputDesc("var").GetDataType() != ge::DT_FLOAT16) {
    return NOT_CHANGED;
  }
  return SUCCESS;
}

Status ApplyKerasMomentumPreCheck(ge::NodePtr node) {
  OP_LOGI(node->GetType().c_str(), "Current Node name is :%s", node->GetName().c_str());
  ge::OpDescPtr current_op = node->GetOpDesc();

  if (current_op->GetInputDesc("var").GetDataType() != ge::DT_FLOAT &&
      current_op->GetInputDesc("var").GetDataType() != ge::DT_FLOAT16) {
    return NOT_CHANGED;
  }
  return SUCCESS;
}

Status SparseApplyFtrlPreCheck(ge::NodePtr node) {
  OP_LOGI(node->GetType().c_str(), "Current Node name is :%s", node->GetName().c_str());
  ge::OpDescPtr ftrl_op = node->GetOpDesc();

  if (ftrl_op->GetInputDesc("var").GetDataType() != ge::DT_FLOAT) {
    return NOT_CHANGED;
  }
  return SUCCESS;
}

Status SparseApplyFtrlV2PreCheck(ge::NodePtr node) {
  OP_LOGI(node->GetType().c_str(), "Current Node name is :%s", node->GetName().c_str());
  ge::OpDescPtr ftrlv2_op = node->GetOpDesc();

  if (ftrlv2_op->GetInputDesc("var").GetDataType() != ge::DT_FLOAT) {
    return NOT_CHANGED;
  }
  return SUCCESS;
}

Status SparseApplyAdagradV2PreCheck(ge::NodePtr node) {
  OP_LOGI(node->GetType().c_str(), "Current Node name is :%s", node->GetName().c_str());
  ge::OpDescPtr adagradv2_op = node->GetOpDesc();

  if (adagradv2_op->GetInputDesc("var").GetDataType() != ge::DT_FLOAT) {
    return NOT_CHANGED;
  }
  return SUCCESS;
}

Status SparseApplyAdadeltaPreCheck(ge::NodePtr node) {
  OP_LOGI(node->GetType().c_str(), "Current Node name is :%s", node->GetName().c_str());
  ge::OpDescPtr adadelta_op = node->GetOpDesc();

  if (adadelta_op->GetInputDesc("var").GetDataType() != ge::DT_FLOAT) {
    return NOT_CHANGED;
  }
  return SUCCESS;
}
}  // namespace fe
