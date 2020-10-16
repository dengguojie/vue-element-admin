/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief precheck functions.
 *
 * @version 1.0
 *
 */
#include "fusion_precheck_func.h"
#include "op_log.h"

namespace fe {
bool ApplyRmsPropPreCheck(ge::NodePtr node) {
  OP_LOGI(node->GetType().c_str(), "Current Node name is :%s", node->GetName().c_str());
  ge::OpDescPtr apply_rms_prop_op = node->GetOpDesc();

  if (apply_rms_prop_op->GetInputDesc("var").GetDataType() != ge::DT_FLOAT) {
    return false;
  }

  return true;
}

bool FusedMulApplyMomentumPreCheck(ge::NodePtr node) {
  OP_LOGI(node->GetType().c_str(), "Current Node name is :%s", node->GetName().c_str());
  ge::OpDescPtr fused_mul_apply_momentum_op = node->GetOpDesc();

  if (fused_mul_apply_momentum_op->GetInputDesc("var").GetDataType() !=
          ge::DT_FLOAT &&
      fused_mul_apply_momentum_op->GetInputDesc("var").GetDataType() !=
          ge::DT_FLOAT16) {
    return false;
  }
  return true;
}

bool FusedMulApplyMomentumExternPreCheck(ge::NodePtr node) {
  OP_LOGI(node->GetType().c_str(), "Current Node name is :%s", node->GetName().c_str());
  ge::OpDescPtr fused_mul_apply_momentum_extern_op = node->GetOpDesc();

  if (fused_mul_apply_momentum_extern_op->GetInputDesc("var").GetDataType() !=
      ge::DT_FLOAT) {
    return false;
  }
  return true;
}

bool SparseApplyRmsPropPreCheck(ge::NodePtr node) {
  OP_LOGI(node->GetType().c_str(), "Current Node name is :%s", node->GetName().c_str());
  ge::OpDescPtr apply_rms_prop_op = node->GetOpDesc();

  if (apply_rms_prop_op->GetInputDesc("var").GetDataType() != ge::DT_FLOAT) {
    return false;
  }

  return true;
}

bool ApplyAdagradV2PreCheck(ge::NodePtr node) {
  OP_LOGI(node->GetType().c_str(), "Current Node name is :%s", node->GetName().c_str());
  ge::OpDescPtr apply_adagradv2_op = node->GetOpDesc();

  if (apply_adagradv2_op->GetInputDesc("var").GetDataType() != ge::DT_FLOAT &&
      apply_adagradv2_op->GetInputDesc("var").GetDataType() != ge::DT_FLOAT16) {
    return false;
  }
  return true;
}

bool ApplyKerasMomentumPreCheck(ge::NodePtr node) {
  OP_LOGI(node->GetType().c_str(), "Current Node name is :%s", node->GetName().c_str());
  ge::OpDescPtr current_op = node->GetOpDesc();

  if (current_op->GetInputDesc("var").GetDataType() != ge::DT_FLOAT &&
      current_op->GetInputDesc("var").GetDataType() != ge::DT_FLOAT16) {
    return false;
  }
  return true;
}

bool SparseApplyFtrlPreCheck(ge::NodePtr node) {
  OP_LOGI(node->GetType().c_str(), "Current Node name is :%s", node->GetName().c_str());
  ge::OpDescPtr ftrl_op = node->GetOpDesc();

  if (ftrl_op->GetInputDesc("var").GetDataType() != ge::DT_FLOAT) {
    return false;
  }
  return true;
}

bool SparseApplyFtrlV2PreCheck(ge::NodePtr node) {
  OP_LOGI(node->GetType().c_str(), "Current Node name is :%s", node->GetName().c_str());
  ge::OpDescPtr ftrlv2_op = node->GetOpDesc();

  if (ftrlv2_op->GetInputDesc("var").GetDataType() != ge::DT_FLOAT) {
    return false;
  }
  return true;
}

bool SparseApplyAdagradV2PreCheck(ge::NodePtr node) {
  OP_LOGI(node->GetType().c_str(), "Current Node name is :%s", node->GetName().c_str());
  ge::OpDescPtr adagradv2_op = node->GetOpDesc();

  if (adagradv2_op->GetInputDesc("var").GetDataType() != ge::DT_FLOAT) {
    return false;
  }
  return true;
}

bool SparseApplyAdadeltaPreCheck(ge::NodePtr node) {
  OP_LOGI(node->GetType().c_str(), "Current Node name is :%s", node->GetName().c_str());
  ge::OpDescPtr adadelta_op = node->GetOpDesc();

  if (adadelta_op->GetInputDesc("var").GetDataType() != ge::DT_FLOAT) {
    return false;
  }
  return true;
}
}  // namespace fe
