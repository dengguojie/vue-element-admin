/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.
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
 * \file eltwise_plugin.cpp
 * \brief
 */
#include <string>
#include "proto/caffe/caffe.pb.h"
#include "register/register.h"
#include "graph/utils/op_desc_utils.h"
#include "op_log.h"
#include "../../op_proto/util/error_util.h"

namespace domi {
void PrintfInfo(const caffe::LayerParameter* layer) {
  if (layer->eltwise_param().coeff_size() == 0 || layer->eltwise_param().coeff_size() == layer->bottom_size()) {
    OP_LOGI("Eltwise",
            "Coeff size(%d) check fail,"
            "Eltwise Layer takes one coefficient per bottom blob.",
            layer->eltwise_param().coeff_size());
  }
  if (layer->eltwise_param().operation() != 0 || layer->eltwise_param().coeff_size() == 0) {
    OP_LOGI("Eltwise", "Eltwise layer only takes coefficients for summation.");
  }
}

Status ParseParamsEltwise(const Message* op_src, ge::Operator& op_dest) {
  auto layer = dynamic_cast<const caffe::LayerParameter*>(op_src);
  if (nullptr == layer) {
    OP_LOGE("Eltwise", "Dynamic cast op_src to LayerParameter failed");
    return FAILED;
  }

  PrintfInfo(layer);

  if (layer->has_eltwise_param() && layer->eltwise_param().has_operation()) {
    switch (layer->eltwise_param().operation()) {
      case 0:
        op_dest.SetAttr("mode", static_cast<int64_t>(0));
        break;
      case 1:
        op_dest.SetAttr("mode", static_cast<int64_t>(1));
        break;
      case 2:
        op_dest.SetAttr("mode", static_cast<int64_t>(2));
        break;
      default:
        ge::OpsAttrValueErrReport(op_dest.GetName(), "operation", "0 or 1 or 2", "unsupported");
        OP_LOGE("Eltwise",
                "Eltwise parse params fail,"
                "unsupported opration %d.",
                layer->eltwise_param().operation());
        return PARAM_INVALID;
    }
  } else {
    op_dest.SetAttr("mode", static_cast<int64_t>(1));
  }
  if (layer->eltwise_param().coeff_size() != 0) {
    vector<float> v_coeff;
    for (int i = 0; i < layer->eltwise_param().coeff_size(); ++i) {
      v_coeff.push_back(layer->eltwise_param().coeff(i));
    }
    op_dest.SetAttr("coeff", (v_coeff));
  } else {
    vector<float> v_coeff;
    op_dest.SetAttr("coeff", (v_coeff));
  }
  int n = layer->bottom_size();
  OP_LOGI("Eltwise", "[PLUGIN_Eltwise]--------------bottom_size=%d---------------", n);
  op_dest.SetAttr("N", n);
  std::shared_ptr<ge::OpDesc> op_desc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);
  op_desc->AddDynamicInputDesc("x", n);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("Eltwise")
    .FrameworkType(CAFFE)
    .OriginOpType("Eltwise")
    .ParseParamsFn(ParseParamsEltwise)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
