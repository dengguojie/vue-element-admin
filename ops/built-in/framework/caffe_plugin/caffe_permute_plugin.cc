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
 * \file caffe_permute_plugin.cpp
 * \brief
 */
#include <string>
#include "proto/caffe/caffe.pb.h"
#include "register/register.h"
#include "op_log.h"
#include "../../op_proto/util/error_util.h"

namespace domi {
Status ParseParamsPermute(const Message* op_src, ge::Operator& op_dest) {
  OP_LOGI("Permute", "ParseParamsPermute start");
  auto layer = dynamic_cast<const caffe::LayerParameter*>(op_src);
  if (layer == nullptr) {
    OP_LOGE("Permute", "Static cast op_src to LayerParameter failed");
    return FAILED;
  }
  vector<int64_t> orders;
  const caffe::PermuteParameter& permute_param = layer->permute_param();
  if (0 >= permute_param.order_size()) {
    OP_LOGI("Permute", "Permute layer orders size is less 0, set Defeaut value!");
    orders.push_back(static_cast<int64_t>(0));
  } else {
    // new orders
    for (int i = 0; i < permute_param.order_size(); ++i) {
      uint32_t order = permute_param.order(i);
      if (std::find(orders.begin(), orders.end(), order) != orders.end()) {
        ge::OpsAttrValueErrReport(op_dest.GetName(), "order", "unrepeatable", "duplicate, [" + to_string(order) + "]");
        OP_LOGE("Permute", "there are duplicate orders");
        return FAILED;
      }
      orders.push_back(static_cast<int64_t>(order));
    }
  }

  /* Permute Attr */
  const std::string EXP_ATTR_ORDER = "order";
  op_dest.SetAttr(EXP_ATTR_ORDER, orders);
  OP_LOGI("Permute", "--ParseParamsPermute  end--");

  return SUCCESS;
}

// register Permute op info to GE
REGISTER_CUSTOM_OP("Permute")
    .FrameworkType(CAFFE)
    .OriginOpType("Permute")
    .ParseParamsFn(ParseParamsPermute);
}  // namespace domi
