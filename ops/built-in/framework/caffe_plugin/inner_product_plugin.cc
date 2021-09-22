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
 * \file inner_product_plugin.cpp
 * \brief
 */
#include "proto/caffe/caffe.pb.h"
#include "register/register.h"
#include "op_log.h"
#include "../../op_proto/util/error_util.h"

namespace domi {
// #### Set param in attr for transfer
Status ParseParamsInnerProduct(const Message* op_src, ge::Operator& op_dest) {
  OP_LOGI("InnerProduct", "Start into the ParseParamsInnerProduct!");
  auto layer = dynamic_cast<const caffe::LayerParameter*>(op_src);

  if (layer == nullptr) {
    OP_LOGE("InnerProduct", "Dynamic cast op_src to LayerParameter failed");
    return FAILED;
  }

  const caffe::InnerProductParameter& inner_product_param = layer->inner_product_param();

  // Parse num_output
  if (!inner_product_param.has_num_output()) {
    ge::OpsGetAttrErrReport("InnerProduct", "num_output");
    OP_LOGE("InnerProduct", "Parse num_output for %s failed.", layer->name().c_str());
    return PARAM_INVALID;
  } else {
    int32_t num_output = static_cast<uint32_t>(inner_product_param.num_output());
    op_dest.SetAttr("num_output", num_output);
  }

  // Parse axis info
  if (!inner_product_param.has_axis()) {
    op_dest.SetAttr("axis", static_cast<int64_t>(1));
  } else {
    int32_t axis = static_cast<uint32_t>(inner_product_param.axis());
    op_dest.SetAttr("axis", axis);
  }

  // Parse transpose
  if (!inner_product_param.has_transpose()) {
    op_dest.SetAttr("transpose", false);
  } else {
    bool transpose = inner_product_param.transpose();
    op_dest.SetAttr("transpose", transpose);
  }

  op_dest.SetAttr("alpha", static_cast<int64_t>(1));
  op_dest.SetAttr("beta", static_cast<int64_t>(0));

  OP_LOGI("InnerProduct", "End of the arseParamsInnerProduct!");

  return SUCCESS;
}

REGISTER_CUSTOM_OP("FullyConnection")
    .FrameworkType(CAFFE)                    // type: CAFFE, TENSORFLOW
    .OriginOpType("InnerProduct")            // name in caffe module
    .ParseParamsFn(ParseParamsInnerProduct)  // AutoMappingFn for Tensorflow, ParseParamsFn need to realize for caffe
    .ImplyType(ImplyType::TVM);
}  // namespace domi
