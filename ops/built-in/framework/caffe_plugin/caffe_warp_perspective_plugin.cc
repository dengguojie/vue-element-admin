/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2018. All rights reserved.
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
 * \file caffe_warp_perspective_plugin.cpp
 * \brief
 */
#include "proto/caffe/caffe.pb.h"
#include "register/register.h"
#include "op_log.h"
#include "../../op_proto/util/error_util.h"

namespace domi {
// Caffe ParseParams
Status ParseParams_WarpPerspective(const Message* op_origin, ge::Operator& op_dest) {
  auto layer = dynamic_cast<const caffe::LayerParameter*>(op_origin);

  if (nullptr == layer) {
    OP_LOGE("WarpPerspective", "Dynamic cast op_src to LayerParameter failed.");
    return FAILED;
  }

  const caffe::WarpPerspectiveParameter& param = layer->warp_perspective_param();
  if (param.has_out_height()) {
    op_dest.SetAttr("out_height", static_cast<int>(param.out_height()));
  } else {
    ge::OpsGetCompileParamsErrReport("WarpPerspective", "out_height");
    OP_LOGE("WarpPerspective Get out_height failed.");
    return FAILED;
  }

  if (param.has_out_width()) {
    op_dest.SetAttr("out_width", static_cast<int>(param.out_width()));
  } else {
    ge::OpsGetCompileParamsErrReport("WarpPerspective", "out_width");
    OP_LOGE("WarpPerspective Get out_width failed.");
    return FAILED;
  }

  if (param.has_constant()) {
    op_dest.SetAttr("constant", static_cast<float>(param.constant()));
  }

  if (param.has_border_type()) {
    op_dest.SetAttr("border_type", (string)param.border_type());
  }

  return SUCCESS;
}

REGISTER_CUSTOM_OP("WarpPerspective")
    .FrameworkType(CAFFE)             // Enumerated type. The options are as follows: CAFFE, TENSORFLOW
    .OriginOpType("WarpPerspective")  // // Reduction indicates the type name of the operator in the caffe framework.
    .ParseParamsFn(ParseParams_WarpPerspective)  // AutoMappingFn indicates automatic mapping the parameters of op.
    .ImplyType(ImplyType::TVM);

}  // namespace domi
