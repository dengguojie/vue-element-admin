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
 * \file spatial_transformer_d_plugin.cpp
 * \brief
 */
#include "op_log.h"
#include "proto/caffe/caffe.pb.h"
#include "register/register.h"
#include "graph/operator.h"

namespace domi {
const int DEFAULT_THETA_INDEX_TWO = 2;
const int DEFAULT_THETA_INDEX_THREE = 3;
const int DEFAULT_THETA_INDEX_FOUR = 4;
const int DEFAULT_THETA_INDEX_FIVE = 5;

Status ParseParamsSpatialTransformer(const Message* op_src, ge::Operator& op_dest) {
  OP_LOGI("SpatialTransformer", "[PLUGIN_STN]------------ParseParams SpatialTransformer Start---------------");
  auto layer = dynamic_cast<const caffe::LayerParameter*>(op_src);

  if (layer == nullptr) {
    OP_LOGE("SpatialTransformer", "Dynamic cast op_src to LayerParameter failed.");
    return FAILED;
  }

  const caffe::SpatialTransformerParameter& param = layer->st_param();

  // set output size
  std::vector<int64_t> output_size = {-1, -1};
  if (param.has_output_h()) {
    output_size[0] = param.output_h();
  }
  if (param.has_output_w()) {
    output_size[1] = param.output_w();
  }
  op_dest.SetAttr("output_size", (output_size));

  // set default_theta and use_default_theta
  std::vector<float> default_theta;
  std::vector<bool> use_default_theta = {false, false, false, false, false, false};
  if (param.has_theta_1_1()) {
    default_theta.push_back(param.theta_1_1());
    use_default_theta[0] = true;
  }
  if (param.has_theta_1_2()) {
    default_theta.push_back(param.theta_1_2());
    use_default_theta[1] = true;
  }
  if (param.has_theta_1_3()) {
    default_theta.push_back(param.theta_1_3());
    use_default_theta[DEFAULT_THETA_INDEX_TWO] = true;
  }
  if (param.has_theta_2_1()) {
    default_theta.push_back(param.theta_2_1());
    use_default_theta[DEFAULT_THETA_INDEX_THREE] = true;
  }
  if (param.has_theta_2_2()) {
    default_theta.push_back(param.theta_2_2());
    use_default_theta[DEFAULT_THETA_INDEX_FOUR] = true;
  }
  if (param.has_theta_2_3()) {
    default_theta.push_back(param.theta_2_3());
    use_default_theta[DEFAULT_THETA_INDEX_FIVE] = true;
  }
  op_dest.SetAttr("default_theta", (default_theta));
  op_dest.SetAttr("use_default_theta", (use_default_theta));

  // set align_corners
  op_dest.SetAttr("align_corners", false);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("SpatialTransformerD")
    .FrameworkType(CAFFE)
    .OriginOpType("SpatialTransformer")
    .ParseParamsFn(ParseParamsSpatialTransformer)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
