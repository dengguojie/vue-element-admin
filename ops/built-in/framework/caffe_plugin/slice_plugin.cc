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

/*!
 * \file slice_plugin.cpp
 * \brief
 */
#include "proto/caffe/caffe.pb.h"
#include "register/register.h"
#include "graph/utils/op_desc_utils.h"
#include "op_log.h"
#include "../../op_proto/util/error_util.h"

namespace domi {

Status ParseParamsSlice(const Message* op_src, ge::Operator& op) {
  auto layer = static_cast<const caffe::LayerParameter*>(op_src);
  if (nullptr == layer) {
    OP_LOGE(op.GetName().c_str(), "convert src op failed.");
    return FAILED;
  }

  const caffe::SliceParameter& slice_param = layer->slice_param();

  if (slice_param.has_axis() && slice_param.has_slice_dim()) {
    ge::OpsInputShapeErrReport(op.GetName(), "set either axis or slice_dim should be specified",
                               "axis and slice_dim", "set both");
    OP_LOGE(op.GetName().c_str(), "Either axis or slice_dim should be specified; not both.");
    return FAILED;
  }
  int32_t split_dim;
  if (slice_param.has_axis()) {
    split_dim = slice_param.axis();
    op.SetAttr("split_dim", split_dim);
  } else if (slice_param.has_slice_dim()) {
    split_dim = slice_param.slice_dim();
    op.SetAttr("split_dim", split_dim);
  } else {
    split_dim = static_cast<int32_t>(1);
    op.SetAttr("split_dim", split_dim);
  }
  OP_LOGI(op.GetName().c_str(), "[PLUGIN_Slice]--------------split_dim=%d---------------", split_dim);
  int n = layer->top_size();
  op.SetAttr("num_split", n);
  OP_LOGI(op.GetName().c_str(), "[PLUGIN_Slice]--------------num_split=%d---------------", n);
  std::vector<int64_t> vec;
  int sliceSize = slice_param.slice_point_size();
  OP_LOGI(op.GetName().c_str(), "[PLUGIN_Slice]--------------sliceSize=%d---------------", sliceSize);
  if (0 == sliceSize) {
    op.SetAttr("size_splits", vec);
  } else {
    vec.push_back(slice_param.slice_point(0));
    for (int i = 1; i < sliceSize; ++i) {
      vec.push_back(slice_param.slice_point(i) - slice_param.slice_point(i - 1));
    }
    op.SetAttr("size_splits", vec);
  }
  std::shared_ptr<ge::OpDesc> op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  op_desc->AddDynamicOutputDesc("y", n);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("SplitVD")
    .FrameworkType(CAFFE)             // type: CAFFE, TENSORFLOW
    .OriginOpType("Slice")            // name in caffe module
    .ParseParamsFn(ParseParamsSlice)  // AutoMappingFn for Tensorflow, ParseParamsFn need to realize for caffe
    .ImplyType(ImplyType::TVM);
}  // namespace domi
