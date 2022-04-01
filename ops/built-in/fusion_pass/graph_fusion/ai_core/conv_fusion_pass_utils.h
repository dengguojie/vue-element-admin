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
 * \file conv_fusion_pass_utils.h
 * \brief util function for conv fusion pass
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_CONV_FUSION_PASS_UTILS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_CONV_FUSION_PASS_UTILS_H_

#include <string>
#include <map>
#include <vector>
#include "graph/debug/ge_attr_define.h"
#include "graph/op_desc.h"
#include "graph/utils/graph_utils.h"

namespace fe {
struct GroupDict {
  int64_t real_g;
  int64_t mag_factor;
  int64_t cin1_g;
  int64_t cout_g;
  int64_t cin_ori;
  int64_t cout_ori;
  int64_t groups;
};
class ConvFusionPassUtils {
 public:
  static bool CalculateGroup(int64_t in_channel, int64_t out_channel, int64_t groups, GroupDict& group_dict);

  static bool GetResizeDepthwiseFilter(const std::vector<int64_t>& ori_shape, const ge::Format& format, int groups,
                                       std::vector<int64_t>& resize_shape, std::vector<int64_t>& fractal_shape);

  static bool ReplaceOutputAnchor(const ge::NodePtr& pre_node, uint32_t pre_idx,
                                  const ge::NodePtr& back_node, uint32_t back_idx);

  static ge::OpDescPtr CreateReshape(const string& node_name, const ge::GeTensorDesc& reshape_input_desc,
                                     const ge::GeTensorDesc& reshape_output_desc);

  static int64_t LCM(int64_t numL, int64_t numR);
};
}  // namespace fe
#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_CONV_FUSION_PASS_UTILS_H_