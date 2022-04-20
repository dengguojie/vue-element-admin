/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
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
 * \file fusion_pre_trans_func.cc
 * \brief fusion pre-transdata and cube node
 */
#include <string>
#include "fusion_pre_trans_func.h"
#include "common/util/platform_info.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_base.h"


namespace fe {
static const string kFusedPreTrans = "FusedTransdataBeforeCube";
static const char kOpTypeTransdata[] = "TransData";
static const std::unordered_map<ge::Format, std::unordered_set<ge::Format>> kSupportTransFormat = {
  {ge::FORMAT_ND, {ge::FORMAT_FRACTAL_NZ}},
  {ge::FORMAT_NHWC, {ge::FORMAT_NC1HWC0, ge::FORMAT_FRACTAL_Z}}
};
static const int64_t kMaxSupportLength = 65536;

void FusePreTransdata(std::vector<ge::NodePtr>& cube_nodes, std::vector<ge::NodePtr>& fusion_nodes) {
  OP_LOGD(kFusedPreTrans.c_str(), "Fusing transdata before cube node begin.");
  // check soc version
  PlatformInfo platform_info;
  OptionalInfo optional_info;
  FUSION_PASS_CHECK(
          PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, optional_info) != SUCCESS,
          OP_LOGW(kFusedPreTrans.c_str(), "Get platform info failed."), return);
  map<string, vector<string>> intrinsic_map = platform_info.ai_core_intrinsic_dtype_map;
  bool support_l0c2out = (intrinsic_map.find("Intrinsic_fix_pipe_l0c2out") != intrinsic_map.end());
  FUSION_PASS_CHECK(!support_l0c2out,
                    OP_LOGD(kFusedPreTrans.c_str(), "Only support Soc with fixpipe unit."), return);

  for (const auto &cube_node : cube_nodes) {
    auto in_nodes = cube_node->GetInDataNodes();
    FUSION_PASS_CHECK(in_nodes.size() < 2,
                      OP_LOGW(kFusedPreTrans.c_str(), "Cube Node should have at least 2 inputs."),
                      return);
    // recognize the transdata nodes
    for (const auto &in_node : in_nodes) {
      if (in_node->GetType() == kOpTypeTransdata &&
          in_node->GetInDataNodes().size() == 1 &&
          in_node->GetOutDataNodesSize() == 1) {
        ge::Format input_format = in_node->GetOpDesc()->GetInputDesc(0).GetFormat();
        ge::Format output_format = in_node->GetOpDesc()->GetOutputDesc(0).GetFormat();
        // only support NHWC -> NC1HWC0 && ND -> FRACTAL_NZ && NHWC -> FRACTAL_Z
        if (kSupportTransFormat.count(input_format) &&
            kSupportTransFormat.at(input_format).count(output_format)) {
          // support fuse transdata when D axis smaller than 65536
          vector<int64_t> input_dims = in_node->GetOpDesc()->GetInputDesc(0).GetShape().GetDims();
          int64_t d_axis_length = input_dims[input_dims.size() - 1];
          if (output_format == ge::FORMAT_FRACTAL_Z) {
            // in this situation, input_format is NHWC
            int64_t h_dim = input_dims[1];
            int64_t w_dim = input_dims[2];
            d_axis_length = d_axis_length * h_dim * w_dim;
          }
          if (d_axis_length < kMaxSupportLength) {
            fusion_nodes.insert(fusion_nodes.begin(), in_node);
          } else {
            OP_LOGW(kFusedPreTrans.c_str(),
                    "Only support fuse transdata when D axis smaller than 65536.");
          }
        }
      }
    }
  }
  OP_LOGD(kFusedPreTrans.c_str(), "Fusing transdata before cube node end.");
}
}  // namespace fe