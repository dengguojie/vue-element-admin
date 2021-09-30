/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
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
 * \file a_a_matmul_nz_to_nd_fusion_pass.h
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_A_A_MATMUL_NZ_TO_ND_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_A_A_MATMUL_NZ_TO_ND_FUSION_PASS_H_

#include <vector>

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class AAMatMulNzToNdFusionPass : public PatternFusionBasePass {
 protected:
  static const string kNameFusionPass;
  Status DoFusion(ge::ComputeGraph& graph);
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) override;
  bool CheckFormatOfTransData(const ge::NodePtr node_ptr_transdata, const string& expect_src_format,
                              const string& expect_dst_format);
  bool IsAligned();
  bool IsLinkRelationshipCorrect();
  bool IsNumOfNodesCorrect(const ge::ComputeGraph& graph);
  bool IsStaticShape();
  bool NeedFusion(const ge::ComputeGraph& graph);
  vector<FusionPattern*> DefinePatterns() override;
  void RestoreOriginalValues();

 private:
  ge::NodePtr node_ptr_data_0 = nullptr;
  ge::NodePtr node_ptr_transdata_0 = nullptr;
  ge::NodePtr node_ptr_data_1 = nullptr;
  ge::NodePtr node_ptr_transdata_1 = nullptr;
  ge::NodePtr node_ptr_matmul = nullptr;
  ge::NodePtr node_ptr_transdata_out = nullptr;
  ge::NodePtr node_ptr_netoutput = nullptr;

  ge::GeShape in_shape_matmul_0;
  ge::GeShape in_shape_matmul_1;
  ge::GeShape out_shape_matmul_0;
  std::vector<std::pair<int64_t, int64_t>> in_range_matmul_0;
  std::vector<std::pair<int64_t, int64_t>> in_range_matmul_1;
  std::vector<std::pair<int64_t, int64_t>> out_range_matmul_0;
};

const string AAMatMulNzToNdFusionPass::kNameFusionPass = "AAMatMulNzToNdFusionPass";
}  // namespace fe

#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_A_A_MATMUL_NZ_TO_ND_FUSION_PASS_H_
