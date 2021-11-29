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
 * \file batch_matmul_v2_non_aligned_fusion_pass.h
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_BATCH_MATMUL_V2_NON_ALIGNED_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_BATCH_MATMUL_V2_NON_ALIGNED_FUSION_PASS_H_

#include <vector>

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class BatchMatMulNonAlignedFusionPass : public PatternFusionBasePass {
 protected:
  static const string kNameFusionPass;
  Status Fusion(ge::ComputeGraph &graph, Mapping &mapping, vector<ge::NodePtr> &fusion_nodes) override;
  vector<FusionPattern *> DefinePatterns() override;

 private:
  ge::NodePtr batchmatmul_1_node = nullptr;
  ge::NodePtr batchmatmul_2_node = nullptr;
  ge::NodePtr batchmatmul_3_node = nullptr;
  ge::NodePtr transpose_1_node = nullptr;
  ge::NodePtr transpose_2_node = nullptr;
  ge::NodePtr reshape_1_node = nullptr;
  ge::NodePtr reshape_2_node = nullptr;
  ge::NodePtr add_1_node = nullptr;
  ge::NodePtr add_2_node = nullptr;

  Status CreatePadDNode(ge::ComputeGraph &graph, const ge::OutDataAnchorPtr &out_anchor, const vector<int64_t> &shape,
                        ge::NodePtr &pad_node, vector<vector<int64_t>> &paddings);
  Status CreateReshapeNode(ge::ComputeGraph &graph, const ge::OutDataAnchorPtr &out_anchor,
                           const vector<int64_t> &shape, ge::NodePtr &shape_node);
  ge::OpDescPtr CreateListConstDesc(const string &name, vector<int64_t> values);
  Status CreateReshapePadReshape(ge::ComputeGraph &graph, const ge::InDataAnchorPtr &dst_anchor,
                                 map<string, vector<int64_t>> &shape_dict, vector<vector<int64_t>> &paddings);
  Status UpdateConst(ge::NodePtr &shape_node, vector<int64_t> &const_shape);
  Status UpdateAllShape(ge::NodePtr &cur_node, ge::NodePtr &end_node);
  Status GetNodes(const Mapping &mapping);
  Status DoFusionPattern1(ge::ComputeGraph &graph);
  Status DoFusionPattern2(ge::ComputeGraph &graph);
  Status CheckStaticShape();
  Status CheckNodeShape(ge::NodePtr &node);
  Status CheckPerm(ge::NodePtr &transpose_node, vector<int64_t> &perm_list);
  Status CheckBatchMatMul();
  Status CheckInsertLocPattern1();
  Status CheckInsertLocPattern2();
  Status CheckReshapePattern1();
  Status CheckReshapePattern2();
};

const string BatchMatMulNonAlignedFusionPass::kNameFusionPass = "BatchMatMulNonAlignedFusionPass";
}  // namespace fe

#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_BATCH_MATMUL_V2_NON_ALIGNED_FUSION_PASS_H_