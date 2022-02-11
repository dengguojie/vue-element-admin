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
 * \file matmul_transdata_ub_fusion.h
 * \brief matmul + transdata ub fusion pass
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_UB_FUSION_AI_CORE_MATMUL_TRANSDATA_H
#define OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_UB_FUSION_AI_CORE_MATMUL_TRANSDATA_H

#include <vector>

#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_base.h"

namespace fe {
class MatmulTransdataFusionPass : public BufferFusionPassBase {
 public:
  explicit MatmulTransdataFusionPass() {
  }

  ~MatmulTransdataFusionPass() override {
  }

 protected:
  bool CheckFormatOfTransData(const ge::NodePtr& node_ptr_transdata, const char *expect_src_format,
                              const char *expect_dst_format) const;
  vector<BufferFusionPattern*> DefinePatterns() override;
  bool DelInputTransdata(ge::NodePtr& node_ptr_transdata, const uint32_t idx);
  bool DelOutputTransdata() const;
  bool DoFusion();
  Status GetFusionNodes(const BufferFusionMapping& mapping, vector<ge::NodePtr>& fusion_nodes) override;
  bool IsAligned() const;
  void IsOutTransdataCorrect(const ge::Node::Vistor<ge::NodePtr>& out_node_matmuls);
  bool IsLinkRelationshipCorrect();
  bool IsOutOfInTransdataCorrect();
  bool IsStaticShape() const;
  bool ModifyTransdataInControlEdge(const ge::NodePtr& node_ptr_transdata) const;
  bool ModifyTransdataOutControlEdge(const ge::NodePtr& node_ptr_transdata) const;
  bool NeedFusion();

 private:
  ge::NodePtr in_ptr_0 = nullptr;
  ge::NodePtr transdata_ptr_0 = nullptr;
  ge::NodePtr in_ptr_1 = nullptr;
  ge::NodePtr transdata_ptr_1 = nullptr;
  ge::NodePtr matmul_node_ptr = nullptr;
  ge::NodePtr out_transdata_ptr = nullptr;

  void SetSplitInfo(const BufferFusionMapping &mapping, std::vector<ge::NodePtr> &fusion_nodes);
};
}  // namespace fe

#endif  // OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_UB_FUSION_AI_CORE_MATMUL_TRANSDATA_H
