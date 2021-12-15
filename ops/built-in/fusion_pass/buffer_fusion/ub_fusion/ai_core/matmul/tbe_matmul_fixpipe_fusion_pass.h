/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_UB_FUSION_AI_CORE_MATMUL_TBE_MATMUL_FIXPIPE_FUSION_PASS_H
#define OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_UB_FUSION_AI_CORE_MATMUL_TBE_MATMUL_FIXPIPE_FUSION_PASS_H

#include <vector>
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_base.h"


namespace fe {

class TbeMatmulFixpipeFusionPass : public BufferFusionPassBase {
 public:
  explicit TbeMatmulFixpipeFusionPass() {}

  ~TbeMatmulFixpipeFusionPass() override {}

 protected:
  /*
   * @brief: define transdata_cube fusion pattern
   *
   *  (Transdata)? + Cube + (Dequent/Quent/Requent)? + (Elemwise)? + (Transdata)?
   *  pattern limit:
   *          1.Transdata,Dequent/Quent/Requent,Elemwise are optional,Cube is required.
   *          2.Elemwise supports LeakyRelu,Relu,PRelu
   *          3.Cube supports Matmul,Conv2d,Conv_dx,Conv_dw.
   *          4.Matmul only support Matmul
   *
   * @return BufferFusionPattern: return all valid patterns
   */
  vector<BufferFusionPattern *> DefinePatterns() override;

  /*
   * @brief: parse nodes matched in mapping and call DoFusion
   * @param [in] graph: original graph
   * @param [out] mapping: nodes matched by pattern
   * @return bool: fusion status ok or not.
   */
  Status GetFusionNodes(const BufferFusionMapping &mapping, vector <ge::NodePtr> &fusion_nodes) override;

 private:
  bool MatmulSupportTrans(const ge::NodePtr &node, const bool &is_input) const;
  bool IsInWhiteListOfElemwiseOp(const vector<ge::NodePtr> &elemwise_nodes);
  void CheckCubeSupportTransNodes(const vector<ge::NodePtr> &cube_nodes, const vector<ge::NodePtr> &transdata1_nodes,
                                  const vector<ge::NodePtr> &transdata2_nodes, vector<ge::NodePtr> &fusion_nodes);
};
}


#endif //OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_UB_FUSION_AI_CORE_MATMUL_TBE_MATMUL_FIXPIPE_FUSION_PASS_H
