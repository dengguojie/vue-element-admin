/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2022. All rights reserved.
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
 * \file double_ascend_antiquant_add_relu_ascend_quant_fusion_pass.h
 * \brief ascend_antiquant * 2 & add & relu & ascend_quant fusion pass
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_DOUBLE_ASCEND_ANTIQUANT_ADD_RELU_ASCEND_QUANT_FUSION_PASS_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_DOUBLE_ASCEND_ANTIQUANT_ADD_RELU_ASCEND_QUANT_FUSION_PASS_H_

#include <string>
#include <vector>

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class XQuantFusionBasePass {
  protected:
    XQuantFusionBasePass() {};
    virtual ~XQuantFusionBasePass() {};

  public:
    void SetNode(const ge::NodePtr node) {
      // define in .h file to be an inline function
      this->node = node;
    };

    ge::NodePtr GetNode() const {
      return this->node;
    };

    virtual Status GetInputAttributes();

  protected:
    ge::OpDescPtr CreateOpDesc(const std::string& name, const std::string& type,
                               const ge::GeTensorDesc& inputDesc, const ge::GeTensorDesc& outputDesc) const;
    ge::NodePtr CreateAddsNode(ge::ComputeGraph& graph, vector<ge::NodePtr>& fusionNodes,
                              const std::string& name, const ge::GeTensorDesc& inputDesc) const;
    ge::NodePtr CreateMulsNode(ge::ComputeGraph& graph, vector<ge::NodePtr>& fusionNodes,
                               const std::string& name, const ge::GeTensorDesc& inputDesc) const;
    ge::NodePtr CreateCastNode(ge::ComputeGraph& graph, vector<ge::NodePtr>& fusionNodes,
                               const std::string& name, const ge::GeTensorDesc& inputDesc,
                               const ge::GeTensorDesc& outputDesc) const;
    Status LinkNode(const ge::NodePtr from, const ge::NodePtr to) const;
    Status ReplaceNode(const ge::NodePtr originalNode,
                       const ge::NodePtr newFirstNode, const ge::NodePtr newLastNode) const;

  protected:
    // original node
    ge::NodePtr node = nullptr;

    // fusion node
    ge::NodePtr castNode = nullptr;
    ge::NodePtr addsNode = nullptr;
    ge::NodePtr mulsNode1 = nullptr;
    ge::NodePtr mulsNode2 = nullptr;

    // input attributes
    ge::Operator::OpFloat scale;
    ge::Operator::OpFloat offset;
    ge::Operator::OpBool sqrtMode = false;
};

class AscendAntiQuantFusionPass : public XQuantFusionBasePass {
  public:
    bool IsMatch() const;
    Status Fusion(ge::ComputeGraph& graph, vector<ge::NodePtr>& fusionNodes);

  private:
    Status UpdateEdges() const;
    Status CreateNodes(ge::ComputeGraph& graph, vector<ge::NodePtr>& fusionNodes);
};

class AscendQuantFusionPass : public XQuantFusionBasePass {
  public:
    bool IsMatch() const;
    Status GetInputAttributes() override;
    Status Fusion(ge::ComputeGraph& graph, vector<ge::NodePtr>& fusionNodes);

  private:
    Status UpdateEdges() const;
    Status CreateNodes(ge::ComputeGraph& graph, vector<ge::NodePtr>& fusionNodes);
    ge::NodePtr CreateConvertNode(ge::ComputeGraph& graph, vector<ge::NodePtr>& fusionNodes,
                                  const std::string& name, const ge::GeTensorDesc& inputDesc) const;

  private:
    // fusion node
    ge::NodePtr convertNode = nullptr;

    // input attributes
    ge::Operator::OpAscendString roundMode = "Round";
    ge::Operator::OpInt dstType = ge::DataType::DT_INT8;
};

class DoubleAscendAntiQuantAddReluAscendQuantFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern*> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) override;

 private:
  bool IsMatch();
  bool IsAddReluOutputMatch() const;
  Status ChangeAddReluShape();

  // pattern matched nodes
  ge::NodePtr addNode = nullptr;
  ge::NodePtr reluNode = nullptr;
  AscendAntiQuantFusionPass antiQuantFP1;
  AscendAntiQuantFusionPass antiQuantFP2;
  AscendQuantFusionPass quantFP;
};
}  // namespace fe
#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_DOUBLE_ASCEND_ANTIQUANT_ADD_RELU_ASCEND_QUANT_FUSION_PASS_H_
