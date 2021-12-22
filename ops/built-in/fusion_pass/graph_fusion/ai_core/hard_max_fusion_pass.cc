/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permission and
 * limitations under the License.
 */
 
 /*!
  * \file hard_max_fusion_pass.h
  * \brief hard_max fusion pass
  */
  
#include "hard_max_fusion_pass.h"
 
#include <vector>
#include <memory>
#include "fp16_t.hpp"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "error_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "tbe_fusion_pass_util.h"
 
using namespace std;
using namespace ge;
 
namespace fe {
    const std::string HardMaxPass::PATTERN_FUSEDNODE = "HardMax";
    const int64_t SCALAR_SHAPE_SIZE = 1;
    const int32_t ON_VALUE_DATA = 1;
    const int32_t OFF_VALUE_DATA = 0;
    const uint16_t UINT_NUM_ZERO = 0;
	 
vector<FusionPattern *> HardMaxPass::DefinePatterns()
{
    vector<FusionPattern *> patterns;
    FusionPattern *pattern = (new (std::nothrow) FusionPattern("HardMaxPass"));
    FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT("HardMaxPass", "new pattern error"),
                      return patterns);
    pattern->AddOpDesc(PATTERN_FUSEDNODE, { "HardMax" }).SetOutput(PATTERN_FUSEDNODE);
    patterns.push_back(pattern);
    return patterns;
}
 
Status HardMaxPass::CreateArgMaxDNode(ge::ComputeGraph &graph, ge::NodePtr &fused_node, ge::NodePtr &new_node,
                                      int64_t &depth, int64_t &dim)
{
    ge::OpDescPtr new_desc = nullptr;
    FUSION_PASS_MAKE_SHARED((new_desc = std::make_shared<ge::OpDesc>("ArgMaxD_For_HardMax", "ArgMaxD")),
        return INTERNAL_ERROR);
    Operator op = ge::OpDescUtils::CreateOperatorFromNode(fused_node);
    ge::GeTensorDesc input_desc = fused_node->GetOpDesc()->GetInputDesc(0);
    ge::GeTensorDesc output_desc = fused_node->GetOpDesc()->GetInputDesc(0);
    int64_t input_size = input_desc.GetShape().GetDimNum();
    vector<int64_t> input_shape_vec = input_desc.GetShape().GetDims();
    vector<int64_t> output_shape_vec;
    int64_t dimension = 0;
    auto ret = op.GetAttr("axis", dimension);
    FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT("HardMaxPass",
                      "CreateArgMax GetAttr dimension fail"),
                      return FAILED);
    if (dimension < 0) {
        dimension += input_size;
    }
    dim = dimension;
	
    for (int64_t i = 0; i < input_size; i++) {
        if (i == dim) {
            continue;
            } else {
                    output_shape_vec.push_back(input_shape_vec[i]);
                }
        }
    ge::GeShape output_shape(output_shape_vec);
    output_desc.SetShape(output_shape);
    output_desc.SetOriginShape(output_shape);
    depth = input_desc.GetShape().GetDim(dimension);
    output_desc.SetDataType(ge::DT_INT32);
    output_desc.SetOriginDataType(ge::DT_INT32);
    ret = new_desc->AddInputDesc("x", input_desc);
    FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT("HardMaxPass", "CreateArgmax AddInputDesc fail"),
                      return FAILED);
    ret = new_desc->AddOutputDesc("y", output_desc);
    FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT("HardMaxPass", "CreateArgmax AddOutputDesc fail"),
                      return FAILED);
    new_node = graph.AddNode(new_desc);
    Operator new_op = ge::OpDescUtils::CreateOperatorFromNode(new_node);
    new_op.SetAttr("dimension", dimension);
    return SUCCESS;
}

Status HardMaxPass::CreateOneHotDNode(ge::ComputeGraph &graph, ge::NodePtr &fused_node, ge::NodePtr &argmax_node,
                                      ge::NodePtr &new_node, int64_t depth, int64_t dim)
{
    ge::OpDescPtr new_desc = nullptr;
    FUSION_PASS_MAKE_SHARED((new_desc = std::make_shared<ge::OpDesc>("OneHotD_For_HardMax", "OneHotD")),
        return INTERNAL_ERROR);
    ge::GeTensorDesc input_desc = argmax_node->GetOpDesc()->GetOutputDesc(0);
    ge::GeTensorDesc output_desc = fused_node->GetOpDesc()->GetInputDesc(0);
    auto ret = new_desc->AddInputDesc("x", input_desc);
    FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT("HardMaxPass",
                      "CreateOneHotDNode AddInputDesc one fail."),
        return FAILED);
    ret = new_desc->AddOutputDesc("y", output_desc);
    FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT("HardMaxPass",
                      "CreateOneHotDNode AddInputDesc one fail."),
        return FAILED);
    new_node = graph.AddNode(new_desc);
    Operator new_op = ge::OpDescUtils::CreateOperatorFromNode(new_node);
    new_op.SetAttr("axis", dim);
    new_op.SetAttr("depth", depth);
    return SUCCESS;
}

Status HardMaxPass::SetConstDesc(vector<int64_t> &tensor_shape, ge::GeTensorDesc &tensor_desc,
	                         ge::GeTensorDesc &des_desc) const
{
    ge::GeShape ten_shapes(tensor_shape);
    tensor_desc.SetOriginFormat(des_desc.GetOriginFormat());
    tensor_desc.SetFormat(des_desc.GetFormat());
    tensor_desc.SetOriginDataType(des_desc.GetOriginDataType());
    tensor_desc.SetDataType(des_desc.GetDataType());
    tensor_desc.SetOriginShape(ten_shapes);
    tensor_desc.SetShape(ten_shapes);
    return SUCCESS;
}

int64_t GetDimN(const vector<int64_t> &shapes)
{
    auto shape_lens = shapes.size();
    int64_t dim_num = 1;
    for (size_t i = 0; i < shape_lens; i++) {
        dim_num = dim_num * shapes[i];
        }
    return dim_num;
}

Status AssistDataGen(int32_t data, uint16_t *output)
{
    if (output == nullptr) {
        VECTOR_FUSION_INNER_ERR_REPORT("Output", "output pointer is null!");
        return FAILED;
        }
	
    output[0] = data;
    return SUCCESS;
}

Status HardMaxPass::OnValueConstNode(vector<int64_t> &on_value_tensor_shape, ge::GeTensorDesc &input_desc_one,
	                             ge::GeTensorPtr &assit_on_value_ptr, int32_t on_value,
 ge::GeTensorDesc &on_value_tensor_desc) const
{
    int64_t on_value_dim_num = GetDimN(on_value_tensor_shape);
    Status ret = SetConstDesc(on_value_tensor_shape, on_value_tensor_desc, input_desc_one);
    unique_ptr<uint16_t[]> on_value_assit(new (std::nothrow) uint16_t[on_value_dim_num]());
    FUSION_PASS_CHECK(on_value_assit.get() == nullptr, VECTOR_FUSION_INNER_ERR_REPORT("HardMaxPass",
                      "on_value_assit is NULL"),
        return PARAM_INVALID);
		
    ret = NnSet(on_value_dim_num, UINT_NUM_ZERO, *reinterpret_cast<uint16_t *>(on_value_assit.get()));
    FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT("HardMaxPass", "NnSet failed."), return ret);
    ret = AssistDataGen(on_value, on_value_assit.get());
    FUSION_PASS_MAKE_SHARED((assit_on_value_ptr = std::make_shared<ge::GeTensor>(on_value_tensor_desc,
        reinterpret_cast<uint8_t *>(on_value_assit.get()), on_value_dim_num * sizeof(uint16_t))),
        assit_on_value_ptr = nullptr;
        return PARAM_INVALID);
    return SUCCESS;
}

Status HardMaxPass::OffValueConstNode(vector<int64_t> &off_value_tensor_shape, ge::GeTensorDesc &input_desc_one,
        ge::GeTensorPtr &assit_off_value_ptr, int32_t off_value, ge::GeTensorDesc &off_value_tensor_desc) const
{   
    int64_t off_value_dim_num = GetDimN(off_value_tensor_shape);
    Status ret = SetConstDesc(off_value_tensor_shape, off_value_tensor_desc, input_desc_one);
    unique_ptr<uint16_t[]> off_value_assit(new (std::nothrow) uint16_t[off_value_dim_num]());
    FUSION_PASS_CHECK(off_value_assit.get() == nullptr, VECTOR_FUSION_INNER_ERR_REPORT("HardMaxPass",
                      "off_value_assit is NULL"),
        return PARAM_INVALID);
    ret = NnSet(off_value_dim_num, UINT_NUM_ZERO, *reinterpret_cast<uint16_t *>(off_value_assit.get()));
    FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT("HardMaxPass", "NnSet failed."), return ret);
    ret = AssistDataGen(off_value, off_value_assit.get());
    FUSION_PASS_MAKE_SHARED((assit_off_value_ptr = std::make_shared<ge::GeTensor>(off_value_tensor_desc,
        reinterpret_cast<uint8_t *>(off_value_assit.get()), off_value_dim_num * sizeof(uint16_t))),
        assit_off_value_ptr = nullptr;
        return PARAM_INVALID);
    return SUCCESS;
}

Status HardMaxPass::AddEdgeToOneHotDForOut(ge::NodePtr &fused_node, ge::NodePtr &one_hot_d_node) const
{
    Status ret = SUCCESS;
    for (auto in_data_anchor : fused_node->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
        ret = ge::GraphUtils::RemoveEdge(fused_node->GetOutDataAnchor(0), in_data_anchor);
        FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT("HardMaxPass",
                          "AddEdgeToOneHotDForOut removeEdge fail"),
        return FAILED);
        ret = ge::GraphUtils::AddEdge(one_hot_d_node->GetOutDataAnchor(0), in_data_anchor);
        FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT("HardMaxPass",
                          "AddEdgeToOneHotDForOut addEdge fail"),
                          return FAILED);
    }
    return SUCCESS;
}

Status HardMaxPass::RemoveFusedNode(ge::ComputeGraph &graph, ge::NodePtr &fused_node) const
{
    for (auto in_anchor : fused_node->GetAllInDataAnchors()) {
        if (in_anchor != nullptr) {
            in_anchor->UnlinkAll();
        }
    }
	
    for (auto out_anchor : fused_node->GetAllOutDataAnchors()) {
        if (out_anchor != nullptr) {
            out_anchor->UnlinkAll();
        }
    }
	
    FUSION_PASS_CHECK(graph.RemoveNode(fused_node) != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT("HardMaxPass",
                      "RemoveFusedNode error"),
        return FAILED);
    return SUCCESS;
}

Status HardMaxPass::Fusion(ge::ComputeGraph &graph, Mapping &mapping, vector<ge::NodePtr> &fusion_nodes)
{
    ge::NodePtr fused_node = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
    FUSION_PASS_CHECK(fused_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT("HardMaxPass", "Fusion GetNode Error"),
                      return PARAM_INVALID);
	
    ge::NodePtr one_hot_d_node;
    ge::NodePtr argmax_d_node;
    int64_t depth = 0;
    int64_t dim = 0;
    auto ret = CreateArgMaxDNode(graph, fused_node, argmax_d_node, depth, dim);
    FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT("HardMaxPass", "Fusion fail"), return FAILED);
	
    ret = CreateOneHotDNode(graph, fused_node, argmax_d_node, one_hot_d_node, depth, dim);
    FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT("HardMaxPass", "Fusion CreateOneHotDNode fail"),
                      return FAILED);
	
    fusion_nodes.push_back(argmax_d_node);
    fusion_nodes.push_back(one_hot_d_node);
	
    ret = ge::GraphUtils::AddEdge(fused_node->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                  argmax_d_node->GetInDataAnchor(0));
    FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT("HardMaxPass", "AddEdge to argmax_d_node fail"),
                      return FAILED);
	
    ret = ge::GraphUtils::AddEdge(argmax_d_node->GetOutDataAnchor(0), one_hot_d_node->GetInDataAnchor(0));
    FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT("HardMaxPass", "AddEdge to one_hot_d_node fail"),
                      return FAILED);
	
    ret = AddEdgeToOneHotDForOut(fused_node, one_hot_d_node);
    FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT("HardMaxPass", "AddEdgeToOneHotDForOut fail"),
                      return FAILED);
	
    vector<int64_t> on_and_off_shape;
    on_and_off_shape.push_back(SCALAR_SHAPE_SIZE);
    ge::GeTensorPtr assit_on_value_ptr = nullptr;
    ge::GeTensorDesc on_value_tensor_desc(GeShape(on_and_off_shape), ge::FORMAT_ND, ge::DT_INT32);
    ge::GeTensorDesc input_desc_one(GeShape(on_and_off_shape), ge::FORMAT_ND, ge::DT_INT32);
    int32_t on_value = ON_VALUE_DATA;
    ret = OnValueConstNode(on_and_off_shape, input_desc_one, assit_on_value_ptr, on_value, on_value_tensor_desc);
    ge::GeTensorPtr assit_off_value_ptr = nullptr;
    ge::GeTensorDesc off_value_tensor_desc(GeShape(on_and_off_shape), ge::FORMAT_ND, ge::DT_INT32);
    int32_t off_value = OFF_VALUE_DATA;
    ret = OffValueConstNode(on_and_off_shape, input_desc_one, assit_off_value_ptr, off_value, off_value_tensor_desc);
	
    vector<ge::GeTensorPtr> value_weights = { assit_on_value_ptr, assit_off_value_ptr };
    ge::OpDescUtils::SetWeights(one_hot_d_node, value_weights);
    auto const_input_nodes = OpDescUtils::GetConstInputs(one_hot_d_node);
	
    if (const_input_nodes.size() <= 0) {
        VECTOR_FUSION_INNER_ERR_REPORT("HardMaxPass", "GetConstInputs Error");
        return PARAM_INVALID;
    }
    NodePtr const_on_value_input = const_input_nodes[0];
    const_on_value_input->GetOpDesc()->SetType("Const");
	
    NodePtr const_off_value_input = const_input_nodes[1];
    const_off_value_input->GetOpDesc()->SetType("Const");
	
    ret = RemoveFusedNode(graph, fused_node);
    FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT("HardMaxPass", "RemoveFusedNode fail"),
                      return FAILED);
	
    return SUCCESS;
}

REGISTER_PASS("HardMaxPass", BUILT_IN_GRAPH_PASS, HardMaxPass);
} //  namespace ge



