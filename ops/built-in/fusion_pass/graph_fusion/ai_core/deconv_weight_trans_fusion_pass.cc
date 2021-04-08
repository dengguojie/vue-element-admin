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
 * \file deconv_weight_trans_fusion_pass.cpp
 * \brief deconv weight trans fusion pass(weight -> deconv ===> weight ->
 *   reshape -> transpose -> reshape -> reverse -> reshape -> deconv)
 */
#include "deconv_weight_trans_fusion_pass.h"

#include <cmath>
#include <string>
#include <vector>

#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "../../../op_proto/util/error_util.h"
#include "common/util/error_manager/error_manager.h"

using namespace ge;

namespace fe {
namespace {
const string DECONV = "Deconvolution";
const string CONV2D_TRANSPOSE = "Conv2DTransposeD";
const string PATTERN_DECONV = "DeconvolutionInt8";
static const std::string CONSTANTOP = "Const";
static const std::string CONSTANT = "Constant";
static const int CONST_VECTOR_LEN = 5;
static const int CONST_DIM4_NUM = 4;
static const int CONST_DIM3_NUM = 3;
static const int CONST_DIM2_NUM = 2;
static const int CONST_DIM1_NUM = 1;
static const int CONST_DIM0_NUM = 0;
}  // namespace

vector<FusionPattern*> DeconvWeightTransFusionPass::DefinePatterns() {
  OP_LOGI(FUSED_OP_TYPE.c_str(),
          "Enter DeconvWeightTransFusionPass::DefinePatterns.");
  vector<FusionPattern*> patterns;
  FusionPattern* pattern =
      new (std::nothrow) FusionPattern("DeconvWeightTransFusionPass");
  FUSION_PASS_CHECK(
      pattern == nullptr,
      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
      return patterns);
  pattern->AddOpDesc(PATTERN_DECONV, {DECONV, CONV2D_TRANSPOSE})
      .SetOutput(PATTERN_DECONV);

  patterns.push_back(pattern);

  return patterns;
}

/* weight not 4D, need to complement 1
 * 1D -> C
 * 2D format: HWCN  -->CN
 *            NCHW/NHWC  -->CH
 * 3D -> CHW
 */
static Status GetShapeByFormat(const ge::Format& format,
                               const ge::GeShape& old_shape,
                               int64_t& number, int64_t& channel,
                               int64_t& height, int64_t& weight) {
  if (old_shape.GetDimNum() == 1) {
    channel = old_shape.GetDim(0);
    number = 1;
    height = 1;
    weight = 1;
  } else if (old_shape.GetDimNum() == CONST_DIM2_NUM) {
    if (format == ge::FORMAT_HWCN) {
      channel = old_shape.GetDim(0);
      number = old_shape.GetDim(1);
      height = 1;
      weight = 1;
    } else {
      channel = old_shape.GetDim(0);
      height = old_shape.GetDim(1);
      number = 1;
      weight = 1;
    }
  } else if (old_shape.GetDimNum() == CONST_DIM3_NUM) {
    channel = old_shape.GetDim(0);
    height = old_shape.GetDim(1);
    weight = old_shape.GetDim(CONST_DIM2_NUM);
    number = 1;
  } else {
    if (format == ge::FORMAT_NCHW) {
      number = old_shape.GetDim(0);
      channel = old_shape.GetDim(1);
      height = old_shape.GetDim(CONST_DIM2_NUM);
      weight = old_shape.GetDim(CONST_DIM3_NUM);
    } else if (format == ge::FORMAT_HWCN) {
      number = old_shape.GetDim(CONST_DIM3_NUM);
      channel = old_shape.GetDim(CONST_DIM2_NUM);
      height = old_shape.GetDim(0);
      weight = old_shape.GetDim(1);
    } else if (format == ge::FORMAT_NHWC) {
      number = old_shape.GetDim(0);
      channel = old_shape.GetDim(CONST_DIM3_NUM);
      height = old_shape.GetDim(1);
      weight = old_shape.GetDim(CONST_DIM2_NUM);
    } else {
      return FAILED;
    }
  }
  return SUCCESS;
}
//Dimensions complement
void DeconvWeightTransFusionPass::
    GetShapeUsedByIntermediateProcessInDeconvWeightTrans(
        const ge::Format& filter_format, const vector<int64_t>& shape_GNCHW,
        vector<int64_t>& complement_dimension, vector<int64_t>& reshape_in,
        vector<int64_t>& permute_shape, vector<int64_t>& reverse_axis,
        vector<int64_t>& reshape_out) {
  if (shape_GNCHW.size() != CONST_VECTOR_LEN) {
    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "size of shape_GNCHW not equal 5");
    return;
  }
  complement_dimension.resize(CONST_VECTOR_LEN);
  reshape_in.resize(CONST_DIM4_NUM);
  permute_shape.resize(CONST_VECTOR_LEN);
  reverse_axis.resize(1);
  reshape_out.resize(CONST_DIM4_NUM);
  int64_t groups = shape_GNCHW[CONST_DIM0_NUM];
  int64_t number = shape_GNCHW[CONST_DIM1_NUM];
  int64_t channel = shape_GNCHW[CONST_DIM2_NUM];
  int64_t height = shape_GNCHW[CONST_DIM3_NUM];
  int64_t weight = shape_GNCHW[CONST_DIM4_NUM];
  if (filter_format == ge::FORMAT_HWCN) {
    complement_dimension[CONST_DIM0_NUM] = height;
    complement_dimension[CONST_DIM1_NUM] = weight;
    complement_dimension[CONST_DIM2_NUM] = channel;
    complement_dimension[CONST_DIM3_NUM] = groups;
    complement_dimension[CONST_DIM4_NUM] = number;
    permute_shape[CONST_DIM0_NUM] = CONST_DIM0_NUM;
    permute_shape[CONST_DIM1_NUM] = CONST_DIM1_NUM;
    permute_shape[CONST_DIM2_NUM] = CONST_DIM4_NUM;
    permute_shape[CONST_DIM3_NUM] = CONST_DIM3_NUM;
    permute_shape[CONST_DIM4_NUM] = CONST_DIM2_NUM;
    reshape_in[CONST_DIM0_NUM] = height * weight;
    reshape_in[CONST_DIM1_NUM] = number;
    reshape_in[CONST_DIM2_NUM] = groups;
    reshape_in[CONST_DIM3_NUM] = channel;
    reverse_axis[CONST_DIM0_NUM] = CONST_DIM0_NUM;
    reshape_out[CONST_DIM0_NUM] = height;
    reshape_out[CONST_DIM1_NUM] = weight;
    reshape_out[CONST_DIM2_NUM] = number;
    reshape_out[CONST_DIM3_NUM] = groups * channel;
  } else if (filter_format == ge::FORMAT_NCHW) {
    complement_dimension[CONST_DIM0_NUM] = groups;
    complement_dimension[CONST_DIM1_NUM] = number;
    complement_dimension[CONST_DIM2_NUM] = channel;
    complement_dimension[CONST_DIM3_NUM] = height;
    complement_dimension[CONST_DIM4_NUM] = weight;
    permute_shape[CONST_DIM0_NUM] = CONST_DIM0_NUM;
    permute_shape[CONST_DIM1_NUM] = CONST_DIM2_NUM;
    permute_shape[CONST_DIM2_NUM] = CONST_DIM1_NUM;
    permute_shape[CONST_DIM3_NUM] = CONST_DIM3_NUM;
    permute_shape[CONST_DIM4_NUM] = CONST_DIM4_NUM;
    reshape_in[CONST_DIM0_NUM] = groups;
    reshape_in[CONST_DIM1_NUM] = channel;
    reshape_in[CONST_DIM2_NUM] = number;
    reshape_in[CONST_DIM3_NUM] = height * weight;
    reverse_axis[CONST_DIM0_NUM] = CONST_DIM3_NUM;
    reshape_out[CONST_DIM0_NUM] = groups * channel;
    reshape_out[CONST_DIM1_NUM] = number;
    reshape_out[CONST_DIM2_NUM] = height;
    reshape_out[CONST_DIM3_NUM] = weight;
  } else if (filter_format == ge::FORMAT_NHWC) {
    complement_dimension[CONST_DIM0_NUM] = groups;
    complement_dimension[CONST_DIM1_NUM] = number;
    complement_dimension[CONST_DIM2_NUM] = height;
    complement_dimension[CONST_DIM3_NUM] = weight;
    complement_dimension[CONST_DIM4_NUM] = channel;
    permute_shape[CONST_DIM0_NUM] = CONST_DIM0_NUM;
    permute_shape[CONST_DIM1_NUM] = CONST_DIM4_NUM;
    permute_shape[CONST_DIM2_NUM] = CONST_DIM2_NUM;
    permute_shape[CONST_DIM3_NUM] = CONST_DIM3_NUM;
    permute_shape[CONST_DIM4_NUM] = CONST_DIM1_NUM;
    reshape_in[CONST_DIM0_NUM] = groups;
    reshape_in[CONST_DIM1_NUM] = channel;
    reshape_in[CONST_DIM2_NUM] = height * weight;
    reshape_in[CONST_DIM3_NUM] = number;
    reverse_axis[CONST_DIM0_NUM] = CONST_DIM2_NUM;
    reshape_out[CONST_DIM0_NUM] = groups * channel;
    reshape_out[CONST_DIM1_NUM] = height;
    reshape_out[CONST_DIM2_NUM] = weight;
    reshape_out[CONST_DIM3_NUM] = number;
  }
}

static Status GenerateTransposeNode(ge::ComputeGraph& graph,
                                    ge::GeTensorDesc& previous_out_desc,
                                    ge::GeTensorDesc& next_in_desc,
                                    const vector<int64_t>& perm,
                                    ge::NodePtr& transpose_node,
                                    const std::string& basename) {
  vector<int64_t> next_in_shape(CONST_VECTOR_LEN);
  for (size_t i = 0; i < perm.size(); ++i) {
    next_in_shape[i] = previous_out_desc.GetShape().GetDim(perm[i]);
  }
  ge::OpDescPtr transpose_desc;
  FUSION_PASS_MAKE_SHARED(
      (transpose_desc = std::make_shared<ge::OpDesc>(
           basename + "_const_fold_transpose_nc", "TransposeD")),
      return FAILED);
  previous_out_desc.SetFormat(ge::FORMAT_ND);
  previous_out_desc.SetOriginFormat(ge::FORMAT_ND);
  transpose_desc->AddInputDesc("x", previous_out_desc);
  next_in_desc.SetFormat(ge::FORMAT_ND);
  next_in_desc.SetOriginFormat(ge::FORMAT_ND);
  next_in_desc.SetShape(ge::GeShape(next_in_shape));
  next_in_desc.SetOriginShape(ge::GeShape(next_in_shape));
  transpose_desc->AddOutputDesc("y", next_in_desc);
  ge::AttrUtils::SetListInt(transpose_desc, "perm", perm);
  transpose_node = graph.AddNode(transpose_desc);
  return SUCCESS;
}

static Status GenerateReshapeNode(ge::ComputeGraph& graph,
                                  ge::GeTensorDesc& previous_out_desc,
                                  ge::GeTensorDesc& next_in_desc,
                                  const vector<int64_t>& shape,
                                  ge::NodePtr& shape_node,
                                  const std::string& name,
                                  const std::string& basename) {
  ge::OpDescPtr reshape_desc;
  FUSION_PASS_MAKE_SHARED((reshape_desc = std::make_shared<ge::OpDesc>(
                               basename + "_const_fold_" + name, "Reshape")),
                          return FAILED);
  reshape_desc->AddInputDesc("x", previous_out_desc);
  next_in_desc.SetShape(ge::GeShape(shape));
  next_in_desc.SetOriginShape(ge::GeShape(shape));
  reshape_desc->AddOutputDesc("y", next_in_desc);
  ge::AttrUtils::SetListInt(reshape_desc, "shape", shape);
  shape_node = graph.AddNode(reshape_desc);
  return SUCCESS;
}

static Status GenerateReverseNode(ge::ComputeGraph& graph,
                                  ge::GeTensorDesc& previous_out_desc,
                                  ge::GeTensorDesc& next_in_desc,
                                  const vector<int64_t>& axis,
                                  ge::NodePtr& reverse_node,
                                  const std::string& basename) {
  ge::OpDescPtr reverse_desc;
  FUSION_PASS_MAKE_SHARED(
      (reverse_desc = std::make_shared<ge::OpDesc>(
           basename + "_const_fold_reverse_hw", "ReverseV2D")),
      return FAILED);
  reverse_desc->AddInputDesc("x", previous_out_desc);
  next_in_desc.SetShape(previous_out_desc.GetShape());
  next_in_desc.SetOriginShape(previous_out_desc.GetShape());
  reverse_desc->AddOutputDesc("y", next_in_desc);
  ge::AttrUtils::SetListInt(reverse_desc, "axis", axis);
  reverse_node = graph.AddNode(reverse_desc);
  return SUCCESS;
}

static Status GenerateReFormatNode(ge::ComputeGraph& graph,
                                   ge::GeTensorDesc& previous_out_desc,
                                   ge::GeTensorDesc& next_in_desc,
                                   const ge::Format& format,
                                   ge::NodePtr& reformat_node,
                                   const std::string& basename) {
  ge::OpDescPtr reformat_desc;
  FUSION_PASS_MAKE_SHARED((reformat_desc = std::make_shared<ge::OpDesc>(
                               basename + "_const_fold_reformat", "ReFormat")),
                          return FAILED);
  reformat_desc->AddInputDesc("x", previous_out_desc);
  next_in_desc.SetShape(previous_out_desc.GetShape());
  next_in_desc.SetOriginShape(previous_out_desc.GetShape());
  next_in_desc.SetFormat(format);
  next_in_desc.SetOriginFormat(format);
  reformat_desc->AddOutputDesc("y", next_in_desc);
  reformat_node = graph.AddNode(reformat_desc);
  return SUCCESS;
}

Status DeconvWeightTransFusionPass::Relink(
    ge::NodePtr filter_node, ge::NodePtr dim_comp_node, ge::NodePtr transpose_node,
    ge::NodePtr reformat_node, ge::NodePtr reshape_in_node,
    ge::NodePtr reverse_node, ge::NodePtr reshape_out_node,
    ge::NodePtr deconv_node) {
  // weight -> Deconvolution
  // weight -> [complement_dimension] -> transpose -> [reshape_in -> reverse -> reshape_out] ->
  // deconv
  int filter_anchor = 1;
  FUSION_PASS_CHECK(
      ge::GraphUtils::RemoveEdge(filter_node->GetOutDataAnchor(0),
                                 deconv_node->GetInDataAnchor(filter_anchor)) !=
          SUCCESS,
      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
              "fail to remove edge between filter_node and deconv_node"),
      return FAILED);

  OutDataAnchorPtr dim_comp_out_anchor = nullptr;
  if (dim_comp_node != nullptr) {
    FUSION_PASS_CHECK(
        ge::GraphUtils::AddEdge(filter_node->GetOutDataAnchor(0),
                                dim_comp_node->GetInDataAnchor(0)) != SUCCESS,
        CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                "fail to add edge between filter_node and dim_comp_node"),
        return FAILED);
    dim_comp_out_anchor = dim_comp_node->GetOutDataAnchor(0);
  } else {
    dim_comp_out_anchor = filter_node->GetOutDataAnchor(0);
  }
  FUSION_PASS_CHECK(
      ge::GraphUtils::AddEdge(dim_comp_out_anchor,
                              transpose_node->GetInDataAnchor(0)) != SUCCESS,
      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
              "fail to add edge between dim_comp_node and transpose_node"),
      return FAILED);

  FUSION_PASS_CHECK(
      ge::GraphUtils::AddEdge(transpose_node->GetOutDataAnchor(0),
                              reformat_node->GetInDataAnchor(0)) != SUCCESS,
      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
              "fail to add edge between dim_comp_node and transpose_node"),
      return FAILED);

  if (reshape_in_node != nullptr) {
    FUSION_PASS_CHECK(
        ge::GraphUtils::AddEdge(reformat_node->GetOutDataAnchor(0),
                                reshape_in_node->GetInDataAnchor(0)) != SUCCESS,
        CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                "fail to add edge between transpose_node and reshape_in_node"),
        return FAILED);
    FUSION_PASS_CHECK(
        ge::GraphUtils::AddEdge(reshape_in_node->GetOutDataAnchor(0),
                                reverse_node->GetInDataAnchor(0)) != SUCCESS,
        CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                "fail to add edge between reshape_in_node and reverse_node"),
        return FAILED);
    FUSION_PASS_CHECK(
        ge::GraphUtils::AddEdge(reverse_node->GetOutDataAnchor(0),
                                reshape_out_node->GetInDataAnchor(0)) != SUCCESS,
        CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                "fail to add edge between reverse_node and reshape_out_node"),
        return FAILED);    
    FUSION_PASS_CHECK(
        ge::GraphUtils::AddEdge(reshape_out_node->GetOutDataAnchor(0),
                                deconv_node->GetInDataAnchor(filter_anchor)) !=
            SUCCESS,
        CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                "fail to add edge between reshape_out_node and deconv_node"),
        return FAILED);
  } else {
    FUSION_PASS_CHECK(
        ge::GraphUtils::AddEdge(reformat_node->GetOutDataAnchor(0),
                                reshape_out_node->GetInDataAnchor(0)) != SUCCESS,
        CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                "fail to add edge between transpose_node and reshape_out_node"),
        return FAILED);   
    FUSION_PASS_CHECK(
        ge::GraphUtils::AddEdge(reshape_out_node->GetOutDataAnchor(0),
                                deconv_node->GetInDataAnchor(filter_anchor)) !=
            SUCCESS,
        CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                "fail to add edge between reshape_out_node and deconv_node"),
        return FAILED);
  }
  FUSION_PASS_CHECK(
      deconv_node->GetOpDesc()->UpdateInputDesc(
          filter_anchor, reshape_out_node->GetOpDesc()->GetOutputDesc(0)) != SUCCESS,
      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
              "fail to update input description of deconv"),
      return FAILED);

  return SUCCESS;
}

int64_t DeconvWeightTransFusionPass::GetGroups(ge::OpDescPtr &deconv_desc) {
  int64_t groups = 1;
  bool hasGroup = ge::AttrUtils::GetInt(deconv_desc, "groups", groups);
  return hasGroup ? groups : 1;
}

int64_t GetBatchCeilGroups(int64_t& groups, int64_t& number) {
  if (number % groups != 0) {
    return FAILED;
  } else {
    number = number / groups;
  }
  return SUCCESS;
}

Status DeconvWeightTransFusionPass::Fusion(ge::ComputeGraph& graph,
                                           Mapping& mapping,
                                           vector<ge::NodePtr>& fusion_nodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Enter DeconvWeightTransFusionPass.");
  ge::NodePtr deconv_node = GetNodeFromMapping(PATTERN_DECONV, mapping);

  // pattern
  // originFormat: NCHW,HWCN,NHWC
  // for example: NCHW
  // weight(NCHW) -> | dim completion -> transpose -> reshape in -> reverse ->
  // reshape out | -> Deconvolution
  //                 | |
  //                 | |
  //                 | |

  // input: x, filter, bias
  int input_anchor = 0;
  int filter_anchor = 1;

  // prerequisite
  ge::NodePtr input_node =
      deconv_node->GetInDataAnchor(input_anchor)->GetPeerOutAnchor()->GetOwnerNode();
  int input_index = deconv_node->GetInDataAnchor(input_anchor)->GetPeerOutAnchor()->GetIdx();
  ge::NodePtr filter_node = deconv_node->GetInDataAnchor(filter_anchor)
                               ->GetPeerOutAnchor()
                               ->GetOwnerNode();
  int filter_index =
      deconv_node->GetInDataAnchor(filter_anchor)->GetPeerOutAnchor()->GetIdx();
  if (filter_node->GetOpDesc()->GetOutputDesc(filter_index).GetDataType() !=
          ge::DT_INT8 ||
      input_node->GetOpDesc()->GetOutputDesc(input_index).GetDataType() != ge::DT_INT8) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "The dtype of weight or x is not int8.");
    return NOT_CHANGED;
  }
  std::string type = ge::NodeUtils::GetInConstNodeTypeCrossSubgraph(filter_node);
  if (type != CONSTANT && type != CONSTANTOP) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "The type of weight is not constant.");
    return NOT_CHANGED;
  }

  int64_t number = 0, channel = 0, height = 0, weight = 0;
  auto op_desc = deconv_node->GetOpDesc();
  int64_t groups = GetGroups(op_desc);
  FUSION_PASS_CHECK(groups == 0,
                    OP_LOGW(FUSED_OP_TYPE.c_str(),
                            "Groups can not be 0."),
                    return NOT_CHANGED);
  ge::GeTensorDesc deconv_weight_in_desc =
      deconv_node->GetOpDesc()->GetInputDesc(filter_anchor);
  ge::GeTensorDesc filter_out_desc =
      filter_node->GetOpDesc()->GetOutputDesc(filter_index);
  ge::GeShape filter_shape = filter_out_desc.GetShape();
  ge::Format filter_format = filter_out_desc.GetFormat();
  FUSION_PASS_CHECK(GetShapeByFormat(filter_format, filter_shape, number, channel,
                                     height, weight) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(),
                            "Not support this format %d.", filter_format),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(GetBatchCeilGroups(groups, number) != SUCCESS,
                    OP_LOGW(FUSED_OP_TYPE.c_str(),
                            "Batch of filter cannot divide groups"),
                    return NOT_CHANGED);

  if (PatternFusionUtil::IsUnknownShape(number) ||
      PatternFusionUtil::IsUnknownShape(channel) ||
      PatternFusionUtil::IsUnknownShape(height) ||
      PatternFusionUtil::IsUnknownShape(weight)) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "DeconvWeightTransFusionPass cannot be applied for unknown shape.");
    return NOT_CHANGED;
  }
  vector<int64_t> complement_dimension(CONST_VECTOR_LEN), reshape_in(CONST_DIM4_NUM),
      permute_shape(CONST_VECTOR_LEN), reverse_axis(1), reshape_out(CONST_DIM4_NUM);
  GetShapeUsedByIntermediateProcessInDeconvWeightTrans(
      filter_format, {groups, number, channel, height, weight}, complement_dimension, reshape_in,
      permute_shape, reverse_axis, reshape_out);

  ge::NodePtr dim_comp_node = nullptr, transpose_node = nullptr,
              reformat_node = nullptr, reshape_in_node = nullptr,
              reverse_node = nullptr, reshape_out_node = nullptr;

  auto basename = filter_node->GetName();

  // 1. dimension completion
  FUSION_PASS_CHECK(
      GenerateReshapeNode(graph, filter_out_desc, deconv_weight_in_desc, complement_dimension,
                          dim_comp_node, "dimension_completion",
                          basename) != SUCCESS,
      CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(),
              "fail to generate dimension completion node"),
      return FAILED);
  ge::GeTensorDesc dim_comp_out_desc;
  if (dim_comp_node != nullptr) {
    dim_comp_out_desc = dim_comp_node->GetOpDesc()->GetOutputDesc(0);
  } else {
    dim_comp_out_desc = filter_out_desc;
  }

  // 2. transpose number,channel
  FUSION_PASS_CHECK(
      GenerateTransposeNode(graph, dim_comp_out_desc, deconv_weight_in_desc,
                            permute_shape, transpose_node, basename) != SUCCESS,
      CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "fail to generate transpose node"),
      return FAILED);
  ge::GeTensorDesc transpose_out_desc =
      transpose_node->GetOpDesc()->GetOutputDesc(0);
  FUSION_PASS_CHECK(
      GenerateReFormatNode(graph, transpose_out_desc, deconv_weight_in_desc,
                           filter_format, reformat_node, basename) != SUCCESS,
      CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "fail to generate reformat node"),
      return FAILED);
  ge::GeTensorDesc reformat_desc = reformat_node->GetOpDesc()->GetOutputDesc(0);

  if (height != 1 || weight != 1) {
    // 3. fuse height, weight
    FUSION_PASS_CHECK(
        GenerateReshapeNode(graph, reformat_desc, deconv_weight_in_desc, reshape_in,
                            reshape_in_node, "reshape_in", basename) != SUCCESS,
        CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "fail to generate reshape in node"),
        return FAILED);
    ge::GeTensorDesc reshape_in_out_desc =
        reshape_in_node->GetOpDesc()->GetOutputDesc(0);

    // 4. reverse height*weight
    FUSION_PASS_CHECK(
        GenerateReverseNode(graph, reshape_in_out_desc, deconv_weight_in_desc,
                            reverse_axis, reverse_node, basename) != SUCCESS,
        CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "fail to generate reverse node"),
        return FAILED);
    ge::GeTensorDesc reverse_out_desc =
        reverse_node->GetOpDesc()->GetOutputDesc(0);

    // 5. anti-fusion height*weight
    FUSION_PASS_CHECK(
        GenerateReshapeNode(graph, reverse_out_desc, deconv_weight_in_desc,
                            reshape_out, reshape_out_node, "reshape_out",
                            basename) != SUCCESS,
        CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "fail to generate reshape out node"),
        return FAILED);
  } else {
    FUSION_PASS_CHECK(
        GenerateReshapeNode(graph, reformat_desc, deconv_weight_in_desc,
                            reshape_out, reshape_out_node, "reshape_out",
                            basename) != SUCCESS,
        CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "fail to generate reshape out node"),
        return FAILED);
  }
  FUSION_PASS_CHECK(
      Relink(filter_node, dim_comp_node, transpose_node, reformat_node,
             reshape_in_node, reverse_node, reshape_out_node, deconv_node) != SUCCESS,
      CommonRuntimeErrLog(FUSED_OP_TYPE.c_str(), "fail to relink nodes"), return FAILED);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "End DeconvWeightTransFusionPass.");
  return SUCCESS;
}

REGISTER_PASS("DeconvWeightTransFusionPass", BUILT_IN_GRAPH_PASS,
              DeconvWeightTransFusionPass);
}  // namespace fe
