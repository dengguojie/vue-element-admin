/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain new_offset_name copy of the License at
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
 * \file deformable_conv2d_fusion_pass.cpp
 * \brief convert split+conv2d+concat to group conv2d
 */
#include "deformable_conv2d_fusion_pass.h"
#include <vector>
#include <sstream>
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "error_util.h"

using namespace ge;

namespace fe {

static const char kPatternDfmConv2D[] = "deformable_conv2d";
static const char kDfmConv2DType[] = "DeformableConv2D";
static const char kDfmOffsetType[] = "DeformableOffsets";
static const char kConv2DType[] = "Conv2D";
static const char kAttrKsize[] = "ksize";
static const char kAttrStrides[] = "strides";
static const char kAttrPads[] = "pads";
static const char kAttrDilations[] = "dilations";
static const char kAttrGroups[] = "groups";
static const char kAttrDataFmt[] = "data_format";
static const char kAttrDfmGroups[] = "deformable_groups";
static const char kAttrModulated[] = "modulated";
static const char kAttrOrgFmt[] = "origin_format";
static const std::set<string> kFilterFmt = {"HWCN", "NCHW"};

/*!
  * @brief Define deformable_conv2d pattern.
  *
  * inputs filter    offsets bias(if exist)
  *       \    \       /    /
  *               \/
  *        deformable_conv2d
  *               |
  *
  * @return vector<FusionPattern*> All valid patterns.
  */
vector<FusionPattern*> DeformableConv2dPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("DeformableConv2dPass");
  FUSION_PASS_CHECK(pattern == nullptr, CUBE_CALL_ERR_REPORT(fused_op_type_.c_str(), "new new_offset_name pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(kPatternDfmConv2D, {kDfmConv2DType})
      .SetOutput(kPatternDfmConv2D);
  patterns.push_back(pattern);

  return patterns;
}

/*!
  * @brief Add new offset node and set right attributes.
  *
  * deformable_offset args:
  *   tensor:
  *      x:          use deformable_conv2d
  *      offsets:    use deformable_conv2d
  *      y:          output(H/W)*kernel_size(H/W)
  *   attrs:
  *      [ksize]:             kernel_size(H,W)
  *      [strides]:           use deformable_conv2d
  *      [pads]:              use deformable_conv2d
  *      [dilations]:         use deformable_conv2d
  *      [data_format]:       use deformable_conv2d
  *      [dfm_groups]:        use deformable_conv2d
  *      [modulated]:         use deformable_conv2d
  *
  * @param dfm_conv_node DeformableConv2D node.
  * @param offset_desc New DeforableOffsets node.
  * @param with_bias Whether DeformableConv2D has bias.
  * @return bool Whether the DeforableOffsets desc is created successfully.
  */
bool DeformableConv2dPass::AddOffsetDesc(ge::NodePtr& dfm_conv_node, ge::OpDescPtr& offset_desc, bool with_bias){
  OpDescPtr dfm_conv_desc = dfm_conv_node->GetOpDesc();
  auto x_tensor = dfm_conv_desc->GetInputDesc(0);
  auto filter_tensor = dfm_conv_desc->GetInputDesc(1);
  auto offset_tensor = dfm_conv_desc->GetInputDesc(2);
  std::vector<std::string> offset_in_name = {"x", "offsets"};
  std::vector<GeTensorDesc> offset_in = {x_tensor, offset_tensor};
  for (size_t i = 0; i < offset_in.size(); ++i) {
    Status add_res = offset_desc->AddInputDesc(offset_in_name[i], offset_in[i]);
    FUSION_PASS_CHECK(add_res != GRAPH_SUCCESS,
                      OP_LOGW(fused_op_type_.c_str(), "add defomable_offset input failed"), return false);
  }
  std::string fmt_str;
  AttrUtils::GetStr(filter_tensor, kAttrOrgFmt, fmt_str);
  FUSION_PASS_CHECK(kFilterFmt.find(fmt_str) == kFilterFmt.end(),
                    OP_LOGW(fused_op_type_.c_str(), "unsupport filter format %s", fmt_str.c_str()), return false);
  size_t pos_h = fmt_str.find('H');
  size_t pos_w = fmt_str.find('W');
  if (pos_h == std::string::npos || pos_w == std::string::npos) {
    return false;
  }
  vector<int64_t> filter_shape = filter_tensor.GetOriginShape().GetDims();
  FUSION_PASS_CHECK(filter_shape.size() != 4,
                    OP_LOGW(fused_op_type_.c_str(), "filter shape is not 4D"), return false);
  std::vector<int64_t> ksize = {filter_shape[pos_h], filter_shape[pos_w]};
  AttrUtils::SetListInt(offset_desc, kAttrKsize, ksize);

  std::vector<std::string> offset_attrs = {kAttrStrides, kAttrPads, kAttrDilations};
  for (auto it = offset_attrs.begin(); it != offset_attrs.end(); ++it) {
    std::vector<int64_t> attr;
    FUSION_PASS_CHECK(!AttrUtils::GetListInt(dfm_conv_desc, *it, attr),
                      OP_LOGW(fused_op_type_.c_str(), "get %s attr failed", (*it).c_str()), return false);
    AttrUtils::SetListInt(offset_desc, *it, attr);
  };
  std::string data_format;
  FUSION_PASS_CHECK(!AttrUtils::GetStr(dfm_conv_desc, kAttrDataFmt, data_format),
                    OP_LOGW(fused_op_type_.c_str(), "get %s attr failed", kAttrDataFmt), return false);
  AttrUtils::SetStr(offset_desc, kAttrDataFmt, data_format);

  int64_t dfm_groups;
  FUSION_PASS_CHECK(!AttrUtils::GetInt(dfm_conv_desc, kAttrDfmGroups, dfm_groups),
                    OP_LOGW(fused_op_type_.c_str(), "get %s attr failed", kAttrDfmGroups), return false);
  AttrUtils::SetInt(offset_desc, kAttrDfmGroups, dfm_groups);

  bool modulated;
  FUSION_PASS_CHECK(!AttrUtils::GetBool(dfm_conv_desc, kAttrModulated, modulated),
                    OP_LOGW(fused_op_type_.c_str(), "get %s attr failed", kAttrModulated), return false);
  AttrUtils::SetBool(offset_desc, kAttrModulated, modulated);

  AttrUtils::GetStr(x_tensor, kAttrOrgFmt, fmt_str);
  pos_h = fmt_str.find('H');
  pos_w = fmt_str.find('W');
  if (pos_h == std::string::npos || pos_w == std::string::npos) {
    return false;
  }
  vector<int64_t> y_shape = x_tensor.GetOriginShape().GetDims();
  FUSION_PASS_CHECK(y_shape.size() != 4,
                    OP_LOGW(fused_op_type_.c_str(), "input x shape is not 4D"), return false);
  auto y_tensor = dfm_conv_desc->GetOutputDesc(0);
  vector<int64_t> out_shape = y_tensor.GetOriginShape().GetDims();
  FUSION_PASS_CHECK(out_shape.size() != 4,
                    OP_LOGW(fused_op_type_.c_str(), "output y shape is not 4D"), return false);
  AttrUtils::GetStr(y_tensor, kAttrOrgFmt, fmt_str);
  size_t out_pos_h = fmt_str.find('H');
  size_t out_pos_w = fmt_str.find('W');
  if (out_pos_h == std::string::npos || out_pos_w == std::string::npos) {
    return false;
  }
  if (PatternFusionUtil::IsUnknownShape(out_shape[out_pos_h]) ||
      PatternFusionUtil::IsUnknownShape(out_shape[out_pos_w]) ||
      PatternFusionUtil::IsUnknownShape(ksize[0]) ||
      PatternFusionUtil::IsUnknownShape(ksize[1])) {
    CUBE_INNER_ERR_REPORT(fused_op_type_.c_str(), "AvgPool1DFusionPass cannot be applied for unknown shape.");
    return false;
  }
  y_shape[pos_h] = out_shape[out_pos_h] * ksize[0];
  y_shape[pos_w] = out_shape[out_pos_w] * ksize[1];
  x_tensor.SetShape(ge::GeShape(y_shape));
  x_tensor.SetOriginShape(ge::GeShape(y_shape));
  Status add_res = offset_desc->AddOutputDesc("y", x_tensor);
  FUSION_PASS_CHECK(add_res != GRAPH_SUCCESS,
                    OP_LOGW(fused_op_type_.c_str(), "add defomable_offset output failed"), return false);
  return true;
}

/*!
  * @brief Add new conv2d node and set right attributes.
  *
  * conv2d args:
  *   tensor:
  *      x:              use deformable_offset output
  *      filter:         use deformable_conv2d
  *      bias:           use deformable_conv2d
  *      y:              use deformable_conv2d
  *   attrs:
  *      [strides]:      kernel_size(H/W)
  *      [pads]:         keep defualt
  *      [dilations]:    keep defualt
  *      [data_format]:  use deformable_conv2d
  *
  * @param dfm_conv_node DeformableConv2D node.
  * @param conv_desc New conv2d node.
  * @param with_bias Whether DeformableConv2D has bias.
  * @return bool Whether the DeforableOffsets desc is created successfully.
  */
bool DeformableConv2dPass::AddConvDesc(ge::NodePtr& dfm_conv_node, ge::OpDescPtr& conv_desc, bool with_bias){
  OpDescPtr dfm_conv_desc = dfm_conv_node->GetOpDesc();
  auto x_tensor = dfm_conv_desc->GetInputDesc(0);
  auto filter_tensor = dfm_conv_desc->GetInputDesc(1);
  std::string fmt_str;
  AttrUtils::GetStr(filter_tensor, kAttrOrgFmt, fmt_str);
  FUSION_PASS_CHECK(kFilterFmt.find(fmt_str) == kFilterFmt.end(),
                    OP_LOGW(fused_op_type_.c_str(), "unsupport filter format %s", fmt_str.c_str()), return false);
  size_t pos_h = fmt_str.find('H');
  size_t pos_w = fmt_str.find('W');
  if (pos_h == std::string::npos || pos_w == std::string::npos) {
    return false;
  }
  vector<int64_t> filter_shape = filter_tensor.GetOriginShape().GetDims();
  FUSION_PASS_CHECK(filter_shape.size() != 4,
                    OP_LOGW(fused_op_type_.c_str(), "filter shape is not 4D"), return false);
  std::vector<int64_t> ksize;
  ksize.push_back(filter_shape[pos_h]);
  ksize.push_back(filter_shape[pos_w]);

  AttrUtils::GetStr(x_tensor, kAttrOrgFmt, fmt_str);
  pos_h = fmt_str.find('H');
  pos_w = fmt_str.find('W');
  if (pos_h == std::string::npos || pos_w == std::string::npos) {
    return false;
  }
  vector<int64_t> x_shape = x_tensor.GetOriginShape().GetDims();
  FUSION_PASS_CHECK(x_shape.size() != 4,
                    OP_LOGW(fused_op_type_.c_str(), "input x shape is not 4D"), return false);

  auto y_tensor = dfm_conv_desc->GetOutputDesc(0);
  vector<int64_t> out_shape = y_tensor.GetOriginShape().GetDims();
  FUSION_PASS_CHECK(out_shape.size() != 4,
                    OP_LOGW(fused_op_type_.c_str(), "output y shape is not 4D"), return false);
  AttrUtils::GetStr(y_tensor, kAttrOrgFmt, fmt_str);
  size_t out_pos_h = fmt_str.find('H');
  size_t out_pos_w = fmt_str.find('W');
  if (out_pos_h == std::string::npos || out_pos_w == std::string::npos) {
    return false;
  }
  x_shape[pos_h] = out_shape[out_pos_h] * ksize[0];
  x_shape[pos_w] = out_shape[out_pos_w] * ksize[1];
  x_tensor.SetShape(ge::GeShape(x_shape));
  x_tensor.SetOriginShape(ge::GeShape(x_shape));
  std::vector<std::string> conv_in_name = {"x", "filter", "bias", "offset_w"};
  std::vector<GeTensorDesc> conv_in;
  conv_in.push_back(x_tensor);
  conv_in.push_back(filter_tensor);
  if (with_bias) {
    conv_in.push_back(dfm_conv_desc->GetInputDesc(3));
  }
  for (size_t i = 0; i < conv_in.size(); ++i) {
    Status add_res = conv_desc->AddInputDesc(conv_in_name[i], conv_in[i]);
    FUSION_PASS_CHECK(add_res != GRAPH_SUCCESS,
                      OP_LOGW(fused_op_type_.c_str(), "add conv2d input failed"), return false);
  }
  Status add_res = conv_desc->AddOutputDesc("y", dfm_conv_desc->GetOutputDesc(0));
  FUSION_PASS_CHECK(add_res != GRAPH_SUCCESS,
                    OP_LOGW(fused_op_type_.c_str(), "add conv2d output failed"), return false);

  std::vector<int64_t> strides;
  AttrUtils::GetListInt(dfm_conv_desc, kAttrStrides, strides);
  FUSION_PASS_CHECK(strides.size() != 4, CUBE_INNER_ERR_REPORT(fused_op_type_.c_str(), "get strides attr failed"), return false);
  strides[pos_h] = ksize[0];
  strides[pos_w] = ksize[1];
  AttrUtils::SetListInt(conv_desc, kAttrStrides, strides);
  std::vector<int64_t> def_pads(4);
  AttrUtils::SetListInt(conv_desc, kAttrPads, def_pads);
  std::vector<int64_t> def_dilations(4, 1);
  AttrUtils::SetListInt(conv_desc, kAttrDilations, def_dilations);
  int64_t groups;
  FUSION_PASS_CHECK(!AttrUtils::GetInt(dfm_conv_desc, kAttrGroups, groups),
                    OP_LOGW(fused_op_type_.c_str(), "get %s attr failed", kAttrGroups), return false);
  AttrUtils::SetInt(conv_desc, kAttrGroups, groups);
  std::string data_format;
  FUSION_PASS_CHECK(!AttrUtils::GetStr(dfm_conv_desc, kAttrDataFmt, data_format),
                    OP_LOGW(fused_op_type_.c_str(), "get %s attr failed", kAttrDataFmt), return false);
  AttrUtils::SetStr(conv_desc, kAttrDataFmt, data_format);
  return true;
}

/*!
  * @brief Transform deformable_conv2d to deformable_offset + conv2d.
  *
  * inputs    offsets
  *      \    /
  *        \/
  * deformable_offset  filter bias(if exist)
  *                  \   |   /
  *                     \ /
  *                   conv2d
  *                      |
  *
  * @param graph The whole graph.
  * @param mapping Matched nodes of defined pattern.
  * @param new_nodes New nodes added to graph.
  * @return Status Graph processing result.
  */
Status DeformableConv2dPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, std::vector<NodePtr>& new_nodes) {
  OP_LOGD(fused_op_type_.c_str(), "Enter DeformableConv2dPass.");
  NodePtr dfm_conv_node = GetNodeFromMapping(kPatternDfmConv2D, mapping);
  OpDescPtr dfm_conv_desc = dfm_conv_node->GetOpDesc();
  auto dfm_conv_in_name = dfm_conv_desc->GetAllOutputName();
  auto node_name = dfm_conv_desc->GetName();
  bool with_bias;
  size_t in_count = dfm_conv_node->GetInDataNodes().size();
  if (in_count == 3) {
    with_bias = false;
  } else if (in_count == 4) {
    with_bias = true;
  } else {
    OP_LOGW(fused_op_type_.c_str(), "unsupport DeformableConv2d %u inputs", in_count);
    return NOT_CHANGED;
  }
  const std::string new_offset_name = node_name + "_offset";
  OpDescPtr offset_desc(new ge::OpDesc(new_offset_name, kDfmOffsetType));
  FUSION_PASS_CHECK(offset_desc == nullptr,
                    OP_LOGW(fused_op_type_.c_str(), "create new offset desc failed"), return NOT_CHANGED);
  FUSION_PASS_CHECK(!AddOffsetDesc(dfm_conv_node, offset_desc, with_bias),
                    OP_LOGW(fused_op_type_.c_str(), "add new deformable_offset desc failed"), return NOT_CHANGED);
  NodePtr new_offset = graph.AddNode(offset_desc);
  FUSION_PASS_CHECK(new_offset == nullptr,
                    OP_LOGW(fused_op_type_.c_str(), "new offset is null,fusion failed"), return NOT_CHANGED);
  new_nodes.push_back(new_offset);
  const std::string new_conv_name = node_name + "_conv2d";
  OpDescPtr conv_desc(new ge::OpDesc(new_conv_name, kConv2DType));
  FUSION_PASS_CHECK(conv_desc == nullptr,
                    OP_LOGW(fused_op_type_.c_str(), "create new conv2d desc failed"), return NOT_CHANGED);
  FUSION_PASS_CHECK(!AddConvDesc(dfm_conv_node, conv_desc, with_bias),
                    OP_LOGW(fused_op_type_.c_str(), "add new conv2d desc failed"), return NOT_CHANGED);
  NodePtr new_conv = graph.AddNode(conv_desc);
  new_nodes.push_back(new_conv);
  std::vector<string> offset_in_data = {"x", "offsets"};
  for (size_t i = 0; i < offset_in_data.size(); ++i) {
    int idx = dfm_conv_desc->GetInputIndexByName(offset_in_data[i]);
    auto src_anchor = dfm_conv_node->GetInDataAnchor(idx)->GetPeerOutAnchor();
    Status add_res = GraphUtils::AddEdge(src_anchor, new_offset->GetInDataAnchor(i));
    FUSION_PASS_CHECK(add_res != GRAPH_SUCCESS,
                      CUBE_INNER_ERR_REPORT(fused_op_type_.c_str(), "add edge of new deformable_offset inputs failed"), return FAILED);
  }
  std::vector<string> conv_in_data = {"filter"};
  if (with_bias) {
    conv_in_data.push_back("bias");
  }
  for (size_t i = 0; i < conv_in_data.size(); ++i) {
    int idx = dfm_conv_desc->GetInputIndexByName(conv_in_data[i]);
    auto src_anchor = dfm_conv_node->GetInDataAnchor(idx)->GetPeerOutAnchor();
    Status add_res = GraphUtils::AddEdge(src_anchor, new_conv->GetInDataAnchor(i + 1));
    FUSION_PASS_CHECK(add_res != GRAPH_SUCCESS,
                      CUBE_INNER_ERR_REPORT(fused_op_type_.c_str(), "add edge of new conv2d inputs failed"), return FAILED);
  }
  Status add_res = GraphUtils::AddEdge(new_offset->GetOutDataAnchor(0), new_conv->GetInDataAnchor(0));
  FUSION_PASS_CHECK(add_res != GRAPH_SUCCESS,
                    CUBE_INNER_ERR_REPORT(fused_op_type_.c_str(), "add edge from deformable_offset to conv2d failed"), return FAILED);
  auto out_anchor = dfm_conv_node->GetOutDataAnchor(0);
  auto peer_in_anchor = out_anchor->GetPeerInDataAnchors();
  for (size_t i = 0; i < peer_in_anchor.size(); ++i) {
    out_anchor->Unlink(peer_in_anchor.at(i));
    Status add_res = GraphUtils::AddEdge(new_conv->GetOutDataAnchor(0), peer_in_anchor.at(i));
    FUSION_PASS_CHECK(add_res != GRAPH_SUCCESS,
                      CUBE_INNER_ERR_REPORT(fused_op_type_.c_str(), "add conv2d output edge failed"), return FAILED);
  }
  FUSION_PASS_CHECK(graph.RemoveNode(dfm_conv_node) != GRAPH_SUCCESS,
                    CUBE_INNER_ERR_REPORT(fused_op_type_.c_str(), "remove deformable_conv2d node failed"), return FAILED);
  OP_LOGD(fused_op_type_.c_str(), "Leave DeformableConv2dPass.");

  return SUCCESS;
}

REGISTER_PASS("ADeformableConv2dPass", BUILT_IN_GRAPH_PASS, DeformableConv2dPass);
}  // namespace fe
