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
 * \file conv3d_bp_filter_group_fusion_pass.cc
 * \brief
 */
#include "conv3d_bp_filter_group_fusion_pass.h"

#include <string>
#include <vector>
#include <map>
#include <cmath>

#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "op_log.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "securec.h"
#include "pattern_fusion_util.h"
#include "common/util/error_manager/error_manager.h"
#include "../../../op_proto/util/error_util.h"
#include "anchor_util.h"

using namespace std;
using namespace ge;

namespace fe {
static const string CONV3D_DW = "Conv3DBackpropFilterD";
static const string CONV3D_DW_DYNAMIC = "Conv3DBackpropFilter";
static const string PATTERN_CONV3D_DW_GROUP = "Conv3DBpFilterGroup";
static constexpr int CHANNEL_MIN  = 16;
static constexpr int CONV3D_ORI_DIMS = 5;

vector<FusionPattern*> Conv3DBpFilterGroupFusionPass::DefinePatterns() {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "enter Conv3DBpFilterGroupFusionPass::DefinePatterns.");
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern(PATTERN_CONV3D_DW_GROUP);
  FUSION_PASS_CHECK(pattern == nullptr,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  // define origin graph
  pattern->AddOpDesc(PATTERN_CONV3D_DW_GROUP, {CONV3D_DW, CONV3D_DW_DYNAMIC})
    .SetOutput(PATTERN_CONV3D_DW_GROUP);

  patterns.push_back(pattern);

  return patterns;
}

Status Conv3DBpFilterGroupFusionPass::GetChannelValue(const ge::OpDescPtr& dw_desc,
                                                      const std::string& name, int64_t& channel) {
  FUSION_PASS_CHECK(dw_desc == nullptr,
                    CUBE_CALL_ERR_REPORT(FUSED_OP_TYPE.c_str(), "dw_desc is NULL."),
                    return FAILED);
  GeTensorDesc in_desc = dw_desc->GetInputDesc(name);
  auto format = in_desc.GetOriginFormat();
  auto dims = in_desc.GetOriginShape().GetDims();

  if (dims.size() != CONV3D_ORI_DIMS) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "The original dimension of the input is not equal to 5.");
    return FAILED;
  }

  if (format == ge::FORMAT_NDHWC) {
    channel = dims[4];
  } else if (format == ge::FORMAT_NCDHW) {
    channel = dims[1];
  } else {
    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
      "original format[%d] of the input is unsupported.", format);
    return FAILED;
  }

  return SUCCESS;
}

int64_t Conv3DBpFilterGroupFusionPass::LCM(int64_t numL, int64_t numR) {
  if (numR == 0 || numL == 0) {
    return 1;
  }
  int64_t product = numL * numR;
  while (numL % numR != 0) {
    int64_t tmp = numL % numR;
    numL = numR;
    numR = tmp;
  }

  return product / numR;
}

Status Conv3DBpFilterGroupFusionPass::CalculateGroup(int64_t in_channel, int64_t out_channel, int64_t groups,
                                                     std::map<std::string, int64_t>& group_map) {
  if (in_channel % groups != 0 || out_channel % groups != 0) {
    OP_LOGW(FUSED_OP_TYPE.c_str(),
      "The number of input channel(%lld) or output channel(%lld) of "
      "the dw node is not divisible by groups(%lld).",
      in_channel, out_channel, groups);
    return FAILED;
  }

  int64_t mag_factor0 = LCM(in_channel / groups, CHANNEL_MIN) / (in_channel / groups);
  int64_t mag_factor1 = LCM(out_channel / groups, CHANNEL_MIN) / (out_channel / groups);
  int64_t mag_factor = min(LCM(mag_factor0, mag_factor1), groups);
  int64_t cin1_g = (mag_factor * in_channel / groups + CHANNEL_MIN - 1) / CHANNEL_MIN;
  int64_t cout_g = (mag_factor * out_channel / groups + CHANNEL_MIN - 1) / CHANNEL_MIN * CHANNEL_MIN;
  int64_t real_g = (groups + mag_factor - 1) / mag_factor;

  group_map["real_g"] = real_g;
  group_map["mag_factor"] = mag_factor;
  group_map["cin1_g"] = cin1_g;
  group_map["cout_g"] = cout_g;
  group_map["cin_ori"] = in_channel;
  group_map["cout_ori"] = out_channel;
  group_map["groups"] = groups;

  OP_LOGI(FUSED_OP_TYPE.c_str(),
    "The group_map value: real_g[%lld], mag_factor[%lld], cin1_g[%lld], "
    "cout_g[%lld], in_channel[%lld], out_channel[%lld], groups[%lld].",
    real_g, mag_factor, cin1_g, cout_g, in_channel, out_channel, groups);

  return SUCCESS;
}

Status Conv3DBpFilterGroupFusionPass::TransOutDims2dhwcn(const ge::OpDescPtr& dw_desc,
                                                         std::vector<int64_t>& dims) {
  GeTensorDesc out_desc = dw_desc->GetOutputDesc(0);
  auto out_format = out_desc.GetOriginFormat();
  auto out_dims = out_desc.GetOriginShape().GetDims();
  if (out_dims.size() != CONV3D_ORI_DIMS) {
    OP_LOGW(FUSED_OP_TYPE.c_str(),
            "The original shape of the output is not 5d.");
    return FAILED;
  }

  if (out_format == ge::FORMAT_NDHWC) {
    dims.push_back(out_dims[1]);
    dims.push_back(out_dims[2]);
    dims.push_back(out_dims[3]);
    dims.push_back(out_dims[4]);
    dims.push_back(out_dims[0]);
  } else if (out_format == ge::FORMAT_NCDHW) {
    dims.push_back(out_dims[2]);
    dims.push_back(out_dims[3]);
    dims.push_back(out_dims[4]);
    dims.push_back(out_dims[1]);
    dims.push_back(out_dims[0]);
  } else if (out_format == ge::FORMAT_DHWCN) {
    dims = out_dims;
  } else {
    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                          "The original format of the output is [%d], which is unsupportable.", out_format);
    return FAILED;
  }

  return SUCCESS;
}

void Conv3DBpFilterGroupFusionPass::SetMultiplierValue(float* data, const std::vector<int64_t>& dims,
                                                       const std::map<std::string, int64_t>& group_map) {
  int64_t kernel_depth =  dims[0];
  int64_t kernel_height =  dims[1];
  int64_t kernel_width =  dims[2];

  int64_t real_g = group_map.at("real_g");
  int64_t cin1_g = group_map.at("cin1_g");
  int64_t cout_g = group_map.at("cout_g");
  int64_t in_channel = group_map.at("cin_ori");
  int64_t out_channel = group_map.at("cout_ori");
  int64_t groups = group_map.at("groups");

  int64_t cin_group = in_channel / groups;
  int64_t cout_group = out_channel /groups;
  int64_t g_base = kernel_depth * cin1_g * kernel_height * kernel_width * cout_g * CHANNEL_MIN;
  int64_t d_base = cin1_g * kernel_height * kernel_width * cout_g * CHANNEL_MIN;
  int64_t cin_base = kernel_height * kernel_width * cout_g * CHANNEL_MIN;
  int64_t hw_base = cout_g * CHANNEL_MIN;
  for (int64_t g = 0; g < real_g; ++g) {
    int64_t g_offset = g * g_base;
    for (int64_t d = 0; d < kernel_depth; ++d) {
      int64_t d_offset = d * d_base;
      for (int64_t cin = 0; cin < cin1_g; ++cin) {
        int64_t cin_offset = cin * cin_base;
        for (int64_t hw = 0; hw < kernel_height * kernel_width; ++hw) {
          int64_t hw_offset = hw * hw_base;
          for (int64_t cout = 0; cout < cout_g; ++cout) {
            int64_t cout_offset = cout * CHANNEL_MIN;
            for (int64_t c0 = 0; c0 < CHANNEL_MIN; ++c0) {
              int64_t cin_index = g * cin1_g * CHANNEL_MIN + cin * CHANNEL_MIN + c0;
              int64_t cout_index = g * cout_g + cout;
              if (cin_index < in_channel && cout_index < out_channel &&
                  cin_index / cin_group == cout_index / cout_group) {
                *(data + g_offset + d_offset + cin_offset + hw_offset + cout_offset + c0) = 1.0;
              }
            }
          }
        }
      }
    }
  }
}

bool Conv3DBpFilterGroupFusionPass::GenMultiplier(ge::ComputeGraph& graph,
                                                  const std::vector<int64_t>& dims,
                                                  const std::map<std::string, int64_t>& group_map,
                                                  ge::NodePtr& const_node) {
  int64_t kernel_depth = dims[0];
  int64_t kernel_height = dims[1];
  int64_t kernel_width = dims[2];

  int64_t real_g = group_map.at("real_g");
  int64_t cin1_g = group_map.at("cin1_g");
  int64_t cout_g = group_map.at("cout_g");

  GeTensorPtr multiplier_ptr{nullptr};
  int64_t multiplier_size = real_g * kernel_depth * cin1_g * kernel_height * kernel_width * cout_g * CHANNEL_MIN;
  unique_ptr<float[]> multiplier_mem(new (std::nothrow) float[multiplier_size]());
  FUSION_PASS_CHECK(multiplier_mem.get() == nullptr,
                    CUBE_CALL_ERR_REPORT(FUSED_OP_TYPE.c_str(), "multiplier is NULL."),
                    return false);
  float* data_ptr = multiplier_mem.get();
  FUSION_PASS_CHECK(
    0 != memset_s(data_ptr, multiplier_size * sizeof(float),
                  0, multiplier_size * sizeof(float)),
    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "memset failed."),
    return false);

  SetMultiplierValue(data_ptr, dims, group_map);

  vector<int64_t> const_dims{real_g * kernel_depth * cin1_g * kernel_height * kernel_width,
                             cout_g / CHANNEL_MIN, CHANNEL_MIN, CHANNEL_MIN};
  GeShape const_shape(const_dims);
  GeTensorDesc const_tensor_desc(const_shape, FORMAT_FRACTAL_Z_3D, DT_FLOAT);

  GeShape ori_shape(dims);
  const_tensor_desc.SetOriginShape(ori_shape);
  const_tensor_desc.SetOriginFormat(FORMAT_DHWCN);
  const_tensor_desc.SetOriginDataType(DT_FLOAT);

  FUSION_PASS_MAKE_SHARED(
    (multiplier_ptr = std::make_shared<GeTensor>(const_tensor_desc,
                                                 reinterpret_cast<uint8_t*>(data_ptr),
                                                 multiplier_size * sizeof(float))),
    return false);

  OpDescPtr const_opdesc = OpDescUtils::CreateConstOp(multiplier_ptr);
  FUSION_PASS_CHECK(
    const_opdesc == nullptr,
    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Create const op desc failed."),
    return false);
  const_node = graph.AddNode(const_opdesc);

  return true;
}

bool Conv3DBpFilterGroupFusionPass::GenerateMulNode(ge::ComputeGraph& graph,
                                                    const std::vector<int64_t>& dims,
                                                    const std::map<std::string, int64_t>& group_map,
                                                    const ge::OpDescPtr& conv_desc,
                                                    ge::NodePtr& mul_node) {
  OpDescPtr mul_desc;
  string conv_name = conv_desc->GetName();
  GeTensorDesc conv_out_desc = conv_desc->GetOutputDesc(0);
  GeTensorDesc mul_in_desc = conv_out_desc;
  GeTensorDesc mul_out_desc = conv_out_desc;

  FUSION_PASS_MAKE_SHARED((mul_desc = std::make_shared<ge::OpDesc>(conv_name + "_mul", "Mul")), return false);
  FUSION_PASS_CHECK(mul_desc == nullptr,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "mul_desc is null."),
                    return false);
  mul_desc->AddInputDesc(mul_in_desc);
  mul_desc->AddInputDesc(mul_in_desc);
  mul_desc->AddOutputDesc(mul_out_desc);

  mul_node = graph.AddNode(mul_desc);

  return true;
}

bool Conv3DBpFilterGroupFusionPass::Relink(ge::NodePtr& conv_node,
  ge::NodePtr& mul_node, ge::NodePtr& const_node) {
  Node::Vistor<NodePtr> out_nodes = conv_node->GetOutAllNodes();
  std::vector<int> outAnchorIndexes;
  ge::OutDataAnchorPtr convNodePtr = conv_node->GetOutDataAnchor(0);
  FUSION_PASS_CHECK(convNodePtr == nullptr,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "convNodePtr is null."),
                    return false);
  FUSION_PASS_CHECK(convNodePtr->GetPeerInDataAnchors().size() < out_nodes.size(),
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "in data is null."),
                    return false);
  ge::InDataAnchorPtr inDataPtr = nullptr;
  for (size_t i = 0; i < out_nodes.size(); ++i) {
    inDataPtr = convNodePtr->GetPeerInDataAnchors().at(i);
    FUSION_PASS_CHECK(inDataPtr == nullptr,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "inDataPtr is null."),
                    return false);
    outAnchorIndexes.push_back(inDataPtr->GetIdx());
  }
  for (auto out_anchor : conv_node->GetAllOutDataAnchors()) {
    if (out_anchor != nullptr) {
      out_anchor->UnlinkAll();
    }
  }

  graphStatus status = GraphUtils::AddEdge(const_node->GetOutAnchor(0), mul_node->GetInDataAnchor(0));
  FUSION_PASS_CHECK(status != GRAPH_SUCCESS,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add const data to mul edge fail."),
                    return false);
  status = GraphUtils::AddEdge(conv_node->GetOutAnchor(0), mul_node->GetInDataAnchor(1));
  FUSION_PASS_CHECK(status != GRAPH_SUCCESS,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add conv to mul edge fail."),
                    return false);
  for (size_t i = 0; i < out_nodes.size(); ++i) {
    status = GraphUtils::AddEdge(mul_node->GetOutAnchor(0),
                                 out_nodes.at(i)->GetInDataAnchor(outAnchorIndexes[i]));
    FUSION_PASS_CHECK(status != GRAPH_SUCCESS,
                      CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add mul to output edge fail."),
                      return false);
  }

  return true;
}

Status Conv3DBpFilterGroupFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                           vector<ge::NodePtr>& new_nodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "enter Conv3DBpFilterGroupFusionPass::Fusion");
  NodePtr dw_node = GetNodeFromMapping(PATTERN_CONV3D_DW_GROUP, mapping);
  FUSION_PASS_CHECK(dw_node == nullptr,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "dw_node is null."),
                    return FAILED);
  auto dw_desc = dw_node->GetOpDesc();
  FUSION_PASS_CHECK(dw_desc == nullptr,
                    CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "dw_desc is null."),
                    return FAILED);

  int64_t groups =1;
  bool has_group_flag = ge::AttrUtils::GetInt(dw_desc, "groups", groups);
  if (!has_group_flag || groups <= 1) {
    OP_LOGD(FUSED_OP_TYPE.c_str(),
      "The dw node[name=%s, type=%s] doesn't have the attribute "
      "groups, or the value is not more than 1.",
      dw_desc->GetName().c_str(), dw_desc->GetType().c_str());
    return NOT_CHANGED;
  }

  int64_t dx_channel = 1;
  FUSION_PASS_CHECK(
    SUCCESS != GetChannelValue(dw_desc, "x", dx_channel),
    OP_LOGW(
      FUSED_OP_TYPE.c_str(),
      "Get the channel value of the feature map failed."),
    return NOT_CHANGED);

  int64_t dy_channel = 1;
  FUSION_PASS_CHECK(
    SUCCESS != GetChannelValue(dw_desc, "out_backprop", dy_channel),
    OP_LOGW(FUSED_OP_TYPE.c_str(),
      "Get the channel value of the out backprop failed."),
    return NOT_CHANGED);

  if (PatternFusionUtil::IsUnknownShape(dx_channel) ||
      PatternFusionUtil::IsUnknownShape(dy_channel)) {
    OP_LOGW(FUSED_OP_TYPE.c_str(),
             "Conv3DBpFilterGroupFusionPass cannot be applied for unkonwn shape.");
    return NOT_CHANGED;
  }

  std::map<std::string, int64_t> group_map;
  FUSION_PASS_CHECK(
    SUCCESS != CalculateGroup(dx_channel, dy_channel, groups, group_map),
    OP_LOGW(FUSED_OP_TYPE.c_str(),
      "Conv3DBpFilterGroupFusionPass calculate new group failed."),
    return NOT_CHANGED);

  if (group_map["mag_factor"] <= 1) {
    OP_LOGD(FUSED_OP_TYPE.c_str(),
            "Conv3DBpFilterGroupFusionPass do not need to insert Mul node.");
    return NOT_CHANGED;
  }

  std::vector<int64_t> out_dims;
  FUSION_PASS_CHECK(
    SUCCESS != TransOutDims2dhwcn(dw_desc, out_dims),
    OP_LOGW(FUSED_OP_TYPE.c_str(),
      "Conv3DBpFilterGroupFusionPass get out dims failed."),
    return NOT_CHANGED);

  ge::NodePtr mul_node;
  FUSION_PASS_CHECK(
    !GenerateMulNode(graph, out_dims, group_map, dw_desc, mul_node),
    OP_LOGW(FUSED_OP_TYPE.c_str(),
      "Conv3DBpFilterGroupFusionPass generate mul node failed."),
    return NOT_CHANGED);

  ge::NodePtr const_node;
  FUSION_PASS_CHECK(
    !GenMultiplier(graph, out_dims, group_map, const_node),
    OP_LOGW(FUSED_OP_TYPE.c_str(),
      "Conv3DBpFilterGroupFusionPass generate const node failed."),
    return NOT_CHANGED);

  FUSION_PASS_CHECK(
    !Relink(dw_node, mul_node, const_node),
    OP_LOGW(FUSED_OP_TYPE.c_str(),
      "Conv3DBpFilterGroupFusionPass link nodes failed."),
    return NOT_CHANGED);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Conv3DBpFilterGroupFusionPass pass handle success!!!!");

  return SUCCESS;
}
REGISTER_PASS("Conv3DBpFilterGroupFusionPass", BUILT_IN_GRAPH_PASS, Conv3DBpFilterGroupFusionPass);
}  // namespace fe
