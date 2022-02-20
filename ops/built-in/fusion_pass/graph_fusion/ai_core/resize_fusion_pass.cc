/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 *
 * @brief resize fusion pass(resize -->
 * ResizeNearestNeighborV2D/ResizeBilinearV2D)
 *
 */

#include "resize_fusion_pass.h"

#include <algorithm>
#include <cmath>
#include <map>
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
#include "tbe_fusion_pass_util.h"

using namespace ge;
namespace fe {
static const char *fused_node_resize = "Resize";
static const std::string PATTERN_FUSEDNODE = "ResizeD";
static std::string mode_names[] = {"nearest", "linear"};

static bool DeleteInput(ge::ComputeGraph &graph, ge::NodePtr &node,
                        ge::OpDescPtr &desc, uint32_t index) {
  ge::InDataAnchorPtr anchor = node->GetInDataAnchor(index);
  ge::OutDataAnchorPtr const_anchor = anchor->GetPeerOutAnchor();
  ge::NodeUtils::ClearInDataAnchor(node, anchor);
  if (const_anchor != nullptr) {
    ge::GraphUtils::RemoveEdge(const_anchor, anchor);
    ge::NodePtr const_node = const_anchor->GetOwnerNode();
    if (PatternFusionUtil::GetOutEdgeSize(const_node) == 0) {
      FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(const_node),
                        OP_LOGE("Resize", "Remove Node[%s] failed",
                                const_node->GetName().c_str()),
                        return false);
      OP_LOGD("Resize", "Remove Node:[%s].", const_node->GetName().c_str());
    } else {
      OP_LOGE("Reszie", "Node:[%s] have output link to other node.",
              const_node->GetName().c_str());
    }
  }
  if (!ge::OpDescUtils::ClearInputDesc(desc, index)) {
    OP_LOGE("Resize", "fail to clear input desc[%d]", index);
  }
  return true;
}

static void DeleteAttr(ge::OpDescPtr &desc) {
  desc->DelAttr("coordinate_transformation_mode");
  desc->DelAttr("cubic_coeff_a");
  desc->DelAttr("exclude_outside");
  desc->DelAttr("extrapolation_value");
  desc->DelAttr("mode");
  desc->DelAttr("nearest_mode");
}

static bool MatchModeName(const ge::Operator &op, std::string &mode_name) {
  std::string *ptr;
  if (op.GetAttr("mode", mode_name) != GRAPH_SUCCESS) {
    OP_LOGE(op.GetName().c_str(), "Get attr mode failed.");
  }
  int8_t names_size = sizeof(mode_names) / sizeof(std::string);
  ptr = find(mode_names, mode_names + names_size, mode_name.c_str());
  if (ptr == mode_names + names_size) {
    OP_LOGE(op.GetName().c_str(), "mode name is not nearest or linear.");
    return false;
  }
  return true;
}

ge::NodePtr CopyNewNode(ge::ComputeGraph &graph, ge::OpDescPtr &desc,
                        const std::string &mode_name) {
  std::string fusion_op_type;
  ge::OpDescPtr resize_fused_desc = AttrUtils::CloneOpDesc(desc);
  if (mode_name == "nearest") {
    fusion_op_type = "ResizeNearestNeighborV2D";
  } else if (mode_name == "linear") {
    fusion_op_type = "ResizeBilinearV2D";
  } else {
    OP_LOGE("Reszie", "Unsupported interpolation mode");
  }
  desc->SetType(fusion_op_type);
  ge::NodePtr fused_node = graph.AddNode(resize_fused_desc);
  return fused_node;
}

vector<FusionPattern *> ResizeFusionPass::DefinePatterns() {
  vector<FusionPattern *> patterns;

  FusionPattern *pattern = new (std::nothrow)
      FusionPattern("ReszieToResizeNearestNeighborV2DFusion");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE("Reszie", "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_FUSEDNODE, {fused_node_resize})
      .SetOutput(PATTERN_FUSEDNODE);

  patterns.push_back(pattern);

  return patterns;
}

Status ResizeFusionPass::Fusion(ge::ComputeGraph &graph, Mapping &mapping,
                                vector<ge::NodePtr> &fusionNodes) {
  // get resize node and resize-desc
  ge::NodePtr fuse_node = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(fuse_node == nullptr,
                    OP_LOGE("Resize", "fuse_node is null, fusion failed."),
                    return PARAM_INVALID);
  ge::OpDescPtr fuse_desc = fuse_node->GetOpDesc();
  FUSION_PASS_CHECK(
      fuse_desc == nullptr,
      OP_LOGE("Resize", "fuse_node's OpDesc is null, fusion failed."),
      return PARAM_INVALID);

  // get the attr_mode
  Operator op = ge::OpDescUtils::CreateOperatorFromNode(fuse_node);
  std::string mode_name;
  if (!MatchModeName(op, mode_name)) {
    return PARAM_INVALID;
  }

  Format input_format = fuse_desc->GetInputDesc("x").GetFormat();
  std::vector<int64_t> dims =
      fuse_desc->GetOutputDesc("y").GetShape().GetDims();
  for (int64_t ele : dims) {
    FUSION_PASS_CHECK(
        ele == UNKNOWN_DIM,
        OP_LOGE("Resize", "It is unknown output shape, not changed"),
        return NOT_CHANGED);
  }

  // acquire output_size
  std::vector<int64_t> output_size;
  if (input_format == FORMAT_NHWC) {
    output_size.push_back(dims[1]);  // 1 is index
    output_size.push_back(dims[2]);  // 2 is index
  } else if (input_format == FORMAT_NCHW) {
    output_size.push_back(dims[2]);  // 2 is index
    output_size.push_back(dims[3]);  // 3 is index
  } else {
    OP_LOGW(op.GetName().c_str(),
            "Not supported this format%d, output tensor will be wrong",
            input_format);
  }

  // delete input_roi,input_scales,input_sizes
  for (int i = 0; i < 3; i++) {  // 3 is lenth of input
    DeleteInput(graph, fuse_node, fuse_desc, 1);
  }

  // set attr_align_corners,attr_half_pixel_centers,attr_size
  std::string coordinate_transformation_mode_value;
  if (op.GetAttr("coordinate_transformation_mode",
                 coordinate_transformation_mode_value) != GRAPH_SUCCESS) {
    OP_LOGI(op.GetName().c_str(),
            "Get attr coordinate transformation mode failed, set to default.");
  }
  bool half_pixel_centers = false;
  bool align_corners = false;
  if (coordinate_transformation_mode_value == "pytorch_half_pixel") {
    half_pixel_centers = true;
  } else if (coordinate_transformation_mode_value == "align_corners"){
    align_corners = true;
  }
  ge::AttrUtils::SetBool(fuse_desc, "half_pixel_centers", half_pixel_centers);
  ge::AttrUtils::SetBool(fuse_desc, "align_corners", align_corners);
  ge::AttrUtils::SetListInt(fuse_desc, "size", output_size);

  // delete attr_coordinate_transformation_mode,attr_cubic_coeff_a,
  // attr_exclude_outside,attr_extrapolation_value,attr_mode,attr_nearest_mode
  DeleteAttr(fuse_desc);

  // copy to new node
  ge::NodePtr resize_fuse_node = CopyNewNode(graph, fuse_desc, mode_name);

  fusionNodes.push_back(resize_fuse_node);

  return SUCCESS;
}

REGISTER_PASS("ResizeFusionPass", BUILT_IN_GRAPH_PASS, ResizeFusionPass);
}  // namespace fe
