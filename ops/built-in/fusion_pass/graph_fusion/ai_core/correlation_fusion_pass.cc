/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
 * \file correlation_fusion_pass.cpp
 * \brief
 */
#include "correlation_fusion_pass.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <sstream>
#include "graph/utils/graph_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"

using namespace std;
using namespace ge;
namespace fe {
static const char *FUSED_NODE = "Correlation";
static const std::string PATTERN_FUSED_NODE = "Correlation";
static const std::string PASS_NAME = "CorrelationFusionPass";

static Status generate_reshape_node(ge::ComputeGraph &graph, ge::GeTensorDesc &prev_out_desc,
                                    ge::GeTensorDesc &next_in_desc, ge::GeShape &shape, ge::NodePtr &shape_node,
                                    const std::string &name, const std::string &basename)
{
  ge::OpDescPtr reshape_desc = std::make_shared<ge::OpDesc>(basename + "_const_fold_" + name, "Reshape");
  reshape_desc->AddInputDesc("x", prev_out_desc);
  next_in_desc.SetShape(shape);
  next_in_desc.SetOriginShape(shape);
  reshape_desc->AddOutputDesc("y", next_in_desc);
  ge::AttrUtils::SetListInt(reshape_desc, "shape", shape.GetDims());
  shape_node = graph.AddNode(reshape_desc);
  return SUCCESS;
}

// split the N axis for x
static Status generate_split_node_for_x(ComputeGraph &graph, OpDescPtr conv_desc, int groups,
                                        NodePtr &split_node, GeTensorDesc &split_out_desc)
{
  OpDescPtr slice_desc;
  string conv_op_name = conv_desc->GetName();
  GeTensorDesc input_desc = conv_desc->GetInputDesc(1);
  GeShape input_shape = input_desc.GetShape();
  GeShape split_out_shape = input_shape;
  split_out_shape.SetDim(0, 1);

  slice_desc = std::make_shared<ge::OpDesc>(conv_op_name+"_x_split", "SplitD");
  AttrUtils::SetInt(slice_desc, "split_dim", 0);
  AttrUtils::SetInt(slice_desc, "num_split", groups);
  split_out_desc = input_desc;
  split_out_desc.SetShape(split_out_shape);
  split_out_desc.SetOriginShape(split_out_shape);
  slice_desc->AddInputDesc(input_desc);
  for (int i = 0; i < groups; i++) {
    slice_desc->AddOutputDesc(split_out_desc);
  }
  split_node = graph.AddNode(slice_desc);

  return SUCCESS;
}

static Status generate_split_node_for_filter(ComputeGraph &graph, OpDescPtr conv_desc, int groups,
                                             NodePtr &split_node, GeTensorDesc &split_out_desc)
{
  OpDescPtr slice_desc;
  string conv_op_name = conv_desc->GetName();
  GeTensorDesc input_desc = conv_desc->GetInputDesc(0);
  GeShape input_shape = input_desc.GetShape();
  Format format = input_desc.GetFormat();
  GeShape split_out_shape = input_shape;
  int32_t dim_0 = 0;
  int32_t dim_1 = 1;
  int32_t dim_3 = 3;

  slice_desc = std::make_shared<ge::OpDesc>(conv_op_name+"_filter_split", "SplitD");
  AttrUtils::SetInt(slice_desc, "num_split", groups);
  if (format == FORMAT_NCHW) {
    AttrUtils::SetInt(slice_desc, "split_dim", 0);
    split_out_shape.SetDim(dim_0, dim_1);
  } else if (format == FORMAT_NHWC) {
    AttrUtils::SetInt(slice_desc, "split_dim", 0);
    split_out_shape.SetDim(dim_0, dim_1);
  } else if (format == FORMAT_HWCN) {
    AttrUtils::SetInt(slice_desc, "split_dim", 3);
    split_out_shape.SetDim(dim_3, dim_1);
  }

  split_out_desc = input_desc;
  split_out_desc.SetShape(split_out_shape);
  split_out_desc.SetOriginShape(split_out_shape);
  slice_desc->AddInputDesc(input_desc);
  for (int i = 0; i < groups; i++) {
    slice_desc->AddOutputDesc(split_out_desc);
  }
  split_node = graph.AddNode(slice_desc);

  return SUCCESS;
}

static Status generate_new_conv_nodes(ComputeGraph &graph, OpDescPtr conv_desc,
                                      std::map <string, uint32_t> &input_names,
                                      int groups, const GeTensorDesc &split_out_desc_x,
                                      const GeTensorDesc &split_out_desc_filter,
                                      vector<NodePtr> &new_conv_nodes,
                                      GeTensorDesc &new_conv_out_desc)
{
  string conv_op_name = conv_desc->GetName();
  for (int64_t i = 0; i < groups; i++) {
    ostringstream new_conv_name;
    new_conv_name << conv_op_name << "_conv2d_" << i;
    OpDescPtr new_conv_desc = AttrUtils::CopyOpDesc(conv_desc);
    if (new_conv_desc == nullptr) {
      OP_LOGE(PASS_NAME.c_str(), "Node:%s's OpDesc is null, fusion failed.", conv_desc->GetName().c_str());
      return PARAM_INVALID;
    }
    // set type as Conv2D
    new_conv_desc->SetType("Conv2D");
    new_conv_desc->SetName(new_conv_name.str());
    // update input names
    new_conv_desc->UpdateInputName(input_names);
    // x
    new_conv_desc->UpdateInputDesc(0, split_out_desc_x);
    // filter
    new_conv_desc->UpdateInputDesc(1, split_out_desc_filter);
    // y
    new_conv_desc->UpdateOutputDesc(0, new_conv_out_desc);
    // set attr pads and strides of Conv2D
    AttrUtils::SetListInt(new_conv_desc, "pads", {0, 0, 0, 0});
    AttrUtils::SetListInt(new_conv_desc, "strides", {1, 1, 1, 1});
    AttrUtils::SetListInt(new_conv_desc, "dilations", {1, 1, 1, 1});
    AttrUtils::SetInt(new_conv_desc, "groups", 1);
    AttrUtils::SetInt(new_conv_desc, "offset_x", 0);
    AttrUtils::SetStr(new_conv_desc, "data_format", "NHWC");

    NodePtr new_conv_node = graph.AddNode(new_conv_desc);
    new_conv_nodes.push_back(new_conv_node);
  }

  return SUCCESS;
}

static Status generate_concat_node(ComputeGraph &graph, OpDescPtr conv_desc, int64_t groups,
                                   GeTensorDesc &new_conv_out_desc, GeTensorDesc &concat_out_desc,
                                   NodePtr &concat_node)
{
  string conv_op_name = conv_desc->GetName();
  OpDescPtr concat_desc = std::make_shared<ge::OpDesc>(conv_op_name+"_concat", "ConcatD");
  for (int i = 0; i < groups; i++) {
    concat_desc->AddInputDesc(new_conv_out_desc);
  }
  concat_desc->AddOutputDesc(concat_out_desc);
  AttrUtils::SetInt(concat_desc, "concat_dim", 0); // N axis concat
  concat_node = graph.AddNode(concat_desc);

  return SUCCESS;
}

vector<FusionPattern *> CorrelationFusionPass::DefinePatterns()
{
  vector < FusionPattern *> patterns;
  FusionPattern *pattern = new(std::nothrow) FusionPattern("CorrelationFusionPass");
  if (pattern == nullptr) {
    OP_LOGE(PASS_NAME.c_str(), "new a pattern object failed.");
    return patterns;
  }
  pattern->AddOpDesc(PATTERN_FUSED_NODE, {FUSED_NODE}).SetOutput(PATTERN_FUSED_NODE);
  patterns.push_back(pattern);

  return patterns;
}

Status CorrelationFusionPass::Fusion(ge::ComputeGraph &graph, Mapping &mapping, vector<ge::NodePtr> &newNodes)
{
  // get the NodePtr of Correlation
  ge::NodePtr fused_node = GetNodeFromMapping(PATTERN_FUSED_NODE, mapping);
  if (fused_node == nullptr) {
    OP_LOGE(PASS_NAME.c_str(), "fused_node is null, fusion failed.");
    return PARAM_INVALID;
  }

  // get the OpDescPtr of Correlation
  ge::OpDescPtr fused_desc = fused_node->GetOpDesc();
  if (fused_desc == nullptr) {
    OP_LOGE(PASS_NAME.c_str(), "fused_node's OpDesc is null, fusion failed.");
    return PARAM_INVALID;
  }

  OP_LOGI(PASS_NAME.c_str(), "CorrelationFusionPass start");
  // get the attr
  int32_t groups = 1;
  ge::AttrUtils::GetInt(fused_desc, "groups", groups);

  // get the info of x
  ge::GeTensorDesc x_tensor_desc = fused_desc->GetInputDesc(1);
  Format x_format = x_tensor_desc.GetFormat();
  ge::GeShape x_shape = x_tensor_desc.GetShape();

  // get the info of filter
  ge::GeTensorDesc k_tensor_desc = fused_desc->GetInputDesc(0);
  Format k_format = k_tensor_desc.GetFormat();
  ge::GeShape k_shape = k_tensor_desc.GetShape();

  // get the info of y
  ge::GeTensorDesc y_tensor_desc = fused_desc->GetOutputDesc(0);
  Format y_format = y_tensor_desc.GetFormat();
  ge::GeShape y_shape = y_tensor_desc.GetShape();
  DataType y_data_type = y_tensor_desc.GetDataType();

  int32_t in = 0;
  int32_t ic = 0;
  int32_t kn = 0;
  int32_t kc = 0;
  int32_t dim_0 = 0;
  int32_t dim_1 = 1;
  int32_t dim_2 = 2;
  int32_t dim_3 = 3;
  if (x_format == FORMAT_NCHW) {
    in = x_shape.GetDim(dim_0);
    ic = x_shape.GetDim(dim_1);
  } else if (x_format == FORMAT_NHWC) {
    in = x_shape.GetDim(dim_0);
    ic = x_shape.GetDim(dim_3);
  } else {
    OP_LOGE(PASS_NAME.c_str(), "Node:%s's input x format should be NCHW/NHWC, fusion failed.",
            fused_node->GetName().c_str());
    return PARAM_INVALID;
  }

  if (k_format == FORMAT_NCHW) {
    kn = k_shape.GetDim(dim_0);
    kc = k_shape.GetDim(dim_1);
  } else if (k_format == FORMAT_NHWC) {
    kn = k_shape.GetDim(dim_0);
    kc = k_shape.GetDim(dim_3);
  } else if (k_format == FORMAT_HWCN) {
    kn = k_shape.GetDim(dim_3);
    kc = k_shape.GetDim(dim_2);
  } else {
    OP_LOGE(PASS_NAME.c_str(), "Node:%s's input filter format should be NCHW/NHWC/HWCN, fusion failed.",
            fused_node->GetName().c_str());
    return PARAM_INVALID;
  }

  int32_t in_revised = 0;
  int32_t ic_revised = 0;
  int32_t kn_revised = 0;
  int32_t kc_revised = 0;
  int32_t yn_revised = 0;
  int32_t yc_revised = 0;

  if (groups == 1) {
    in_revised = 1;
    ic_revised = in * ic;
    kn_revised = int(kn * kc / ic);
    kc_revised = ic;
    yn_revised = kn;
    yc_revised = int(kc / ic);
  } else if (groups == ic) {
    in_revised = 1;
    ic_revised = in * ic;
    kn_revised = kn * kc;
    kc_revised = 1;
    yn_revised = kn;
    yc_revised = kc;
  } else {
    OP_LOGE(PASS_NAME.c_str(), "Node:%s's attr groups is not valid, fusion failed.",
            fused_node->GetName().c_str());
    return PARAM_INVALID;
  }

  ge::GeShape x_shape_revised = x_shape;
  ge::GeShape k_shape_revised = k_shape;
  ge::GeShape y_shape_revised = y_shape;
  ge::GeShape y_shape_revised_before_reshape = y_shape;

  // update x shape
  if (x_format == FORMAT_NCHW) {
    x_shape_revised.SetDim(dim_0, in_revised);
    x_shape_revised.SetDim(dim_1, ic_revised);
  } else if (x_format == FORMAT_NHWC) {
    x_shape_revised.SetDim(dim_0, in_revised);
    x_shape_revised.SetDim(dim_3, ic_revised);
  }

  // update filter shape
  if (k_format == FORMAT_NCHW) {
    k_shape_revised.SetDim(dim_0, kn_revised);
    k_shape_revised.SetDim(dim_1, kc_revised);
  } else if (k_format == FORMAT_NHWC) {
    k_shape_revised.SetDim(dim_0, kn_revised);
    k_shape_revised.SetDim(dim_3, kc_revised);
  } else if (k_format == FORMAT_HWCN) {
    k_shape_revised.SetDim(dim_3, kn_revised);
    k_shape_revised.SetDim(dim_2, kc_revised);
  }

  // update y shape
  if (y_format == FORMAT_NCHW) {
    y_shape_revised.SetDim(dim_0, yn_revised);
    y_shape_revised.SetDim(dim_1, yc_revised);

    y_shape_revised_before_reshape.SetDim(dim_0, dim_1);
    y_shape_revised_before_reshape.SetDim(dim_1, yn_revised * yc_revised);
  } else if (y_format == FORMAT_NHWC) {
    y_shape_revised.SetDim(dim_0, yn_revised);
    y_shape_revised.SetDim(dim_3, yc_revised);

    y_shape_revised_before_reshape.SetDim(dim_0, dim_1);
    y_shape_revised_before_reshape.SetDim(dim_3, yn_revised * yc_revised);
  } else {
    OP_LOGE(PASS_NAME.c_str(), "Node:%s's output y format should be NCHW/NHWC, fusion failed.",
            fused_node->GetName().c_str());
    return PARAM_INVALID;
  }

  // get input names
  std::map < string, uint32_t > input_names = fused_desc->GetAllInputName();
  // in Correlation, the order is filter, x
  // in Conv2D, the order is x, fiter
  // so switch the order of x and filter
  uint32_t tmp = input_names["x"];
  input_names["x"] = input_names["filter"];
  input_names["filter"] = tmp;

  if(groups == 1 && kn > 1 && kc > 1){
    // SplitD and ConcatD flow
    int idx = 0;
    // generate split node
    NodePtr split_node_x = nullptr, split_node_filter = nullptr;
    GeTensorDesc split_out_desc_x, split_out_desc_filter;

    if (SUCCESS != generate_split_node_for_x(graph, fused_desc, kn, split_node_x, split_out_desc_x)) {
      OP_LOGE(PASS_NAME.c_str(), "Generate split node for x fail [for fusion node:%s]",
              fused_node->GetName().c_str());
      return FAILED;
    }
    if (SUCCESS != generate_split_node_for_filter(graph, fused_desc, kn, split_node_filter,
        split_out_desc_filter)) {
      OP_LOGE(PASS_NAME.c_str(), "Generate split node for filter fail [for fusion node:%s]",
              fused_node->GetName().c_str());
      return FAILED;
    }

    vector<NodePtr> new_conv_nodes;
    GeTensorDesc new_conv_out_desc = y_tensor_desc;
    ge::GeShape new_conv_out_shape = y_shape_revised;
    if (y_format == FORMAT_NCHW) {
      new_conv_out_shape.SetDim(dim_0, dim_1);
      new_conv_out_shape.SetDim(dim_1, dim_1);
    } else if (y_format == FORMAT_NHWC) {
      new_conv_out_shape.SetDim(dim_0, dim_1);
      new_conv_out_shape.SetDim(dim_3, dim_1);
    }
    new_conv_out_desc.SetOriginShape(new_conv_out_shape);
    new_conv_out_desc.SetShape(new_conv_out_shape);
    if (SUCCESS != generate_new_conv_nodes(graph, fused_desc, input_names, kn, split_out_desc_x,
        split_out_desc_filter, new_conv_nodes, new_conv_out_desc)) {
      OP_LOGE(PASS_NAME.c_str(), "Generate new conv nodes fail [for fusion node:%s]",
              fused_node->GetName().c_str());
      return FAILED;
    }

    // generate concat node
    NodePtr concat_node;
    GeTensorDesc concat_out_desc = y_tensor_desc;
    concat_out_desc.SetOriginShape(y_shape_revised);
    concat_out_desc.SetShape(y_shape_revised);
    if (SUCCESS != generate_concat_node(graph, fused_desc, kn, new_conv_out_desc, concat_out_desc, concat_node)) {
      OP_LOGE(PASS_NAME.c_str(), "Generate concat node fail [for fusion node:%s]", fused_node->GetName().c_str());
      return FAILED;
    }

    // add edges from input x of Correction to split_node_x
    if (SUCCESS != ge::GraphUtils::AddEdge(fused_node->GetInDataAnchor(1)->GetPeerOutAnchor(),
        split_node_x->GetInDataAnchor(0))) {
      OP_LOGE(PASS_NAME.c_str(), "Add edge from node:%s's x to node:%s's input[0] failed.",
              fused_node->GetName().c_str(), split_node_x->GetName().c_str());
      return FAILED;
    }
    OP_LOGD(PASS_NAME.c_str(), "Add edge from node:%s's x to node:%s's input[0].",
            fused_node->GetName().c_str(), split_node_x->GetName().c_str());

    // add edges from input filter of Correction to split_node_filter
    if (SUCCESS != ge::GraphUtils::AddEdge(fused_node->GetInDataAnchor(0)->GetPeerOutAnchor(),
        split_node_filter->GetInDataAnchor(0))) {
      OP_LOGE(PASS_NAME.c_str(), "Add edge from node:%s's filter to node:%s's input[0] failed.",
              fused_node->GetName().c_str(), split_node_filter->GetName().c_str());
      return FAILED;
    }
    OP_LOGD(PASS_NAME.c_str(), "Add edge from node:%s's filter to node:%s's input[0].",
            fused_node->GetName().c_str(), split_node_filter->GetName().c_str());

    // add edges from split_node_x to new_conv_nodes's input 0
    idx = 0;
    for (NodePtr conv_node : new_conv_nodes) {
      if (SUCCESS != ge::GraphUtils::AddEdge(split_node_x->GetOutDataAnchor(idx),
          conv_node->GetInDataAnchor(0))) {
        OP_LOGE(PASS_NAME.c_str(), "Add edge from node:%s's output[0] to node:%s's input[0] failed.",
                split_node_x->GetName().c_str(), conv_node->GetName().c_str());
        return FAILED;
      }
      OP_LOGD(PASS_NAME.c_str(), "Add edge from node:%s's output[0] to node:%s's input[0].",
              split_node_x->GetName().c_str(), conv_node->GetName().c_str());
      idx++;
    }

    // add edges from split_node_filter to new_conv_nodes's input 1
    idx = 0;
    for (NodePtr conv_node : new_conv_nodes) {
      if (SUCCESS != ge::GraphUtils::AddEdge(split_node_filter->GetOutDataAnchor(idx),
          conv_node->GetInDataAnchor(1))) {
        OP_LOGE(PASS_NAME.c_str(), "Add edge from node:%s's output[0] to node:%s's input[1] failed.",
                split_node_filter->GetName().c_str(), conv_node->GetName().c_str());
        return FAILED;
      }
      OP_LOGD(PASS_NAME.c_str(), "Add edge from node:%s's output[0] to node:%s's input[1].",
              split_node_filter->GetName().c_str(), conv_node->GetName().c_str());
      idx++;
    }

    // add edges from new_conv_nodes to concat_node
    idx = 0;
    for (NodePtr conv_node : new_conv_nodes) {
      if (SUCCESS != ge::GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0),
          concat_node->GetInDataAnchor(idx))) {
        OP_LOGE(PASS_NAME.c_str(), "Add edge from node:%s's output[0] to node:%s's input[0] failed.",
                conv_node->GetName().c_str(), concat_node->GetName().c_str());
        return FAILED;
      }
      OP_LOGD(PASS_NAME.c_str(), "Add edge from node:%s's output[0] to node:%s's input[0].",
              conv_node->GetName().c_str(), concat_node->GetName().c_str());
      idx++;
    }

    // add edges from concat_node to output of Correction
    if (fused_node->GetOutDataAnchor(0)->GetPeerInDataAnchors().size() > 0) {
      for (InDataAnchorPtr in_anchor_ptr : fused_node->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
        in_anchor_ptr->UnlinkAll();
        if (SUCCESS != ge::GraphUtils::AddEdge(concat_node->GetOutDataAnchor(0), in_anchor_ptr)) {
          OP_LOGE(PASS_NAME.c_str(), "Add edge from node:%s's output[0] to node:%s's output[0] failed.",
                  concat_node->GetName().c_str(), fused_node->GetName().c_str());
          return FAILED;
        }
        OP_LOGD(PASS_NAME.c_str(), "Add edge from node:%s's output[0] to node:%s's output[0].",
                concat_node->GetName().c_str(), fused_node->GetName().c_str());
      }
    }

    // unlink all input of Correlation
    for (auto in_anchor : fused_node->GetAllInDataAnchors()) {
      if (in_anchor != nullptr) {
        in_anchor->UnlinkAll();
      }
    }

    // remove Correlation from graph
    if (ge::GRAPH_SUCCESS != graph.RemoveNode(fused_node)) {
      OP_LOGE(PASS_NAME.c_str(), "remove fused_node node[%s] failed", fused_node->GetName().c_str());
      return FAILED;
    }

    // add nodes in newNodes
    newNodes.push_back(split_node_x);
    newNodes.push_back(split_node_filter);
    for (NodePtr conv_node : new_conv_nodes) {
      newNodes.push_back(conv_node);
    }
    newNodes.push_back(concat_node);
  } else {
    // DepthwiseConv2D flow
    x_tensor_desc.SetOriginShape(x_shape_revised);
    k_tensor_desc.SetOriginShape(k_shape_revised);
    y_tensor_desc.SetOriginShape(y_shape_revised_before_reshape);
    x_tensor_desc.SetShape(x_shape_revised);
    k_tensor_desc.SetShape(k_shape_revised);
    y_tensor_desc.SetShape(y_shape_revised_before_reshape);

    // get the peer out of x and filter
    auto k_peer = fused_node->GetInDataAnchor(0)->GetPeerOutAnchor();
    auto x_peer = fused_node->GetInDataAnchor(1)->GetPeerOutAnchor();

    // unlink all input of Correlation
    for (auto in_anchor : fused_node->GetAllInDataAnchors()) {
      if (in_anchor != nullptr) {
        in_anchor->UnlinkAll();
      }
    }

    // switch the pos of x and filter
    // link input x to Conv2D input 0
    if (SUCCESS != ge::GraphUtils::AddEdge(x_peer, fused_node->GetInDataAnchor(0))) {
      OP_LOGE(PASS_NAME.c_str(), "Add edge from x to fusion node:%s's input[0] failed.",
              fused_node->GetName().c_str());
      return FAILED;
    }
    OP_LOGD(PASS_NAME.c_str(), "Add edge from x to fusion node:%s's input[0].",
            fused_node->GetName().c_str());

    // link input filter to Conv2D input 1
    if (SUCCESS != ge::GraphUtils::AddEdge(k_peer, fused_node->GetInDataAnchor(1))) {
      OP_LOGE(PASS_NAME.c_str(), "Add edge from filter to fusion node:%s's input[1] failed.",
              fused_node->GetName().c_str());
      return FAILED;
    }

    OP_LOGD(PASS_NAME.c_str(), "Add edge from filter to fusion node:%s's input[1].",
            fused_node->GetName().c_str());

    // set type as Conv2D
    fused_desc->SetType("Conv2D");

    // update tensor desc
    fused_desc->UpdateInputDesc(0, x_tensor_desc);
    fused_desc->UpdateInputDesc(1, k_tensor_desc);
    fused_desc->UpdateOutputDesc(0, y_tensor_desc);

    // update input names
    fused_desc->UpdateInputName(input_names);

    // set attr groups of Conv2D
    if (groups == 1) {
      ge::AttrUtils::SetInt(fused_desc, "groups", kn);
    } else {
      ge::AttrUtils::SetInt(fused_desc, "groups", kn * kc);
    }

    // set attr pads and strides of Conv2D
    ge::AttrUtils::SetListInt(fused_desc, "pads", {0, 0, 0, 0});
    ge::AttrUtils::SetListInt(fused_desc, "strides", {1, 1, 1, 1});
    ge::AttrUtils::SetListInt(fused_desc, "dilations", {1, 1, 1, 1});
    ge::AttrUtils::SetInt(fused_desc, "offset_x", 0);
    ge::AttrUtils::SetStr(fused_desc, "data_format", "NHWC");

    if (kn > 1) {
      // generate reshape node
      ge::GeTensorDesc reshape_out_tensor_desc = ge::GeTensorDesc(y_shape_revised, y_format, y_data_type);
      reshape_out_tensor_desc.SetOriginShape(y_shape_revised);

      ge::NodePtr reshape_node = nullptr;
      if (generate_reshape_node(graph, y_tensor_desc, reshape_out_tensor_desc,
                                y_shape_revised, reshape_node, "reshape", fused_node->GetName()) != SUCCESS) {
        OP_LOGE(PASS_NAME.c_str(), "Fail to generate reshape node for fusion node:%s's",
                fused_node->GetName().c_str());
        return FAILED;
      }
      OP_LOGD(PASS_NAME.c_str(), "Generate reshape node for fusion node:%s's", 
              fused_node->GetName().c_str());

      // add reshape node to the graph
      // connect the output 0 of reshape_node to output 0 of fused_node
      if (fused_node->GetOutDataAnchor(0)->GetPeerInDataAnchors().size() > 0) {
        for (InDataAnchorPtr in_anchor_ptr : fused_node->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
          in_anchor_ptr->UnlinkAll();
          if (SUCCESS != ge::GraphUtils::AddEdge(reshape_node->GetOutDataAnchor(0), in_anchor_ptr)) {
            OP_LOGE(PASS_NAME.c_str(), 
                    "Add edge from reshape node:%s's output[0] to fusion node:%s's output[0] failed.",
                    reshape_node->GetName().c_str(), fused_node->GetName().c_str());
            return FAILED;
          }
          OP_LOGD(PASS_NAME.c_str(), "Add edge from reshape node:%s's output[0] to fusion node:%s's output[0].",
                  reshape_node->GetName().c_str(), fused_node->GetName().c_str());
        }
      }

      // connect the output 0 of fused_node to input of reshape_node
      if (SUCCESS != ge::GraphUtils::AddEdge(fused_node->GetOutDataAnchor(0),
                                             reshape_node->GetInDataAnchor(0))) {
        OP_LOGE(PASS_NAME.c_str(), "Add edge from fused node:%s's output[0] to reshape node:%s's input[0] failed.",
                fused_node->GetName().c_str(), reshape_node->GetName().c_str());
        return FAILED;
      }
      OP_LOGD(PASS_NAME.c_str(), "Add edge from fused node:%s's output[0] to reshape node:%s's input[0].",
              fused_node->GetName().c_str(), reshape_node->GetName().c_str());

      // add nodes in newNodes
      newNodes.push_back(reshape_node);
    }
  }
  return SUCCESS;
}

REGISTER_PASS("CorrelationFusionPass", BUILT_IN_GRAPH_PASS, CorrelationFusionPass);
}
