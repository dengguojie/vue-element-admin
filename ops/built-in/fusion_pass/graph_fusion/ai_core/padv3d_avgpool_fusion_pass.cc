/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
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
 * \file padv3d_pooling_fusion_pass.h
 * \brief padv3d + pooling fusion pass
 */
#include "padv3d_avgpool_fusion_pass.h"

#include <iostream>
#include <vector>
#include <string>
#include <map>

#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"

using namespace ge;
namespace fe {
static const char *PADV3D = "PadV3D";
static const char *PADV3 = "PadV3";
static const char *POOLING = "AvgPoolV2";
static const char *POOLING3D = "AvgPool3DD";
static const std::string PATTERN_PADV3D = "PadV3D";
static const std::string PATTERN_POOLING = "AvgPoolV2";

vector<FusionPattern*> Padv3dAvgpoolFusionPass::DefinePatterns() {
  vector < FusionPattern * > patterns;
  FusionPattern *pattern =
      new (std::nothrow) FusionPattern("Padv3dAvgpoolFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), 
                    "new a pattern object failed."), return patterns);

  pattern->AddOpDesc(PATTERN_PADV3D, {PADV3D, PADV3})
      .AddOpDesc(PATTERN_POOLING, {POOLING, POOLING3D})
      .SetInputs(PATTERN_POOLING, {PATTERN_PADV3D})
      .SetOutput(PATTERN_POOLING);

  patterns.push_back(pattern);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define Padv3dAvgpoolFusionPass pattern end");
  return patterns;
}

Status Padv3dAvgpoolFusionPass::Fusion(ge::ComputeGraph& graph,
                                        Mapping& mapping,
                                        vector<ge::NodePtr> &fusionNodes) {
  // get all nodes
  ge::NodePtr pad_node = GetNodeFromMapping(PATTERN_PADV3D, mapping);
  ge::NodePtr pooling_node = GetNodeFromMapping(PATTERN_POOLING, mapping);
  FUSION_PASS_CHECK(pad_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), 
                    "pad_node is null, fusion failed."), return PARAM_INVALID);
  FUSION_PASS_CHECK(pooling_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), 
                    "pooling_node is null, fusion failed."), return PARAM_INVALID);

  // check output link
  FUSION_PASS_CHECK(pad_node->GetOutDataAnchor(0)->GetPeerAnchorsSize() != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "PADV3D_node output size is [%d], which not equal to 1.",
                            pad_node->GetOutDataAnchor(0)->GetPeerAnchorsSize()),
                    return NOT_CHANGED);

  // get all node's desc
  ge::OpDescPtr pad_desc = pad_node->GetOpDesc();
  ge::OpDescPtr Pooling_desc = pooling_node->GetOpDesc();
  FUSION_PASS_CHECK(pad_desc == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), 
                    "pad_node's OpDesc is null, fusion failed."), return PARAM_INVALID);
  FUSION_PASS_CHECK(Pooling_desc == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), 
                    "pooling_node's OpDesc is null, fusion failed."), return PARAM_INVALID);

  // get shape and format
  ge::GeTensorDesc input_desc = pad_desc->GetInputDesc(0);
  ge::Format input_format = input_desc.GetFormat();
  
  // get op
  Operator op_pad = ge::OpDescUtils::CreateOperatorFromNode(pad_node);
  Operator op_Pooling = ge::OpDescUtils::CreateOperatorFromNode(pooling_node);

  // attr:paddings
  std::vector<std::vector<int64_t>> paddings;
  if (ge::GRAPH_SUCCESS != op_pad.GetAttr("paddings", paddings)) {
    Tensor data;
    if (GRAPH_SUCCESS != op_pad.GetInputConstData("paddings", data)) {
      OP_LOGW(FUSED_OP_TYPE.c_str(), "not get paddings value");
      return NOT_CHANGED;
    }
    auto dtype = pad_desc->GetInputDesc("paddings").GetDataType();
    if (dtype == ge::DT_INT32) {
      int32_t* const_data_ptr = (int32_t*)data.GetData();
      size_t const_data_size = data.GetSize() / sizeof(int32_t);
      for (size_t i = 0; i < const_data_size; i += 2) {
        std::vector<int64_t> val;
        val.push_back((int32_t)((*(const_data_ptr + i))));
        val.push_back((int32_t)((*(const_data_ptr + i + 1))));
        paddings.emplace_back(val);
      }
    } else if (dtype == ge::DT_INT64) {
      int64_t* const_data_ptr = (int64_t*)data.GetData();
      size_t const_data_size = data.GetSize() / sizeof(int64_t);
      for (size_t i = 0; i < const_data_size; i += 2) {
        std::vector<int64_t> val;
        val.push_back((int64_t)((*(const_data_ptr + i))));
        val.push_back((int64_t)((*(const_data_ptr + i + 1))));
        paddings.emplace_back(val);
      }
    } else {
      OP_LOGE(FUSED_OP_TYPE.c_str(), "paddings dtype should be int32 or int64");
      return GRAPH_FAILED;
    }
  }

  bool paddings_contiguous = false;
  op_pad.GetAttr("paddings_contiguous", paddings_contiguous);

 
  // verify
  if (CheckFormatAndPading(input_format, paddings, paddings_contiguous) != SUCCESS) {
    return NOT_CHANGED;
  }

  // attr:pad
  std::vector<int32_t> new_pad;
  UpdateAttrPads(input_format, paddings, new_pad, paddings_contiguous);

  std::vector<int32_t> ksize;
  op_Pooling.GetAttr("ksize", ksize);
  if (CheckPadAndKsize(input_format, new_pad, ksize) != SUCCESS) {
    return NOT_CHANGED;
  }

  op_Pooling.SetAttr("pads", new_pad);
  Pooling_desc->UpdateInputDesc(0, input_desc);
  FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(pooling_node->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                               pooling_node->GetInDataAnchor(0)) != SUCCESS,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove out data edge failed."), return FAILED);

  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(
      pad_node->GetInDataAnchor(0)->GetPeerOutAnchor(),
      pooling_node->GetInDataAnchor(0)) != SUCCESS,
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
          pad_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
          pooling_node->GetName().c_str()),
      return FAILED);

  // delete fused nodes
  FUSION_PASS_CHECK(graph.RemoveNode(pad_node) != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(),
                    "Remove pad_node failed."), return FAILED);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Padv3dAvgpoolFusionPass graph fusion success!");
  return SUCCESS;
}

void Padv3dAvgpoolFusionPass::UpdateAttrPads(ge::Format& input_format, std::vector<std::vector<int64_t>>& paddings,
                                             std::vector<int32_t>& new_pad, bool paddings_contiguous) {
  //pad_indexs store HW OR DHW index
  std::vector<std::pair<int, int>> pad_indexs;
  //paddings_contiguous = true ,paddings is x1_begin,x1_end,x2_begin,x2_end...
  //paddings_contiguous = false, paddings is x1_begin,x2_begin... x1_end,x2_end...
  if (paddings_contiguous) {
    if (input_format == FORMAT_NHWC) {
      pad_indexs.push_back(std::pair<int, int>(1, 0));
      pad_indexs.push_back(std::pair<int, int>(1, 1));
      pad_indexs.push_back(std::pair<int, int>(2, 0));
      pad_indexs.push_back(std::pair<int, int>(2, 1));
    } else if (input_format == FORMAT_NCHW) {
      pad_indexs.push_back(std::pair<int, int>(2, 0));
      pad_indexs.push_back(std::pair<int, int>(2, 1));
      pad_indexs.push_back(std::pair<int, int>(3, 0));
      pad_indexs.push_back(std::pair<int, int>(3, 1));
    } else if (input_format == FORMAT_NCDHW) {
      pad_indexs.push_back(std::pair<int, int>(2, 0));
      pad_indexs.push_back(std::pair<int, int>(2, 1));
      pad_indexs.push_back(std::pair<int, int>(3, 0));
      pad_indexs.push_back(std::pair<int, int>(3, 1));
      pad_indexs.push_back(std::pair<int, int>(4, 0));
      pad_indexs.push_back(std::pair<int, int>(4, 1));
    } else if (input_format == FORMAT_NDHWC){
      pad_indexs.push_back(std::pair<int, int>(1, 0));
      pad_indexs.push_back(std::pair<int, int>(1, 1));
      pad_indexs.push_back(std::pair<int, int>(2, 0));
      pad_indexs.push_back(std::pair<int, int>(2, 1));
      pad_indexs.push_back(std::pair<int, int>(3, 0));
      pad_indexs.push_back(std::pair<int, int>(3, 1));
    }
  } else {
    if (input_format == FORMAT_NHWC) {
      pad_indexs.push_back(std::pair<int, int>(0, 1));
      pad_indexs.push_back(std::pair<int, int>(2, 1));
      pad_indexs.push_back(std::pair<int, int>(2, 0));
      pad_indexs.push_back(std::pair<int, int>(3, 0));
    } else if (input_format == FORMAT_NCHW) {
      pad_indexs.push_back(std::pair<int, int>(1, 0));
      pad_indexs.push_back(std::pair<int, int>(3, 0));
      pad_indexs.push_back(std::pair<int, int>(1, 1));
      pad_indexs.push_back(std::pair<int, int>(3, 1));
    } else if (input_format == FORMAT_NCDHW) {
      pad_indexs.push_back(std::pair<int, int>(1, 0));
      pad_indexs.push_back(std::pair<int, int>(3, 1));
      pad_indexs.push_back(std::pair<int, int>(1, 1));
      pad_indexs.push_back(std::pair<int, int>(4, 0));
      pad_indexs.push_back(std::pair<int, int>(2, 0));
      pad_indexs.push_back(std::pair<int, int>(4, 1));
    } else if (input_format == FORMAT_NDHWC) {
      pad_indexs.push_back(std::pair<int, int>(0, 1));
      pad_indexs.push_back(std::pair<int, int>(3, 0));
      pad_indexs.push_back(std::pair<int, int>(1, 0));
      pad_indexs.push_back(std::pair<int, int>(3, 1));
      pad_indexs.push_back(std::pair<int, int>(1, 1));
      pad_indexs.push_back(std::pair<int, int>(4, 0));
    }
  }

  int size = pad_indexs.size();
  for (int i = 0; i < size; ++i) {
    new_pad.push_back(paddings[pad_indexs[i].first][pad_indexs[i].second]);
  } 
}

Status Padv3dAvgpoolFusionPass::CheckFormatAndPading(ge::Format& input_format, std::vector<std::vector<int64_t>>& paddings,
                                                     bool paddings_contiguous) {
  if ((input_format != FORMAT_NHWC) && (input_format != FORMAT_NCHW) && 
      (input_format != FORMAT_NCDHW) && (input_format != FORMAT_NDHWC)) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "input format is not match.");
    return NOT_CHANGED;
  }
  
  int size = paddings.size();
  if (size != 4 && size != 5) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "paddding size is not match, shoud be is 4 0r  5cur is %d.", size);
    return NOT_CHANGED;
  }

  for (int i = 0; i < size; ++i) {
    if (paddings[i].size() != 2) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "paddings[%d] size shoud be 2, cur is %d", i, paddings[i].size());
      return NOT_CHANGED;
    }
  }

  //pad_indexs store NC index
  std::vector<std::pair<int, int>> pad_indexs;
  //paddings_contiguous = true ,paddings is x1_begin,x1_end,x2_begin,x2_end...
  //paddings_contiguous = false, paddings is x1_begin,x2_begin... x1_end,x2_end...
  if (paddings_contiguous) {
    if (input_format == FORMAT_NHWC) {
      pad_indexs.push_back(std::pair<int, int>(0, 0));
      pad_indexs.push_back(std::pair<int, int>(0, 1));
      pad_indexs.push_back(std::pair<int, int>(3, 0));
      pad_indexs.push_back(std::pair<int, int>(3, 1));
    } else if (input_format == FORMAT_NCHW || input_format == FORMAT_NCDHW) {
      pad_indexs.push_back(std::pair<int, int>(0, 0));
      pad_indexs.push_back(std::pair<int, int>(0, 1));
      pad_indexs.push_back(std::pair<int, int>(1, 0));
      pad_indexs.push_back(std::pair<int, int>(1, 1));
    } else {
      pad_indexs.push_back(std::pair<int, int>(0, 0));
      pad_indexs.push_back(std::pair<int, int>(0, 1));
      pad_indexs.push_back(std::pair<int, int>(4, 0));
      pad_indexs.push_back(std::pair<int, int>(4, 1));
    }
  } else {
    if (input_format == FORMAT_NHWC) {
      pad_indexs.push_back(std::pair<int, int>(0, 0));
      pad_indexs.push_back(std::pair<int, int>(1, 1));
      pad_indexs.push_back(std::pair<int, int>(2, 0));
      pad_indexs.push_back(std::pair<int, int>(3, 1));
    } else if (input_format == FORMAT_NCHW) {
      pad_indexs.push_back(std::pair<int, int>(0, 0));
      pad_indexs.push_back(std::pair<int, int>(0, 1));
      pad_indexs.push_back(std::pair<int, int>(2, 0));
      pad_indexs.push_back(std::pair<int, int>(2, 1));
    } else if (input_format == FORMAT_NCDHW) {
      pad_indexs.push_back(std::pair<int, int>(0, 0));
      pad_indexs.push_back(std::pair<int, int>(0, 1));
      pad_indexs.push_back(std::pair<int, int>(2, 1));
      pad_indexs.push_back(std::pair<int, int>(3, 0));
    } else if (input_format == FORMAT_NDHWC) {
      pad_indexs.push_back(std::pair<int, int>(0, 0));
      pad_indexs.push_back(std::pair<int, int>(2, 0));
      pad_indexs.push_back(std::pair<int, int>(2, 1));
      pad_indexs.push_back(std::pair<int, int>(4, 1));
    }
  }  

  int index_size = pad_indexs.size();
  for (int i = 0; i < index_size; ++i) {
    if (pad_indexs[i].first >= size) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "the pad_index of paddings are not match.");
      return NOT_CHANGED;
    }

    if (paddings[pad_indexs[i].first][pad_indexs[i].second] != 0) {
      OP_LOGI(FUSED_OP_TYPE.c_str(),
              "the pad_index of paddings are not match. is shoud be 0, cur is %d, format is %d, pad_index is[%d][%d]", 
              paddings[pad_indexs[i].first][pad_indexs[i].second], input_format, pad_indexs[i].first, 
              pad_indexs[i].second);
      return NOT_CHANGED;
    }
  }
  return SUCCESS;
}

Status Padv3dAvgpoolFusionPass::CheckPadAndKsize(ge::Format& input_format, std::vector<int32_t>& pads, std::vector<int32_t>& ksize) {
  if (ksize.size() < 4) {
    return NOT_CHANGED;
  }

  int start = 0;
  if (pads.size() != 4) {
    start = 2;
  }
  int ksize_w = 0;
  int ksize_h = 0;
  if (input_format == ge::FORMAT_NHWC) {
    ksize_h = ksize[1];
    ksize_w = ksize[2];
  } else if (input_format == ge::FORMAT_NCHW || input_format == ge::FORMAT_NDHWC) {
    ksize_h = ksize[2];
    ksize_w = ksize[3];
  } else if (input_format == ge::FORMAT_NCDHW) {
    ksize_h = ksize[3];
    ksize_w = ksize[4];
  }

  if (pads[start] >= ksize_h || pads[start + 1] >= ksize_h) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "pads start[%d] >=ksize_h[%d] or pads end[%d] >= ksize_h[%d]",
            pads[start], ksize_h, pads[start + 1], ksize_h);
    return NOT_CHANGED;
  }

  start += 2;
  if (pads[start] >= ksize_w || pads[start + 1] >= ksize_w) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "pads start[%d] >=ksize_w[%d] or pads end[%d] >= ksize_w[%d]",
            pads[start], ksize_w, pads[start + 1], ksize_w);
    return NOT_CHANGED;
  }
  return SUCCESS;
}

REGISTER_PASS("Padv3dAvgpoolFusionPass", BUILT_IN_GRAPH_PASS, Padv3dAvgpoolFusionPass);
}
