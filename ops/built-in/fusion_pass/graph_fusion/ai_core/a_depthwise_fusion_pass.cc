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
 * \file a_depthwise_fusion_pass.cpp
 * \brief a_depthwise_fusion_pass
 */
#include "a_depthwise_fusion_pass.h"

#include <iostream>
#include <vector>
#include <string>
#include <map>

#include "quant_host_cpu_op_common.h"
#include "op_log.h"
#include "fp16_t.hpp"
#include "graph/debug/ge_attr_define.h"
#include "graph/types.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"
#include "anchor_util.h"
#include "securec.h"

using namespace std;
using namespace ge;

namespace fe{
    static const string PATTERN_DEPTHWISE = "DepthwiseConv2D";
    static const char *DEPTHWISE = "DepthwiseConv2D";
    static const char kNetOutputType[] = "NetOutput";
    const int MAX_DIM_NUM = 4;
    const int64_t ALREADY_CHANGED_C = 1;
    static int64_t DEPTHWISE_KERNEL_RESHAPE_NUM = 0;

    vector<FusionPattern *> DepthwiseFusionPass::DefinePatterns() {
        vector<FusionPattern *> patterns;

        // define AvgPoolFusion
        FusionPattern *pattern = new (std::nothrow) FusionPattern("ADepthwiseConv2D");
        FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                        return patterns);
        // define origin graph
        pattern->AddOpDesc(PATTERN_DEPTHWISE, {DEPTHWISE}).SetOutput(PATTERN_DEPTHWISE);
        patterns.push_back(pattern);

        return patterns;
    }

    Status DepthwiseFusionPass::CreateReshapeNode(ge::ComputeGraph& graph, const ge::InDataAnchorPtr & in_anchor,
                                                  const vector<int64_t> & pre_shape, const vector<int64_t> & new_shape,
                                                  ge::NodePtr& shape_node) {
        fe::DEPTHWISE_KERNEL_RESHAPE_NUM ++;
        auto previous_node = in_anchor->GetPeerOutAnchor()->GetOwnerNode();
        auto next_in_node = in_anchor->GetOwnerNode();
        int idx = in_anchor->GetIdx();
        auto next_in_node_desc = next_in_node->GetOpDesc()->GetInputDesc(idx);
        ge::GeTensorDesc previous_node_desc = next_in_node_desc.Clone();
        previous_node_desc.SetShape(ge::GeShape(pre_shape));
        previous_node_desc.SetOriginShape(ge::GeShape(pre_shape));
        next_in_node_desc.SetShape(ge::GeShape(new_shape));
        next_in_node_desc.SetOriginShape(ge::GeShape(new_shape));

        ge::OpDescPtr reshape_desc;
        FUSION_PASS_MAKE_SHARED((reshape_desc = std::make_shared<ge::OpDesc>(
                previous_node->GetName() + "/Reshape_" + std::to_string(fe::DEPTHWISE_KERNEL_RESHAPE_NUM), "Reshape")),
                return FAILED);
        FUSION_PASS_CHECK(reshape_desc->AddInputDesc("x", previous_node_desc) != GRAPH_SUCCESS,
                          OP_LOGE(FUSED_OP_TYPE.c_str(), "failed to add input desc x to reshape."), return FAILED);
        FUSION_PASS_CHECK(reshape_desc->AddOutputDesc("y", next_in_node_desc) != GRAPH_SUCCESS,
                          OP_LOGE(FUSED_OP_TYPE.c_str(), "failed to add input desc y to reshape."), return FAILED);
        ge::AttrUtils::SetListInt(reshape_desc, "shape", new_shape);

        auto new_shape_node = graph.AddNode(reshape_desc);
        FUSION_PASS_CHECK(new_shape_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "failed to add reshape to graph."),
        return FAILED);
        shape_node = new_shape_node;
        return SUCCESS;
    }

    Status DepthwiseFusionPass::InsertNode(const ge::OutDataAnchorPtr &src, const ge::InDataAnchorPtr &dst,
                                            ge::NodePtr& new_node) {
        ge::NodePtr src_node = src->GetOwnerNode();
        ge::NodePtr dst_node = dst->GetOwnerNode();
        if(ge::GraphUtils::RemoveEdge(src, dst) != SUCCESS) {
            OP_LOGE(dst_node->GetName().c_str(), "Remove ori_filter edge error.");
            return FAILED;
        }
        if(ge::GraphUtils::AddEdge(src, new_node->GetInDataAnchor(0)) != SUCCESS) {
            OP_LOGE(src_node->GetName().c_str(), "Add edge to node %s failed.", new_node->GetName().c_str());
            return FAILED;
        }
        if(ge::GraphUtils::AddEdge(new_node->GetOutDataAnchor(0), dst)!= SUCCESS) {
            OP_LOGE(new_node->GetName().c_str(), "Add edge to node %s failed.", dst_node->GetName().c_str());
            return FAILED;
        }
        return SUCCESS;
    }

    Status DepthwiseFusionPass::Fusion(ge::ComputeGraph &graph, Mapping &mapping, vector<ge::NodePtr> &fusionNodes) {
        OP_LOGD("Enter DepthwiseFusionPass");
        fe::DEPTHWISE_KERNEL_RESHAPE_NUM = 0;
        ge::NodePtr depthwise_node = GetNodeFromMapping(PATTERN_DEPTHWISE, mapping);
        OpDescPtr depthwise_desc = depthwise_node->GetOpDesc();
        OP_LOGD(depthwise_desc->GetName().c_str(), "dealing with");
        auto filterDesc = GetCurrNodeInputDesc(depthwise_node, 1);
        FUSION_PASS_CHECK(filterDesc == nullptr,
                          CUBE_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "filterDesc is null"),
                          return FAILED);
        auto filter_shape = filterDesc->GetOriginShape().GetDims();
        int64_t n = 0;
        int64_t c = 0;
        int64_t h = 0;
        int64_t w = 0;
        ge::GeTensorDesc tensor_desc = depthwise_desc->GetInputDesc(1);
        ge::Format origin_format = tensor_desc.GetOriginFormat();
        if (filter_shape.size() == 4) {
            // NCHW format already transfer to conv2d at torch
            // caffe has no depthwise conv
            // tf NCHW should not have to change
            // tf torch NHWC format do not need to process
            if (origin_format == FORMAT_NHWC) {
                OP_LOGD(FUSED_OP_TYPE.c_str(), "in FORMAT_NHWC, before swap N, H, W, C: [%d, %d, %d, %d]",
                        (int)filter_shape[3], (int)filter_shape[0], (int)filter_shape[1], (int)filter_shape[2]);
                return NOT_CHANGED;
            } else if (origin_format == FORMAT_HWCN) {
                OP_LOGD(FUSED_OP_TYPE.c_str(), "in FORMAT_HWCN, before swap H, W, C, N: [%d, %d, %d, %d]",
                        (int)filter_shape[0], (int)filter_shape[1], (int)filter_shape[2], (int)filter_shape[3]);
                n = filter_shape[3];
                h = filter_shape[0];
                w = filter_shape[1];
                c = filter_shape[2];
                if (c == 1){
                    OP_LOGD(FUSED_OP_TYPE.c_str(), "The input1 of depthwiseConv2d has satisfied that c == 1");
                    return NOT_CHANGED;
                }
                vector<int64_t> pre_shape = {h, w, c, n};
                vector<int64_t> new_shape = {h, w, 1, c*n};
                auto in_anchor = depthwise_node->GetInDataAnchor(1);
                FUSION_PASS_CHECK(in_anchor == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(),
                                "Failed to get in data anchor 1."), return FAILED);
                auto out_anchor = in_anchor->GetPeerOutAnchor();
                FUSION_PASS_CHECK(out_anchor == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(),
                                "Failed to get out data anchor 1."), return FAILED);

                ge::NodePtr filter_ori_node = out_anchor->GetOwnerNode();
                std::string filter_ori_node_type = filter_ori_node->GetType().c_str();
                auto filter_out_all = filter_ori_node->GetAllOutDataAnchors();
                for (auto iter_out : filter_out_all){
                    auto in_anchors_dst = iter_out->GetPeerInDataAnchors();
                    for (auto iter_in : in_anchors_dst) {
                        ge::NodePtr filter_reshape_node = nullptr;
                        ge::NodePtr cur_node = iter_in->GetOwnerNode();
                        std::string cur_node_type = cur_node->GetType().c_str();
                        if (cur_node_type == "QuantBiasOptimization" || cur_node_type == "DepthwiseConv2D") {
                            auto create_res = CreateReshapeNode(graph, iter_in, pre_shape, new_shape, filter_reshape_node);
                            FUSION_PASS_CHECK(create_res == FAILED,
                                            OP_LOGE(FUSED_OP_TYPE.c_str(), "Create reshape node failed"), 
                                            return FAILED);
                            ge::GeTensorDescPtr fusedNodeInputDescPtr = cur_node->GetOpDesc()->MutableInputDesc(1);
                            FUSION_PASS_CHECK(fusedNodeInputDescPtr == nullptr,
                                            OP_LOGE(FUSED_OP_TYPE.c_str(), "fusedNodeInputDescPtr is null."),
                                            return FAILED);
                            fusedNodeInputDescPtr->SetShape(ge::GeShape(new_shape));
                            fusedNodeInputDescPtr->SetOriginShape(ge::GeShape(new_shape));
                            Status ret = InsertNode(iter_out, iter_in, filter_reshape_node);
                            if (ret != SUCCESS) {
                                OP_LOGE(filter_reshape_node->GetType().c_str(), "Add node %s failed.",
                                        filter_reshape_node->GetName().c_str());
                                return FAILED;
                            }
                        } else {
                            continue;
                        }
                    }
                }
            }
        }else{
            OP_LOGE(FUSED_OP_TYPE.c_str(), "dim_info is not right, please check!");
            return FAILED;
        }
        OP_LOGD("Leave DepthwiseFusionPass");
        return SUCCESS;
    }
  REGISTER_PASS("ADepthwiseFusionPass", BUILT_IN_GRAPH_PASS, DepthwiseFusionPass);
} // namespace fe
