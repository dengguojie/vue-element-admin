/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
 * \file pattern_fusion_util.h
 * \brief add a control edge from source node to dest node Provide some
 *   basic methods for fusion pass (include change a const input to attr)
 */

#ifndef OPS_BUILT_IN_FUSION_PASS_COMMON_PATTERN_FUSION_UTIL_H_
#define OPS_BUILT_IN_FUSION_PASS_COMMON_PATTERN_FUSION_UTIL_H_

#include "securec.h"
#include "graph/compute_graph.h"
#include "graph/tensor.h"
#include "graph/utils/type_utils.h"
#include "op_log.h"
#include "error_util.h"
#include "register/graph_optimizer/graph_optimize_register_error_codes.h"

using namespace ge;
namespace fe {
struct PassAttrInfo {
  int attrIndex;
  std::string attrName;
  std::string attrType;
};

struct PassInputInfo {
  uint32_t inputOpDescIndex;
  std::string inputOpDescName;
};

#define FUSION_PASS_MAKE_SHARED(exec_expr0, exec_expr1) \
  do {                                                  \
    try {                                               \
      exec_expr0;                                       \
    } catch (...) {                                     \
      exec_expr1;                                       \
    }                                                   \
  } while (0)

#define FUSION_PASS_CHECK(cond, log_func, return_expr) \
  do {                                                 \
    if (cond) {                                        \
      log_func;                                        \
      return_expr;                                     \
    }                                                  \
  } while (0)

template <typename Dtype>
Status NnSet(const int32_t n, const Dtype alpha, Dtype& output1) {
  Dtype* output = &output1;
  FUSION_PASS_CHECK(output == nullptr, OP_LOGE("NnSet", "output is null"), return FAILED);

  if (alpha == 0) {
    const size_t total_size = sizeof(Dtype) * n;
    const size_t step = SECUREC_MEM_MAX_LEN;
    const size_t loop_times = total_size / step;
    const size_t tail_count = total_size % step;
    char* addr = reinterpret_cast<char*>(output);
    for (size_t i = 0; i < loop_times; ++i) {
      auto ret = memset_s(addr + i * step, step, 0, step);
      FUSION_PASS_CHECK(ret != EOK, OP_LOGE("NnSet", "memset fail."), return FAILED);
    }

    auto ret = memset_s(addr + loop_times * step, tail_count, 0, tail_count);
    FUSION_PASS_CHECK(ret != EOK, OP_LOGE("NnSet", "memset fail."), return FAILED);
  }

  for (int32_t i = 0; i < n; ++i) {
    output[i] = alpha;
  }
  return SUCCESS;
}

class PatternFusionUtil {
 public:
  static Status ConstToAttrWithNode(ge::ComputeGraph& graph, ge::NodePtr& fusedNode, std::string fusionOpType,
                                    std::vector<PassAttrInfo>& attrInfo, ge::NodePtr& fusionNode);
  static size_t GetOutEdgeSize(NodePtr node);
  static ge::OpDescPtr GetFusionOpDesc(ge::NodePtr fusedNodePtr, std::string fusionOpType,
                                       std::vector<PassAttrInfo>& attrInfos);
  static Status SetOutputDescAttrForDataDump(ge::NodePtr fusedNode, ge::NodePtr fusionNode);
  static Status RecordOriginalNamesForConstToAttr(ge::NodePtr& fusedNode, std::vector<PassAttrInfo>& attrInfos,
                                                  std::vector<ge::NodePtr>& originalNodes);
  static Status AddInputToOutput(ge::NodePtr node, std::vector<PassInputInfo>& inputInfoVec);
  static Status ParseChannelIdx(ge::GeTensorDesc& tensorDesc, size_t& channelIdx);
  static Status ParseNChannelIdx(ge::GeTensorDesc& tensorDesc, size_t& channelIdx);
  static Status ProcessGroupPadding(ComputeGraph& graph, const NodePtr& groupConvNode, int64_t groups);

  static Status RemoveInputEdge(ge::NodePtr node);

  static Status SetWeightByIndex(ge::NodePtr node, ge::GeTensorPtr tensor, const uint32_t &index,
                                 ge::ComputeGraph &graph);

  static Status UpdateInputAndOutputName(const ge::OpDescPtr opDescPtr);

  static bool IsUnknownShape(const int64_t& shape);
  /**
  * @ingroup fe
  * @brief add a control edge from source node to dest node
  */
  static Status LinkControlEdge(ge::NodePtr srcNode, ge::NodePtr dstNode);

  static Status CopyMultiReferenceConstNode(ge::ComputeGraph& graph, ge::NodePtr nodePtr);

  static ge::NodePtr InsertSingleNode(ge::ComputeGraph &graph, ge::NodePtr &src_node, const string &op_type,
                                      const bool &is_input, const int32_t &index, vector<ge::NodePtr> &fusion_nodes);

  static Status InsertSliceDNodes(ComputeGraph& graph, NodePtr srcNode, unsigned int constIdx,
                                  const vector<NodePtr>& newConvNodes, int64_t group, size_t sliceDimIdx);

 private:
  static void SetConstValueToAttrWithType(ge::OpDescPtr op_desc, const ge::Tensor& const_tensor, const DataType& dtype,
                                          PassAttrInfo& attrInfo);
  static bool FindAttrInfoByIndex(vector<PassAttrInfo>& attrInfos, int index, PassAttrInfo& retAttrinfo);
  static Status GenGroupPaddingTensor(ge::GeTensorDesc& inTensor, ge::GeTensorDesc& outTensor, int64_t groups,
                                      const NodePtr& weightNode);
  static NodePtr AddGroupPaddingNode(ComputeGraph& graph, ge::GeTensorDesc& inTensor, ge::GeTensorDesc& outTensor,
                                     string nodeName);
  static ge::NodePtr InsertInputNode(ge::ComputeGraph &graph, ge::NodePtr &src_node, const string &op_type,
                                     const int32_t &index, std::atomic<uint64_t> &name_id);
  static ge::NodePtr InsertOutputNode(ge::ComputeGraph &graph, ge::NodePtr &src_node, const string &op_type,
                                      const int32_t &index, std::atomic<uint64_t> &name_id);
  static Status LinkControlAnchorForConst(ge::NodePtr oneConstNode, ge::NodePtr fusionNode);
};

}  // namespace fe

#endif  // OPS_BUILT_IN_FUSION_PASS_COMMON_PATTERN_FUSION_UTIL_H_
