/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief Provide some basic methods for fusion pass (include change a const input to attr)
 *
 */

#ifndef FE_PATTER_FUSION_UTIL_H
#define FE_PATTER_FUSION_UTIL_H

#include "securec.h"
#include "graph/compute_graph.h"
#include "graph/tensor.h"
#include "graph/utils/type_utils.h"
#include "op_log.h"
#include "register/graph_optimizer/graph_optimize_register_error_codes.h"

using namespace ge;
namespace fe
{
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
  do {\
    try {\
      exec_expr0;\
    }\
    catch(...) {\
      exec_expr1;\
    }\
  } while (0)

#define FUSION_PASS_CHECK(cond, log_func, return_expr) \
  do {\
    if (cond) {\
      log_func;\
      return_expr;\
    }\
  } while (0)

  template <typename Dtype>
  Status NnSet(const int32_t n, const Dtype alpha, Dtype &output1) {
    Dtype *output = &output1;
    FUSION_PASS_CHECK(output == nullptr, OP_LOGE("NnSet", "output is null"), return FAILED);

    if (alpha == 0) {
      memset_s(output, sizeof(Dtype) * n, 0, sizeof(Dtype) * n);
    }

    for (int32_t i = 0; i < n; ++i) {
      output[i] = alpha;
    }
    return SUCCESS;
  }

class PatternFusionUtil {
 public:
  Status ConstToAttr(ge::ComputeGraph& graph,
                     ge::NodePtr fusedNode,
                     std::string fusionOpType,
                     std::map<int16_t , std::string> attrInfo);

  static Status ConstToAttrWithType(ge::ComputeGraph& graph,
                                    ge::NodePtr& fusedNode,
                                    std::string fusionOpType,
                                    std::vector<PassAttrInfo>& attrInfo);
  static Status ConstToAttrWithNode(ge::ComputeGraph& graph,
                                    ge::NodePtr& fusedNode,
                                    std::string fusionOpType,
                                    std::vector<PassAttrInfo>& attrInfo,
                                    ge::NodePtr& fusionNode);
  static size_t GetOutEdgeSize(NodePtr node);
  static ge::OpDescPtr GetFusionOpDesc(ge::NodePtr fusedNodePtr,
                                       std::string fusionOpType,
                                       std::vector<PassAttrInfo>& attrInfos);
  static Status SetOutputDescAttrForDataDump(ge::NodePtr fusedNode,
                                             ge::NodePtr fusionNode);
  static Status RecordOriginalNamesForConstToAttr(
      ge::NodePtr& fusedNode, std::vector<PassAttrInfo>& attrInfos,
      std::vector<ge::NodePtr> &originalNodes);
  static Status InsertSliceDNodes(ComputeGraph &graph,
                                  NodePtr srcNode,
                                  unsigned int constIdx,
                                  const vector<NodePtr> &newConvNodes,
                                  int64_t group,
                                  size_t sliceDimIdx);
  static Status AddInputToOutput(ge::NodePtr node, std::vector<PassInputInfo> &inputInfoVec);
  static Status ParseChannelIdx (ge::GeTensorDesc &tensorDesc, size_t &channelIdx);
  static Status ProcessGroupPadding(ComputeGraph &graph, const NodePtr& groupConvNode, int64_t groups);

  static Status RemoveInputEdge(ge::NodePtr node);

 /**
 * @ingroup fe
 * @brief add a control edge from source node to dest node
 */
  static Status LinkControlEdge(ge::NodePtr srcNode, ge::NodePtr dstNode);

  static  Status CopyMultiReferenceConstNode(ge::ComputeGraph &graph, ge::NodePtr nodePtr);

 private:
  void SetConstValueToAttr(ge::OpDescPtr op_desc,
                           const ge::Tensor& const_tensor,
                           const DataType& dtype,
                           std::string attr_name);
  static void SetConstValueToAttrWithType(ge::OpDescPtr op_desc,
                                          const ge::Tensor& const_tensor,
                                          const DataType& dtype,
                                          PassAttrInfo& attrInfo);
  static bool FindAttrInfoByIndex(vector<PassAttrInfo>& attrInfos , int index, PassAttrInfo& retAttrinfo);
  static Status GenGroupPaddingTensor(ge::GeTensorDesc &inTensor,
                                    ge::GeTensorDesc &outTensor,
                                    int64_t groups, const NodePtr& weightNode);
  static NodePtr AddGroupPaddingNode(ComputeGraph &graph,
                                     ge::GeTensorDesc &inTensor,
                                     ge::GeTensorDesc &outTensor,
                                     string nodeName);
};

}  // namespace fe

#endif  // FE_PATTER_FUSION_UTIL_H
