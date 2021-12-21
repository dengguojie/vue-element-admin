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
 * \file tbe_fusion_pass_util.h
 * \brief
 */
#ifndef OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_TBE_FUSION_PASS_UTIL_H_
#define OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_TBE_FUSION_PASS_UTIL_H_

#include <vector>
#include <string>

#include "securec.h"
#include "graph/tensor.h"
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
/**
 * Insert a transpose before one input of one op
 * @param [in] fusedNode which node will be inserted
 * @param [in] inputIndex which input index will be inserted
 * @param [in] permList transpose list
 * @param [in] graph
 * @return status whether insert success
 */
Status AddTransposeBeforeNode(const ge::NodePtr& fusedNode, const int64_t& inputIndex, const vector<int64_t>& permList,
                              ge::ComputeGraph& graph);

/**
 * Insert a transpose after one output of one op
 * @param [in] fusedNode which node will be inserted
 * @param [in] inputIndex which output index will be inserted
 * @param [in] permList transpose list
 * @param [in] graph
 * @return status whether insert success
 */

Status AddTransposeAfterNode(const ge::NodePtr& fusedNode, const int64_t& outputIndex, const vector<int64_t>& permList,
                             ge::ComputeGraph& graph);

/**
 * Insert a cast after one output of one op
 * @param [in] fusedNode which node will be inserted
 * @param [in] outputIndex which output index will be inserted
 * @param [in] dst_type cast dst_type
 * @param [in] graph
 * @return status whether insert success
 */
Status AddCastAfterNode(const ge::NodePtr& fusedNode, const int64_t& outputIndex, const ge::DataType& dst_type,
                        ge::ComputeGraph& graph);

class TbeFusionPassUtil {
 public:
  /**
   * Get int type const value from tensor data
   * @param [in] data const tensor data
   * @param [in] data_type DT_INT8, DT_INT16, DT_INT32, DT_INT64
   * @param [out] const_values const int values
   * @return true:success, false:failed.
   */
  static bool GetConstIntData(const ge::Tensor& data, ge::DataType data_type, std::vector<int64_t>& const_values);

  /**
   * Get int type const value from tensor data
   * @param [in] op Operator
   * @param [in] name name of the input
   * @param [out] values const int values
   * @return true:success, false:failed.
   */
  static bool GetConstIntData(const ge::Operator& op, const std::string& name, std::vector<int64_t>& values);

  /**
   * update the attr is_input_const for one op node
   * @param [in] fuse_node which node will be update the attr is_input_const
   * @return true:success, false:failed.
   */
  static bool UpdateAttrIsInputConst(const ge::NodePtr& fuse_node);

  /**
   * check the tensor desc is null or not
   * @param [in] tensor_desc which will not changed
   * @return true:success, false:failed.
   */
  static bool IsEmptyTensor(const ge::GeTensorDesc& tensor_desc);
};

}  // namespace fe

#endif  // OPS_BUILT_IN_FUSION_PASS_GRAPH_FUSION_AI_CORE_TBE_FUSION_PASS_UTIL_H_
