/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
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
 * \file matmul_atmoic_add_ub_fusion.h
 * \brief matmul use atmoic add fusion pattern
 */

#ifndef OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_UB_FUSION_AI_CORE_MATMUL_MATMUL_ATOMIC_ADD_UB_FUSION_H
#define OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_UB_FUSION_AI_CORE_MATMUL_MATMUL_ATOMIC_ADD_UB_FUSION_H

#define ATOMIC_ADD_DISABLE 0
#define ATOMIC_ADD_ENABLE 1
#define ATOMIC_ADD_NEED_CAST 2
#define ATOMIC_ADD_NEED_TRANSDATA 4

#include <utility>
#include <vector>

#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_base.h"

namespace fe {
class MatmulAtomicAddUbFusion : public BufferFusionPassBase {
 public:
  explicit MatmulAtomicAddUbFusion() {}

  ~MatmulAtomicAddUbFusion() {}
  bool EnableAtomicAdd(const ge::NodePtr &matmul_node);
  bool NeedSplitK(const ge::NodePtr &matmul_node);
  Status GetBandWidth(int64_t &hbm_bandwidth, int64_t &l2_bandwidth);
  bool computePerf(vector<int64_t> shapes, vector<int> block_dims, int64_t cur_bandwidth, int64_t hbm_bandwidth,
                   ge::DataType out_dtype, ge::Format out_format, float &cur_cost);
  bool getValueByKey(std::unordered_map<ge::DataType, int> ori_map, ge::DataType traget_key, int &target_value);
  vector<int64_t> GetMatMulDims(const ge::NodePtr &matmul_node);
  int AtomicAddType(const ge::NodePtr &matmul_node);
  Status IsDynamic(const ge::NodePtr &matmul_node, bool &is_dynamic, bool &is_no_range);
  bool IsTheRangeOfNoRange(const vector<std::pair<int64_t, int64_t>> &range_data);
  Status GenerateCastNode(ge::NodePtr &matmul_node, ge::NodePtr &cast_node);
  Status GenerateTransDataNode(ge::NodePtr &matmul_node, ge::NodePtr &transdata_node);
  Status AddSuperKernelId(ge::NodePtr &matmul_node);
  Status AddCustomNode(int cur_add_node_type, ge::NodePtr &matmul_node, vector<ge::NodePtr> &fusion_nodes);

 protected:
  vector<BufferFusionPattern *> DefinePatterns() override;
  /*
   * @brief: parse nodes matched in mapping and call DoFusion
   * @param [in] graph: original graph
   * @param [out] mapping: nodes matched by pattern
   * @return bool: fusion status ok or not.
   */
  Status GetFusionNodes(const BufferFusionMapping &mapping, vector<ge::NodePtr> &fusion_nodes) override;

 private:
  const string kFusedOpType = "MatMulAtomicAddUbFusion";
  bool is_dynamic_flag = false;
  bool is_no_range = false;
  int64_t block_in = 16;
  int64_t block_reduce = 16;
  int64_t block_out = 16;
  int core_num = 1;
  int l2_size = 1;

  std::unordered_map<int, int> soc_hbm_bandwidth_info = {{8, 250}, {32, 1100}};

  std::unordered_map<int, int> soc_l2_bandwidth_info = {{8, 1300}, {32, 3300}};

  std::unordered_map<ge::DataType, int> bytes_dtype = {{ge::DT_FLOAT16, 2}, {ge::DT_FLOAT, 4}};
};
}  // namespace fe

#endif  // OPS_BUILT_IN_FUSION_PASS_BUFFER_FUSION_UB_FUSION_AI_CORE_MATMUL_MATMUL_ATOMIC_ADD_UB_FUSION_H
