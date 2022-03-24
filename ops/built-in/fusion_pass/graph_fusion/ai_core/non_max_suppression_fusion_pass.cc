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
 * \file non_max_suppression_fusion_pass.cc
 * \brief non_max_suppressionv6 --> non_max_suppressionv6)
 */
#include "non_max_suppression_fusion_pass.h"
#include <vector>
#include <memory>
#include "fp16_t.hpp"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "tbe_fusion_pass_util.h"

using namespace std;
using namespace ge;

namespace fe {
  const std::string NonMaxSuppressionV6Pass::PATTERN_FUSEDNODE = "NonMaxSuppressionFusedNode";
  const uint16_t UINT_NUM_ZERO = 0;
  const uint8_t MAX_OUTPUT_SIZE_IDX = 2;
  const int64_t CHANNEL = 4;
  const int64_t SHAPE_SIZE = 4;
  vector<FusionPattern *> NonMaxSuppressionV6Pass::DefinePatterns() {
    vector<FusionPattern*> patterns;
    FusionPattern* pattern = (new(std::nothrow) FusionPattern("NonMaxSuppressionV6Fusion"));
    FUSION_PASS_CHECK(pattern == nullptr,
                      OP_LOGE("NonMaxSuppressionPass",  "new pattern error"),
                      return patterns);
    pattern->AddOpDesc(PATTERN_FUSEDNODE, {"NonMaxSuppressionV6"})
            .SetOutput(PATTERN_FUSEDNODE);
    patterns.push_back(pattern);
    return patterns;
  }

  Status NonMaxSuppressionV6Pass::SetConstDesc(vector<int64_t> &tensor_shape,
                                               ge::GeTensorDesc &tensor_desc,
                                               const ge::GeTensorDesc &desDesc) const {
    // Define auxiliary matrix idx shape
    ge::GeShape tenShapes(tensor_shape);
    tensor_desc.SetOriginFormat(desDesc.GetOriginFormat());
    tensor_desc.SetFormat(desDesc.GetFormat());
    tensor_desc.SetOriginDataType(ge::DT_FLOAT16);
    tensor_desc.SetDataType(ge::DT_FLOAT16);
    tensor_desc.SetOriginShape(tenShapes);
    tensor_desc.SetShape(tenShapes);
    return SUCCESS;
  }

  int64_t GetNmsDims(const vector<int64_t> &shapes) {
    auto shape_lens = shapes.size();
    int64_t dim_num = 1;
    for (size_t i = 0; i < shape_lens; i++) {
      dim_num = dim_num * shapes[i];
    }
    return dim_num;
  }

  Status AssistGen(vector<float> data, uint16_t* const output) {
    if (output == nullptr) {
      OP_LOGE("NonMaxSuppression", "output pointer is null!");
      return FAILED;
    }
    auto size_data = data.size();
    for (size_t i = 0; i < size_data; i++) {
      fp16_t tmp;
      tmp = data[i];
      output[i] = tmp.val;
    }
    return SUCCESS;
  }

  void AssistIndexGen(vector<int64_t> &shape, vector<float> &index_id) {
    const int64_t batch_len = shape[0];
    const int64_t class_len = shape[1];
    const int64_t score_len = shape[2];
    for (int64_t i = 0; i < batch_len; i++) {
      for (int64_t j = 0; j < class_len; j++) {
        for (int64_t k = 0; k < score_len; k++) {
          int64_t iIdx = i * class_len * score_len * CHANNEL + j * score_len * CHANNEL + k * CHANNEL;
          int64_t jIdx = i * class_len * score_len * CHANNEL + j * score_len * CHANNEL + k * CHANNEL + 1;
          int64_t kIdx = i * class_len * score_len * CHANNEL + j * score_len * CHANNEL + k * CHANNEL + 2;
          int64_t hIdx = i * class_len * score_len * CHANNEL + j * score_len * CHANNEL + k * CHANNEL + 3;
          index_id[iIdx] = i * 1.0;
          index_id[jIdx] = j * 1.0;
          index_id[kIdx] = (k / 1000) * 1.0;
          index_id[hIdx] = (k % 1000) * 1.0;
        }
      }
    }
  }

  Status NonMaxSuppressionV6Pass::IdxValueConstNode(vector<int64_t> &idx_value_tensor_shape,
                                                    const ge::GeTensorDesc &input_desc1,
                                                    ge::GeTensorPtr &assit_index_value_ptr,
                                                    ge::GeTensorDesc &idx_value_tensor_desc) const {
    int64_t idx_value_dim_num = GetNmsDims(idx_value_tensor_shape);
    if (idx_value_dim_num < 0) {
      OP_LOGE("NonMaxSuppressionPass", "The shape of score cannot be negative.");
      return FAILED;
    }
    vector<float> index_id(idx_value_dim_num);
    AssistIndexGen(idx_value_tensor_shape, index_id);
    Status ret = SetConstDesc(idx_value_tensor_shape, idx_value_tensor_desc, input_desc1);
    unique_ptr<uint16_t[]> idx_value_assit(new (std::nothrow) uint16_t[idx_value_dim_num]());
    FUSION_PASS_CHECK(idx_value_assit.get() == nullptr,
        OP_LOGE("NonMaxSuppressionPass", "idx_value_assit is NULL"),
        return PARAM_INVALID);

    ret = NnSet(idx_value_dim_num, UINT_NUM_ZERO, *reinterpret_cast<uint16_t *>(idx_value_assit.get()));
    FUSION_PASS_CHECK(ret != SUCCESS,
        OP_LOGE("NonMaxSuppressionPass", "NnSet failed."),
        return ret);
    ret = AssistGen(index_id, idx_value_assit.get());
    FUSION_PASS_MAKE_SHARED((assit_index_value_ptr = std::make_shared<ge::GeTensor>(idx_value_tensor_desc,
        reinterpret_cast<uint8_t *>(idx_value_assit.get()),
        idx_value_dim_num * sizeof(uint16_t))),
        assit_index_value_ptr = nullptr;
        return PARAM_INVALID);
    return SUCCESS;
  }

  bool NonMaxSuppressionV6Pass::GetConstValue(const Tensor &const_tensor,
                                              const DataType &dtype,
                                              std::vector<int32_t>& const_data) {
    if (dtype == ge::DT_INT64) {
      const int64_t* const_data_ptr = reinterpret_cast<const int64_t*>(const_tensor.GetData());
      size_t size = const_tensor.GetSize() / sizeof(int64_t);
      for (size_t i = 0; i < size; ++i) {
        const_data.push_back((static_cast<int32_t>(*(const_data_ptr + i))));
        OP_LOGD("NonMaxSuppressionPass", "const data int64 fusion pass ====== %d", (int64_t)(*(const_data_ptr + i)));
      }
    } else {
      OP_LOGE("NonMaxSuppressionPass", "not support this type");
      return false;
    }
    return true;
  }

  Status NonMaxSuppressionV6Pass::Fusion(ge::ComputeGraph &graph,
                                         Mapping &mapping,
                                         vector<ge::NodePtr> &fusion_nodes) {
    ge::NodePtr fused_node = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
    FUSION_PASS_CHECK(fused_node == nullptr,
                      OP_LOGE("NonMaxSuppressionPass", "Fusion GetNode Error"),
                      return PARAM_INVALID);
    ge::OpDescPtr fuse_desc = fused_node->GetOpDesc();
    FUSION_PASS_CHECK(fuse_desc == nullptr,
                        OP_LOGE("NonMaxSuppressionPass", "fuse_node's OpDesc is null, fusion failed."),
                        return PARAM_INVALID);
    ge::Tensor max_ouput_size_tensor;
    vector<int32_t> size_tensor_list;
    Operator op = ge::OpDescUtils::CreateOperatorFromNode(fused_node);
    if (fuse_desc->MutableInputDesc(fuse_desc->GetInputIndexByName("max_output_size")) != nullptr) {
      if (op.GetInputConstData("max_output_size", max_ouput_size_tensor) != GRAPH_SUCCESS) {
          OP_LOGE("NonMaxSuppressionPass", "Get constValue failed of [max_output_size]");
          return GRAPH_FAILED;
      }
      const char* max_output_size = "max_output_size";
      DataType dtype = op.GetInputDescByName(max_output_size).GetDataType();
      GetConstValue(max_ouput_size_tensor, dtype, size_tensor_list);
      // update op input origin type
      int index = fuse_desc->GetInputIndexByName("max_output_size");
      GeTensorDescPtr output_tensor_desc = fuse_desc->MutableInputDesc(index);
      output_tensor_desc->SetOriginDataType(ge::DT_INT32);
      output_tensor_desc->SetDataType(ge::DT_INT32);
    }
    Format input_format = fuse_desc->GetInputDesc("boxes").GetFormat();
    vector<int64_t> index_shape = fuse_desc->GetInputDesc("scores").GetShape().GetDims();
    index_shape.push_back(CHANNEL);

    ge::GeTensorPtr assit_index_value_ptr = nullptr;
    ge::GeTensorDesc idx_value_tensor_desc(GeShape(index_shape), input_format, ge::DT_FLOAT16);
    ge::GeTensorDesc input_desc1 = fuse_desc->GetInputDesc(0);
    auto ret = IdxValueConstNode(index_shape, input_desc1, assit_index_value_ptr, idx_value_tensor_desc);
    FUSION_PASS_CHECK(ret != SUCCESS,
                      OP_LOGE("NonMaxSuppressionPass", "generate const value of idx fail"),
                      return FAILED);
    vector<ge::GeTensorPtr> const_tensor_vector = ge::OpDescUtils::MutableWeights(fused_node);
    const_tensor_vector.push_back(assit_index_value_ptr);
    ge::OpDescUtils::SetWeights(fused_node, const_tensor_vector);
    auto const_input_nodes = OpDescUtils::GetConstInputs(fused_node);
    if (const_input_nodes.size() <= 0) {
      OP_LOGE("NonMaxSuppressionPass", "GetConstInputs Error");
      return PARAM_INVALID;
    }
    NodePtr const_idx_value_input = const_input_nodes[const_input_nodes.size()-1];
    const_idx_value_input->GetOpDesc()->SetType("Const");
    fuse_desc->SetType("NonMaxSuppressionV7");
    fusion_nodes.push_back(fused_node);
    return SUCCESS;
  }

  REGISTER_PASS("NonMaxSuppressionV6Fusion", BUILT_IN_GRAPH_PASS, NonMaxSuppressionV6Pass);
}
