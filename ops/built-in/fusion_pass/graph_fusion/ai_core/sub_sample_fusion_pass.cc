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
 * \file sub_sample_fusion_pass.cc
 * \brief sub_sample --> sub_sample_labels)
 */
#include "sub_sample_fusion_pass.h"
#include <vector>
#include <memory>
#include <iostream>
#include <algorithm>
#include <random>
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
const std::string SubSamplePass::PATTERN_FUSEDNODE = "SubSampleFusedNode";

vector<FusionPattern*> SubSamplePass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = (new (std::nothrow) FusionPattern("SubSampleFusion"));
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE("SubSamplePass", "new pattern error"), return patterns);
  pattern->AddOpDesc(PATTERN_FUSEDNODE, {"SubSample"}).SetOutput(PATTERN_FUSEDNODE);
  patterns.push_back(pattern);
  return patterns;
}

// gen shuffle matrix
void shuffle_matrix_gen(int64_t labels_size,vector<int32_t> &shuffle_matrix) {
  for (int32_t i = 0; i < labels_size; i++) {
    shuffle_matrix.push_back(i);
  }
  std::shuffle(std::begin(shuffle_matrix), std::end(shuffle_matrix), std::random_device());
}

// set node tensor desc
void SubSamplePass::set_node_tensor_desc(ge::GeTensorDesc &tensorDesc,
                                              vector<int64_t> &dims,
                                              const ge::DataType &dtype,
                                              const ge::Format &format) const {
    ge::GeShape shape(dims);
    tensorDesc.SetShape(shape);
    tensorDesc.SetDataType(dtype);
    tensorDesc.SetFormat(format);
    tensorDesc.SetOriginShape(shape);
    tensorDesc.SetOriginDataType(dtype);
    tensorDesc.SetOriginFormat(format);
    return;
}

// assist data to output
Status assist_data_gen(vector<int32_t> data, int32_t *output) {
  if (output == nullptr) {
    OP_LOGE("Sub Sample", "output pointer is null!");
    return FAILED;
  }
  auto size_data = data.size();
  for (size_t i = 0; i < size_data; i++) {
    output[i] = data[i];
  }
  return SUCCESS;
}

Status SubSamplePass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {

  ge::NodePtr fusedNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(fusedNode == nullptr, OP_LOGE("SubSamplePass", "Fusion GetNode Error"),
                    return PARAM_INVALID);

  ge::OpDescPtr fuseDesc = fusedNode->GetOpDesc();
  FUSION_PASS_CHECK(fuseDesc == nullptr, OP_LOGE("SubSamplePass", "fuse_node's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);
  auto label_shape_temp = fuseDesc->GetInputDesc(0).GetShape().GetDims();

  vector<int64_t> label_shape;
  label_shape.push_back(label_shape_temp[0]);
  // shuffle labels
  int64_t label_size = label_shape[0];
  vector<int32_t> shuffle_matrix;
  shuffle_matrix_gen(label_size, shuffle_matrix);
  // create const node 
  int64_t size = label_size;
  unique_ptr<int32_t[]> data(new (std::nothrow) int32_t[size]());

  const int32_t init_value = 0;
  if (NnSet(size, init_value, *reinterpret_cast<int32_t *>(data.get())) != SUCCESS) {
      OP_LOGE("SubSamplePass", "NnSet data failed.");
      return FAILED;
  }
  // update assist data
  Status ret = assist_data_gen(shuffle_matrix, data.get());
  if (ret == FAILED) {
    OP_LOGE("SubSamplePass", "get assist data failed.");
    return FAILED;
  }
  ge::GeTensorDesc assist_const_desc;
  this->set_node_tensor_desc(assist_const_desc, label_shape, ge::DT_INT32, ge::FORMAT_ND);
  ge::GeTensorPtr weight_ptr =
      std::make_shared<ge::GeTensor>(assist_const_desc, reinterpret_cast<uint8_t *>(data.get()), size * sizeof(int32_t));
  if (!weight_ptr) {
      OP_LOGE("SubSamplePass", "create shuffle matrix weight failed.");
      return FAILED;
  }
  vector<ge::GeTensorPtr> weights = {weight_ptr};
  ge::OpDescUtils::SetWeights(fusedNode, weights);
  auto const_input_nodes = OpDescUtils::GetConstInputs(fusedNode);
  if (const_input_nodes.empty()) {
      OP_LOGE("SubSamplePass", "ConstInputNodes is empty, fusion failed.");
      return PARAM_INVALID;
  }
  NodePtr const_input = const_input_nodes[0];
  const_input->GetOpDesc()->SetType("Const");
  fuseDesc->SetType("SubSampleLabels");
  fusionNodes.push_back(fusedNode);
  return SUCCESS;
}
REGISTER_PASS("SubSamplePass", BUILT_IN_GRAPH_PASS, SubSamplePass);
}  // namespace fe