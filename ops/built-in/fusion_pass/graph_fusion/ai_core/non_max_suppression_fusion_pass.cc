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
    vector<FusionPattern *> NonMaxSuppressionV6Pass::DefinePatterns()
    {
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
    Status NonMaxSuppressionV6Pass::SetConstDesc(vector<int64_t> &tensorShape,
        ge::GeTensorDesc &tensorDesc,
        const ge::GeTensorDesc &desDesc) const {
        // 定义辅助矩阵输入idx shape
        ge::GeShape tenShapes(tensorShape);
        tensorDesc.SetOriginFormat(desDesc.GetOriginFormat());
        tensorDesc.SetFormat(desDesc.GetFormat());
        tensorDesc.SetOriginDataType(ge::DT_FLOAT16);  //desDesc.GetOriginDataType()
        tensorDesc.SetDataType(ge::DT_FLOAT16);  //desDesc.GetDataType()
        tensorDesc.SetOriginShape(tenShapes);
        tensorDesc.SetShape(tenShapes);
        return SUCCESS;
    }

    int64_t GetNmsDims(const vector<int64_t> &shapes) {
        auto shapeLens = shapes.size();
        int64_t dimNum = 1;
        for (size_t i = 0; i < shapeLens; i++) {
            dimNum = dimNum * shapes[i];
        }
        return dimNum;
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

    void AssistIndexGen(vector<int64_t> &shape, vector<float> &index_id)
    {
        const int64_t batchLen = shape[0];
        const int64_t classLen = shape[1];
        const int64_t scoreLen = shape[2];
        for (int64_t i = 0; i < batchLen; i++) {
            for (int64_t j = 0; j < classLen; j++) {
                for (int64_t k = 0; k < scoreLen; k++) {
                    int64_t iIdx = i * classLen * scoreLen * CHANNEL + j * scoreLen * CHANNEL + k * CHANNEL;
                    int64_t jIdx = i * classLen * scoreLen * CHANNEL + j * scoreLen * CHANNEL + k * CHANNEL + 1;
                    int64_t kIdx = i * classLen * scoreLen * CHANNEL + j * scoreLen * CHANNEL + k * CHANNEL + 2;
                    int64_t hIdx = i * classLen * scoreLen * CHANNEL + j * scoreLen * CHANNEL + k * CHANNEL + 3;
                    index_id[iIdx] = i * 1.0;
                    index_id[jIdx] = j * 1.0;
                    index_id[kIdx] = (k / 1000) * 1.0;
                    index_id[hIdx] = (k % 1000) * 1.0;
                }
            }
        }
    }

    Status NonMaxSuppressionV6Pass::IdxValueConstNode(vector<int64_t> &IdxValueTensorShape,
        const ge::GeTensorDesc &inputDesc1,
        ge::GeTensorPtr &assitIndexValuePtr,
        ge::GeTensorDesc &IdxValueTensorDesc) const
    {
        int64_t IdxValueDimNum = GetNmsDims(IdxValueTensorShape);
        vector<float> index_id(IdxValueDimNum);
        AssistIndexGen(IdxValueTensorShape, index_id);
        Status ret = SetConstDesc(IdxValueTensorShape, IdxValueTensorDesc, inputDesc1);
        unique_ptr<uint16_t[]> IdxValueAssit(new (std::nothrow) uint16_t[IdxValueDimNum]());
        FUSION_PASS_CHECK(IdxValueAssit.get() == nullptr,
            OP_LOGE("NonMaxSuppressionPass", "IdxValueAssit is NULL"),
            return PARAM_INVALID);

        ret = NnSet(IdxValueDimNum, UINT_NUM_ZERO, *reinterpret_cast<uint16_t *>(IdxValueAssit.get()));
        FUSION_PASS_CHECK(ret != SUCCESS,
            OP_LOGE("NonMaxSuppressionPass", "NnSet failed."),
            return ret);
        ret = AssistGen(index_id, IdxValueAssit.get());
        FUSION_PASS_MAKE_SHARED((assitIndexValuePtr = std::make_shared<ge::GeTensor>(IdxValueTensorDesc,
            reinterpret_cast<uint8_t *>(IdxValueAssit.get()),
            IdxValueDimNum * sizeof(uint16_t))),
            assitIndexValuePtr = nullptr;
            return PARAM_INVALID);
        return SUCCESS;
    }

    bool NonMaxSuppressionV6Pass::GetConstValue(const Tensor &const_tensor,
        const DataType &dtype, std::vector<int32_t>& const_data)
    {
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

    Status NonMaxSuppressionV6Pass::Fusion(ge::ComputeGraph &graph, Mapping &mapping, vector<ge::NodePtr> &fusionNodes)
    {
        ge::NodePtr fusedNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
        FUSION_PASS_CHECK(fusedNode == nullptr,
                          OP_LOGE("NonMaxSuppressionPass", "Fusion GetNode Error"),
                          return PARAM_INVALID);
        ge::OpDescPtr fuseDesc = fusedNode->GetOpDesc();
        FUSION_PASS_CHECK(fuseDesc == nullptr,
                            OP_LOGE("NonMaxSuppressionPass", "fuse_node's OpDesc is null, fusion failed."),
                            return PARAM_INVALID);
        ge::Tensor maxOuputSizeTensor;
        vector<int32_t> sizeTensorList;
        Operator op = ge::OpDescUtils::CreateOperatorFromNode(fusedNode);
        if (fuseDesc->MutableInputDesc(fuseDesc->GetInputIndexByName("max_output_size")) != nullptr) {
            if (op.GetInputConstData("max_output_size", maxOuputSizeTensor) != GRAPH_SUCCESS) {
                OP_LOGE("NonMaxSuppressionPass", "Get constValue failed of [max_output_size]");
                return GRAPH_FAILED;
            }
            const char* max_output_size = "max_output_size";
            DataType dtype = op.GetInputDescByName(max_output_size).GetDataType();
            GetConstValue(maxOuputSizeTensor, dtype, sizeTensorList);
            // update op input origin type
            int index = fuseDesc->GetInputIndexByName("max_output_size");
            GeTensorDescPtr output_tensor_desc = fuseDesc->MutableInputDesc(index);
            output_tensor_desc->SetOriginDataType(ge::DT_INT32);
            output_tensor_desc->SetDataType(ge::DT_INT32);
        }
        Format inputFormat = fuseDesc->GetInputDesc("boxes").GetFormat();
        vector<int64_t> indexShape = fuseDesc->GetInputDesc("scores").GetShape().GetDims();
        indexShape.push_back(CHANNEL);

        ge::GeTensorPtr assitIndexValuePtr = nullptr;
        ge::GeTensorDesc IdxValueTensorDesc(GeShape(indexShape), inputFormat, ge::DT_FLOAT16);
        ge::GeTensorDesc inputDesc1 = fuseDesc->GetInputDesc(0);
        auto ret = IdxValueConstNode(indexShape, inputDesc1, assitIndexValuePtr, IdxValueTensorDesc);
        FUSION_PASS_CHECK(ret != SUCCESS,
                          OP_LOGE("NonMaxSuppressionPass", "generate const value of idx fail"),
                          return FAILED);
        vector<ge::GeTensorPtr> const_tensor_vector = ge::OpDescUtils::MutableWeights(fusedNode);
        const_tensor_vector.push_back(assitIndexValuePtr);
        ge::OpDescUtils::SetWeights(fusedNode, const_tensor_vector);
        auto const_input_nodes = OpDescUtils::GetConstInputs(fusedNode);
        if (const_input_nodes.size() <= 0) {
            OP_LOGE("NonMaxSuppressionPass", "GetConstInputs Error");
            return PARAM_INVALID;
        }
        NodePtr const_idx_value_input = const_input_nodes[const_input_nodes.size()-1];
        const_idx_value_input->GetOpDesc()->SetType("Const");
        fuseDesc->SetType("NonMaxSuppressionV7");  //这里需要注意修改融合后的算子名.
        fusionNodes.push_back(fusedNode);
        return SUCCESS;
    }

    REGISTER_PASS("NonMaxSuppressionV6Fusion", BUILT_IN_GRAPH_PASS, NonMaxSuppressionV6Pass);
}
