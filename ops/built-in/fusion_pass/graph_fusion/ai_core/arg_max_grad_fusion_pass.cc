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
 * \file arg_max_grad_fusion_pass.cpp
 * \brief
 */
#include "arg_max_grad_fusion_pass.h"

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <memory>

#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "securec.h"

#include "op_log.h"
#include "fp16_t.hpp"
#include "pattern_fusion_util.h"

using namespace std;
using namespace ge;

namespace fe {
// the name of tf op
static const int32_t INT_NUM_ZERO = 0;
static const string PATTERN_ARGMAXGRAD = "ArgMaxGrad";
static const std::string CONSTANTOP = "Constant";
static const char* ARGMAXGRAD = "ArgMaxGrad";

/* Generating auxiliary matrix */
template <typename Dtype>
Status assist_int32_help(const int32_t dim, const int32_t shape_len, const int32_t dim_axis_value, 
                         const int32_t first_dim_size, const int32_t last_dim_size,
                         const int32_t last_first_axis_value, const int32_t last_second_axis_value, Dtype& output) {
    int32_t each_block_size = 0;
    int32_t each_block_byte_num = 0;
    Dtype *outbuf = &output;
    Dtype *p = nullptr;

    /* proc the last dim */
    if (shape_len == dim + 1) {
        /* Generate a single row matrix, generate each row like:0 1 2 3...*/
        p = outbuf;
        for (int32_t i = 0; i < last_first_axis_value; i++) {
            p[i] = i;
        }

        /* Generate all row matrix */
        int32_t sum_row_num = first_dim_size;
        each_block_size = last_first_axis_value;
        each_block_byte_num = each_block_size * sizeof(int32_t);
		
        for (int32_t i = 1; i < sum_row_num;) {
            p = outbuf + i * each_block_size;
            if (i < sum_row_num - i) {
                if (EOK != memcpy_s(p, i * each_block_byte_num, outbuf, i * each_block_byte_num)) {
                    return FAILED;
                }
				
                i += i;
            } else {
                if (EOK != memcpy_s(p, (sum_row_num - i) * each_block_byte_num, 
                                    outbuf, (sum_row_num - i) * each_block_byte_num)) {
                    return FAILED;
                }
                break;
            }
        }
    }
  
    /* proc the last second dim */
    else if (shape_len == dim + 2) {
        /* Generates a single matrix element */
        for (int32_t i = 0; i < last_second_axis_value; i++) {
            p = outbuf + i * last_first_axis_value;
            for (int32_t j = 0; j < last_first_axis_value; j++) {
                p[j] = i;
            }
        }

        /* Copy as a unit */
        each_block_size = last_first_axis_value * last_second_axis_value;
        each_block_byte_num = each_block_size * sizeof(int32_t);
        for (int32_t i = 1; i < first_dim_size;) {
            p = outbuf + i * each_block_size;
            if (i < first_dim_size - i) {
                if (EOK != memcpy_s(p, i * each_block_byte_num, outbuf, i * each_block_byte_num)) {
                    return FAILED;
                }
                i += i;
            } else {
                if (EOK != memcpy_s(p, (first_dim_size - i) * each_block_byte_num, 
                                    outbuf, (first_dim_size - i) * each_block_byte_num)) {
                    return FAILED;
                }
                break;
            }
        }
    }
  
    /* Fill in by cell */
    else {
        /* Generating cell block */
        for (int32_t i = 0; i < dim_axis_value; i++) {
            p = outbuf + i * last_dim_size;
            for (int32_t j = 0; j < last_dim_size; j++) {
                p[j] = i;
            }
        }

        /* Copy as a unit block */
        each_block_size = dim_axis_value * last_dim_size;
        each_block_byte_num = each_block_size * sizeof(int32_t);
        for (int32_t i = 1; i < first_dim_size;) {
            p = outbuf + i * each_block_size;
            if (i < first_dim_size - i) {
                if (EOK != memcpy_s(p, i * each_block_byte_num, outbuf, i * each_block_byte_num)) {
                    return FAILED;
                }
                i += i;
            } else {
                if (EOK != memcpy_s(p, (first_dim_size - i) * each_block_byte_num, 
                                    outbuf, (first_dim_size - i) * each_block_byte_num)) {
                    return FAILED;
                }
                break;
            }
        }
    }

    return SUCCESS;
}

vector<FusionPattern*> ArgMaxGradFusionPass::DefinePatterns() {
    vector<FusionPattern*> patterns;

    // arg_max_gradD->arg_max_grad
    // define ArgMaxGradFusion
    FusionPattern* pattern = new (std::nothrow) FusionPattern("ArgMaxGradFusionPass");
    FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

    // define origin graph
    pattern->AddOpDesc(PATTERN_ARGMAXGRAD, {ARGMAXGRAD}).SetOutput(PATTERN_ARGMAXGRAD);

    patterns.push_back(pattern);

    return patterns;
}

Status ArgMaxGradFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
    // arg_max_gradD node
    ge::NodePtr ArgMaxGradVNode = GetNodeFromMapping(PATTERN_ARGMAXGRAD, mapping);
    FUSION_PASS_CHECK(ArgMaxGradVNode == nullptr,
                      OP_LOGE(FUSED_OP_TYPE.c_str(),
                              "ArgMaxGradVNode is null, "
                              "fusion failed."),
                      return PARAM_INVALID);

    // input of arg_max_grad
    ge::OpDescPtr ArgMaxGradDesc = ArgMaxGradVNode->GetOpDesc();
    FUSION_PASS_CHECK(ArgMaxGradDesc == nullptr,
                      OP_LOGE(FUSED_OP_TYPE.c_str(),
                              "ArgMaxGradVNode's OpDesc is "
                              "null, fusion failed."),
                      return PARAM_INVALID);
    ge::OpDescPtr fusionDesc = AttrUtils::CopyOpDesc(ArgMaxGradDesc);

    // get the input desc of the entrance of arg_max_grad node to differentiate between const and var
    ge::GeTensorDesc InputVarTensor = ArgMaxGradVNode->GetOpDesc()->GetInputDesc("var");

    int64_t attr_dim = 0;
    AttrUtils::GetInt(ArgMaxGradDesc, "dimension", attr_dim);

    // get the shape info
    ge::GeShape InputVarShape = InputVarTensor.GetShape();

    // multiples of dims
    int64_t dimNums = 1;
    int32_t shape_len = InputVarShape.GetDimNum();
    if (shape_len <= 0) {
        OP_LOGE(FUSED_OP_TYPE.c_str(), "ArgMaxGradFusionPass shape_len can not be zero.");
        return NOT_CHANGED;
    }

    for (int32_t j = 0; j < shape_len; ++j) {
        if (PatternFusionUtil::IsUnknownShape(InputVarShape.GetDim(j))) {
            OP_LOGE(FUSED_OP_TYPE.c_str(), "ArgMaxGradFusionPass cannot be applied for unknown shape.");
            return NOT_CHANGED;
        }
        dimNums = InputVarShape.GetDim(j) * dimNums;
    }

    vector<int64_t> dimInfo = InputVarShape.GetDims();
    Format assitMatrixFormat = InputVarTensor.GetFormat();

    ge::GeTensorPtr assitPtr = nullptr;
    ge::GeTensorDesc tensorDesc(GeShape(), ge::FORMAT_ND, ge::DT_INT32);

    unique_ptr<int32_t[]> inputAssist(new (std::nothrow) int32_t[dimNums]());
    FUSION_PASS_CHECK(inputAssist.get() == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "inputAssist is NULL"),
                      return PARAM_INVALID);

    /* create assist matrix data */
    int32_t dim = attr_dim;
    if (((dim < 0) && (dim < (0 - shape_len))) || 
        ((dim > 0) && (dim >= shape_len))) {
        OP_LOGE(FUSED_OP_TYPE.c_str(), "ArgMaxGradFusionPass dim should be in range:-shape_len, shape_len).");
        return NOT_CHANGED;
    }

    if (dim < 0) {
        dim = dim + shape_len;
    }
    int32_t dim_axis_value = InputVarShape.GetDim(dim); 
    int32_t first_dim_size = 1;
    int32_t last_dim_size = 1;
    int32_t last_first_axis_value = InputVarShape.GetDim(shape_len - 1);
    int32_t last_second_axis_value = 1;

    if (shape_len > 1) {
        last_second_axis_value = InputVarShape.GetDim(shape_len - 2);
    } 

    if (dim < shape_len - 1) {
        int32_t i = 0;
        while (i < dim) {
            first_dim_size = first_dim_size * InputVarShape.GetDim(i);
            i++;
        }

        i = dim + 1;
        while (i < shape_len) {
            last_dim_size = last_dim_size * InputVarShape.GetDim(i);
            i++;
        }
    } else {
        int32_t i = 0;
        while (i < shape_len - 1) {
            first_dim_size = first_dim_size * InputVarShape.GetDim(i);
            i++;
        }
    }
	
    if (SUCCESS != assist_int32_help(dim, shape_len, dim_axis_value, first_dim_size, last_dim_size,
                                     last_first_axis_value, last_second_axis_value, *inputAssist.get())) {
        OP_LOGE(FUSED_OP_TYPE.c_str(), "ArgMaxGradFusionPass assist_int32 init fail");
        return NOT_CHANGED;
    }

    // define the shape of auxiliary matrix
    ge::GeShape assitShape = InputVarShape;
    tensorDesc.SetShape(assitShape);
    tensorDesc.SetOriginShape(assitShape);
    tensorDesc.SetFormat(assitMatrixFormat);
    tensorDesc.SetOriginFormat(assitMatrixFormat);
    tensorDesc.SetDataType(ge::DT_INT32);
    tensorDesc.SetOriginDataType(ge::DT_INT32);
    FUSION_PASS_MAKE_SHARED((assitPtr = std::make_shared<ge::GeTensor>(
                            tensorDesc, reinterpret_cast<uint8_t*>(inputAssist.get()), dimNums * sizeof(int32_t))),
                            assitPtr = nullptr;
                            return PARAM_INVALID);

    // check op support
    vector<ge::GeTensorPtr> weights = {assitPtr};
    ge::OpDescUtils::SetWeights(ArgMaxGradVNode, weights);
    auto constInputNodes = OpDescUtils::GetConstInputs(ArgMaxGradVNode);
    NodePtr constInput = constInputNodes[0];
    constInput->GetOpDesc()->SetType(CONSTANTOP);
    ArgMaxGradDesc->SetType("ArgMaxGradD");

    return SUCCESS;
}

REGISTER_PASS("ArgMaxGradFusionPass", BUILT_IN_GRAPH_PASS, ArgMaxGradFusionPass);
}  // namespace fe
