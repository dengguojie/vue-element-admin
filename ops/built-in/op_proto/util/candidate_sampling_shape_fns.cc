/* *
 * Copyright (C)  2019. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this
 * file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file candidate_sampling_shape_fns.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include "candidate_sampling_shape_fns.h"
#include <vector>
#include "op_log.h"

namespace ge {
graphStatus CandidateSamplerShape(Operator &op)
{
    bool judge = false;
    int64_t numTrue = 0;
    op.GetAttr("num_true", numTrue);
    if (numTrue < 1) {
        OP_LOGE(op.GetName().c_str(), "Attr num_true must >= 1.");
        return GRAPH_FAILED;
    }

    int64_t numSampled = 0;
    op.GetAttr("num_sampled", numSampled);
    if (numSampled < 1) {
        OP_LOGE(op.GetName().c_str(), "Attr num_sampled must >= 1.");
        return GRAPH_FAILED;
    }

    int64_t rangeMax = 0;
    op.GetAttr("range_max", rangeMax);
    if (rangeMax < 1) {
        OP_LOGE(op.GetName().c_str(), "Attr range_max must >= 1.");
        return GRAPH_FAILED;
    }

    Shape trueClasses;
    if (WithRank(op.GetInputDesc(0), 2, trueClasses, op.GetName().c_str()) != GRAPH_SUCCESS) {
        OP_LOGE(op.GetName().c_str(), "input true_classes must be 2-D.");
        return GRAPH_FAILED;
    }

    int64_t batchSize = op.GetInputDesc(0).GetShape().GetDim(0);

    std::vector<int64_t> sampledDims;
    sampledDims.reserve(1);
    sampledDims.push_back(numSampled);

    std::vector<int64_t> trueDims;
    trueDims.reserve(2);
    trueDims.push_back(batchSize);
    trueDims.push_back(numTrue);

    TensorDesc candidateDesc = op.GetOutputDesc("sampled_candidates");
    candidateDesc.SetShape(Shape(sampledDims));
    candidateDesc.SetDataType(DT_INT64);
    judge = (op.UpdateOutputDesc("sampled_candidates", candidateDesc) != GRAPH_SUCCESS);
    if (judge) {
        OP_LOGE(op.GetName().c_str(), "fail to update output sampled_candidates.");
        return GRAPH_FAILED;
    }

    TensorDesc trueDesc = op.GetOutputDesc("true_expected_count");
    trueDesc.SetShape(Shape(trueDims));
    trueDesc.SetDataType(DT_FLOAT);
    judge = (op.UpdateOutputDesc("true_expected_count", trueDesc) != GRAPH_SUCCESS);

    if (judge) {
        OP_LOGE(op.GetName().c_str(), "fail to update output true_expected_count.");
        return GRAPH_FAILED;
    }

    TensorDesc sampledDesc = op.GetOutputDesc("sampled_expected_count");
    sampledDesc.SetShape(Shape(sampledDims));
    sampledDesc.SetDataType(DT_FLOAT);
    judge = (op.UpdateOutputDesc("sampled_expected_count", sampledDesc) != GRAPH_SUCCESS);
    if (judge) {
        OP_LOGE(op.GetName().c_str(), "fail to update output sampled_expected_count.");
        return GRAPH_FAILED;
    }

    return GRAPH_SUCCESS;
}
}