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
 * \file batch_norm.cc
 * \brief
 */
#include "batch_norm.h"
#include <string>
#include <vector>
#include "util/util.h"
#include "op_log.h"
#include "util/error_util.h"
#include "graph/utils/node_utils.h"

namespace ge {
// -----------------------------BatchNorm------------------------------
IMPLEMT_VERIFIER(BatchNorm, BatchNormVerify) {
    if (!CheckTwoInputDtypeSame(op, "scale", "offset")) {
        return GRAPH_FAILED;
    }
        return GRAPH_SUCCESS;
}

IMPLEMT_INFERFUNC(BatchNorm, BatchNormInferShape) {
    std::string data_format;
    if (op.GetAttr("data_format", data_format) == GRAPH_SUCCESS) {
        if (data_format != "NHWC" && data_format != "NCHW") {
            string expected_format_list = ConcatString("NHWC, NCHW");
            OpsInputFormatErrReport(op.GetName(), "data_format", expected_format_list, data_format);
            OP_LOGE(op.GetName().c_str(),
                    "data_format only "
                    "support 'NHWC' and 'NCHW'.");
            return GRAPH_FAILED;
        }
    }
    if (!OneInOneOutDynamicInfer(op, "x", {"y"})) {
        return GRAPH_FAILED;
    }
    if (!OneInOneOutDynamicInfer(op, "scale", {"batch_mean", "batch_variance", "reserve_space_1", "reserve_space_2"})) {
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

INFER_FUNC_REG(BatchNorm, BatchNormInferShape);
VERIFY_FUNC_REG(BatchNorm, BatchNormVerify);
}  // namespace ge