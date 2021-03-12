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
#include "sigmoid_cross_entropy_with_logits_grad.h"

namespace ge {
// ----------------SigmoidCrossEntropyWithLogitsGrad-------------------
IMPLEMT_COMMON_INFERFUNC(SigmoidCrossEntropyWithLogitsGradInferShape) {
  if (TwoInOneOutDynamicInferNoBroadcast(op, "predict", "target", {"gradient"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(SigmoidCrossEntropyWithLogitsGrad, SigmoidCrossEntropyWithLogitsGradInferShape);
// ---------------SigmoidCrossEntropyWithLogitsGrad END-----------------

// -------------------SigmoidCrossEntropyWithLogits---------------------
IMPLEMT_COMMON_INFERFUNC(SigmoidCrossEntropyWithLogitsInferShape) {
  if (TwoInOneOutDynamicInferNoBroadcast(op, "predict", "target", {"loss"})) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

COMMON_INFER_FUNC_REG(SigmoidCrossEntropyWithLogits, SigmoidCrossEntropyWithLogitsInferShape);
// ------------------SigmoidCrossEntropyWithLogits END------------------
}  // namespace ge