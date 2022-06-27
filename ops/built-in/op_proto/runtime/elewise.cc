/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
#include "runtime_util.h"

using namespace ge;
namespace ops {
IMPL_OP(Cast).InferShape(InferShapeForOneInOneOut);

IMPL_OP(Tanh).InferShape(InferShapeForOneInOneOut);

IMPL_OP(ZerosLike).InferShape(InferShapeForOneInOneOut);

IMPL_OP(Gelu).InferShape(InferShapeForOneInOneOut);

IMPL_OP(Relu).InferShape(InferShapeForOneInOneOut);

IMPL_OP(Neg).InferShape(InferShapeForOneInOneOut);

IMPL_OP(LogSoftmaxV2).InferShape(InferShapeForOneInOneOut);

IMPL_OP(DropOutDoMask).InferShape(InferShapeForOneInOneOut);

IMPL_OP(SoftmaxV2).InferShape(InferShapeForOneInOneOut);
}  // namespace ops
