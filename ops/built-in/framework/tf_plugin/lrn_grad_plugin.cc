/**
 * Copyright (C)  2020. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file lrn_grad_plugin.cpp
 *
 * @brief tensorflow plugin for lrn_grad
 *
 * @version 1.0
 *
 */

#include "register/register.h"

namespace domi {
REGISTER_CUSTOM_OP("LRNGrad")
  .FrameworkType(TENSORFLOW)
  .OriginOpType("LRNGrad")
  .ParseParamsFn(AutoMappingFn)
  .ImplyType(ImplyType::TVM);
}  // namespace domi

