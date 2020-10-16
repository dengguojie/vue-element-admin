/**
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
 * @file candidate_sampling_shape_fns.h
 *
 * @brief
 *
 * @version 1.0
 *
 */

#ifndef GE_CANDIDATE_SAMPLING_OPS_SHAPE_FNS_H
#define GE_CANDIDATE_SAMPLING_OPS_SHAPE_FNS_H

#include "common_shape_fns.h"

namespace ge {
/**
 * Set output shape that as same as a input for candidate sampling op
 * @param op Operator
 * @return status whether Shape's condition Satisfied
 */
graphStatus CandidateSamplerShape(Operator &op);
}
#endif
