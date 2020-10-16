/* *
 * Copyright (C)  2019. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file array_ops_shape_fns.h
 *
 * @brief
 *
 * @version 1.0
 *
 */
#ifndef GE_ARRAY_OPS_SHAPE_FNS_H
#define GE_ARRAY_OPS_SHAPE_FNS_H

#include "graph/operator.h"

namespace ge {
/* *
 * infer pad op shape
 * @param op Operator which need to infershape
 * @return status whether infershape success
 */
graphStatus PadShapeFn(Operator &op);

/* *
 * infer pad grad op shape
 * @param op Operator which need to infershape
 * @return status whether infershape success
 */
graphStatus PadGradShapeFn(Operator &op);
}

#endif
