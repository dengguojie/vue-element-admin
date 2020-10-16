/**
 * Copyright (C)  2019. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file random_ops_shape_fns.h
 *
 * @brief
 *
 * @version 1.0
 *
 */

#ifndef GE_RANDOM_OPS_SHAPE_FNS_H
#define GE_RANDOM_OPS_SHAPE_FNS_H

#include "common_shape_fns.h"

namespace ge {

/**
 * Set output shape that as same as a input for random op
 * @param op Operator
 * @param shape_name A input shape name
 * @param out_name Output name
 * @return status whether Shape's condition Satisfied
 */
graphStatus RandomShape(Operator &op, const std::string &shape_name,
                        const std::string out_name);

/**
 * Set output shape that as same as a input for random op
 * and set output data type
 * @param op Operator
 * @param shape_name A input shape name
 * @param date_type_attr_name Data type attr name associated to output
 * @param out_name Output name
 * @return status whether Shape's condition Satisfied
 */
graphStatus RandomShapeWithDataType(Operator &op,
                                    const std::string &shape_name,
                                    const std::string &date_type_attr_name,
                                    const std::string &out_name);
}   // namespace ge

#endif  // GE_RANDOM_OPS_SHAPE_FNS_H
