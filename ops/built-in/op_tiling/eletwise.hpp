/*
 * Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <vector>
#include <string>

namespace optiling {

const int32_t BLOCK_SIZE = 32;

/*
 * @brief: tiling function of op
 * @param [in] op_type: op_type of the op
 * @param [in] input_shape_x: input shape of the first operand
 * @param [in] input_shape_y: input shape of the second operand
 * @param [out] output_shape: max value for every shape
 * @return bool: success or not
 */
bool ProduceShapes(const std::string &op_type,
                   std::vector<int64_t> &input_shape_x,
                   std::vector<int64_t> &input_shape_y,
                   std::vector<int64_t> &output_shape);

/*
 * @brief: tiling function of op
 * @param [in] op_type: op_type of the op
 * @param [in] input_shape_x: input shape of the first operand
 * @param [in] input_shape_y: input shape of the second operand
 * @param [out] output_shape: broadcast shapes
 * @return bool: success or not
 */
bool RefineShapesForBroadcast(const std::string &op_type,
                              std::vector<int64_t> &input_shape_x,
                              std::vector<int64_t> &input_shape_y,
                              std::vector<int64_t> &output_shape);

}
