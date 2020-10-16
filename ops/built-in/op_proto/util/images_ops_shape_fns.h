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
 * @file images_ops_shape_fns.h
 *
 * @brief
 *
 * @version 1.0
 *
 */

#ifndef GE_IMAGES_OPS_SHAPE_FNS_H
#define GE_IMAGES_OPS_SHAPE_FNS_H

#include <string>

#include "common_shape_fns.h"

namespace ge {
/**
 * ColorspaceShapeFn, infereshape funtion of colorspace op
 * @param op, Operators that need to reason about shape
 * @param output_name, the name of output
 * @return status whether infer shape success
 */
graphStatus ColorspaceShapeFn(Operator &op, const std::string output_name);

/**
 * ResizeShapeFn, infereshape funtion of image resize op
 * @param op, Operators that need to reason about shape
 * @param input_name, the name of input
 * @param size_input_name, the name of size input name
 * @param output_name, the name of output
 * @return status whether infer shape success
 */
graphStatus ResizeShapeFn(Operator &op,
                          const std::string input_name,
                          const std::string size_input_name,
                          const std::string output_name);

/**
 * SetOutputToSizedImage, set output shape of size image op
 * @param op, Operators that need to set output shape
 * @param batch_dim, the dim of batch
 * @param size_input_name, the name of size input
 * @param channel_dim, the dim of channel
 * @param output_name, the name of output
 * @return status whether set output shape success
 */
graphStatus SetOutputToSizedImage(Operator &op,
                                  const int64_t batch_dim,
                                  const std::string size_input_name,
                                  const int64_t channel_dim,
                                  const std::string output_name);

/**
 * EncodeImageShapeFn, infereshape funtion of EncodeImage op
 * @param op, Operators that need to reason about shape
 * @return status whether infer shape success
 */
graphStatus EncodeImageShapeFn(Operator &op);

/**
 * EncodeImageShapeFn, infereshape funtion of EncodeImage op
 * @param inputs, the list of impu dims
 * @param unknown_dim_val, the definithion of UNKNOWN_DIM
 * @return status whether infer shape success
 */
template<typename T = int64_t>
bool DimsAllEqualOrUnknown(std::initializer_list<T> && inputs, T unknown_dim_val = UNKNOWN_DIM);

}

#endif