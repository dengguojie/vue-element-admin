/**
 * Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0. You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file test_strided_slice_grad_proto.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */

#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "array_ops.h"
#include "selection_ops.h"
#include <string>
#include <vector>
using namespace std;

class strided_slice_grad : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "strided_slice_grad SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "strided_slice_grad TearDown" << std::endl;
  }

  template<typename T>
  void test(const vector<int64_t> &shape,
            const vector<T> &begin,
            const vector<T> &end,
            const vector<T> &strides,
            const vector<int64_t> &dy,
            int32_t begin_mask,
            int32_t end_mask,
            int32_t ellipsis_mask,
            int32_t new_axis_mask,
            int32_t shrink_axis_mask,
            const vector<std::pair<int64_t, int64_t>> &shape_shape_range,
            const vector<std::pair<int64_t, int64_t>> &dy_shape_range,
            vector<int64_t> &output_shape,
            vector<std::pair<int64_t, int64_t>> &output_range) {
    ge::op::StridedSliceGrad op;
    auto shape_tensor_desc = create_desc_shape_range(shape,
                                                     ge::DT_FLOAT16, ge::FORMAT_ND,
                                                     shape,
                                                     ge::FORMAT_ND, shape_shape_range);
    op.UpdateInputDesc("shape", shape_tensor_desc);
    auto dy_tensor_desc = create_desc_shape_range(dy,
                                                  ge::DT_FLOAT16, ge::FORMAT_ND,
                                                  dy,
                                                  ge::FORMAT_ND, dy_shape_range);
    op.UpdateInputDesc("shape", shape_tensor_desc);
    op.UpdateInputDesc("dy", dy_tensor_desc);
    auto begin_dtype = ge::DT_INT32;
    if (sizeof(T) == sizeof(int64_t)) {
      begin_dtype = ge::DT_INT64;
    }

    if (!begin.empty()) {
      ge::Tensor constTensorBegin;
      ge::TensorDesc constDescBegin(ge::Shape({static_cast<int64_t>(begin.size())}), ge::FORMAT_ND, begin_dtype);
      constDescBegin.SetSize(begin.size() * sizeof(T));
      constTensorBegin.SetTensorDesc(constDescBegin);
      constTensorBegin.SetData((uint8_t*)begin.data(), begin.size() * sizeof(T));
      auto input_begin = ge::op::Constant().set_attr_value(constTensorBegin);
      op.set_input_begin(input_begin);
      auto descBegin = op.GetInputDesc("begin");
      descBegin.SetDataType(begin_dtype);
      descBegin.SetShape(ge::Shape({static_cast<int64_t>(begin.size())}));
      descBegin.SetOriginShape(ge::Shape({static_cast<int64_t>(begin.size())}));
      op.UpdateInputDesc("begin", descBegin);
    } else {
      auto descBegin = op.GetInputDesc("begin");
      descBegin.SetDataType(begin_dtype);
      descBegin.SetShape(ge::Shape({-1}));
      descBegin.SetOriginShape(ge::Shape({-1}));
      op.UpdateInputDesc("begin", descBegin);
    }

    if (!end.empty()) {
      ge::Tensor constTensorEnd;
      ge::TensorDesc constDescEnd(ge::Shape({static_cast<int64_t>(end.size())}), ge::FORMAT_ND, begin_dtype);
      constDescEnd.SetShape(ge::Shape({static_cast<int64_t>(end.size())}));
      constDescEnd.SetSize(end.size() * sizeof(T));
      constTensorEnd.SetTensorDesc(constDescEnd);
      constTensorEnd.SetData((uint8_t*)end.data(), end.size() * sizeof(T));
      auto input_size = ge::op::Constant().set_attr_value(constTensorEnd);
      op.set_input_end(input_size);
      auto descEnd = op.GetInputDesc("end");
      descEnd.SetDataType(begin_dtype);
      descEnd.SetShape(ge::Shape({static_cast<int64_t>(end.size())}));
      descEnd.SetOriginShape(ge::Shape({static_cast<int64_t>(end.size())}));
      op.UpdateInputDesc("end", descEnd);
    } else {
      auto descEnd = op.GetInputDesc("end");
      descEnd.SetDataType(begin_dtype);
      descEnd.SetShape(ge::Shape({-1}));
      descEnd.SetOriginShape(ge::Shape({-1}));
      op.UpdateInputDesc("end", descEnd);
    }

    if (!strides.empty()) {
      ge::Tensor constTensorStride;
      ge::TensorDesc constDescStride(ge::Shape({static_cast<int64_t>(strides.size())}), ge::FORMAT_ND, begin_dtype);
      constDescStride.SetShape(ge::Shape({static_cast<int64_t>(strides.size())}));
      constDescStride.SetSize(strides.size() * sizeof(T));
      constTensorStride.SetTensorDesc(constDescStride);
      constTensorStride.SetData((uint8_t*)strides.data(), strides.size() * sizeof(T));
      auto input_size = ge::op::Constant().set_attr_value(constTensorStride);
      op.set_input_strides(input_size);
      auto descStride = op.GetInputDesc("strides");
      descStride.SetDataType(begin_dtype);
      descStride.SetShape(ge::Shape({static_cast<int64_t>(strides.size())}));
      descStride.SetOriginShape(ge::Shape({static_cast<int64_t>(strides.size())}));
      op.UpdateInputDesc("strides", descStride);
    } else {
      auto descStride = op.GetInputDesc("strides");
      descStride.SetDataType(begin_dtype);
      descStride.SetShape(ge::Shape({-1}));
      descStride.SetOriginShape(ge::Shape({-1}));
      op.UpdateInputDesc("strides", descStride);
    }

    op.SetAttr("begin_mask", begin_mask);
    op.SetAttr("end_mask", end_mask);
    op.SetAttr("ellipsis_mask", ellipsis_mask);
    op.SetAttr("new_axis_mask", new_axis_mask);
    op.SetAttr("shrink_axis_mask", shrink_axis_mask);

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc = op.GetOutputDesc("output");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    output_shape = output_desc.GetShape().GetDims();
    EXPECT_EQ(output_desc.GetShapeRange(output_range), ge::GRAPH_SUCCESS);
  }
};

TEST_F(strided_slice_grad, strided_slice_grad_infer_shape_range_001) {
  std::vector<int64_t> output_shape;
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  test<int32_t>({3},
                {},
                {},
                {1, 1, 1},
                {32,13,13},
                0, 0, 0, 0, 0,
                {},
                {},
                output_shape, output_shape_range);

  std::vector<int64_t> expected_output_shape = {-1, -1, -1};
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {{0, -1}, {0, -1}, {0, -1}};
  EXPECT_EQ(output_shape, expected_output_shape);
  EXPECT_EQ(output_shape_range, expected_shape_range);
}
