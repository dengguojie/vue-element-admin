/**
 * Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0. You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file test_strided_slice_proto.cpp
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

class strided_slice_v3 : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "strided_slice_v3 SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "strided_slice_v3 TearDown" << std::endl;
  }

  template<typename T>
  void test_dynamic(const vector<int64_t> &input_shape,
            const vector<T> &begin,
            const vector<T> &end,
            const vector<T> &strides,
            const vector<T> &axes,
            const vector<std::pair<int64_t, int64_t>> &shape_range,
            vector<int64_t> &output_shape,
            vector<std::pair<int64_t, int64_t>> &output_range) {
    ge::op::StridedSliceV3 op;
    auto tensor_desc = create_desc_shape_range(input_shape,
                                               ge::DT_FLOAT16, ge::FORMAT_ND,
                                               input_shape,
                                               ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", tensor_desc);
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
      op.UpdateInputDesc("begin", descBegin);
    } else {
      auto descBegin = op.GetInputDesc("begin");
      descBegin.SetDataType(begin_dtype);
      descBegin.SetShape(ge::Shape({-1}));
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
      op.UpdateInputDesc("end", descEnd);
    } else {
      auto descEnd = op.GetInputDesc("end");
      descEnd.SetDataType(begin_dtype);
      descEnd.SetShape(ge::Shape({-1}));
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
      descStride.SetShape(ge::Shape({static_cast<int64_t>(strides.size())}));;
      op.UpdateInputDesc("strides", descStride);
    } else {
      auto descStride = op.GetInputDesc("strides");
      descStride.SetDataType(begin_dtype);
      descStride.SetShape(ge::Shape({-1}));;
      op.UpdateInputDesc("strides", descStride);
    }

    if (!axes.empty()) {
      ge::Tensor constTensorAxes;
      ge::TensorDesc constDescAxes(ge::Shape({static_cast<int64_t>(axes.size())}), ge::FORMAT_ND, begin_dtype);
      constDescAxes.SetShape(ge::Shape({static_cast<int64_t>(axes.size())}));
      constDescAxes.SetSize(axes.size() * sizeof(T));
      constTensorAxes.SetTensorDesc(constDescAxes);
      constTensorAxes.SetData((uint8_t*)axes.data(), axes.size() * sizeof(T));
      auto input_size = ge::op::Constant().set_attr_value(constTensorAxes);
      op.set_input_axes(input_size);
      auto descAxes = op.GetInputDesc("axes");
      descAxes.SetDataType(begin_dtype);
      descAxes.SetShape(ge::Shape({static_cast<int64_t>(axes.size())}));;
      op.UpdateInputDesc("axes", descAxes);
    } else {
      auto descAxes = op.GetInputDesc("axes");
      descAxes.SetDataType(begin_dtype);
      descAxes.SetShape(ge::Shape({-1}));;
      op.UpdateInputDesc("axes", descAxes);
    }
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    output_shape = output_desc.GetShape().GetDims();
    EXPECT_EQ(output_desc.GetShapeRange(output_range), ge::GRAPH_SUCCESS);
  }

  template<typename T>
  void test_static(const vector<int64_t> &input_shape,
            const vector<T> &begin,
            const vector<T> &end,
            const vector<T> &strides,
            const vector<T> &axes,
            const vector<std::pair<int64_t, int64_t>> &shape_range,
            vector<int64_t> &output_shape,
            vector<std::pair<int64_t, int64_t>> &output_range) {
    ge::op::StridedSliceV3 op;
    auto tensor_desc = create_desc_shape_range(input_shape,
                                               ge::DT_FLOAT16, ge::FORMAT_ND,
                                               input_shape,
                                               ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", tensor_desc);
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
      op.UpdateInputDesc("begin", descBegin);
    } else {
      auto descBegin = op.GetInputDesc("begin");
      descBegin.SetDataType(begin_dtype);
      descBegin.SetShape(ge::Shape({0}));
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
      op.UpdateInputDesc("end", descEnd);
    } else {
      auto descEnd = op.GetInputDesc("end");
      descEnd.SetDataType(begin_dtype);
      descEnd.SetShape(ge::Shape({0}));
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
      descStride.SetShape(ge::Shape({static_cast<int64_t>(strides.size())}));;
      op.UpdateInputDesc("strides", descStride);
    } else {
      auto descStride = op.GetInputDesc("strides");
      descStride.SetDataType(begin_dtype);
      descStride.SetShape(ge::Shape({0}));;
      op.UpdateInputDesc("strides", descStride);
    }

    if (!axes.empty()) {
      ge::Tensor constTensorAxes;
      ge::TensorDesc constDescAxes(ge::Shape({static_cast<int64_t>(axes.size())}), ge::FORMAT_ND, begin_dtype);
      constDescAxes.SetShape(ge::Shape({static_cast<int64_t>(axes.size())}));
      constDescAxes.SetSize(axes.size() * sizeof(T));
      constTensorAxes.SetTensorDesc(constDescAxes);
      constTensorAxes.SetData((uint8_t*)axes.data(), axes.size() * sizeof(T));
      auto input_size = ge::op::Constant().set_attr_value(constTensorAxes);
      op.set_input_axes(input_size);
      auto descAxes = op.GetInputDesc("axes");
      descAxes.SetDataType(begin_dtype);
      descAxes.SetShape(ge::Shape({static_cast<int64_t>(axes.size())}));;
      op.UpdateInputDesc("axes", descAxes);
    } else {
      auto descAxes = op.GetInputDesc("axes");
      descAxes.SetDataType(begin_dtype);
      descAxes.SetShape(ge::Shape({0}));;
      op.UpdateInputDesc("axes", descAxes);
    }
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    output_shape = output_desc.GetShape().GetDims();
    EXPECT_EQ(output_desc.GetShapeRange(output_range), ge::GRAPH_SUCCESS);
  }
};

//main case for strided_slice_v3
TEST_F(strided_slice_v3, strided_slice_v3_infer_shape_range_head) {
  std::vector<int64_t> output_shape;
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  test_dynamic<int32_t>({9, 10, 11, 12},
                {},
                {8, 7},
                {1, 1},
                {0, 1},
                {},
                output_shape, output_shape_range);

  std::vector<int64_t> expected_output_shape = {-1, -1, -1, -1};
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {{0, 8}, {0, 7}, {0, 11}, {0, 12}};
  EXPECT_EQ(output_shape, expected_output_shape);
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

//main case for strided_slice_v3
TEST_F(strided_slice_v3, strided_slice_v3_infer_shape_range_tail) {
  std::vector<int64_t> output_shape;
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  test_dynamic<int32_t>({9, 10, 11, 12},
                {0, 0},
                {8, 7},
                {1, 1},
                {2, 3},
                {},
                output_shape, output_shape_range);

  std::vector<int64_t> expected_output_shape = {9, 10, 8, 7};
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {};
  EXPECT_EQ(output_shape, expected_output_shape);
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(strided_slice_v3, strided_slice_v3_infer_shape_range_no_all) {
  std::vector<int64_t> output_shape;
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  test_dynamic<int32_t>({9, 10, 11, 12},
                {},
                {},
                {},
                {},
                {},
                output_shape, output_shape_range);

  std::vector<int64_t> expected_output_shape = {-2};
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {};
  EXPECT_EQ(output_shape, expected_output_shape);
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(strided_slice_v3, strided_slice_v3_infer_shape_range_begin_len_neg) {
  std::vector<int64_t> output_shape;
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  test_dynamic<int32_t>({-1, -1, -1},
                {-3, -3, -3},
                {-6, -6, -6},
                {-1, -1, -1},
                {0, 1, 2},
                {{2, 100}, {4, 100}, {6, 100}},
                output_shape, output_shape_range);

  std::vector<int64_t> expected_output_shape = {-1, -1, -1};
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {{0, 3},{2, 3},{3, 3}};
  EXPECT_EQ(output_shape, expected_output_shape);
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(strided_slice_v3, strided_slice_v3_infer_shape_range_no_stride_no_end) {
  std::vector<int64_t> output_shape;
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  test_static<int32_t>({9, 10, 11, 12} ,
                {},
                {8, 7},
                {},
                {},
                {},
                output_shape, output_shape_range);

  //std::vector<int64_t> expected_output_shape = {-1, -1, 11, 12};
  std::vector<int64_t> expected_output_shape = {-2};
  //std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {{0, 8},{0, 7},{11, 11},{12, 12}};
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {};
  EXPECT_EQ(output_shape, expected_output_shape);
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(strided_slice_v3, strided_slice_v3_infer_shape_range_zero_input) {
  std::vector<int64_t> output_shape;
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  test_dynamic<int32_t>({0},
                {-3, -3, -3},
                {-6, -6, -6},
                {-1, -1, -1},
                {0, 1, 2},
                {{2, 100}, {4, 100}, {6, 100}},
                output_shape, output_shape_range);

  std::vector<int64_t> expected_output_shape = {0};
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {};
  EXPECT_EQ(output_shape, expected_output_shape);
  EXPECT_EQ(output_shape_range, expected_shape_range);
}