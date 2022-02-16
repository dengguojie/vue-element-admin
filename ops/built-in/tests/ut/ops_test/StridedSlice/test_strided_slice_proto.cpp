/**
 * Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0. You may not use this file except in compliance with the
 License.

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

#include <gtest/gtest.h>

#include <iostream>
#include <string>
#include <vector>

#include "array_ops.h"
#include "op_proto_test_util.h"
#include "selection_ops.h"
#include "graph/debug/ge_attr_define.h"
#include "utils/op_desc_utils.h"
#include "utils/attr_utils.h"
using namespace std;

class strided_slice : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "strided_slice SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "strided_slice TearDown" << std::endl;
  }

  template <typename T>
  void test(const vector<int64_t>& input_shape, const vector<T>& begin, const vector<T>& end, const vector<T>& strides,
            int32_t begin_mask, int32_t end_mask, int32_t ellipsis_mask, int32_t new_axis_mask,
            int32_t shrink_axis_mask, const vector<std::pair<int64_t, int64_t>>& shape_range,
            vector<int64_t>& output_shape, vector<std::pair<int64_t, int64_t>>& output_range) {
    ge::op::StridedSlice op;
    auto tensor_desc =
        create_desc_shape_range(input_shape, ge::DT_FLOAT16, ge::FORMAT_ND, input_shape, ge::FORMAT_ND, shape_range);
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
    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    output_shape = output_desc.GetShape().GetDims();
    EXPECT_EQ(output_desc.GetShapeRange(output_range), ge::GRAPH_SUCCESS);
  }

  template <typename T>
  void test_data_slice(const vector<int64_t>& input_shape, const vector<int64_t>& ori_shape, const vector<T>& begin,
                       const vector<T>& end, const vector<T>& strides, int32_t begin_mask, int32_t end_mask,
                       int32_t ellipsis_mask, int32_t new_axis_mask, int32_t shrink_axis_mask, ge::Format format,
                       ge::Format ori_format, const vector<vector<int64_t>>& output_data_slice,
                       vector<vector<int64_t>>* input_data_slice) {
    ge::op::StridedSliceD op;
    auto tensor_desc =
        create_desc_shape_range(input_shape, ge::DT_FLOAT16, format, ori_shape, ori_format, {});
    op.UpdateInputDesc("x", tensor_desc);
    op.SetAttr("begin", begin);
    op.SetAttr("end", end);
    op.SetAttr("strides", strides);
    op.SetAttr("begin_mask", begin_mask);
    op.SetAttr("end_mask", end_mask);
    op.SetAttr("ellipsis_mask", ellipsis_mask);
    op.SetAttr("new_axis_mask", new_axis_mask);
    op.SetAttr("shrink_axis_mask", shrink_axis_mask);

    auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc(0);
    ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, output_data_slice);
    auto status = op_desc->InferDataSlice();
    auto tensor_desc_x0 = op_desc->MutableInputDesc(0);
    ge::AttrUtils::GetListListInt(tensor_desc_x0, ge::ATTR_NAME_DATA_SLICE, *input_data_slice);
  }
};

TEST_F(strided_slice, strided_slice_infer_shape_fp16) {
  std::vector<int64_t> output_shape;
  std::vector<int64_t> expected_output_shape = {-1};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {
      {1, 36},
  };
  test<int32_t>({-1}, {0}, {36}, {1}, 0, 0, 0, 0, 0, {{1, 100}}, output_shape, output_shape_range);
  EXPECT_EQ(output_shape, expected_output_shape);
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(strided_slice, strided_slice_infer_shape2_fp16) {
  std::vector<int64_t> output_shape;
  std::vector<int64_t> expected_output_shape = {16, -1, -1, -1, 128};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {
      {16, 16}, {2, 100}, {1, 36}, {1, 36}, {128, 128},
  };
  test<int32_t>({16, -1, -1, -1, 128, 2}, {0, 0}, {0, 1}, {1, 1}, 0, 0, 1, 0, 2,
                {{2, 100}, {2, 100}, {1, 36}, {1, 36}, {1, 36}, {1, 36}}, output_shape, output_shape_range);
  EXPECT_EQ(output_shape, expected_output_shape);
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(strided_slice, strided_slice_infer_shape3_fp16) {
  std::vector<int64_t> output_shape;
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  test<int32_t>({16, 76, 76, 3, 128, 2}, {0, 0}, {0, 1}, {1, 1}, 0, 0, 1, 0, 2,
                {{2, 100}, {2, 100}, {1, 76}, {1, 76}, {1, 128}, {1, 36}}, output_shape, output_shape_range);

  std::vector<int64_t> expected_output_shape = {16, 76, 76, 3, 128};
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {};
  EXPECT_EQ(output_shape, expected_output_shape);
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(strided_slice, strided_slice_infer_shape4_fp16) {
  std::vector<int64_t> output_shape;
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  test<int32_t>({-1, 13, 13, 3, 2}, {0, 0}, {0, 1}, {1, 1}, 0, 0, 0, 1, 0,
                {{2, 100}, {2, 100}, {1, 76}, {1, 76}, {1, 128}}, output_shape, output_shape_range);

  std::vector<int64_t> expected_output_shape = {1, -1, 13, 13, 3, 2};
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {{1, 1}, {1, 1}, {13, 13}, {13, 13}, {3, 3}, {2, 2}};
  EXPECT_EQ(output_shape, expected_output_shape);
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(strided_slice, strided_slice_infer_shape_range_00) {
  std::vector<int64_t> output_shape;
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  test<int32_t>({-1, -1, -1}, {6, 6, 6}, {3, 3, 3}, {-1, -1, -1}, 0, 0, 0, 0, 0, {{2, 100}, {6, 100}, {7, 100}},
                output_shape, output_shape_range);

  std::vector<int64_t> expected_output_shape = {-1, -1, -1};
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {{0, 3}, {2, 3}, {3, 3}};
  EXPECT_EQ(output_shape, expected_output_shape);
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(strided_slice, strided_slice_infer_shape_range_01) {
  std::vector<int64_t> output_shape;
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  test<int32_t>({-1, -1, -1}, {-3, -3, -3}, {-6, -6, -6}, {-1, -1, -1}, 0, 0, 0, 0, 0, {{2, 100}, {4, 100}, {6, 100}},
                output_shape, output_shape_range);

  std::vector<int64_t> expected_output_shape = {-1, -1, -1};
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {{0, 3}, {2, 3}, {3, 3}};
  EXPECT_EQ(output_shape, expected_output_shape);
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(strided_slice, strided_slice_infer_shape_range_10) {
  std::vector<int64_t> output_shape;
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  test<int32_t>({-1, -1, -1}, {4, 5, 6}, {6, 8, 9}, {1, 2, 1}, 0b101, 0b011, 0, 0, 0,
                {
                    {2, 100},
                    {2, 100},
                    {36, 76},
                },
                output_shape, output_shape_range);

  std::vector<int64_t> expected_output_shape = {-1, -1, -1};
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {
      {2, 100},
      {0, 48},
      {9, 9},
  };
  EXPECT_EQ(output_shape, expected_output_shape);
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(strided_slice, strided_slice_infer_shape_range_11) {
  std::vector<int64_t> output_shape;
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  test<int32_t>({-1, -1, -1}, {3, 3, 3}, {6, 6, 6}, {1, 1, 1}, 0, 0, 0, 0, 0, {{2, 100}, {5, 100}, {6, 100}},
                output_shape, output_shape_range);

  std::vector<int64_t> expected_output_shape = {-1, -1, -1};
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {{0, 3}, {2, 3}, {3, 3}};
  EXPECT_EQ(output_shape, expected_output_shape);
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(strided_slice, strided_slice_infer_shape_range_12) {
  std::vector<int64_t> output_shape;
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  test<int32_t>({-1, -1, -1}, {-6, -6, -6}, {-3, -3, -3}, {1, 1, 1}, 0, 0, 0, 0, 0, {{2, 100}, {5, 100}, {6, 100}},
                output_shape, output_shape_range);

  std::vector<int64_t> expected_output_shape = {-1, -1, -1};
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {{0, 3}, {2, 3}, {3, 3}};
  EXPECT_EQ(output_shape, expected_output_shape);
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(strided_slice, strided_slice_infer_shape_range_13) {
  std::vector<int64_t> output_shape;
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  test<int32_t>({-1, -1, -1}, {3, 3, 3}, {-1, -1, -1}, {1, 1, 1}, 0, 0, 0, 0, 0, {{2, 100}, {5, 100}, {6, 100}},
                output_shape, output_shape_range);

  std::vector<int64_t> expected_output_shape = {-1, -1, -1};
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {{0, 96}, {1, 96}, {2, 96}};
  EXPECT_EQ(output_shape, expected_output_shape);
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(strided_slice, strided_slice_infer_shape_range_14) {
  std::vector<int64_t> output_shape;
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  test<int32_t>({-1, -1, -1}, {3, 3, 3}, {-5, -5, -5}, {1, 1, 1}, 0, 0, 0, 0, 0, {{2, 100}, {5, 100}, {6, 100}},
                output_shape, output_shape_range);

  std::vector<int64_t> expected_output_shape = {-1, -1, -1};
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {{0, 92}, {0, 92}, {0, 92}};
  EXPECT_EQ(output_shape, expected_output_shape);
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(strided_slice, strided_slice_infer_shape_range_15) {
  std::vector<int64_t> output_shape;
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  test<int32_t>({-1, -1, -1}, {-2, -2, -2}, {55, 55, 55}, {1, 1, 1}, 0, 0, 0, 0, 0, {{2, 100}, {5, 100}, {6, 100}},
                output_shape, output_shape_range);

  std::vector<int64_t> expected_output_shape = {-1, -1, -1};
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {{0, 2}, {0, 2}, {0, 2}};
  EXPECT_EQ(output_shape, expected_output_shape);
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(strided_slice, strided_slice_infer_shape_range_16) {
  std::vector<int64_t> output_shape;
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  test<int32_t>({-1, -1, -1}, {-2, -2, -2}, {1, 1, 1}, {1, 1, 1}, 0, 0, 0, 0, 0, {{2, 100}, {5, 100}, {6, 100}},
                output_shape, output_shape_range);

  std::vector<int64_t> expected_output_shape = {-1, -1, -1};
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {{0, 1}, {0, 0}, {0, 0}};
  EXPECT_EQ(output_shape, expected_output_shape);
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(strided_slice, strided_slice_infer_shape_range_20) {
  std::vector<int64_t> output_shape;
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  test<int32_t>({-1, -1, -1}, {4, 5, 6}, {6, 8, 9}, {1, 2, 1}, 0b101, 0b011, 0, 0, 0,
                {
                    {2, -1},
                    {2, -1},
                    {36, -1},
                },
                output_shape, output_shape_range);

  std::vector<int64_t> expected_output_shape = {-1, -1, -1};
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {
      {2, -1},
      {0, -1},
      {9, 9},
  };
  EXPECT_EQ(output_shape, expected_output_shape);
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(strided_slice, strided_slice_infer_shape_range_21) {
  std::vector<int64_t> output_shape;
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  test<int32_t>({-1, -1, -1}, {3, 0, 3}, {6, 0, 0}, {1, 1, 1}, 0, 0, 0, 0, 0, {{2, -1}, {5, -1}, {6, -1}}, output_shape,
                output_shape_range);

  std::vector<int64_t> expected_output_shape = {-1, -1, -1};
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {{0, 3}, {0, 0}, {0, 0}};
  EXPECT_EQ(output_shape, expected_output_shape);
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(strided_slice, strided_slice_infer_shape_range_22) {
  std::vector<int64_t> output_shape;
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  test<int32_t>({-1, -1, -1}, {-5, -5, -5}, {-2, -2, -2}, {1, 1, 1}, 0, 0, 0, 0, 0, {{2, -1}, {5, -1}, {6, -1}},
                output_shape, output_shape_range);

  std::vector<int64_t> expected_output_shape = {-1, -1, -1};
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {{0, 3}, {3, 3}, {3, 3}};
  EXPECT_EQ(output_shape, expected_output_shape);
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(strided_slice, strided_slice_infer_shape_range_23) {
  std::vector<int64_t> output_shape;
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  test<int32_t>({-1, -1, -1}, {-5, -5, -5}, {2, 2, 55}, {1, 1, 1}, 0, 0, 0, 0, 0, {{1, -1}, {5, -1}, {6, -1}},
                output_shape, output_shape_range);

  std::vector<int64_t> expected_output_shape = {-1, -1, -1};
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {{0, 2}, {0, 2}, {0, 5}};
  EXPECT_EQ(output_shape, expected_output_shape);
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(strided_slice, strided_slice_infer_shape_range_24) {
  std::vector<int64_t> output_shape;
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  test<int32_t>({-1, -1, -1}, {5, 0, 5}, {-2, -2, -55}, {1, 1, 1}, 0, 0, 0, 0, 0, {{1, -1}, {5, -1}, {100, -1}},
                output_shape, output_shape_range);

  std::vector<int64_t> expected_output_shape = {-1, -1, -1};
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {{0, -1}, {3, -1}, {40, -1}};
  EXPECT_EQ(output_shape, expected_output_shape);
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(strided_slice, strided_slice_infer_shape_range_25) {
  std::vector<int64_t> output_shape;
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  test<int32_t>({-1, -1, -1}, {-2, -2, -2}, {55, 55, 55}, {1, 1, 1}, 0, 0, 0, 0, 0, {{2, -1}, {5, -1}, {6, -1}},
                output_shape, output_shape_range);

  std::vector<int64_t> expected_output_shape = {-1, -1, -1};
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {{0, 2}, {0, 2}, {0, 2}};
  EXPECT_EQ(output_shape, expected_output_shape);
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(strided_slice, strided_slice_infer_shape_range_26) {
  std::vector<int64_t> output_shape;
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  test<int32_t>({-1, -1, -1}, {-2, -2, -2}, {1, 1, 1}, {1, 1, 1}, 0, 0, 0, 0, 0, {{2, -1}, {5, -1}, {6, -1}},
                output_shape, output_shape_range);

  std::vector<int64_t> expected_output_shape = {-1, -1, -1};
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {{0, 1}, {0, 0}, {0, 0}};
  EXPECT_EQ(output_shape, expected_output_shape);
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(strided_slice, strided_slice_infer_shape_range_30) {
  std::vector<int64_t> output_shape;
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  test<int32_t>({-1, -1, -1}, {}, {6, 8, 9}, {1, 2, 1}, 0b101, 0b011, 0, 0, 0,
                {
                    {2, -1},
                    {2, -1},
                    {36, -1},
                },
                output_shape, output_shape_range);

  std::vector<int64_t> expected_output_shape = {-1, -1, -1};
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {
      {2, -1},
      {0, -1},
      {9, 9},
  };
  EXPECT_EQ(output_shape, expected_output_shape);
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(strided_slice, strided_slice_infer_shape_range_31) {
  std::vector<int64_t> output_shape;
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  test<int32_t>({-1, -1, -1}, {}, {5, 0, 6}, {1, 1, 1}, 0, 0, 0, 0, 0, {{2, -1}, {5, -1}, {6, -1}}, output_shape,
                output_shape_range);

  std::vector<int64_t> expected_output_shape = {-1, -1, -1};
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {{0, 5}, {0, 0}, {0, 6}};
  EXPECT_EQ(output_shape, expected_output_shape);
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(strided_slice, strided_slice_infer_shape_range_32) {
  std::vector<int64_t> output_shape;
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  test<int32_t>({-1, -1, -1}, {1, 1, 1}, {}, {1, 1, 1}, 0, 0, 0, 0, 0, {{2, -1}, {5, 10}, {6, -1}}, output_shape,
                output_shape_range);

  std::vector<int64_t> expected_output_shape = {-1, -1, -1};
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {{0, -1}, {0, 9}, {0, -1}};
  EXPECT_EQ(output_shape, expected_output_shape);
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(strided_slice, strided_slice_infer_shape_range_33) {
  std::vector<int64_t> output_shape;
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  test<int32_t>({-1, -1, -1}, {1, 1, 1}, {5, 6, 7}, {}, 0, 0, 0, 0, 0, {{2, -1}, {5, 10}, {6, -1}}, output_shape,
                output_shape_range);

  std::vector<int64_t> expected_output_shape = {-2};
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {};
  EXPECT_EQ(output_shape, expected_output_shape);
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(strided_slice, strided_slice_infer_shape_range_34) {
  std::vector<int64_t> output_shape;
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  test<int32_t>({9, 10, 11, 12}, {0, 0}, {}, {1, 1}, 1, 2, 0, 0, 0, {}, output_shape, output_shape_range);

  std::vector<int64_t> expected_output_shape = {-1, -1, 11, 12};
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {{0, 9}, {10, 10}, {11, 11}, {12, 12}};
  EXPECT_EQ(output_shape, expected_output_shape);
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(strided_slice, strided_slice_infer_shape_range_35) {
  std::vector<int64_t> output_shape;
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  test<int32_t>({9, 10, 11, 12}, {0, 0}, {}, {1, 1}, 1, 2, 0, 0, 2, {}, output_shape, output_shape_range);

  std::vector<int64_t> expected_output_shape = {-1, 11, 12};
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {{0, 9}, {11, 11}, {12, 12}};
  EXPECT_EQ(output_shape, expected_output_shape);
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(strided_slice, strided_slice_infer_shape_range_36) {
  std::vector<int64_t> output_shape;
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  test<int32_t>({9, 10, 11, 12}, {0, 0}, {}, {1, 1}, 1, 2, 0, 2, 1, {}, output_shape, output_shape_range);

  std::vector<int64_t> expected_output_shape = {1, 10, 11, 12};
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {};
  EXPECT_EQ(output_shape, expected_output_shape);
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(strided_slice, strided_slice_infer_shape_range_37) {
  std::vector<int64_t> output_shape;
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  test<int32_t>({9, 10, 11, 12}, {0, 0}, {}, {1, 1}, 1, 2, 2, 1, 2, {}, output_shape, output_shape_range);

  std::vector<int64_t> expected_output_shape = {1, 9, 10, 11, 12};
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {};
  EXPECT_EQ(output_shape, expected_output_shape);
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(strided_slice, strided_slice_infer_shape_range_38) {
  std::vector<int64_t> output_shape;
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  test<int32_t>({9, 10, 11, 12}, {}, {}, {1, 1}, 1, 2, 2, 1, 2, {}, output_shape, output_shape_range);

  std::vector<int64_t> expected_output_shape = {1, 9, 10, 11, 12};
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {};
  EXPECT_EQ(output_shape, expected_output_shape);
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(strided_slice, strided_slice_infer_shape_range_39) {
  std::vector<int64_t> output_shape;
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  test<int32_t>({9, 10, 11, 12}, {0, 2}, {}, {1, 1}, 0, 0, 0, 1, 2, {}, output_shape, output_shape_range);

  std::vector<int64_t> expected_output_shape = {1, 10, 11, 12};
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {};
  EXPECT_EQ(output_shape, expected_output_shape);
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(strided_slice, strided_slice_infer_shape_range_40) {
  std::vector<int64_t> output_shape;
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  test<int32_t>({9, 10, 11, 12}, {0, 2}, {}, {1, 1}, 0, 0, 0, 0, 0, {}, output_shape, output_shape_range);

  std::vector<int64_t> expected_output_shape = {-1, -1, 11, 12};
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {{0, 9}, {0, 8}, {11, 11}, {12, 12}};
  EXPECT_EQ(output_shape, expected_output_shape);
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(strided_slice, strided_slice_infer_shape_range_41) {
  std::vector<int64_t> output_shape;
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  test<int32_t>({9, 10, 11, 12}, {}, {8, 7}, {1, 1}, 0, 0, 0, 0, 0, {}, output_shape, output_shape_range);

  std::vector<int64_t> expected_output_shape = {-1, -1, 11, 12};
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {{0, 8}, {0, 7}, {11, 11}, {12, 12}};
  EXPECT_EQ(output_shape, expected_output_shape);
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(strided_slice, strided_slice_infer_shape_range_42) {
  std::vector<int64_t> output_shape;
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  test<int32_t>({9, 10, 11, 12}, {}, {}, {}, 0, 0, 0, 0, 0, {}, output_shape, output_shape_range);

  std::vector<int64_t> expected_output_shape = {-2};
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {};
  EXPECT_EQ(output_shape, expected_output_shape);
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(strided_slice, strided_slice_infer_shape_range_43) {
  std::vector<int64_t> output_shape;
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  test<int32_t>({32, 13, 13, 3, 32}, {}, {}, {1, 1}, 0, 0, 0, 1, 0, {}, output_shape, output_shape_range);

  std::vector<int64_t> expected_output_shape = {1, -1, 13, 13, 3, 32};
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {{1, 1},   {0, 32}, {13, 13},
                                                                   {13, 13}, {3, 3},  {32, 32}};
  EXPECT_EQ(output_shape, expected_output_shape);
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(strided_slice, strided_slice_infer_shape_range_44) {
  std::vector<int64_t> output_shape;
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  test<int32_t>({9, 10, 11, 12}, {0, 0}, {1, 1}, {}, 1, 2, 0, 0, 0, {}, output_shape, output_shape_range);

  std::vector<int64_t> expected_output_shape = {-2};
  std::vector<std::pair<int64_t, int64_t>> expected_shape_range = {};
  EXPECT_EQ(output_shape, expected_output_shape);
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

TEST_F(strided_slice, concatd_data_slice_infer1) {
  std::vector<std::vector<int64_t>> output_slice = {{0, 5}, {0, 5}, {0, 5}, {0, 6}};
  std::vector<std::vector<int64_t>> input_slice;
  test_data_slice<int32_t>({9, 10, 11, 12}, {9, 10, 11, 12}, {0, 0}, {1, 1}, {}, 3, 3, 0, 0, 0, ge::Format::FORMAT_ND,
                           ge::Format::FORMAT_ND, output_slice, &input_slice);
  EXPECT_EQ(output_slice, input_slice);
}

TEST_F(strided_slice, concatd_data_slice_infer2) {
  std::vector<std::vector<int64_t>> output_slice = {{0, 5}, {0, 1}, {0, 5}, {0, 5}};
  std::vector<std::vector<int64_t>> input_slice;
  test_data_slice<int32_t>({9, 2, 11, 12, 16}, {9, 11, 12, 32}, {0, 0}, {1, 1}, {}, 2, 2, 0, 0, 0,
                           ge::Format::FORMAT_NC1HWC0, ge::Format::FORMAT_ND, output_slice, &input_slice);
  EXPECT_EQ(output_slice, input_slice);
}

TEST_F(strided_slice, concatd_data_slice_infer3) {
  std::vector<std::vector<int64_t>> output_slice = {};
  std::vector<std::vector<int64_t>> input_slice;
  test_data_slice<int32_t>({9, 10, 11, 12}, {9, 10, 11, 12}, {0, 0}, {1, 1}, {}, 3, 3, 0, 1, 0, ge::Format::FORMAT_ND,
                           ge::Format::FORMAT_ND, output_slice, &input_slice);
  EXPECT_EQ(output_slice, input_slice);
}

TEST_F(strided_slice, concatd_data_slice_infer4) {
  std::vector<std::vector<int64_t>> output_slice = {};
  std::vector<std::vector<int64_t>> input_slice;
  test_data_slice<int32_t>({9, 10, 11, 12}, {9, 10, 11, 12}, {0, 0}, {1, 1}, {}, 3, 3, 0, 0, 1, ge::Format::FORMAT_ND,
                           ge::Format::FORMAT_ND, output_slice, &input_slice);
  EXPECT_EQ(output_slice, input_slice);
}

TEST_F(strided_slice, concatd_data_slice_infer5) {
  std::vector<std::vector<int64_t>> output_slice = {{0, 5}, {0, 1}, {0, 5}, {0, 5}};
  std::vector<std::vector<int64_t>> input_slice;
  test_data_slice<int32_t>({9, 10, 11, 12}, {9, 10, 11, 12}, {0, 0}, {1, 1}, {}, 3, 3, 0, 0, 1, ge::Format::FORMAT_ND,
                           ge::Format::FORMAT_ND, output_slice, &input_slice);
  EXPECT_NE(output_slice, input_slice);
}
