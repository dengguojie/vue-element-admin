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
 * @file test_strided_slice_v2_proto.cpp
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

class strided_slice_v2 : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "strided_slice_v2 SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "strided_slice_v2 TearDown" << std::endl;
  }

  template<typename T>
  void test(ge::op::StridedSliceV2 op,
            const vector<int64_t> &input_shape,
            const vector<T> &begin,
            const vector<T> &end,
            const vector<T> &axes,
            const vector<T> &strides,
            int32_t begin_mask,
            int32_t end_mask,
            int32_t ellipsis_mask,
            int32_t new_axis_mask,
            int32_t shrink_axis_mask,
            const vector<std::pair<int64_t, int64_t>> &shape_range) {
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

    op.SetAttr("begin_mask", begin_mask);
    op.SetAttr("end_mask", end_mask);
    op.SetAttr("ellipsis_mask", ellipsis_mask);
    op.SetAttr("new_axis_mask", new_axis_mask);
    op.SetAttr("shrink_axis_mask", shrink_axis_mask);
  }
};


TEST_F(strided_slice_v2, strided_slice_v2_infer_shape3_fp16) {
  ge::op::StridedSliceV2 op;
  test<int32_t>(op,
                {20, 10, 5},   // x
                {0, 0, 3},     // begin
                {},            // end
                {0, 1, 2},     // axes
                {1, 1, 1},     // strides
                0, 0, 0, 0, 0, // some masks
                {{2, 100}, {2, 100}, {1, 76}}  // shape_range
                );

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

