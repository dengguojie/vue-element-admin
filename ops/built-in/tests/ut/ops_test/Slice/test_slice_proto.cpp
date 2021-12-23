/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file test_slice_proto.cpp
 * \brief ut test for slice proto
 */

#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "array_ops.h"
#include "selection_ops.h"
using namespace std;

class Slice : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "slice SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "slice TearDown" << std::endl;
  }

  template <typename T>
  void test(const vector<int64_t> &input_shape,
            const vector<T> &offset,
            const vector<T> &size,
            const vector<int64_t> &ori_shape,
            const vector<std::pair<int64_t, int64_t>> &shape_range,
            vector<int64_t> &output_shape,
            vector<std::pair<int64_t, int64_t>> &output_range) {
    ge::op::Slice op;
    auto tensor_desc = create_desc_shape_range(input_shape,
                                               ge::DT_FLOAT16, ge::FORMAT_ND,
                                               ori_shape,
                                               ge::FORMAT_ND, shape_range);
    op.UpdateInputDesc("x", tensor_desc);
    auto offset_dtype = ge::DT_INT32;
    if (sizeof(T) == sizeof(int64_t)) {
      offset_dtype = ge::DT_INT64;
    }

    if (!offset.empty()) {
      ge::Tensor constTensorOffset;
      ge::TensorDesc constDescOffset(ge::Shape(), ge::FORMAT_ND, offset_dtype);
      constDescOffset.SetSize(offset.size() * sizeof(T));
      constTensorOffset.SetTensorDesc(constDescOffset);
      constTensorOffset.SetData((uint8_t*)offset.data(), offset.size() * sizeof(T));
      auto input_offset = ge::op::Constant().set_attr_value(constTensorOffset);
      op.set_input_offsets(input_offset);
      auto descOffset = op.GetInputDesc("offsets");
      descOffset.SetDataType(offset_dtype);
      op.UpdateInputDesc("offsets", descOffset);
    }

    if (!size.empty()) {
      ge::Tensor constTensorSize;
      ge::TensorDesc constDescSize(ge::Shape(), ge::FORMAT_ND, offset_dtype);
      constDescSize.SetSize(size.size() * sizeof(T));
      constTensorSize.SetTensorDesc(constDescSize);
      constTensorSize.SetData((uint8_t*)size.data(), size.size() * sizeof(T));
      auto input_size = ge::op::Constant().set_attr_value(constTensorSize);
      op.set_input_size(input_size);
      auto descSize = op.GetInputDesc("size");
      descSize.SetDataType(offset_dtype);
      op.UpdateInputDesc("size", descSize);
    }

    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    output_shape = output_desc.GetShape().GetDims();
    EXPECT_EQ(output_desc.GetShapeRange(output_range), ge::GRAPH_SUCCESS);
  }
};

TEST_F(Slice, slice_infer_shape_static_shape1) {
  vector<int64_t> output_shape;
  vector<std::pair<int64_t, int64_t>> output_range;
  test<int32_t>({38, 39}, {0, 2}, {36, 37}, {38, 39}, {}, output_shape, output_range);

  std::vector<int64_t> expected_output_shape = {36, 37};
  EXPECT_EQ(output_shape, expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {
  };
  EXPECT_EQ(output_range, expected_shape_range);
}

TEST_F(Slice, slice_infer_shape_static_shape2) {
  vector<int64_t> output_shape;
  vector<std::pair<int64_t, int64_t>> output_range;
  test<int32_t>({38, 39}, {0, 2}, {36, -1}, {38, 39}, {}, output_shape, output_range);

  std::vector<int64_t> expected_output_shape = {36, 37};
  EXPECT_EQ(output_shape, expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {
  };
  EXPECT_EQ(output_range, expected_shape_range);
}

TEST_F(Slice, slice_infer_shape_dynamic_unknow_dim_shape1) {
  vector<int64_t> output_shape;
  vector<std::pair<int64_t, int64_t>> output_range;
  test<int32_t>({38, -1}, {0, 2}, {36, 37}, {38, 39}, {{38, 38}, {2, 100}}, output_shape, output_range);

  std::vector<int64_t> expected_output_shape = {36, 37};
  EXPECT_EQ(output_shape, expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {
  };
  EXPECT_EQ(output_range, expected_shape_range);
}

TEST_F(Slice, slice_infer_dynamic_unknow_dim_shape2) {
  vector<int64_t> output_shape;
  vector<std::pair<int64_t, int64_t>> output_range;
  test<int32_t>({38, 39}, {0, 0}, {}, {38, 39}, {{2, 100}, {2, 100}}, output_shape, output_range);

  std::vector<int64_t> expected_output_shape = {-1, -1};
  EXPECT_EQ(output_shape, expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {
      {0, 38},
      {0, 39},
  };
  EXPECT_EQ(output_range, expected_shape_range);
}

TEST_F(Slice, slice_infer_dynamic_unknow_dim_shape3) {
  vector<int64_t> output_shape;
  vector<std::pair<int64_t, int64_t>> output_range;
  test<int32_t>({38, -1}, {0, 0}, {}, {38, 39}, {{2, 100}, {2, 100}}, output_shape, output_range);

  std::vector<int64_t> expected_output_shape = {-1, -1};
  EXPECT_EQ(output_shape, expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {
      {0, 38},
      {0, 100},
  };
  EXPECT_EQ(output_range, expected_shape_range);
}

TEST_F(Slice, slice_infer_dynamic_unknow_dim_shape4) {
  vector<int64_t> output_shape;
  vector<std::pair<int64_t, int64_t>> output_range;
  test<int64_t>({38, -1}, {0, 0}, {}, {38, 39}, {{2, 100}, {2, 100}}, output_shape, output_range);

  std::vector<int64_t> expected_output_shape = {-1, -1};
  EXPECT_EQ(output_shape, expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {
      {0, 38},
      {0, 100},
  };
  EXPECT_EQ(output_range, expected_shape_range);
}

TEST_F(Slice, slice_infer_dynamic_unknow_dim_shape5) {
  vector<int64_t> output_shape;
  vector<std::pair<int64_t, int64_t>> output_range;
  test<int64_t>({38, -1}, {2, 3}, {}, {38, 39}, {{2, 100}, {2, 100}}, output_shape, output_range);

  std::vector<int64_t> expected_output_shape = {-1, -1};
  EXPECT_EQ(output_shape, expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {
      {0, 36},
      {0, 97},
  };
  EXPECT_EQ(output_range, expected_shape_range);
}

TEST_F(Slice, slice_infer_dynamic_unknow_dim_shape6) {
  vector<int64_t> output_shape;
  vector<std::pair<int64_t, int64_t>> output_range;
  test<int64_t>({38, -1}, {}, {2, 3}, {38, 39}, {{2, 100}, {2, 100}}, output_shape, output_range);

  std::vector<int64_t> expected_output_shape = {2, 3};
  EXPECT_EQ(output_shape, expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {
  };
  EXPECT_EQ(output_range, expected_shape_range);
}

TEST_F(Slice, slice_infer_dynamic_unknow_dim_shape7) {
  vector<int64_t> output_shape;
  vector<std::pair<int64_t, int64_t>> output_range;
  test<int64_t>({38, -1}, {}, {}, {38, 39}, {{38, 38}, {2, 100}}, output_shape, output_range);

  std::vector<int64_t> expected_output_shape = {-1, -1};
  EXPECT_EQ(output_shape, expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {
      {0, 38}, {0, 100}
  };
  EXPECT_EQ(output_range, expected_shape_range);
}

TEST_F(Slice, slice_infer_dynamic_unknow_dim_shape8) {
  vector<int64_t> output_shape;
  vector<std::pair<int64_t, int64_t>> output_range;
  test<int64_t>({38, -1}, {1, 2}, {5, -1}, {38, 39}, {{38, 38}, {2, 100}}, output_shape, output_range);

  std::vector<int64_t> expected_output_shape = {5, -1};
  EXPECT_EQ(output_shape, expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {
      {5, 5}, {0, 100}
  };
  EXPECT_EQ(output_range, expected_shape_range);
}

TEST_F(Slice, slice_infer_dynamic_unknow_rank_shape1) {
  vector<int64_t> output_shape;
  vector<std::pair<int64_t, int64_t>> output_range;
  test<int32_t>({-2}, {0, 2}, {36, 37}, {38, 39}, {}, output_shape, output_range);

  std::vector<int64_t> expected_output_shape = {36, 37};
  EXPECT_EQ(output_shape, expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {
  };
  EXPECT_EQ(output_range, expected_shape_range);
}

TEST_F(Slice, slice_infer_dynamic_unknow_rank_shape2) {
  vector<int64_t> output_shape;
  vector<std::pair<int64_t, int64_t>> output_range;
  test<int32_t>({-2}, {0, 2}, {}, {38, 39}, {}, output_shape, output_range);

  std::vector<int64_t> expected_output_shape = {-1, -1};
  EXPECT_EQ(output_shape, expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {
      {0, -1},
      {0, -1}
  };
  EXPECT_EQ(output_range, expected_shape_range);
}

TEST_F(Slice, slice_infer_dynamic_unknow_rank_shape3) {
  vector<int64_t> output_shape;
  vector<std::pair<int64_t, int64_t>> output_range;
  test<int32_t>({-2}, {}, {2, 3}, {38, 39}, {}, output_shape, output_range);

  std::vector<int64_t> expected_output_shape = {2, 3};
  EXPECT_EQ(output_shape, expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {
  };
  EXPECT_EQ(output_range, expected_shape_range);
}

TEST_F(Slice, slice_infer_dynamic_unknow_rank_shape4) {
  vector<int64_t> output_shape;
  vector<std::pair<int64_t, int64_t>> output_range;
  test<int32_t>({-2}, {}, {2, -1}, {38, 39}, {}, output_shape, output_range);

  std::vector<int64_t> expected_output_shape = {2, -1};
  EXPECT_EQ(output_shape, expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {
      {2, 2}, {0, -1}
  };
  EXPECT_EQ(output_range, expected_shape_range);
}

TEST_F(Slice, slice_infer_dynamic_unknow_rank_shape5) {
  vector<int64_t> output_shape;
  vector<std::pair<int64_t, int64_t>> output_range;
  test<int32_t>({-2}, {}, {}, {38, 39}, {}, output_shape, output_range);

  std::vector<int64_t> expected_output_shape = {-2};
  EXPECT_EQ(output_shape, expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {
  };
  EXPECT_EQ(output_range, expected_shape_range);
}

TEST_F(Slice, slice_infer_dynamic_unknow_shape9) {
  vector<int64_t> output_shape;
  vector<std::pair<int64_t, int64_t>> output_range;
  test<int32_t>({4,-1,-1,9}, {}, {-1,8,8,-1}, {4,-1,-1,9}, {{4, 4}, {1,10}, {1,10}, {9,9}},
  output_shape, output_range);

  std::vector<int64_t> expected_output_shape = {4,8,8,9};
  EXPECT_EQ(output_shape, expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {
  };
  EXPECT_EQ(output_range, expected_shape_range);
}
