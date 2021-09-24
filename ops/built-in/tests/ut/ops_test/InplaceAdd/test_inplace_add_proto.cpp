#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "array_ops.h"
#include "selection_ops.h"
#include "common/utils/ut_op_util.h"

// ----------------InplaceAdd-------------------
class inplace_add : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "inplace_add SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "inplace_add TearDown" << std::endl;
  }
};

using namespace ut_util;

TEST_F(inplace_add, inplace_add_infershape_diff_test) {
  ge::op::InplaceAdd op;
  op.UpdateInputDesc("x", create_desc({2, 3, 4}, ge::DT_FLOAT16));
  op.UpdateInputDesc("v", create_desc({4, 3, 4}, ge::DT_FLOAT16));
  vector<uint32_t> value = {0, 1, 2, 3};
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(op, indices, vector<int64_t>({4}), ge::DT_INT32, FORMAT_ND, value);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {2, 3, 4};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(inplace_add, inplace_add_test_infershape_same_test) {
  ge::op::InplaceAdd op;
  op.UpdateInputDesc("x", create_desc({2, 5, 6}, ge::DT_FLOAT));
  op.UpdateInputDesc("v", create_desc({2, 5, 6}, ge::DT_FLOAT));
  vector<uint32_t> value = {0, 0};
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(op, indices, vector<int64_t>({2}), ge::DT_INT32, FORMAT_ND, value);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {2, 5, 6};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

// ----------------InplaceAddD-------------------

class inplace_add_d : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "inplace_add_d SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "inplace_add_d TearDown" << std::endl;
  }
};

TEST_F(inplace_add_d, inplace_add_d_infershape_diff_test) {
  ge::op::InplaceAddD op;
  op.UpdateInputDesc("x", create_desc({2, 3, 4}, ge::DT_FLOAT16));
  op.UpdateInputDesc("v", create_desc({4, 3, 4}, ge::DT_FLOAT16));
  std::vector<int64_t> indices_value = {0, 1, 2, 3};
  op.SetAttr("indices",indices_value);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);

  std::vector<int64_t> expected_output_shape = {2, 3, 4};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
