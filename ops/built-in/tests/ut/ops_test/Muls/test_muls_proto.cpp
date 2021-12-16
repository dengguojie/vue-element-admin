#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "elewise_calculation_ops.h"

class muls : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "muls Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "muls Proto Test TearDown" << std::endl;
  }
};

TEST_F(muls, muls_infershape_diff_test){
  ge::op::Muls op;
  op.UpdateInputDesc("x", create_desc({4, 3, 4}, ge::DT_FLOAT16));
  
  float value = 3.0;
  op.SetAttr("value", value);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {4, 3, 4};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(muls, muls_infershape_same_test){
  ge::op::Muls op;
  op.UpdateInputDesc("x", create_desc({4, 3, 4}, ge::DT_FLOAT16));

  float value = 3.0;
  op.SetAttr("value", value);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {4, 3, 4};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

// ----------------dynamic Muls-------------------
class dynamic_muls : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "dynamic_muls SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "dynamic_muls TearDown" << std::endl;
    }
};

TEST_F(dynamic_muls, muls_infershape_test_1) {
ge::op::Muls op;
op.UpdateInputDesc("x", create_desc_shape_range({3, 4, 5, 6, -1}, ge::DT_INT32, ge::FORMAT_ND, {3, 4, 5, 6, -1}, ge::FORMAT_ND, {{3,3},{4,4},{5,5},{6,6},{3,8}}));
auto ret = op.InferShapeAndType();
EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
auto output_desc = op.GetOutputDesc("y");
EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
std::vector<int64_t> expected_output_shape = {3, 4, 5, 6, -1};
EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{3,3},{4,4},{5,5},{6,6},{3,8}};
std::vector<std::pair<int64_t, int64_t>> output_shape_range;
output_desc.GetShapeRange(output_shape_range);
EXPECT_EQ(output_shape_range, expected_output_shape_range);
}

