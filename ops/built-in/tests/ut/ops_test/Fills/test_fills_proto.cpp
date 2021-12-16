#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "elewise_calculation_ops.h"

class fills : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "fills Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "fills Proto Test TearDown" << std::endl;
  }
};

TEST_F(fills, fills_infershape_diff_test){
  ge::op::Fills op;
  op.UpdateInputDesc("x", create_desc_shape_range({-1, -1},
                                                  ge::DT_FLOAT16,
                                                  ge::FORMAT_ND,
                                                  {1, 1},
                                                  ge::FORMAT_ND,
                                                  {{1, 1},{1, 1}}));
  
  float value = 1.0;
  op.SetAttr("value", value);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {-1, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);

  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {{1,1},{1,1}};
  EXPECT_EQ(output_shape_range, expected_shape_range);
}

