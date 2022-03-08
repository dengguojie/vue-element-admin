#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "pad_ops.h"

class fill : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "fill Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "fill Proto Test TearDown" << std::endl;
  }
};

TEST_F(fill, fill_infershape_diff_test){
  ge::op::Fill op;
  op.UpdateInputDesc("dims", create_desc_shape_range({2, 2},
                                                  ge::DT_FLOAT16,
                                                  ge::FORMAT_ND,
                                                  {2, 2},
                                                  ge::FORMAT_ND,
                                                  {{2, 2},{2, 2}}));
  op.UpdateInputDesc("value", create_desc_shape_range({1},
                                                  ge::DT_FLOAT16,
                                                  ge::FORMAT_ND,
                                                  {1},
                                                  ge::FORMAT_ND,
                                                  {{1, 1}}));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);

  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> expected_shape_range = {{0,-1},{0,-1}};
  EXPECT_EQ(output_shape_range, expected_shape_range);
}
