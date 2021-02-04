#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "elewise_calculation_ops.h"

class OnesLike : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ones_like SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ones_like TearDown" << std::endl;
  }
};

TEST_F(OnesLike, ones_like_infershape_test){
  ge::op::OnesLike op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2,10},{3,10},{4,10}};
  auto tensor_desc_x = create_desc_shape_range({-1,-1,-1},
											  ge::DT_FLOAT16, ge::FORMAT_ND,
											  {2,3,4},
											  ge::FORMAT_ND,shape_range);
  
  op.UpdateInputDesc("x",tensor_desc_x);
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {-1, -1, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
  std::vector<std::pair<int64_t,int64_t>> excepted_shape_range = {
	  {2,10},{3,10},{4,10}
  };
  EXPECT_EQ(output_shape_range, excepted_shape_range); 
}