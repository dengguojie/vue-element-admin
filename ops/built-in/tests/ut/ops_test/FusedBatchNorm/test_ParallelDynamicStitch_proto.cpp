#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "data_flow_ops.h"

class ParallelDynamicStitch : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ParallelDynamicStitch SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ParallelDynamicStitch TearDown" << std::endl;
  }
};

TEST_F(ParallelDynamicStitch, ParallelDynamicStitch_infer_shape_1) {
  ge::op::ParallelDynamicStitch op; 
  op.SetAttr("N", 2);  
  op.UpdateInputDesc("indices0",  create_desc({3}, ge::DT_INT32));
  op.UpdateInputDesc("indices1",  create_desc({2}, ge::DT_INT64));                                                                                                                                                                               
  op.UpdateInputDesc("x0", create_desc({3}, ge::DT_FLOAT));
  op.UpdateInputDesc("x1", create_desc({2},ge::DT_FLOAT));

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
  std::vector<int64_t> expected_output_shape = {1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

 