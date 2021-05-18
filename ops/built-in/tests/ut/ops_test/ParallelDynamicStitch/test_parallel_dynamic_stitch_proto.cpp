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

TEST_F(ParallelDynamicStitch, ParallelDynamicStitch_infershape_test_1) {
  ge::op::ParallelDynamicStitch op;
  auto tensor_desc = create_desc_with_ori({-1, -1},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {-1, -1},
                                             ge::FORMAT_ND);
  op.create_dynamic_input_indices(2);
  op.create_dynamic_input_x(2);
  op.UpdateDynamicInputDesc("indices", 0, tensor_desc);
  op.UpdateDynamicInputDesc("indices", 1, tensor_desc);
  op.UpdateDynamicInputDesc("x", 0, tensor_desc);
  op.UpdateDynamicInputDesc("x", 1, tensor_desc);
  op.SetAttr("N",1);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(ParallelDynamicStitch, ParallelDynamicStitch_infershape_test_2) {
  ge::op::ParallelDynamicStitch op;
  auto tensor_desc = create_desc_with_ori({-1, -1},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {-1, -1},
                                             ge::FORMAT_ND);
  op.create_dynamic_input_indices(2);
  op.create_dynamic_input_x(2);
  op.UpdateDynamicInputDesc("indices", 0, tensor_desc);
  op.UpdateDynamicInputDesc("indices", 1, tensor_desc);
  op.UpdateDynamicInputDesc("x", 0, tensor_desc);
  op.UpdateDynamicInputDesc("x", 1, tensor_desc);
  op.SetAttr("N",0);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

 TEST_F(ParallelDynamicStitch, ParallelDynamicStitch_infershape_test_3) {
  ge::op::ParallelDynamicStitch op;
  auto tensor_desc1 = create_desc_with_ori({2, 3},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {2, 3},
                                             ge::FORMAT_ND);
  auto tensor_desc2 = create_desc_with_ori({2},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {2},
                                             ge::FORMAT_ND);
  op.create_dynamic_input_indices(2);
  op.create_dynamic_input_x(1);
  op.UpdateDynamicInputDesc("indices", 0, tensor_desc1);
  op.UpdateDynamicInputDesc("indices", 1, tensor_desc2);
  op.UpdateDynamicInputDesc("x", 0, tensor_desc2);
  op.SetAttr("N",1);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

 TEST_F(ParallelDynamicStitch, ParallelDynamicStitch_infershape_test_4) {
  ge::op::ParallelDynamicStitch op;
  auto tensor_desc1 = create_desc_with_ori({2, 3},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {2, 3},
                                             ge::FORMAT_ND);
  auto tensor_desc2 = create_desc_with_ori({1},
                                             ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {1},
                                             ge::FORMAT_ND);
  op.create_dynamic_input_indices(2);
  op.create_dynamic_input_x(2);
  op.UpdateDynamicInputDesc("indices", 0, tensor_desc1);
  op.UpdateDynamicInputDesc("indices", 1, tensor_desc2);
  op.UpdateDynamicInputDesc("x", 0, tensor_desc2);
  op.UpdateDynamicInputDesc("x", 1, tensor_desc2);
  op.SetAttr("N",1);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(ParallelDynamicStitch, ParallelDynamicStitch_test_1) {
  ge::op::ParallelDynamicStitch op;
  ge::TensorDesc tensor_desc_indices(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("indices", tensor_desc_indices);
  ge::TensorDesc tensor_desc_x(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("x", tensor_desc_x);
  op.SetAttr("N",1);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}



