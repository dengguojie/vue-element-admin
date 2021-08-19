#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "data_flow_ops.h"

class tensorArrayConcat : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "tensorArrayConcat Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "tensorArrayConcat Proto Test TearDown" << std::endl;
  }
};

TEST_F(tensorArrayConcat, tensorArrayConcat_inference_context_null_failed){
  ge::op::TensorArrayConcat op_tensor_array_concat;

  op_tensor_array_concat.SetAttr("dtype", ge::DT_INT64);
  op_tensor_array_concat.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op_tensor_array_concat.UpdateInputDesc("flow_in", create_desc({}, ge::DT_FLOAT));

  auto ret = op_tensor_array_concat.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(tensorArrayConcat, tensorArrayConcat_infershape_input0_rank_failed){
  ge::op::TensorArrayConcat op;
  op.SetAttr("dtype", ge::DT_INT64);
  op.UpdateInputDesc("handle", create_desc({2}, ge::DT_RESOURCE));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(tensorArrayConcat, tensorArrayConcat_infershape_input1_rank_failed){
  ge::op::TensorArrayConcat op;
  op.SetAttr("dtype", ge::DT_INT64);
  op.UpdateInputDesc("flow_in", create_desc({2}, ge::DT_FLOAT));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(tensorArrayConcat, tensorArrayConcat_infershape_attr_dtype_failed){
  ge::op::TensorArrayConcat op;
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op.UpdateInputDesc("flow_in", create_desc({}, ge::DT_FLOAT));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}