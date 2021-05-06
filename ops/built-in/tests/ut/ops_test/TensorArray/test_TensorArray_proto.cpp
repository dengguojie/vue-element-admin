#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "data_flow_ops.h"
#include "inference_context.h"

class tensorArray : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "tensorArray Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "tensorArray Proto Test TearDown" << std::endl;
  }
};

TEST_F(tensorArray, tensorArray_infershape_success){
  ge::op::TensorArray op;
  op.UpdateInputDesc("size", create_desc({}, ge::DT_INT32));
  op.SetAttr("dtype", ge::DT_INT64);
  op.SetAttr("identical_element_shapes", true);
  ge::InferenceContextPtr inferCtxPtr = std::move(ge::InferenceContext::Create());
  op.SetInferenceContext(inferCtxPtr);
  std::vector<std::vector<ge::ShapeAndType>> shapes_and_types;
  auto context = op.GetInferenceContext();
  context->SetOutputHandleShapesAndTypes(shapes_and_types);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(tensorArray, tensorArray_infershape_input0_rank_failed){
  ge::op::TensorArray op;
  op.UpdateInputDesc("size", create_desc({2}, ge::DT_INT32));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(tensorArray, tensorArray_infershape_get_attr_identical_element_shapes_failed){
  ge::op::TensorArray op;
  op.UpdateInputDesc("size", create_desc({}, ge::DT_INT32));
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(tensorArray, tensorArray_infershape_get_attr_dtype_failed){
  ge::op::TensorArray op;
  op.UpdateInputDesc("size", create_desc({}, ge::DT_INT32));
  op.SetAttr("identical_element_shapes", true);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(tensorArray, tensorArray_infershape_context_null_failed){
  ge::op::TensorArray op;
  op.UpdateInputDesc("size", create_desc({}, ge::DT_INT32));
  op.SetAttr("dtype", ge::DT_INT64);
  op.SetAttr("identical_element_shapes", true);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}