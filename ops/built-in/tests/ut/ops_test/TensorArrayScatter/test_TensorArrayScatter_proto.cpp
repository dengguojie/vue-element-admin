#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "data_flow_ops.h"
#include "inference_context.h"

class tensorArrayScatter : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "tensorArrayScatter Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "tensorArrayScatter Proto Test TearDown" << std::endl;
  }
};

TEST_F(tensorArrayScatter, tensorArrayScatter_infershape_diff_test){
  ge::op::TensorArrayScatter op;
  ge::InferenceContextPtr inferCtxPtr = std::move(ge::InferenceContext::Create());
  op.SetInferenceContext(inferCtxPtr);
  std::vector<std::vector<ge::ShapeAndType>> shapes_and_types;
  auto context = op.GetInferenceContext();
  context->SetOutputHandleShapesAndTypes(shapes_and_types);
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op.UpdateInputDesc("indices", create_desc({4}, ge::DT_INT32));
  op.UpdateInputDesc("value", create_desc({4}, ge::DT_INT32));
  op.UpdateInputDesc("flow_in", create_desc({}, ge::DT_FLOAT));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}