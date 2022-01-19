#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "data_flow_ops.h"
#include "inference_context.h"
#include "../util/common_shape_fns.h"
#include "op_proto_test_common.h"

class tensorArrayWrite : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "tensorArrayWrite Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "tensorArrayWrite Proto Test TearDown" << std::endl;
  }
};

TEST_F(tensorArrayWrite, tensorArrayWrite_infershape_input0_rank_fail){
  ge::op::TensorArrayWrite op;
  op.UpdateInputDesc("handle", create_desc({2}, ge::DT_RESOURCE));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(tensorArrayWrite, tensorArrayWrite_infershape_input1_rank_fail){
  ge::op::TensorArrayWrite op;
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op.UpdateInputDesc("index", create_desc({2}, ge::DT_INT32));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(tensorArrayWrite, tensorArrayWrite_infershape_input3_rank_fail){
  ge::op::TensorArrayWrite op;
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op.UpdateInputDesc("index", create_desc({}, ge::DT_INT32));
  op.UpdateInputDesc("flow_in", create_desc({4}, ge::DT_FLOAT));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(tensorArrayWrite, tensorArrayWrite_infershape_context_null_fail){
  ge::op::TensorArrayWrite op;
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op.UpdateInputDesc("index", create_desc({}, ge::DT_INT32));
  op.UpdateInputDesc("flow_in", create_desc({}, ge::DT_FLOAT));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(tensorArrayWrite, tensorArrayWrite_infershape_context_null){
  ge::op::TensorArrayWrite op;
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op.UpdateInputDesc("indices", create_desc({4}, ge::DT_INT32));
  op.UpdateInputDesc("flow_in", create_desc({}, ge::DT_FLOAT));
  ge::ResourceContextMgr resource_mgr;
  ge::InferenceContextPtr inferCtxPtr = std::move(ge::InferenceContext::Create(&resource_mgr));
  std::vector<std::vector<ge::ShapeAndType>> shapes_and_types;
  inferCtxPtr->SetOutputHandleShapesAndTypes(shapes_and_types);
  std::vector<std::string> marks = {std::string("tensorArray001")};
  inferCtxPtr->SetMarks(marks);
  ge::AicpuResourceContext *aicpu_resource_context = new ge::AicpuResourceContext();
  aicpu_resource_context->shape_and_range_.clear();
  inferCtxPtr->SetResourceContext(marks[0].c_str(), aicpu_resource_context);
  op.SetInferenceContext(inferCtxPtr);
  op.SetAttr("element_shape", {4});
  op.SetAttr("dtype", ge::DT_INT64);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}