#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "op_proto_test_util.h"
#include "data_flow_ops.h"
#include "inference_context.h"

#include "op_proto_test_common.h"


class unstage : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "unstage Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "unstage Proto Test TearDown" << std::endl;
  }
};

class stage : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "unstage Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "unstage Proto Test TearDown" << std::endl;
  }
};

class MapStage : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "StagePeek SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "StagePeek TearDown" << std::endl;
  }
};

class MapUnstage : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "StagePeek SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "StagePeek TearDown" << std::endl;
  }
};

class MapUnstageNoKey : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "StagePeek SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "StagePeek TearDown" << std::endl;
  }
};

class OrderedMapStage : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "StagePeek SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "StagePeek TearDown" << std::endl;
  }
};

TEST_F(unstage, unstage_infershape_success){
  ge::op::Unstage op;
  op.create_dynamic_output_y(1);
  std::vector<ge::DataType> dtypes = {ge::DT_FLOAT16};
  op.SetAttr("dtypes", dtypes);
  op.SetAttr("container", "container");
  op.SetAttr("shared_name", "shared_name");
  ge::ResourceContextMgr resource_mgr;
  std::vector<std::vector<ge::ShapeAndType>> shapes_and_types;
  ge::InferenceContextPtr inferCtx = std::move(ge::InferenceContext::Create(&resource_mgr));
  inferCtx->SetOutputHandleShapesAndTypes(shapes_and_types);
  op.SetInferenceContext(inferCtx);

  auto ret = op.InferShapeAndType();
  
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(stage, stage_infershape_success){
  ge::op::Stage op;
  op.SetAttr("container", "container");
  op.SetAttr("shared_name", "shared_name");
  const int32_t size = 2;
  op.create_dynamic_input_values(size);
  for (int i = 0; i < size; ++i) {
    op.UpdateDynamicInputDesc("values", i, create_desc({2, 3}, ge::DT_INT64));
  }

  ge::ResourceContextMgr resource_mgr;
  ge::InferenceContextPtr inferCtx = std::move(ge::InferenceContext::Create(&resource_mgr));
  op.SetInferenceContext(inferCtx);

  auto ret = op.InferShapeAndType();
  
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}


TEST_F(stage, stage_unstage_infershape_success){
  ge::op::Stage op;
  op.SetAttr("container", "container");
  op.SetAttr("shared_name", "shared_name");
  const int32_t size = 1;
  op.create_dynamic_input_values(size);
  for (int i = 0; i < size; ++i) {
    op.UpdateDynamicInputDesc("values", i, create_desc({2, 3}, ge::DT_INT64));
  }

  ge::ResourceContextMgr resource_mgr;
  ge::InferenceContextPtr inferCtx = std::move(ge::InferenceContext::Create(&resource_mgr));
  op.SetInferenceContext(inferCtx);

  auto ret = op.InferShapeAndType();
  
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);


  ge::op::Unstage UnstageOp;
  UnstageOp.create_dynamic_output_y(1);
  std::vector<ge::DataType> dtypes = {ge::DT_FLOAT16};
  UnstageOp.SetAttr("dtypes", dtypes);
  UnstageOp.SetAttr("container", "container");
  UnstageOp.SetAttr("shared_name", "shared_name");
  std::vector<std::vector<ge::ShapeAndType>> shapes_and_types;
  ge::InferenceContextPtr UnstageOpinferCtx = std::move(ge::InferenceContext::Create(&resource_mgr));
  UnstageOpinferCtx->SetOutputHandleShapesAndTypes(shapes_and_types);
  UnstageOp.SetInferenceContext(UnstageOpinferCtx);

  auto result = UnstageOp.InferShapeAndType();
  
  EXPECT_EQ(result, ge::GRAPH_SUCCESS);
}

TEST_F(MapStage, mapstage_infershape_success){
  ge::op::MapStage op;
  op.SetAttr("container", "container");
  op.SetAttr("shared_name", "shared_name");
  const int32_t size = 2;
  op.create_dynamic_input_values(size);
  for (int i = 0; i < size; ++i) {
    op.UpdateDynamicInputDesc("values", i, create_desc({2, 3}, ge::DT_INT64));
  }
  ge::ResourceContextMgr resource_mgr;
  ge::InferenceContextPtr inferCtx = std::move(ge::InferenceContext::Create(&resource_mgr));
  op.SetInferenceContext(inferCtx);
  auto ret = op.InferShapeAndType();  
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(MapStage, mapstage_mapunstage_infershape_success){
  ge::op::MapStage op;
  op.SetAttr("container", "container");
  op.SetAttr("shared_name", "shared_name");
  const int32_t size = 1;
  op.create_dynamic_input_values(size);
  for (int i = 0; i < size; ++i) {
    op.UpdateDynamicInputDesc("values", i, create_desc({2, 3}, ge::DT_INT64));
  }

  ge::ResourceContextMgr resource_mgr;
  ge::InferenceContextPtr inferCtx = std::move(ge::InferenceContext::Create(&resource_mgr));
  op.SetInferenceContext(inferCtx);
  auto ret = op.InferShapeAndType();  
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  ge::op::MapUnstage UnstageOp;
  UnstageOp.create_dynamic_output_values(2);
  std::vector<ge::DataType> dtypes = {ge::DT_FLOAT16};
  UnstageOp.SetAttr("dtypes", dtypes);
  UnstageOp.SetAttr("container", "container");
  UnstageOp.SetAttr("shared_name", "shared_name");
  std::vector<std::vector<ge::ShapeAndType>> shapes_and_types;
  ge::InferenceContextPtr UnstageOpinferCtx = std::move(ge::InferenceContext::Create(&resource_mgr));
  UnstageOpinferCtx->SetOutputHandleShapesAndTypes(shapes_and_types);
  UnstageOp.SetInferenceContext(UnstageOpinferCtx);
  auto result = UnstageOp.InferShapeAndType();
  
  EXPECT_EQ(result, ge::GRAPH_SUCCESS);
}


TEST_F(MapStage, mapstage_mapunstageNoKey_infershape_success){
  ge::op::MapStage op;
  op.SetAttr("container", "container");
  op.SetAttr("shared_name", "shared_name");
  const int32_t size = 1;
  op.create_dynamic_input_values(size);
  for (int i = 0; i < size; ++i) {
    op.UpdateDynamicInputDesc("values", i, create_desc({2, 3}, ge::DT_INT64));
  }

  ge::ResourceContextMgr resource_mgr;
  ge::InferenceContextPtr inferCtx = std::move(ge::InferenceContext::Create(&resource_mgr));
  op.SetInferenceContext(inferCtx);
  auto ret = op.InferShapeAndType();  
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  ge::op::MapUnstageNoKey UnstageOp;
  UnstageOp.create_dynamic_output_values(2);
  std::vector<ge::DataType> dtypes = {ge::DT_FLOAT16};
  UnstageOp.SetAttr("dtypes", dtypes);
  UnstageOp.SetAttr("container", "container");
  UnstageOp.SetAttr("shared_name", "shared_name");
  std::vector<std::vector<ge::ShapeAndType>> shapes_and_types;
  ge::InferenceContextPtr UnstageOpinferCtx = std::move(ge::InferenceContext::Create(&resource_mgr));
  UnstageOpinferCtx->SetOutputHandleShapesAndTypes(shapes_and_types);
  UnstageOp.SetInferenceContext(UnstageOpinferCtx);
  auto result = UnstageOp.InferShapeAndType();
  
  EXPECT_EQ(result, ge::GRAPH_SUCCESS);
}

TEST_F(OrderedMapStage, OrderedMapStage_infershape_success){
  ge::op::OrderedMapStage op;
  op.SetAttr("container", "container");
  op.SetAttr("shared_name", "shared_name");
  const int32_t size = 4;
  op.create_dynamic_input_values(size);
  for (int i = 0; i < size; ++i) {
    op.UpdateDynamicInputDesc("values", i, create_desc({2, 3}, ge::DT_INT64));
  }

  ge::ResourceContextMgr resource_mgr;
  ge::InferenceContextPtr inferCtx = std::move(ge::InferenceContext::Create(&resource_mgr));
  op.SetInferenceContext(inferCtx);
  auto ret = op.InferShapeAndType();  
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

