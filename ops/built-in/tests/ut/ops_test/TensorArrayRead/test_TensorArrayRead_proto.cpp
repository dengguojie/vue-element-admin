#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "data_flow_ops.h"
#include "inference_context.h"

#include <string>
#include <map>
#include <mutex>
#include "external/graph/resource_context.h"
#include "graph/ge_error_codes.h"
#include "graph/node.h"

namespace ge {
class ResourceContextMgr {
 public:
  ResourceContextMgr() = default;
  ~ResourceContextMgr() = default;
  /**
   * Given resource_key , return corresponding resource pointer
   * @param resource_key
   * @return orresponding resource pointer
   */
  ResourceContext *GetResourceContext(const std::string &resource_key);
  /**
   * Given resource_key , corresponding resource pointer, set resouce_context with new resource
   * @param resource_key
   * @param context
   * @return status
   */
  graphStatus SetResourceContext(const std::string &resource_key, ResourceContext *context);
  /**
   * Given resource_key , node reiled on this resource, mgr will keep the relation
   * @param resource_key
   * @param node
   * @return status
   */
  graphStatus RegisterNodeReliedOnResource(const std::string &resource_key, NodePtr &node);
  /**
   * Given resource_key , mgr find node reiled on this reousrce.
   * @param resource_key
   * @param read_nodes
   * @return status
   */
  std::unordered_set<NodePtr> &MutableNodesReliedOnResource(const std::string &resource_key);
  /**
   * Resource context need to be cleared when session finalize
   * @return status
   */
  graphStatus ClearContext();
  
 private:
  std::mutex ctx_mu_;
  std::map<std::string, std::unique_ptr<ResourceContext>> resource_keys_to_contexts_;
  std::map<std::string, std::unordered_set<NodePtr>> resource_keys_to_read_nodes_;
};

ResourceContext *ResourceContextMgr::GetResourceContext(const std::string &resource_key) {
  std::lock_guard<std::mutex> lk(ctx_mu_);
  auto iter = resource_keys_to_contexts_.find(resource_key);
  if (iter == resource_keys_to_contexts_.end()) {
    return nullptr;
  }
  return resource_keys_to_contexts_[resource_key].get();
}

graphStatus ResourceContextMgr::SetResourceContext(const std::string &resource_key, ResourceContext *context) {
  std::lock_guard<std::mutex> lk(ctx_mu_);
  resource_keys_to_contexts_[resource_key] = std::unique_ptr<ResourceContext>(context);
  return GRAPH_SUCCESS;
}

graphStatus ResourceContextMgr::RegisterNodeReliedOnResource(const std::string &resource_key, NodePtr &node) {
  std::lock_guard<std::mutex> lk(ctx_mu_);
  resource_keys_to_read_nodes_[resource_key].emplace(node);
  return GRAPH_SUCCESS;
}

std::unordered_set<NodePtr> &ResourceContextMgr::MutableNodesReliedOnResource(const std::string &resource_key) {
  std::lock_guard<std::mutex> lk(ctx_mu_);
  return resource_keys_to_read_nodes_[resource_key];
}

graphStatus ResourceContextMgr::ClearContext() {
  std::lock_guard<std::mutex> lk_resource(ctx_mu_);
  resource_keys_to_contexts_.clear();
  resource_keys_to_read_nodes_.clear();
  return GRAPH_SUCCESS;
}
}

class tensorArrayRead : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "tensorArrayRead Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "tensorArrayRead Proto Test TearDown" << std::endl;
  }
};

TEST_F(tensorArrayRead, tensorArrayRead_infershape_diff_test){
  ge::op::TensorArrayRead op;
  op.SetAttr("dtype", ge::DT_INT64);
  ge::InferenceContextPtr inferCtxPtr = std::move(ge::InferenceContext::Create());
  op.SetInferenceContext(inferCtxPtr);
  std::vector<std::vector<ge::ShapeAndType>> shapes_and_types;
  auto context = op.GetInferenceContext();
  context->SetOutputHandleShapesAndTypes(shapes_and_types);
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op.UpdateInputDesc("index", create_desc({}, ge::DT_INT32));
  op.UpdateInputDesc("flow_in", create_desc({}, ge::DT_FLOAT));
  
  auto ret = op.InferShapeAndType();
}

TEST_F(tensorArrayRead, tensorArrayRead_infershape_input0_rank_failed){
  ge::op::TensorArrayRead op;
  op.SetAttr("dtype", ge::DT_INT64);
  ge::InferenceContextPtr inferCtxPtr = std::move(ge::InferenceContext::Create());
  op.SetInferenceContext(inferCtxPtr);
  std::vector<std::vector<ge::ShapeAndType>> shapes_and_types;
  auto context = op.GetInferenceContext();
  context->SetOutputHandleShapesAndTypes(shapes_and_types);
  op.UpdateInputDesc("handle", create_desc({2}, ge::DT_RESOURCE));
  op.UpdateInputDesc("index", create_desc({}, ge::DT_INT32));
  op.UpdateInputDesc("flow_in", create_desc({}, ge::DT_FLOAT));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(tensorArrayRead, tensorArrayRead_infershape_input1_rank_failed){
  ge::op::TensorArrayRead op;
  op.SetAttr("dtype", ge::DT_INT64);
  ge::InferenceContextPtr inferCtxPtr = std::move(ge::InferenceContext::Create());
  op.SetInferenceContext(inferCtxPtr);
  std::vector<std::vector<ge::ShapeAndType>> shapes_and_types;
  auto context = op.GetInferenceContext();
  context->SetOutputHandleShapesAndTypes(shapes_and_types);
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op.UpdateInputDesc("index", create_desc({2}, ge::DT_INT32));
  op.UpdateInputDesc("flow_in", create_desc({}, ge::DT_FLOAT));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(tensorArrayRead, tensorArrayRead_infershape_input2_rank_failed){
  ge::op::TensorArrayRead op;
  op.SetAttr("dtype", ge::DT_INT64);
  ge::InferenceContextPtr inferCtxPtr = std::move(ge::InferenceContext::Create());
  op.SetInferenceContext(inferCtxPtr);
  std::vector<std::vector<ge::ShapeAndType>> shapes_and_types;
  auto context = op.GetInferenceContext();
  context->SetOutputHandleShapesAndTypes(shapes_and_types);
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op.UpdateInputDesc("index", create_desc({}, ge::DT_INT32));
  op.UpdateInputDesc("flow_in", create_desc({2}, ge::DT_FLOAT));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(tensorArrayRead, tensorArrayRead_infershape_context_null_failed){
  ge::op::TensorArrayRead op;
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op.UpdateInputDesc("index", create_desc({}, ge::DT_INT32));
  op.UpdateInputDesc("flow_in", create_desc({}, ge::DT_FLOAT));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(tensorArrayRead, tensorArrayRead_infershape_attr_dtype_failed){
  ge::op::TensorArrayRead op;
  ge::InferenceContextPtr inferCtxPtr = std::move(ge::InferenceContext::Create());
  op.SetInferenceContext(inferCtxPtr);
  std::vector<std::vector<ge::ShapeAndType>> shapes_and_types;
  auto context = op.GetInferenceContext();
  context->SetOutputHandleShapesAndTypes(shapes_and_types);
  op.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op.UpdateInputDesc("index", create_desc({}, ge::DT_INT32));
  op.UpdateInputDesc("flow_in", create_desc({}, ge::DT_FLOAT));
  
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(tensorArrayRead, tensorArrayRead_infershape_write_and_read_success){
  std::vector<std::string> marks = {std::string("TensorArray001")};
  std::vector<std::vector<ge::ShapeAndType>> shapes_and_types;

  ge::op::TensorArrayWrite op_tensor_array_write;
  op_tensor_array_write.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op_tensor_array_write.UpdateInputDesc("index", create_desc({}, ge::DT_INT32));
  op_tensor_array_write.UpdateInputDesc("value", create_desc({2,2}, ge::DT_FLOAT));
  op_tensor_array_write.UpdateInputDesc("flow_in", create_desc({}, ge::DT_FLOAT));
  auto value_desc = op_tensor_array_write.GetInputDesc("value");
  value_desc.SetShapeRange({{2, 2}, {2, 2}});
  op_tensor_array_write.UpdateInputDesc("value", value_desc);
  ge::ResourceContextMgr resource_mgr;
  ge::InferenceContextPtr inferCtxPtr1 = std::move(ge::InferenceContext::Create(&resource_mgr));
  inferCtxPtr1->SetOutputHandleShapesAndTypes(shapes_and_types);
  inferCtxPtr1->SetMarks(marks);
  op_tensor_array_write.SetInferenceContext(inferCtxPtr1);
  auto ret = op_tensor_array_write.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  ge::op::TensorArrayRead op_tensor_array_read;
  ge::InferenceContextPtr inferCtxPtr2 = std::move(ge::InferenceContext::Create(&resource_mgr));
  inferCtxPtr2->SetOutputHandleShapesAndTypes(shapes_and_types);
  inferCtxPtr2->SetMarks(marks);
  op_tensor_array_read.SetInferenceContext(inferCtxPtr2);
  
  op_tensor_array_read.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op_tensor_array_read.UpdateInputDesc("index", create_desc({}, ge::DT_INT32));
  op_tensor_array_read.UpdateInputDesc("flow_in", create_desc({}, ge::DT_FLOAT));
  op_tensor_array_read.SetAttr("dtype", ge::DT_INT64);
  ret = op_tensor_array_read.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(tensorArrayRead, tensorArrayRead_infershape_write_twice_success){
  std::vector<std::string> marks = {std::string("TensorArray002")};
  std::vector<std::vector<ge::ShapeAndType>> shapes_and_types;

  ge::op::TensorArrayWrite op_tensor_array_write;
  op_tensor_array_write.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op_tensor_array_write.UpdateInputDesc("index", create_desc({}, ge::DT_INT32));
  op_tensor_array_write.UpdateInputDesc("value", create_desc({2,2,4}, ge::DT_FLOAT));
  op_tensor_array_write.UpdateInputDesc("flow_in", create_desc({}, ge::DT_FLOAT));
  auto value_desc = op_tensor_array_write.GetInputDesc("value");
  value_desc.SetShapeRange({{2, 2}, {2, 2}, {4, 4}});
  op_tensor_array_write.UpdateInputDesc("value", value_desc);
  ge::ResourceContextMgr resource_mgr;
  ge::InferenceContextPtr inferCtxPtr1 = std::move(ge::InferenceContext::Create(&resource_mgr));
  inferCtxPtr1->SetOutputHandleShapesAndTypes(shapes_and_types);
  inferCtxPtr1->SetMarks(marks);
  op_tensor_array_write.SetInferenceContext(inferCtxPtr1);
  auto ret = op_tensor_array_write.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  ge::op::TensorArrayWrite op_tensor_array_write2;
  op_tensor_array_write2.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op_tensor_array_write2.UpdateInputDesc("index", create_desc({}, ge::DT_INT32));
  op_tensor_array_write2.UpdateInputDesc("value", create_desc({2,3,2}, ge::DT_FLOAT));
  op_tensor_array_write2.UpdateInputDesc("flow_in", create_desc({}, ge::DT_FLOAT));
  auto value_desc2 = op_tensor_array_write2.GetInputDesc("value");
  value_desc2.SetShapeRange({{2, 2}, {3, 3}, {2, 2}});
  op_tensor_array_write2.UpdateInputDesc("value", value_desc2);
  ge::InferenceContextPtr inferCtxPtr2 = std::move(ge::InferenceContext::Create(&resource_mgr));
  inferCtxPtr2->SetOutputHandleShapesAndTypes(shapes_and_types);
  inferCtxPtr2->SetMarks(marks);
  op_tensor_array_write2.SetInferenceContext(inferCtxPtr2);
  auto ret2 = op_tensor_array_write2.InferShapeAndType();
  EXPECT_EQ(ret2, ge::GRAPH_SUCCESS);
}

TEST_F(tensorArrayRead, tensorArrayRead_infershape_shared_shape_rank_unknown_success){
  std::vector<std::string> marks = {std::string("TensorArray003")};
  std::vector<std::vector<ge::ShapeAndType>> shapes_and_types;

  ge::op::TensorArrayWrite op_tensor_array_write;
  op_tensor_array_write.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op_tensor_array_write.UpdateInputDesc("index", create_desc({}, ge::DT_INT32));
  op_tensor_array_write.UpdateInputDesc("value", create_desc({-2}, ge::DT_FLOAT));
  op_tensor_array_write.UpdateInputDesc("flow_in", create_desc({}, ge::DT_FLOAT));
  auto value_desc = op_tensor_array_write.GetInputDesc("value");
  value_desc.SetShapeRange({{2, 2}, {2, 2}, {4, 4}});
  op_tensor_array_write.UpdateInputDesc("value", value_desc);
  ge::ResourceContextMgr resource_mgr;
  ge::InferenceContextPtr inferCtxPtr1 = std::move(ge::InferenceContext::Create(&resource_mgr));
  inferCtxPtr1->SetOutputHandleShapesAndTypes(shapes_and_types);
  inferCtxPtr1->SetMarks(marks);
  op_tensor_array_write.SetInferenceContext(inferCtxPtr1);
  auto ret = op_tensor_array_write.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  ge::op::TensorArrayWrite op_tensor_array_write2;
  op_tensor_array_write2.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op_tensor_array_write2.UpdateInputDesc("index", create_desc({}, ge::DT_INT32));
  op_tensor_array_write2.UpdateInputDesc("value", create_desc({2,3,2}, ge::DT_FLOAT));
  op_tensor_array_write2.UpdateInputDesc("flow_in", create_desc({}, ge::DT_FLOAT));
  auto value_desc2 = op_tensor_array_write2.GetInputDesc("value");
  value_desc2.SetShapeRange({{2, 2}, {3, 3}, {2, 2}});
  op_tensor_array_write2.UpdateInputDesc("value", value_desc2);
  ge::InferenceContextPtr inferCtxPtr2 = std::move(ge::InferenceContext::Create(&resource_mgr));
  inferCtxPtr2->SetOutputHandleShapesAndTypes(shapes_and_types);
  inferCtxPtr2->SetMarks(marks);
  op_tensor_array_write2.SetInferenceContext(inferCtxPtr2);
  auto ret2 = op_tensor_array_write2.InferShapeAndType();
  EXPECT_EQ(ret2, ge::GRAPH_SUCCESS);
}

TEST_F(tensorArrayRead, tensorArrayRead_infershape_value_shape_rank_unknown_success){
  std::vector<std::string> marks = {std::string("TensorArray004")};
  std::vector<std::vector<ge::ShapeAndType>> shapes_and_types;

  ge::op::TensorArrayWrite op_tensor_array_write;
  op_tensor_array_write.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op_tensor_array_write.UpdateInputDesc("index", create_desc({}, ge::DT_INT32));
  op_tensor_array_write.UpdateInputDesc("value", create_desc({2,2,4}, ge::DT_FLOAT));
  op_tensor_array_write.UpdateInputDesc("flow_in", create_desc({}, ge::DT_FLOAT));
  auto value_desc = op_tensor_array_write.GetInputDesc("value");
  value_desc.SetShapeRange({{2, 2}, {2, 2}, {4, 4}});
  op_tensor_array_write.UpdateInputDesc("value", value_desc);
  ge::ResourceContextMgr resource_mgr;
  ge::InferenceContextPtr inferCtxPtr1 = std::move(ge::InferenceContext::Create(&resource_mgr));
  inferCtxPtr1->SetOutputHandleShapesAndTypes(shapes_and_types);
  inferCtxPtr1->SetMarks(marks);
  op_tensor_array_write.SetInferenceContext(inferCtxPtr1);
  auto ret = op_tensor_array_write.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  ge::op::TensorArrayWrite op_tensor_array_write2;
  op_tensor_array_write2.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op_tensor_array_write2.UpdateInputDesc("index", create_desc({}, ge::DT_INT32));
  op_tensor_array_write2.UpdateInputDesc("value", create_desc({-2}, ge::DT_FLOAT));
  op_tensor_array_write2.UpdateInputDesc("flow_in", create_desc({}, ge::DT_FLOAT));
  auto value_desc2 = op_tensor_array_write2.GetInputDesc("value");
  value_desc2.SetShapeRange({{2, 2}, {3, 3}, {2, 2}});
  op_tensor_array_write2.UpdateInputDesc("value", value_desc2);
  ge::InferenceContextPtr inferCtxPtr2 = std::move(ge::InferenceContext::Create(&resource_mgr));
  inferCtxPtr2->SetOutputHandleShapesAndTypes(shapes_and_types);
  inferCtxPtr2->SetMarks(marks);
  op_tensor_array_write2.SetInferenceContext(inferCtxPtr2);
  auto ret2 = op_tensor_array_write2.InferShapeAndType();
  EXPECT_EQ(ret2, ge::GRAPH_SUCCESS);
}

TEST_F(tensorArrayRead, tensorArrayRead_infershape_shared_shape_num_not_equal_failed){
  std::vector<std::string> marks = {std::string("TensorArray005")};
  std::vector<std::vector<ge::ShapeAndType>> shapes_and_types;

  ge::op::TensorArrayWrite op_tensor_array_write;
  op_tensor_array_write.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op_tensor_array_write.UpdateInputDesc("index", create_desc({}, ge::DT_INT32));
  op_tensor_array_write.UpdateInputDesc("value", create_desc({2,2,4}, ge::DT_FLOAT));
  op_tensor_array_write.UpdateInputDesc("flow_in", create_desc({}, ge::DT_FLOAT));
  auto value_desc = op_tensor_array_write.GetInputDesc("value");
  value_desc.SetShapeRange({{2, 2}, {2, 2}});
  op_tensor_array_write.UpdateInputDesc("value", value_desc);
  ge::ResourceContextMgr resource_mgr;
  ge::InferenceContextPtr inferCtxPtr1 = std::move(ge::InferenceContext::Create(&resource_mgr));
  inferCtxPtr1->SetOutputHandleShapesAndTypes(shapes_and_types);
  inferCtxPtr1->SetMarks(marks);
  op_tensor_array_write.SetInferenceContext(inferCtxPtr1);
  auto ret = op_tensor_array_write.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  ge::op::TensorArrayWrite op_tensor_array_write2;
  op_tensor_array_write2.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op_tensor_array_write2.UpdateInputDesc("index", create_desc({}, ge::DT_INT32));
  op_tensor_array_write2.UpdateInputDesc("value", create_desc({2,3,2}, ge::DT_FLOAT));
  op_tensor_array_write2.UpdateInputDesc("flow_in", create_desc({}, ge::DT_FLOAT));
  auto value_desc2 = op_tensor_array_write2.GetInputDesc("value");
  value_desc2.SetShapeRange({{2, 2}, {3, 3}, {2, 2}});
  op_tensor_array_write2.UpdateInputDesc("value", value_desc2);
  ge::InferenceContextPtr inferCtxPtr2 = std::move(ge::InferenceContext::Create(&resource_mgr));
  inferCtxPtr2->SetOutputHandleShapesAndTypes(shapes_and_types);
  inferCtxPtr2->SetMarks(marks);
  op_tensor_array_write2.SetInferenceContext(inferCtxPtr2);
  auto ret2 = op_tensor_array_write2.InferShapeAndType();
  EXPECT_EQ(ret2, ge::GRAPH_FAILED);
}

TEST_F(tensorArrayRead, tensorArrayRead_infershape_value_shape_num_not_equal_failed){
  std::vector<std::string> marks = {std::string("TensorArray006")};
  std::vector<std::vector<ge::ShapeAndType>> shapes_and_types;

  ge::op::TensorArrayWrite op_tensor_array_write;
  op_tensor_array_write.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op_tensor_array_write.UpdateInputDesc("index", create_desc({}, ge::DT_INT32));
  op_tensor_array_write.UpdateInputDesc("value", create_desc({2,2,4}, ge::DT_FLOAT));
  op_tensor_array_write.UpdateInputDesc("flow_in", create_desc({}, ge::DT_FLOAT));
  auto value_desc = op_tensor_array_write.GetInputDesc("value");
  value_desc.SetShapeRange({{2, 2}, {2, 2}, {4, 4}});
  op_tensor_array_write.UpdateInputDesc("value", value_desc);
  ge::ResourceContextMgr resource_mgr;
  ge::InferenceContextPtr inferCtxPtr1 = std::move(ge::InferenceContext::Create(&resource_mgr));
  inferCtxPtr1->SetOutputHandleShapesAndTypes(shapes_and_types);
  inferCtxPtr1->SetMarks(marks);
  op_tensor_array_write.SetInferenceContext(inferCtxPtr1);
  auto ret = op_tensor_array_write.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  ge::op::TensorArrayWrite op_tensor_array_write2;
  op_tensor_array_write2.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op_tensor_array_write2.UpdateInputDesc("index", create_desc({}, ge::DT_INT32));
  op_tensor_array_write2.UpdateInputDesc("value", create_desc({2,3,2}, ge::DT_FLOAT));
  op_tensor_array_write2.UpdateInputDesc("flow_in", create_desc({}, ge::DT_FLOAT));
  auto value_desc2 = op_tensor_array_write2.GetInputDesc("value");
  value_desc2.SetShapeRange({{2, 2}, {3, 3}});
  op_tensor_array_write2.UpdateInputDesc("value", value_desc2);
  ge::InferenceContextPtr inferCtxPtr2 = std::move(ge::InferenceContext::Create(&resource_mgr));
  inferCtxPtr2->SetOutputHandleShapesAndTypes(shapes_and_types);
  inferCtxPtr2->SetMarks(marks);
  op_tensor_array_write2.SetInferenceContext(inferCtxPtr2);
  auto ret2 = op_tensor_array_write2.InferShapeAndType();
  EXPECT_EQ(ret2, ge::GRAPH_FAILED);
}

TEST_F(tensorArrayRead, tensorArrayRead_infershape_shape_empty_failed){
  std::vector<std::string> marks = {std::string("TensorArray007")};
  std::vector<std::vector<ge::ShapeAndType>> shapes_and_types;

  ge::op::TensorArrayRead op_tensor_array_read;
  ge::ResourceContextMgr resource_mgr;
  ge::InferenceContextPtr inferCtxPtr2 = std::move(ge::InferenceContext::Create(&resource_mgr));
  inferCtxPtr2->SetOutputHandleShapesAndTypes(shapes_and_types);
  inferCtxPtr2->SetMarks(marks);
  ge::AicpuResourceContext *aicpu_resource_context = new ge::AicpuResourceContext();
  aicpu_resource_context->shape_and_range_.clear();
  inferCtxPtr2->SetResourceContext(marks[0].c_str(), aicpu_resource_context);

  op_tensor_array_read.SetInferenceContext(inferCtxPtr2);
  
  op_tensor_array_read.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op_tensor_array_read.UpdateInputDesc("index", create_desc({}, ge::DT_INT32));
  op_tensor_array_read.UpdateInputDesc("flow_in", create_desc({}, ge::DT_FLOAT));
  op_tensor_array_read.SetAttr("dtype", ge::DT_INT64);
  auto ret = op_tensor_array_read.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(tensorArrayRead, tensorArrayRead_infershape_write_and_gather_success){
  std::vector<std::string> marks = {std::string("TensorArray008")};
  std::vector<std::vector<ge::ShapeAndType>> shapes_and_types;

  ge::op::TensorArrayWrite op_tensor_array_write;
  op_tensor_array_write.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op_tensor_array_write.UpdateInputDesc("index", create_desc({}, ge::DT_INT32));
  op_tensor_array_write.UpdateInputDesc("value", create_desc({2,2}, ge::DT_FLOAT));
  op_tensor_array_write.UpdateInputDesc("flow_in", create_desc({}, ge::DT_FLOAT));
  auto value_desc = op_tensor_array_write.GetInputDesc("value");
  value_desc.SetShapeRange({{2, 2}, {2, 2}});
  op_tensor_array_write.UpdateInputDesc("value", value_desc);
  ge::ResourceContextMgr resource_mgr;
  ge::InferenceContextPtr inferCtxPtr1 = std::move(ge::InferenceContext::Create(&resource_mgr));
  inferCtxPtr1->SetOutputHandleShapesAndTypes(shapes_and_types);
  inferCtxPtr1->SetMarks(marks);
  op_tensor_array_write.SetInferenceContext(inferCtxPtr1);
  auto ret = op_tensor_array_write.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  ge::op::TensorArrayGather op_tensor_array_gather;
  op_tensor_array_gather.UpdateInputDesc("handle", create_desc({}, ge::DT_RESOURCE));
  op_tensor_array_gather.UpdateInputDesc("indices", create_desc({4}, ge::DT_INT32));
  op_tensor_array_gather.UpdateInputDesc("flow_in", create_desc({}, ge::DT_FLOAT));
  op_tensor_array_gather.SetAttr("element_shape", {-2});
  op_tensor_array_gather.SetAttr("dtype", ge::DT_INT64);
  ge::InferenceContextPtr inferCtxPtr2 = std::move(ge::InferenceContext::Create(&resource_mgr));
  inferCtxPtr2->SetOutputHandleShapesAndTypes(shapes_and_types);
  inferCtxPtr2->SetMarks(marks);
  op_tensor_array_gather.SetInferenceContext(inferCtxPtr2);

  ret = op_tensor_array_gather.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}



