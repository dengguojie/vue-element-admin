#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "data_flow_ops.h"
#include "array_ops.h"
#include "op_proto_test_util.h"

using namespace ge;
using namespace op;

class barrier_infer_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "barrier_infer_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "barrier_infer_test TearDown" << std::endl;
  }
};

TEST_F(barrier_infer_test, barrier_infer_test_1) {
  //new op
  ge::op::Barrier op;
  // set attr info
  std::vector<ge::DataType> component_types{ge::DT_INT64, ge::DT_INT32};
  op.SetAttr("component_types", component_types);

  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(barrier_infer_test, barrier_infer_test_2) {
  //new op
  ge::op::Barrier op;
  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(barrier_infer_test, barrier_infer_test_3) {
  //new op
  ge::op::Barrier op;
  // set attr info
  std::vector<ge::DataType> component_types;
  op.SetAttr("component_types", component_types);

  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// merge failed testcase
TEST_F(barrier_infer_test, barrier_insert_many_infer_test_1) {
  //new op
  ge::op::BarrierInsertMany op;
  // set input info
  ge::TensorDesc tensor_desc_handle(ge::Shape({2}), ge::FORMAT_ND, ge::DT_STRING_REF);
  op.UpdateInputDesc("handle", tensor_desc_handle);
  ge::TensorDesc tensor_desc_keys(ge::Shape({1}), ge::FORMAT_ND, ge::DT_STRING);
  op.UpdateInputDesc("keys", tensor_desc_keys);
  ge::TensorDesc tensor_desc_values(ge::Shape({3, 4}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("values", tensor_desc_values);

  // set attr info
  op.SetAttr("component_index", ge::DT_INT32);

  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// handle rank failed testcase
TEST_F(barrier_infer_test, barrier_insert_many_infer_test_2) {
  //new op
  ge::op::BarrierInsertMany op;
  // set input info
  ge::TensorDesc tensor_desc_handle(ge::Shape({2, 3}), ge::FORMAT_ND, ge::DT_STRING_REF);
  op.UpdateInputDesc("handle", tensor_desc_handle);
  ge::TensorDesc tensor_desc_keys(ge::Shape({1}), ge::FORMAT_ND, ge::DT_STRING);
  op.UpdateInputDesc("keys", tensor_desc_keys);
  ge::TensorDesc tensor_desc_values(ge::Shape({3, 4}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("values", tensor_desc_values);

  // set attr info
  op.SetAttr("component_index", ge::DT_INT32);

  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// handle dim[0] failed testcase
TEST_F(barrier_infer_test, barrier_insert_many_infer_test_3) {
  //new op
  ge::op::BarrierInsertMany op;
  // set input info
  ge::TensorDesc tensor_desc_handle(ge::Shape({1}), ge::FORMAT_ND, ge::DT_STRING_REF);
  op.UpdateInputDesc("handle", tensor_desc_handle);
  ge::TensorDesc tensor_desc_keys(ge::Shape({1}), ge::FORMAT_ND, ge::DT_STRING);
  op.UpdateInputDesc("keys", tensor_desc_keys);
  ge::TensorDesc tensor_desc_values(ge::Shape({3, 4}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("values", tensor_desc_values);

  // set attr info
  op.SetAttr("component_index", ge::DT_INT32);

  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// keys rank failed testcase
TEST_F(barrier_infer_test, barrier_insert_many_infer_test_4) {
  //new op
  ge::op::BarrierInsertMany op;
  // set input info
  ge::TensorDesc tensor_desc_handle(ge::Shape({2}), ge::FORMAT_ND, ge::DT_STRING_REF);
  op.UpdateInputDesc("handle", tensor_desc_handle);
  ge::TensorDesc tensor_desc_keys(ge::Shape({1, 4}), ge::FORMAT_ND, ge::DT_STRING);
  op.UpdateInputDesc("keys", tensor_desc_keys);
  ge::TensorDesc tensor_desc_values(ge::Shape({3, 4}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("values", tensor_desc_values);

  // set attr info
  op.SetAttr("component_index", ge::DT_INT32);

  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// values rank failed testcase
TEST_F(barrier_infer_test, barrier_insert_many_infer_test_5) {
  //new op
  ge::op::BarrierInsertMany op;
  // set input info
  ge::TensorDesc tensor_desc_handle(ge::Shape({2}), ge::FORMAT_ND, ge::DT_STRING_REF);
  op.UpdateInputDesc("handle", tensor_desc_handle);
  ge::TensorDesc tensor_desc_keys(ge::Shape({1}), ge::FORMAT_ND, ge::DT_STRING);
  op.UpdateInputDesc("keys", tensor_desc_keys);
  ge::TensorDesc tensor_desc_values(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("values", tensor_desc_values);

  // set attr info
  op.SetAttr("component_index", ge::DT_INT32);

  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// get attr[component_types] failed
TEST_F(barrier_infer_test, barrier_take_many_infer_test_1) {
  //new op
  ge::op::BarrierTakeMany op;
  // set input info
  ge::TensorDesc tensor_desc_handle(ge::Shape({2}), ge::FORMAT_ND, ge::DT_STRING_REF);
  op.UpdateInputDesc("handle", tensor_desc_handle);
  ge::TensorDesc tensor_desc_num_elements(ge::Shape({1}), ge::FORMAT_ND, ge::DT_STRING);
  op.UpdateInputDesc("keys", tensor_desc_num_elements);

  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// attr[component_types] size < 1
TEST_F(barrier_infer_test, barrier_take_many_infer_test_2) {
  //new op
  ge::op::BarrierTakeMany op;
  // set input info
  ge::TensorDesc tensor_desc_handle(ge::Shape({2}), ge::FORMAT_ND, ge::DT_STRING_REF);
  op.UpdateInputDesc("handle", tensor_desc_handle);
  ge::TensorDesc tensor_desc_num_elements(ge::Shape({1}), ge::FORMAT_ND, ge::DT_STRING);
  op.UpdateInputDesc("num_elements", tensor_desc_num_elements);

  // set attr info
  std::vector<ge::DataType> component_types;
  op.SetAttr("component_types", component_types);

  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// attr[component_types] type difference
TEST_F(barrier_infer_test, barrier_take_many_infer_test_3) {
  //new op
  ge::op::BarrierTakeMany op;
  // set input info
  ge::TensorDesc tensor_desc_handle(ge::Shape({2}), ge::FORMAT_ND, ge::DT_STRING_REF);
  op.UpdateInputDesc("handle", tensor_desc_handle);
  ge::TensorDesc tensor_desc_num_elements(ge::Shape({1}), ge::FORMAT_ND, ge::DT_STRING);
  op.UpdateInputDesc("num_elements", tensor_desc_num_elements);

  // set attr info
  std::vector<ge::DataType> component_types{ge::DT_INT64, ge::DT_INT32, ge::DT_INT8};
  op.SetAttr("component_types", component_types);

  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

// attr[component_types] type difference
TEST_F(barrier_infer_test, barrier_take_many_infer_test_4) {
  //new op
  ge::op::BarrierTakeMany op;
  // set input info
  ge::TensorDesc tensor_desc_handle(ge::Shape({2}), ge::FORMAT_ND, ge::DT_STRING_REF);
  op.UpdateInputDesc("handle", tensor_desc_handle);
  ge::TensorDesc tensor_desc_num_elements(ge::Shape({1}), ge::FORMAT_ND, ge::DT_STRING);
  op.UpdateInputDesc("num_elements", tensor_desc_num_elements);

  // set attr info
  std::vector<ge::DataType> component_types{ge::DT_INT32, ge::DT_INT32, ge::DT_INT32};
  op.SetAttr("component_types", component_types);

  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(barrier_infer_test, barrier_close_infer_test_1) {
  //new op
  ge::op::BarrierClose op;
  // set input info
  ge::TensorDesc tensor_desc_handle(ge::Shape({2, 3}), ge::FORMAT_ND, ge::DT_STRING_REF);
  op.UpdateInputDesc("handle", tensor_desc_handle);
  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(barrier_infer_test, barrier_close_infer_test_2) {
  //new op
  ge::op::BarrierClose op;
  // set input info
  ge::TensorDesc tensor_desc_handle(ge::Shape({1}), ge::FORMAT_ND, ge::DT_STRING_REF);
  op.UpdateInputDesc("handle", tensor_desc_handle);
  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(barrier_infer_test, barrier_close_infer_test_3) {
  //new op
  ge::op::BarrierClose op;
  // set input info
  ge::TensorDesc tensor_desc_handle(ge::Shape({2}), ge::FORMAT_ND, ge::DT_STRING_REF);
  tensor_desc_handle.SetOriginShape(ge::Shape({2}));
  op.UpdateInputDesc("handle", tensor_desc_handle);
  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(barrier_infer_test, barrier_ready_size_infer_test_1) {
  //new op
  ge::op::BarrierReadySize op;
  // set input info
  ge::TensorDesc tensor_desc_handle(ge::Shape({2, 3}), ge::FORMAT_ND, ge::DT_STRING_REF);
  op.UpdateInputDesc("handle", tensor_desc_handle);
  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(barrier_infer_test, barrier_ready_size_infer_test_2) {
  //new op
  ge::op::BarrierReadySize op;
  // set input info
  ge::TensorDesc tensor_desc_handle(ge::Shape({1}), ge::FORMAT_ND, ge::DT_STRING_REF);
  op.UpdateInputDesc("handle", tensor_desc_handle);
  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(barrier_infer_test, barrier_ready_size_infer_test_3) {
  //new op
  ge::op::BarrierReadySize op;
  // set input info
  ge::TensorDesc tensor_desc_handle(ge::Shape({2}), ge::FORMAT_ND, ge::DT_STRING_REF);
    tensor_desc_handle.SetOriginShape(ge::Shape({2}));
  op.UpdateInputDesc("handle", tensor_desc_handle);
  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(barrier_infer_test, barrier_incomplete_size_infer_test_1) {
  //new op
  ge::op::BarrierIncompleteSize op;
  // set input info
  ge::TensorDesc tensor_desc_handle(ge::Shape({2}), ge::FORMAT_ND, ge::DT_STRING_REF);
    tensor_desc_handle.SetOriginShape(ge::Shape({2}));
  op.UpdateInputDesc("handle", tensor_desc_handle);
  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(barrier_infer_test, barrier_incomplete_size_infer_test_2) {
  //new op
  ge::op::BarrierIncompleteSize op;
  // set input info
  ge::TensorDesc tensor_desc_handle(ge::Shape({2, 3}), ge::FORMAT_ND, ge::DT_STRING_REF);
  op.UpdateInputDesc("handle", tensor_desc_handle);
  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(barrier_infer_test, barrier_incomplete_size_infer_test_3) {
  //new op
  ge::op::BarrierIncompleteSize op;
  // set input info
  ge::TensorDesc tensor_desc_handle(ge::Shape({1}), ge::FORMAT_ND, ge::DT_STRING_REF);
  op.UpdateInputDesc("handle", tensor_desc_handle);
  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}