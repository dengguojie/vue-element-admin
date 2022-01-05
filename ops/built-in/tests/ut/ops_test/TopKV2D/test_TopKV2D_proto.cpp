#include <iostream>
#include "gtest/gtest.h"
#include "op_proto_test_util.h"
#include "selection_ops.h"
#include "array_ops.h"
// ----------------TopKV2D-------------------
using namespace ge;
using namespace op;

class TopKV2DProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "TopKV2D Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "TopKV2D Proto Test TearDown" << std::endl;
  }
};

TEST_F(TopKV2DProtoTest, topkv2d_verify_test001) {
  ge::op::TopKV2D op;
  op.UpdateInputDesc("x", create_desc({1, 16}, ge::DT_FLOAT16));
  op.UpdateInputDesc("k", create_desc({1}, ge::DT_INT32));
  op.UpdateInputDesc("assist_seq", create_desc({1, 16}, ge::DT_FLOAT16));
  op.SetAttr("sorted", true);
  op.SetAttr("dim", -1);
  op.SetAttr("largest", true);
  auto status = op.InferShapeAndType(); 
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

TEST_F(TopKV2DProtoTest, topkv2d_verify_test002) {
  ge::op::TopKV2D op;
  op.UpdateInputDesc("x", create_desc({1, 16}, ge::DT_FLOAT16));
  int32_t *k_data = new int32_t[1];
  k_data[0]=1;
  ge::TensorDesc k_desc(ge::Shape({1}),FORMAT_ND, DT_INT32);
  Tensor k_tensor(k_desc, (uint8_t *)k_data,sizeof(int32_t));
  auto k_const = ge::op::Constant().set_attr_value(k_tensor);
  op.set_input_k(k_const);
  op.UpdateInputDesc("assist_seq", create_desc({1, 16}, ge::DT_FLOAT16));
  op.SetAttr("sorted", true);
  op.SetAttr("dim", -1);
  op.SetAttr("largest", true);
  auto status = op.InferShapeAndType(); 
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}
