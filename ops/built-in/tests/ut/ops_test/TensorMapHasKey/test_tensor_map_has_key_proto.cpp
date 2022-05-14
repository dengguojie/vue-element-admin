#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "elewise_calculation_ops.h"
#include "map_ops.h"
class TensorMapHasKey : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "TensorMapHasKey Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "TensorMapHasKey Proto Test TearDown" << std::endl;
  }
};

TEST_F(TensorMapHasKey, TensorMapHasKey_infershape_test) {
  ge::op::TensorMapHasKey op;
  op.UpdateInputDesc("input_handle", create_desc({}, ge::DT_VARIANT));
  op.UpdateInputDesc("key", create_desc({}, ge::DT_INT32));
  ge::InferenceContextPtr inferCtxPtr = std::move(ge::InferenceContext::Create());
  op.SetInferenceContext(inferCtxPtr);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto has_key = op.GetOutputDescByName("has_key");
  EXPECT_EQ(has_key.GetDataType(), ge::DT_BOOL);
}