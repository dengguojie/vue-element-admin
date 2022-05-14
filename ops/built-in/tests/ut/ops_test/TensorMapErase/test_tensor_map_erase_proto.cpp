#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "elewise_calculation_ops.h"
#include "map_ops.h"
class TensorMapErase : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "TensorMapErase Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "TensorMapErase Proto Test TearDown" << std::endl;
  }
};

TEST_F(TensorMapErase, TensorMapErase_infershape_test) {
  ge::op::TensorMapErase op;
  op.UpdateInputDesc("input_handle", create_desc({}, ge::DT_VARIANT));
  op.UpdateInputDesc("key", create_desc({}, ge::DT_INT32));
  ge::InferenceContextPtr inferCtxPtr = std::move(ge::InferenceContext::Create());
  op.SetInferenceContext(inferCtxPtr);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_handle = op.GetOutputDescByName("output_handle");
  EXPECT_EQ(output_handle.GetDataType(), ge::DT_VARIANT);
}
