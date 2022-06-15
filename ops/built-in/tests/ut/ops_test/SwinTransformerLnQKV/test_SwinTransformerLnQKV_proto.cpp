#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "matrix_calculation_ops.h"

class SwinTransformerLnQKV : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SwinTransformerLnQKV Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SwinTransformerLnQKV Proto Test TearDown" << std::endl;
  }
};



TEST_F(SwinTransformerLnQKV, swin_transformer_ln_qkv_infershape_test_1){
  ge::op::SwinTransformerLnQKV op;
  op.UpdateInputDesc("x", create_desc({8, 9216, 128}, ge::DT_FLOAT16));
  op.SetAttr("head_num", 4);
  op.SetAttr("head_dim", 32);
  op.SetAttr("seq_length", 144);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc_0 = op.GetOutputDesc("query_output");
  EXPECT_EQ(output_desc_0.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape_0 = {512, 4, 144, 32};
  EXPECT_EQ(output_desc_0.GetShape().GetDims(), expected_output_shape_0);

  auto output_desc_1 = op.GetOutputDesc("key_output");
  EXPECT_EQ(output_desc_1.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape_1 = {512, 4, 144, 32};
  EXPECT_EQ(output_desc_1.GetShape().GetDims(), expected_output_shape_1);

  auto output_desc_2 = op.GetOutputDesc("value_output");
  EXPECT_EQ(output_desc_2.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape_2 = {512, 4, 144, 32};
  EXPECT_EQ(output_desc_2.GetShape().GetDims(), expected_output_shape_2);
}
