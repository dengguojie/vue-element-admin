#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "correlation.h"

class CorrelationProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "Correlation Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Correlation Proto Test TearDown" << std::endl;
  }
};

TEST_F(CorrelationProtoTest, Correlation_infershape_test01){
  ge::op::Correlation op;
  
  op.UpdateInputDesc("filter", create_desc_with_ori(
    {1, 32, 5, 5}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
    {1, 32, 5, 5}, ge::FORMAT_NCHW));
  op.UpdateInputDesc("x", create_desc_with_ori(
    {1, 32, 29, 29}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
    {1, 32, 29, 29}, ge::FORMAT_NCHW));
  op.UpdateOutputDesc("y", create_desc_with_ori(
    {1, 1, 25, 25}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
    {1, 1, 25, 25}, ge::FORMAT_NCHW));
  op.SetAttr("groups", 1);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {1, 1, 25, 25};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(CorrelationProtoTest, Correlation_infershape_test02){
  ge::op::Correlation op;
  
  op.UpdateInputDesc("filter", create_desc_with_ori(
    {1, 32, 5, 5}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
    {1, 32, 5, 5}, ge::FORMAT_NCHW));
  op.UpdateInputDesc("x", create_desc_with_ori(
    {1, 32, 29, 29}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
    {1, 32, 29, 29}, ge::FORMAT_NCHW));
  op.UpdateOutputDesc("y", create_desc_with_ori(
    {1, 32, 25, 25}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
    {1, 32, 25, 25}, ge::FORMAT_NCHW));
  
  op.SetAttr("groups", 32);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {1, 32, 25, 25};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(CorrelationProtoTest, Correlation_infershape_test03){
  ge::op::Correlation op;
  
  ge::TensorDesc tensor_filter_desc;
  op.UpdateInputDesc("filter", create_desc_with_ori(
    {1, 5, 5, 32}, ge::DT_FLOAT16, ge::FORMAT_NHWC,
    {1, 5, 5, 32}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("x", create_desc_with_ori(
    {1, 29, 29, 32}, ge::DT_FLOAT16, ge::FORMAT_NHWC,
    {1, 29, 29, 32}, ge::FORMAT_NHWC));
  op.UpdateOutputDesc("y", create_desc_with_ori(
    {1, 25, 25, 1}, ge::DT_FLOAT16, ge::FORMAT_NHWC,
    {1, 25, 25, 1}, ge::FORMAT_NHWC));
  
  op.SetAttr("groups", 1);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {1, 25, 25, 1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(CorrelationProtoTest, Correlation_infershape_test04){
  ge::op::Correlation op;
  
  op.UpdateInputDesc("filter", create_desc_with_ori(
    {1, 5, 5, 32}, ge::DT_FLOAT16, ge::FORMAT_NHWC,
    {1, 5, 5, 32}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("x", create_desc_with_ori(
    {1, 29, 29, 32}, ge::DT_FLOAT16, ge::FORMAT_NHWC,
    {1, 29, 29, 32}, ge::FORMAT_NHWC));
  op.UpdateOutputDesc("y", create_desc_with_ori(
    {1, 25, 25, 32}, ge::DT_FLOAT16, ge::FORMAT_NHWC,
    {1, 25, 25, 32}, ge::FORMAT_NHWC));
  
  op.SetAttr("groups", 32);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {1, 25, 25, 32};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(CorrelationProtoTest, Correlation_verify_test01) {
  ge::op::Correlation op;
  ge::TensorDesc tensor_filter;
  ge::Shape filter_shape({1, 32, 5, 5});
  tensor_filter.SetDataType(ge::DT_FLOAT16);
  tensor_filter.SetShape(filter_shape);
  tensor_filter.SetFormat(ge::FORMAT_NCHW);

  ge::TensorDesc tensor_x;
  ge::Shape x_shape({1, 32, 29, 29});
  tensor_x.SetDataType(ge::DT_FLOAT16);
  tensor_x.SetShape(x_shape);
  tensor_x.SetFormat(ge::FORMAT_NCHW);

  op.SetAttr("groups", 1);

  // [TODO] update op input here
  op.UpdateInputDesc("filter", tensor_filter);
  op.UpdateInputDesc("x", tensor_x);

  // [TODO] call Verify function here
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

TEST_F(CorrelationProtoTest, Correlation_verify_test02) {
  ge::op::Correlation op;
  ge::TensorDesc tensor_filter;
  ge::Shape filter_shape({1, 5, 5, 32});
  tensor_filter.SetDataType(ge::DT_FLOAT16);
  tensor_filter.SetShape(filter_shape);
  tensor_filter.SetFormat(ge::FORMAT_NHWC);

  ge::TensorDesc tensor_x;
  ge::Shape x_shape({1, 29, 29, 32});
  tensor_x.SetDataType(ge::DT_FLOAT16);
  tensor_x.SetShape(x_shape);
  tensor_x.SetFormat(ge::FORMAT_NHWC);

  op.SetAttr("groups", 1);

  // [TODO] update op input here
  op.UpdateInputDesc("filter", tensor_filter);
  op.UpdateInputDesc("x", tensor_x);

  // [TODO] call Verify function here
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

TEST_F(CorrelationProtoTest, Correlation_verify_test03) {
  ge::op::Correlation op;
  ge::TensorDesc tensor_filter;
  ge::Shape filter_shape({1, 32, 5});
  tensor_filter.SetDataType(ge::DT_FLOAT16);
  tensor_filter.SetShape(filter_shape);
  tensor_filter.SetFormat(ge::FORMAT_ND);

  ge::TensorDesc tensor_x;
  ge::Shape x_shape({1, 32, 29, 29});
  tensor_x.SetDataType(ge::DT_FLOAT16);
  tensor_x.SetShape(x_shape);
  tensor_x.SetFormat(ge::FORMAT_NCHW);

  op.SetAttr("groups", 1);

  // [TODO] update op input here
  op.UpdateInputDesc("filter", tensor_filter);
  op.UpdateInputDesc("x", tensor_x);

  // [TODO] call Verify function here
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(CorrelationProtoTest, Correlation_verify_test04) {
  ge::op::Correlation op;
  ge::TensorDesc tensor_filter;
  ge::Shape filter_shape({2, 32, 5, 5});
  tensor_filter.SetDataType(ge::DT_FLOAT16);
  tensor_filter.SetShape(filter_shape);
  tensor_filter.SetFormat(ge::FORMAT_NCHW);

  ge::TensorDesc tensor_x;
  ge::Shape x_shape({1, 32, 29, 29});
  tensor_x.SetDataType(ge::DT_FLOAT16);
  tensor_x.SetShape(x_shape);
  tensor_x.SetFormat(ge::FORMAT_NCHW);

  op.SetAttr("groups", 1);

  // [TODO] update op input here
  op.UpdateInputDesc("filter", tensor_filter);
  op.UpdateInputDesc("x", tensor_x);

  // [TODO] call Verify function here
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(CorrelationProtoTest, Correlation_verify_test05) {
  ge::op::Correlation op;
  ge::TensorDesc tensor_filter;
  ge::Shape filter_shape({1, 16, 5, 5});
  tensor_filter.SetDataType(ge::DT_FLOAT16);
  tensor_filter.SetShape(filter_shape);
  tensor_filter.SetFormat(ge::FORMAT_NCHW);

  ge::TensorDesc tensor_x;
  ge::Shape x_shape({1, 32, 29, 29});
  tensor_x.SetDataType(ge::DT_FLOAT16);
  tensor_x.SetShape(x_shape);
  tensor_x.SetFormat(ge::FORMAT_NCHW);

  op.SetAttr("groups", 1);

  // [TODO] update op input here
  op.UpdateInputDesc("filter", tensor_filter);
  op.UpdateInputDesc("x", tensor_x);

  // [TODO] call Verify function here
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(CorrelationProtoTest, Correlation_verify_test06) {
  ge::op::Correlation op;
  ge::TensorDesc tensor_filter;
  ge::Shape filter_shape({1, 32, 5, 5});
  tensor_filter.SetDataType(ge::DT_FLOAT);
  tensor_filter.SetShape(filter_shape);
  tensor_filter.SetFormat(ge::FORMAT_NCHW);

  ge::TensorDesc tensor_x;
  ge::Shape x_shape({1, 32, 29, 29});
  tensor_x.SetDataType(ge::DT_FLOAT16);
  tensor_x.SetShape(x_shape);
  tensor_x.SetFormat(ge::FORMAT_NCHW);

  op.SetAttr("groups", 1);

  // [TODO] update op input here
  op.UpdateInputDesc("filter", tensor_filter);
  op.UpdateInputDesc("x", tensor_x);

  // [TODO] call Verify function here
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(CorrelationProtoTest, Correlation_verify_test07) {
  ge::op::Correlation op;
  ge::TensorDesc tensor_filter;
  ge::Shape filter_shape({32, 32, 5, 5});
  tensor_filter.SetDataType(ge::DT_FLOAT16);
  tensor_filter.SetShape(filter_shape);
  tensor_filter.SetFormat(ge::FORMAT_NCHW);

  ge::TensorDesc tensor_x;
  ge::Shape x_shape({32, 16, 29, 29});
  tensor_x.SetDataType(ge::DT_FLOAT16);
  tensor_x.SetShape(x_shape);
  tensor_x.SetFormat(ge::FORMAT_NCHW);

  op.SetAttr("groups", 32);

  // [TODO] update op input here
  op.UpdateInputDesc("filter", tensor_filter);
  op.UpdateInputDesc("x", tensor_x);

  // [TODO] call Verify function here
  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}
