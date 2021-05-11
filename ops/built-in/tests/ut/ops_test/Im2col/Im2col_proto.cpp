#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "transformation_ops.h"
#include "graph/utils/type_utils.h"
#include "op_log.h"
#include "op_desc.h"
#include "utils/op_desc_utils.h"
#include "utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/common_error_codes.h"

class Im2colProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "Im2col Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Im2col Proto Test TearDown" << std::endl;
  }
};

TEST_F(Im2colProtoTest, im2col_verify_failed_test) {
  ge::op::Im2col op;
  op.UpdateInputDesc("x",
                     create_desc_with_ori({2, 7, 7, 32}, ge::DT_INT8, ge::FORMAT_NHWC, {2, 7, 7, 32}, ge::FORMAT_NHWC));
  auto verify_fail_res1 = op.VerifyAllAttr(true);
  EXPECT_EQ(verify_fail_res1, ge::GRAPH_FAILED);
  op.SetAttr("ksizes", {3, 3});
  op.SetAttr("strides", {1, 1});
  op.SetAttr("dilations", {2, 2});
  std::string padding = "OTHER";
  op.SetAttr("padding_mode", padding);
  auto verify_fail_res5 = op.VerifyAllAttr(true);
  EXPECT_EQ(verify_fail_res5, ge::GRAPH_FAILED);
}

TEST_F(Im2colProtoTest, im2col_infer_failed_test_1) {
  ge::op::Im2col op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({2, 7, 7, 32}, ge::DT_INT8, ge::FORMAT_NDHWC, {2, 7, 7, 32}, ge::FORMAT_NDHWC));
  op.SetAttr("ksizes", {3, 3});
  op.SetAttr("strides", {1, 1});
  op.SetAttr("dilations", {2, 2});
  auto infer_fail_res1 = op.InferShapeAndType();
  EXPECT_EQ(infer_fail_res1, ge::GRAPH_FAILED);
  std::string padding = "VALID";
  op.SetAttr("padding_mode", padding);
  auto infer_fail_res2 = op.InferShapeAndType();
  EXPECT_EQ(infer_fail_res2, ge::GRAPH_FAILED);
}

TEST_F(Im2colProtoTest, im2col_verify_and_infer_test_1) {
  ge::op::Im2col op;
  op.UpdateInputDesc("x",
                     create_desc_with_ori({2, 7, 7, 32}, ge::DT_INT8, ge::FORMAT_NHWC, {2, 7, 7, 32}, ge::FORMAT_NHWC));
  op.SetAttr("ksizes", {3, 3});
  op.SetAttr("strides", {1, 1});
  op.SetAttr("dilations", {2, 2});
  std::string padding = "VALID";
  op.SetAttr("padding_mode", padding);
  op.SetAttr("pads", {0, 0, 0, 0});

  auto verify_res = op.VerifyAllAttr(true);
  EXPECT_EQ(verify_res, ge::GRAPH_SUCCESS);

  auto infer_res = op.InferShapeAndType();
  EXPECT_EQ(infer_res, ge::GRAPH_SUCCESS);

  auto out_var_desc = op.GetOutputDesc("y");
  EXPECT_EQ(out_var_desc.GetDataType(), ge::DT_INT8);
  std::vector<int64_t> expected_var_output_shape = {2, 3, 3, 288};
  EXPECT_EQ(out_var_desc.GetShape().GetDims(), expected_var_output_shape);
}


TEST_F(Im2colProtoTest, im2col_verify_and_infer_test_2) {
  ge::op::Im2col op;
  op.UpdateInputDesc("x",
                     create_desc_with_ori({2, 32, 7, 7}, ge::DT_INT8, ge::FORMAT_NCHW, {2, 32, 7, 7}, ge::FORMAT_NCHW));
  op.SetAttr("ksizes", {3, 3});
  op.SetAttr("strides", {1, 1});
  op.SetAttr("dilations", {2, 2});
  std::string padding = "SAME";
  op.SetAttr("padding_mode", padding);
  op.SetAttr("pads", {0, 0, 0, 0});

  auto verify_res = op.VerifyAllAttr(true);
  EXPECT_EQ(verify_res, ge::GRAPH_SUCCESS);

  auto infer_res = op.InferShapeAndType();
  EXPECT_EQ(infer_res, ge::GRAPH_SUCCESS);

  auto out_var_desc = op.GetOutputDesc("y");
  EXPECT_EQ(out_var_desc.GetDataType(), ge::DT_INT8);
  std::vector<int64_t> expected_var_output_shape = {2, 288, 7, 7};
  EXPECT_EQ(out_var_desc.GetShape().GetDims(), expected_var_output_shape);
}

TEST_F(Im2colProtoTest, im2col_verify_and_infer_test_3) {
  ge::op::Im2col op;
  op.UpdateInputDesc("x",
                     create_desc_with_ori({2, 32, 7, 7}, ge::DT_INT8, ge::FORMAT_NCHW, {2, 32, 7, 7}, ge::FORMAT_NCHW));
  op.SetAttr("ksizes", {3, 3});
  op.SetAttr("strides", {1, 1});
  op.SetAttr("dilations", {2, 2});
  std::string padding = "CALCULATED";
  op.SetAttr("padding_mode", padding);
  op.SetAttr("pads", {0, 0, 0, 0});

  auto verify_res = op.VerifyAllAttr(true);
  EXPECT_EQ(verify_res, ge::GRAPH_SUCCESS);

  auto infer_res = op.InferShapeAndType();
  EXPECT_EQ(infer_res, ge::GRAPH_SUCCESS);

  auto out_var_desc = op.GetOutputDesc("y");
  EXPECT_EQ(out_var_desc.GetDataType(), ge::DT_INT8);
  std::vector<int64_t> expected_var_output_shape = {2, 288, 3, 3};
  EXPECT_EQ(out_var_desc.GetShape().GetDims(), expected_var_output_shape);
}


TEST_F(Im2colProtoTest, im2col_verify_and_infer_test_4) {
  ge::op::Im2col op;
  op.UpdateInputDesc("x",
                     create_desc_with_ori({2, 32, 7, 7}, ge::DT_INT8, ge::FORMAT_NCHW, {2, 32, 7, 7}, ge::FORMAT_NCHW));
  op.SetAttr("ksizes", {3, 3});
  op.SetAttr("strides", {1, 1});
  op.SetAttr("dilations", {2, 2});
  std::string padding = "CALCULATED";
  op.SetAttr("padding_mode", padding);
  op.SetAttr("pads", {0});

  auto verify_res = op.VerifyAllAttr(true);
  EXPECT_EQ(verify_res, ge::GRAPH_SUCCESS);

  auto infer_res = op.InferShapeAndType();
  EXPECT_EQ(infer_res, ge::GRAPH_SUCCESS);

  auto out_var_desc = op.GetOutputDesc("y");
  EXPECT_EQ(out_var_desc.GetDataType(), ge::DT_INT8);
  std::vector<int64_t> expected_var_output_shape = {2, 288, 3, 3};
  EXPECT_EQ(out_var_desc.GetShape().GetDims(), expected_var_output_shape);
}

TEST_F(Im2colProtoTest, im2col_infer_failed_test_2) {
  ge::op::Im2col op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({2, 7, 7, 32}, ge::DT_INT8, ge::FORMAT_NDHWC, {2, 7, 7, 32}, ge::FORMAT_NDHWC));
  op.SetAttr("ksizes", {3, 3, 3});
  op.SetAttr("strides", {1, 1});
  op.SetAttr("dilations", {2, 2});
  std::string padding = "VALID";
  op.SetAttr("padding_mode", padding);
  auto infer_fail_res1 = op.InferShapeAndType();
  EXPECT_EQ(infer_fail_res1, ge::GRAPH_FAILED);
}

TEST_F(Im2colProtoTest, im2col_infer_failed_test_3) {
  ge::op::Im2col op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({2, 7, 7, 32}, ge::DT_INT8, ge::FORMAT_NDHWC, {2, 7, 7, 32}, ge::FORMAT_NDHWC));
  op.SetAttr("ksizes", {3, 3});
  op.SetAttr("strides", {1, 1, 1});
  op.SetAttr("dilations", {2, 2});
  std::string padding = "VALID";
  op.SetAttr("padding_mode", padding);
  auto infer_fail_res1 = op.InferShapeAndType();
  EXPECT_EQ(infer_fail_res1, ge::GRAPH_FAILED);
}

TEST_F(Im2colProtoTest, im2col_infer_failed_test_4) {
  ge::op::Im2col op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({2, 7, 7, 32}, ge::DT_INT8, ge::FORMAT_NDHWC, {2, 7, 7, 32}, ge::FORMAT_NDHWC));
  op.SetAttr("ksizes", {3, 3});
  op.SetAttr("strides", {1, 1});
  op.SetAttr("dilations", {2, 2, 2});
  std::string padding = "VALID";
  op.SetAttr("padding_mode", padding);
  auto infer_fail_res1 = op.InferShapeAndType();
  EXPECT_EQ(infer_fail_res1, ge::GRAPH_FAILED);
}

TEST_F(Im2colProtoTest, im2col_infer_failed_test_5) {
  ge::op::Im2col op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({2, 7, 7, 32}, ge::DT_INT8, ge::FORMAT_NDHWC, {2, 7, 7, 32}, ge::FORMAT_NDHWC));
  op.SetAttr("ksizes", {3, 3});
  op.SetAttr("strides", {1, 0});
  op.SetAttr("dilations", {2, 2});
  std::string padding = "VALID";
  op.SetAttr("padding_mode", padding);
  auto infer_fail_res1 = op.InferShapeAndType();
  EXPECT_EQ(infer_fail_res1, ge::GRAPH_FAILED);
}

TEST_F(Im2colProtoTest, im2col_infer_failed_test_6) {
  ge::op::Im2col op;
  op.UpdateInputDesc("x",
                     create_desc_with_ori({2, 32, 7, 7}, ge::DT_INT8, ge::FORMAT_NCHW, {2, 32, 7, 7}, ge::FORMAT_NCHW));
  op.SetAttr("ksizes", {3, 3});
  op.SetAttr("strides", {1, 1});
  op.SetAttr("dilations", {2, 2});
  std::string padding = "CALCULATED";
  op.SetAttr("padding_mode", padding);
  op.SetAttr("pads", {0,0,0,0,0});
  auto infer_fail_res1 = op.InferShapeAndType();
  EXPECT_EQ(infer_fail_res1, ge::GRAPH_FAILED);
}

TEST_F(Im2colProtoTest, im2col_data_slice_failed_test) {
  ge::op::Im2col op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({13, 14, 15, 16}, ge::DT_FLOAT, ge::FORMAT_NCHW, {13, 14, 15, 16}, ge::FORMAT_NCHW));
  op.SetAttr("ksizes", {3, 4});
  op.SetAttr("strides", {7, 8});
  op.SetAttr("dilations", {11, 12});
  std::string padding = "SAME";
  op.SetAttr("padding_mode", padding);
  auto infer_res_succ1 = op.InferShapeAndType();
  EXPECT_EQ(infer_res_succ1, ge::GRAPH_SUCCESS);
  std::vector<std::vector<int64_t>> y_data_slice = {{17, 17}, {18}, {19}, {21}, {22}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
  ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
  auto infer_data_slice_fail1 = op_desc->InferDataSlice();
  ge::GeTensorDescPtr tensor_desc_x = op_desc->MutableInputDesc("x");
  std::vector<std::vector<int64_t>> x_data_slice;
  ge::AttrUtils::GetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice);
  std::vector<std::vector<int64_t>> expect_x_data_slice = {{}, {}, {114, 13}, {}, {}};
  EXPECT_EQ(infer_data_slice_fail1, ge::NO_OVERLAP_DIM);
}
TEST_F(Im2colProtoTest, im2col_data_slice_test) {
  ge::op::Im2col op;
  op.UpdateInputDesc(
      "x", create_desc_with_ori({13, 14, 15, 16}, ge::DT_FLOAT, ge::FORMAT_NHWC, {13, 14, 15, 16}, ge::FORMAT_NHWC));
  op.SetAttr("ksizes", {2, 3});
  op.SetAttr("strides", {6, 7});
  op.SetAttr("dilations", {10, 11});
  std::string padding = "VALID";
  op.SetAttr("padding_mode", padding);
  std::vector<std::vector<int64_t>> y_data_slice = {{17}, {18}, {19, 20}, {21}, {22}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
  ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
  auto status = op_desc->InferDataSlice();
  ge::GeTensorDescPtr tensor_desc_x = op_desc->MutableInputDesc("x");
  std::vector<std::vector<int64_t>> x_data_slice;
  ge::AttrUtils::GetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice);
  std::vector<std::vector<int64_t>> expect_x_data_slice = {{}, {}, {114, 13}, {}, {}};
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  EXPECT_EQ(expect_x_data_slice, x_data_slice);
}