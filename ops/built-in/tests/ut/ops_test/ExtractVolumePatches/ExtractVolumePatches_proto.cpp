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

class ExtractVolumePatchesProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ExtractVolumePatches Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ExtractVolumePatches Proto Test TearDown" << std::endl;
  }
};

TEST_F(ExtractVolumePatchesProtoTest, extract_volume_patches_verify_failed_test) {
  ge::op::ExtractVolumePatches op;
  op.UpdateInputDesc("x", create_desc_with_ori({1, 3, 6, 9, 16}, ge::DT_FLOAT16, ge::FORMAT_NDC1HWC0, {1, 3, 6, 9, 16},
                                               ge::FORMAT_NDHWC));
  auto verify_fail_res1 = op.VerifyAllAttr(true);
  EXPECT_EQ(verify_fail_res1, ge::GRAPH_FAILED);
  op.SetAttr("ksizes", {1, 2, 5, 5, 1});
  auto verify_fail_res2 = op.VerifyAllAttr(true);
  EXPECT_EQ(verify_fail_res2, ge::GRAPH_FAILED);
  op.SetAttr("strides", {1, 2, 3, 2, 1});
  auto verify_fail_res3 = op.VerifyAllAttr(true);
  EXPECT_EQ(verify_fail_res3, ge::GRAPH_FAILED);
  std::string padding = "OTHER";
  op.SetAttr("padding", padding);
  auto verify_fail_res5 = op.VerifyAllAttr(true);
  EXPECT_EQ(verify_fail_res5, ge::GRAPH_FAILED);
}
TEST_F(ExtractVolumePatchesProtoTest, extract_volume_patches_infer_failed_test) {
  ge::op::ExtractVolumePatches op;
  op.UpdateInputDesc("x", create_desc_with_ori({1, 3, 6, 9, 16}, ge::DT_FLOAT16, ge::FORMAT_NDC1HWC0, {1, 3, 6, 9, 16},
                                               ge::FORMAT_NDHWC));
  op.SetAttr("ksizes", {1, 2, 5, 5, 1});
  op.SetAttr("strides", {1, 0, 0, 0, 1});
  auto infer_fail_res1 = op.InferShapeAndType();
  EXPECT_EQ(infer_fail_res1, ge::GRAPH_FAILED);
  std::string padding = "VALID";
  op.SetAttr("padding", padding);
  auto infer_fail_res2 = op.InferShapeAndType();
  EXPECT_EQ(infer_fail_res2, ge::GRAPH_FAILED);
}

TEST_F(ExtractVolumePatchesProtoTest, extract_volume_patches_data_slice_failed_test) {
  ge::op::ExtractVolumePatches op;
  op.UpdateInputDesc("x", create_desc_with_ori({13, 14, 15, 16, 17}, ge::DT_FLOAT, ge::FORMAT_NDC1HWC0,
                                               {13, 14, 15, 16, 17}, ge::FORMAT_NCDHW));
  op.SetAttr("ksizes", {1, 2, 3, 4, 5});
  op.SetAttr("strides", {5, 6, 7, 8, 9});
  std::string padding = "VALID";
  op.SetAttr("padding", padding);
  std::vector<std::vector<int64_t>> y_data_slice = {{16, 16}, {17}, {18}, {19}, {21}, {22}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
  ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
  auto status = op_desc->InferDataSlice();
  EXPECT_EQ(status, ge::NO_OVERLAP_DIM);
}

TEST_F(ExtractVolumePatchesProtoTest, extract_volume_patches_verify_and_infer_test) {
  ge::op::ExtractVolumePatches op;
  op.UpdateInputDesc("x", create_desc_with_ori({1, 3, 6, 9, 16}, ge::DT_FLOAT16, ge::FORMAT_NDC1HWC0, {1, 3, 6, 9, 16},
                                               ge::FORMAT_NDHWC));
  op.SetAttr("ksizes", {1, 2, 5, 5, 1});
  op.SetAttr("strides", {1, 2, 3, 2, 1});
  std::string padding = "SAME";
  op.SetAttr("padding", padding);

  auto verify_res = op.VerifyAllAttr(true);
  EXPECT_EQ(verify_res, ge::GRAPH_SUCCESS);

  auto infer_res = op.InferShapeAndType();
  EXPECT_EQ(infer_res, ge::GRAPH_SUCCESS);

  auto out_var_desc = op.GetOutputDesc("y");
  EXPECT_EQ(out_var_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_var_output_shape = {1, 2, 2, 5, 800};
  EXPECT_EQ(out_var_desc.GetShape().GetDims(), expected_var_output_shape);
}

TEST_F(ExtractVolumePatchesProtoTest, extract_volume_patches_data_slice_test1) {
  ge::op::ExtractVolumePatches op;
  op.UpdateInputDesc("x", create_desc_with_ori({13, 14, 15, 16}, ge::DT_FLOAT, ge::FORMAT_NDC1HWC0, {13, 14, 15, 16},
                                               ge::FORMAT_NDHWC));
  op.SetAttr("ksizes", {1, 2, 3, 4});
  op.SetAttr("strides", {5, 6, 7, 8});
  std::string padding = "SAME";
  op.SetAttr("padding", padding);
  std::vector<std::vector<int64_t>> y_data_slice = {{16}, {17}, {18}, {19, 20}, {21}, {22}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
  ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
  auto status = op_desc->InferDataSlice();
  ge::GeTensorDescPtr tensor_desc_x = op_desc->MutableInputDesc("x");
  std::vector<std::vector<int64_t>> x_data_slice;
  ge::AttrUtils::GetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice);
  std::vector<std::vector<int64_t>> expect_x_data_slice = {{}, {}, {}, {133, 14}, {}, {}};
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  EXPECT_EQ(expect_x_data_slice, x_data_slice);
}
TEST_F(ExtractVolumePatchesProtoTest, extract_volume_patches_data_slice_test2) {
  ge::op::ExtractVolumePatches op;
  op.UpdateInputDesc("x", create_desc_with_ori({13, 14, 15, 16}, ge::DT_FLOAT, ge::FORMAT_NDC1HWC0, {13, 14, 15, 16},
                                               ge::FORMAT_NDHWC));
  op.SetAttr("ksizes", {1, 2, 3, 4});
  op.SetAttr("strides", {5, 6, 7, 8});
  std::string padding = "VALID";
  op.SetAttr("padding", padding);
  std::vector<std::vector<int64_t>> y_data_slice = {{16}, {17, 18}, {19}, {20}, {21}, {22}};
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  ge::GeTensorDescPtr tensor_desc_y = op_desc->MutableOutputDesc("y");
  ge::AttrUtils::SetListListInt(tensor_desc_y, ge::ATTR_NAME_DATA_SLICE, y_data_slice);
  auto status = op_desc->InferDataSlice();
  ge::GeTensorDescPtr tensor_desc_x = op_desc->MutableInputDesc("x");
  std::vector<std::vector<int64_t>> x_data_slice;
  ge::AttrUtils::GetListListInt(tensor_desc_x, ge::ATTR_NAME_DATA_SLICE, x_data_slice);
  std::vector<std::vector<int64_t>> expect_x_data_slice = {{}, {102, 13}, {}, {}, {}, {}};
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  EXPECT_EQ(expect_x_data_slice, x_data_slice);
}