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

class UnpackProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "Unpack Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Unpack Proto Test TearDown" << std::endl;
  }
};

TEST_F(UnpackProtoTest, unpack_infershape_test1) {
  ge::op::Unpack op;
  auto op_info = ge::OpDescUtils::GetOpDescFromOperator(op);
  op_info->AddOutputDescForward("y", 4);
  op.UpdateInputDesc("x", create_desc_with_ori({4, 3}, ge::DT_FLOAT16, ge::FORMAT_ND, {2, 3}, ge::FORMAT_ND));
  auto infer_fail_res1 = op.InferShapeAndType();
  EXPECT_EQ(infer_fail_res1, ge::GRAPH_FAILED);
  op.SetAttr("axis", 0);
  auto infer_fail_res2 = op.InferShapeAndType();
  EXPECT_EQ(infer_fail_res2, ge::GRAPH_FAILED);
  op.SetAttr("num", 4);
  auto infer_succ_res = op.InferShapeAndType();
  EXPECT_EQ(infer_succ_res, ge::GRAPH_SUCCESS);
  size_t out_count = op.GetOutputsSize();
  EXPECT_EQ(out_count, 4);
  std::vector<int64_t> out_dims = op.GetOutputDesc(0).GetShape().GetDims();
  bool expect_out = (out_dims == std::vector<int64_t>{3});
  EXPECT_EQ(expect_out, true);
}

TEST_F(UnpackProtoTest, unpack_infershape_test2) {
  ge::op::Unpack op;
  auto op_info = ge::OpDescUtils::GetOpDescFromOperator(op);
  op_info->AddOutputDescForward("y", 2);
  op.UpdateInputDesc("x", create_desc_shape_range({-1, -1, -1}, ge::DT_FLOAT, ge::FORMAT_ND, {-1, -1, -1},
                                                  ge::FORMAT_ND, {{1, 32}, {1, 2}, {1, 20}}));
  op.SetAttr("axis", 1);
  op.SetAttr("num", 2);
  auto infer_succ_res = op.InferShapeAndType();
  EXPECT_EQ(infer_succ_res, ge::GRAPH_SUCCESS);
  size_t out_count = op.GetOutputsSize();
  EXPECT_EQ(out_count, 2);
  std::vector<int64_t> out_dims = op.GetOutputDesc(0).GetShape().GetDims();
  bool expect_out = (out_dims == std::vector<int64_t>{-1, -1});
  EXPECT_EQ(expect_out, true);
  std::vector<std::pair<int64_t, int64_t>> out_range;
  op.GetOutputDesc(0).GetShapeRange(out_range);
  expect_out = (out_range == std::vector<std::pair<int64_t, int64_t>>{{1, 32}, {1, 20}});
  EXPECT_EQ(expect_out, true);
}

TEST_F(UnpackProtoTest, unpack_infershape_test3) {
  ge::op::Unpack op;
  auto op_info = ge::OpDescUtils::GetOpDescFromOperator(op);
  op_info->AddOutputDescForward("y", 2);
  op.UpdateInputDesc("x", create_desc_shape_range({-2}, ge::DT_FLOAT, ge::FORMAT_ND, {-2},
                                                  ge::FORMAT_ND, {}));
  op.SetAttr("axis", 1);
  op.SetAttr("num", 2);
  auto infer_succ_res = op.InferShapeAndType();
  EXPECT_EQ(infer_succ_res, ge::GRAPH_SUCCESS);
  std::vector<int64_t> out_dims = op.GetOutputDesc(0).GetShape().GetDims();
  bool expect_out = (out_dims == std::vector<int64_t>{-2});
  EXPECT_EQ(expect_out, true);
}
