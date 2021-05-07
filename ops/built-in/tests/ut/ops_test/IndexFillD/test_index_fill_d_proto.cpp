#include <gtest/gtest.h>
#include <vector>
#include "selection_ops.h"
#include "op_proto_test_util.h"

#include "utils/op_desc_utils.h"
#include "utils/attr_utils.h"

class IndexFillDdTest : public testing::Test {
  protected:
  static void SetUpTestCase() {
    std::cout << "index_fill_d test SetUp" << std::endl;
    }

  static void TearDownTestCase() {
    std::cout << "index_fill_d test TearDown" << std::endl;
  }
};

TEST_F(IndexFillDdTest, index_fill_d_test_case_1) {
  //   define your op here
  ge::op::IndexFillD index_fill_d_op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2,2},{100,200},{4,8}};
  auto format = ge::FORMAT_ND;
  auto x = create_desc_shape_range({2,100,4},ge::DT_FLOAT16, format,{2,100,4},format,shape_range);

  //   update op input here
  index_fill_d_op.UpdateInputDesc("x", x);
  index_fill_d_op.UpdateInputDesc("assist1", x);
  index_fill_d_op.UpdateInputDesc("assist2", x);

  auto opdesc = ge::OpDescUtils::GetOpDescFromOperator(index_fill_d_op);

  auto ret_Verify = index_fill_d_op.VerifyAllAttr(true);
  EXPECT_EQ(ret_Verify, ge::GRAPH_SUCCESS);

  //   call InferShapeAndType function here
  auto ret = index_fill_d_op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  //   compare dtype and shape of op output
  ge::GeTensorDescPtr output_desc = opdesc->MutableOutputDesc("y");
  EXPECT_EQ(output_desc->GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {2, 100, 4};
  EXPECT_EQ(output_desc->MutableShape().GetDims(), expected_output_shape);
}

TEST_F(IndexFillDdTest, index_fill_d_test_case_2) {
    // define op here
  ge::op::IndexFillD index_fill_d_op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2,2},{100,200},{4,8}};
  auto format = ge::FORMAT_ND;
  auto x = create_desc_shape_range({2,100,4},ge::DT_FLOAT16, format,{2,100,4},format,shape_range);
  auto x_ = create_desc_shape_range({2,100,4},ge::DT_FLOAT, format,{2,100,4},format,shape_range);
  //  update op input here
  index_fill_d_op.UpdateInputDesc("x", x);
  index_fill_d_op.UpdateInputDesc("assist1", x);
  index_fill_d_op.UpdateInputDesc("assist2", x_);

  auto opdesc = ge::OpDescUtils::GetOpDescFromOperator(index_fill_d_op);

  auto ret_Verify = index_fill_d_op.VerifyAllAttr(true);
  EXPECT_EQ(ret_Verify, ge::GRAPH_FAILED);

  auto ret = index_fill_d_op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  //   compare dtype and shape of op output
  ge::GeTensorDescPtr output_desc = opdesc->MutableOutputDesc("y");
  EXPECT_EQ(output_desc->GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {2, 100, 4};
  EXPECT_EQ(output_desc->MutableShape().GetDims(), expected_output_shape);
}