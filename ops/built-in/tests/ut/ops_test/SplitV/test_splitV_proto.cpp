#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "split_combination_ops.h"
#include "array_ops.h"


class SplitVTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SplitVTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SplitVTest TearDown" << std::endl;
  }
};

TEST_F(SplitVTest, splitV_test_infershape_diff_test_1) {
  ge::op::SplitV op;

  ge::Tensor constTensorSizeSplits;
  ge::TensorDesc constDescSizeSplits(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDescSizeSplits.SetSize(2 * sizeof(int32_t));
  constTensorSizeSplits.SetTensorDesc(constDescSizeSplits);
  int32_t constDataSizeSplits[2] = {64, 64};
  constTensorSizeSplits.SetData((uint8_t*)constDataSizeSplits, 2 * sizeof(int32_t));
  auto constSizeSplits = ge::op::Constant().set_attr_value(constTensorSizeSplits);
  op.set_input_size_splits(constSizeSplits);
  op.UpdateInputDesc("size_splits", constDescSizeSplits);

  ge::Tensor constTensorSplitDim;
  ge::TensorDesc constDescSplitDim(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDescSplitDim.SetSize(1 * sizeof(int32_t));
  constTensorSplitDim.SetTensorDesc(constDescSplitDim);
  int32_t constDataSplitDim[1] = {3};
  constTensorSplitDim.SetData((uint8_t*)constDataSplitDim, 1 * sizeof(int32_t));
  auto constSplitDim = ge::op::Constant().set_attr_value(constTensorSplitDim);
  op.set_input_split_dim(constSplitDim);
  op.UpdateInputDesc("split_dim", constDescSplitDim);

  // op.UpdateInputDesc("split_dim", create_desc_shape_range({1}, ge::DT_INT32, ge::FORMAT_ND, {1}, ge::FORMAT_ND,{{1,1}}));

  op.UpdateInputDesc("x", create_desc_shape_range({1, -1, -1, 128}, ge::DT_INT32, ge::FORMAT_ND,
                     {1, -1, -1, 128}, ge::FORMAT_ND,{{1, -1}, {1, -1}, {1, -1}, {128, 128}}));
  op.SetAttr("num_split", 2);

  auto output_desc = op.GetOutputDescByName("y");
  auto output_shape = output_desc.GetShape().GetDims();

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(SplitVTest, splitV_test_infershape_diff_test_2) {
  ge::op::SplitV op;

  op.UpdateInputDesc("size_splits", create_desc_shape_range({1}, ge::DT_INT32, ge::FORMAT_ND, {1}, ge::FORMAT_ND,{{64,64}}));

  ge::Tensor constTensorSplitDim;
  ge::TensorDesc constDescSplitDim(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDescSplitDim.SetSize(1 * sizeof(int32_t));
  constTensorSplitDim.SetTensorDesc(constDescSplitDim);
  int32_t constDataSplitDim[1] = {3};
  constTensorSplitDim.SetData((uint8_t*)constDataSplitDim, 1 * sizeof(int32_t));
  auto constSplitDim = ge::op::Constant().set_attr_value(constTensorSplitDim);
  op.set_input_split_dim(constSplitDim);
  op.UpdateInputDesc("split_dim", constDescSplitDim);

  // op.UpdateInputDesc("split_dim", create_desc_shape_range({1}, ge::DT_INT32, ge::FORMAT_ND, {1}, ge::FORMAT_ND,{{1,1}}));

  op.UpdateInputDesc("x", create_desc_shape_range({1, -1, -1, 128}, ge::DT_INT32, ge::FORMAT_ND,
                     {1, -1, -1, 128}, ge::FORMAT_ND,{{1, -1}, {1, -1}, {1, -1}, {128, 128}}));
  op.SetAttr("num_split", 2);

  auto output_desc = op.GetOutputDescByName("y");
  auto output_shape = output_desc.GetShape().GetDims();

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
