#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "array_ops.h"

class UniqueConsecutiveTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "UniqueConsecutive Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "UniqueConsecutive Proto Test TearDown" << std::endl;
    }
};

TEST_F(UniqueConsecutiveTest, unique_consecutive_infer_shape_test1) {
  ge::op::UniqueConsecutive op;

  ge::DataType dtype = ge::DT_INT64;
  ge::Format format = ge::FORMAT_ND;
  
  auto input_tensor = create_desc_with_ori({3,4}, dtype, format, {3,4}, format);
  
  op.UpdateInputDesc("x", input_tensor);

  op.SetAttr("return_idx", false);
  op.SetAttr("return_counts", false);
  op.SetAttr("axis", 1000);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  auto idx_desc = op.GetOutputDesc("idx");
  auto count_desc = op.GetOutputDesc("count");

  EXPECT_EQ(idx_desc.GetShape().GetDimNum(), 0);
  EXPECT_EQ(count_desc.GetShape().GetDimNum(), 0);
}

TEST_F(UniqueConsecutiveTest, unique_consecutive_infer_shape_test2) {
  ge::op::UniqueConsecutive op;

  ge::DataType dtype = ge::DT_INT64;
  ge::Format format = ge::FORMAT_ND;
  
  auto input_tensor = create_desc_with_ori({-2}, dtype, format, {-2}, format);
  
  op.UpdateInputDesc("x", input_tensor);

  op.SetAttr("return_idx", true);
  op.SetAttr("return_counts", true);
  op.SetAttr("axis", 1000);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  auto idx_desc = op.GetOutputDesc("idx");
  auto count_desc = op.GetOutputDesc("count");
  std::vector<int64_t> expected_idx_shape = {-2};
  std::vector<int64_t> expected_count_shape = {-2};
  EXPECT_EQ(idx_desc.GetShape().GetDims(), expected_idx_shape);
  EXPECT_EQ(count_desc.GetShape().GetDims(), expected_idx_shape);
}

TEST_F(UniqueConsecutiveTest, unique_consecutive_infer_shape_test3) {
  ge::op::UniqueConsecutive op;

  ge::DataType dtype = ge::DT_INT64;
  ge::Format format = ge::FORMAT_ND;
  
  auto input_tensor = create_desc_with_ori({-1, -1, 4}, dtype, format, {-1, -1, 4}, format);
  
  op.UpdateInputDesc("x", input_tensor);

  op.SetAttr("return_idx", true);
  op.SetAttr("return_counts", true);
  op.SetAttr("axis", 1000);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  auto idx_desc = op.GetOutputDesc("idx");
  auto count_desc = op.GetOutputDesc("count");
  std::vector<int64_t> expected_idx_shape = {-2};
  std::vector<int64_t> expected_count_shape = {-2};
  EXPECT_EQ(idx_desc.GetShape().GetDims(), expected_idx_shape);
  EXPECT_EQ(count_desc.GetShape().GetDims(), expected_idx_shape);
}