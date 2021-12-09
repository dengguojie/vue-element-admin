#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "nn_pooling_ops.h"
#include "array_ops.h"
#include "op_proto_test_util.h"

using namespace ge;
using namespace op;

class MaxPoolV2Test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "MaxPoolV2Proto SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MaxPoolV2Proto TearDown" << std::endl;
  }
};

TEST_F(MaxPoolV2Test, MaxPoolV2_infer_shape_ksize_failed) {
  ge::op::MaxPoolV2 op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}};
  auto tensor_desc1 = create_desc_shape_range({2,2},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {2,2},
                                             ge::FORMAT_ND, shape_range);
  auto tensor_desc2 = create_desc_shape_range({2},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {2},
                                             ge::FORMAT_ND, {{}});
  auto tensor_desc3 = create_desc_shape_range({},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {},
                                             ge::FORMAT_ND, {{}});

  op.UpdateInputDesc("x", tensor_desc1);
  op.UpdateInputDesc("ksize", tensor_desc1);
  op.UpdateInputDesc("strides", tensor_desc1);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV2Test, MaxPoolV2_infer_shape_ksize_failed2) {
  ge::op::MaxPoolV2 op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}};
  auto tensor_desc1 = create_desc_shape_range({2,2},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {2,2},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x", tensor_desc1);

  ge::TensorDesc const_desc(ge::Shape({2}), ge::FORMAT_ND, ge::DT_INT32);
  int32_t const_value[2] = {6, 7};
  auto const_op = ge::op::Constant().set_attr_value(
  ge::Tensor(const_desc, (uint8_t *)const_value, 2 * sizeof(int32_t)));
  op.set_input_ksize(const_op);
  op.UpdateInputDesc("ksize", const_desc);

  op.UpdateInputDesc("strides", tensor_desc1);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV2Test, MaxPoolV2_infer_shape_strides_failed) {
  ge::op::MaxPoolV2 op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}};
  auto tensor_desc1 = create_desc_shape_range({2,2},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {2,2},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x", tensor_desc1);

  ge::TensorDesc const_desc(ge::Shape({2}), ge::FORMAT_ND, ge::DT_INT32);
  int32_t const_value[4] = {6, 7, 8, 9};
  auto const_op = ge::op::Constant().set_attr_value(
  ge::Tensor(const_desc, (uint8_t *)const_value, 4 * sizeof(int32_t)));
  op.set_input_ksize(const_op);
  op.UpdateInputDesc("ksize", const_desc);

  op.UpdateInputDesc("strides", tensor_desc1);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV2Test, MaxPoolV2_infer_shape_strides_failed2) {
  ge::op::MaxPoolV2 op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}};
  auto tensor_desc1 = create_desc_shape_range({2,2},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {2,2},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x", tensor_desc1);

  ge::TensorDesc const_desc(ge::Shape({4}), ge::FORMAT_ND, ge::DT_INT32);
  int32_t const_value[4] = {6, 7, 8, 9};
  auto const_op = ge::op::Constant().set_attr_value(
  ge::Tensor(const_desc, (uint8_t *)const_value, 4 * sizeof(int32_t)));
  op.set_input_ksize(const_op);
  op.UpdateInputDesc("ksize", const_desc);

  ge::TensorDesc const_desc1(ge::Shape({2}), ge::FORMAT_ND, ge::DT_INT32);
  int32_t const_value1[2] = {6, 7};
  auto const_op1 = ge::op::Constant().set_attr_value(
  ge::Tensor(const_desc1, (uint8_t *)const_value1, 2 * sizeof(int32_t)));
  op.set_input_strides(const_op1);
  op.UpdateInputDesc("strides", const_desc1);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV2Test, MaxPoolV2_infer_shape_format_failed) {
  ge::op::MaxPoolV2 op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}};
  auto tensor_desc1 = create_desc_shape_range({2,2},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {2,2},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x", tensor_desc1);

  ge::TensorDesc const_desc(ge::Shape({4}), ge::FORMAT_ND, ge::DT_INT32);
  int32_t const_value[4] = {6, 7, 8, 9};
  auto const_op = ge::op::Constant().set_attr_value(
  ge::Tensor(const_desc, (uint8_t *)const_value, 4 * sizeof(int32_t)));
  op.set_input_ksize(const_op);
  op.UpdateInputDesc("ksize", const_desc);

  ge::TensorDesc const_desc1(ge::Shape({4}), ge::FORMAT_ND, ge::DT_INT32);
  int32_t const_value1[4] = {6, 7, 8, 9};
  auto const_op1 = ge::op::Constant().set_attr_value(
  ge::Tensor(const_desc1, (uint8_t *)const_value1, 4 * sizeof(int32_t)));
  op.set_input_strides(const_op1);
  op.UpdateInputDesc("strides", const_desc1);

  op.set_attr_data_format("AAAA");
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV2Test, MaxPoolV2_infer_shape_ksizeList_failed) {
  ge::op::MaxPoolV2 op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}};
  auto tensor_desc1 = create_desc_shape_range({2,2},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {2,2},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x", tensor_desc1);

  ge::TensorDesc const_desc(ge::Shape({4}), ge::FORMAT_ND, ge::DT_INT32);
  int32_t const_value[4] = {6, 7, 8, 9};
  auto const_op = ge::op::Constant().set_attr_value(
  ge::Tensor(const_desc, (uint8_t *)const_value, 4 * sizeof(int32_t)));
  op.set_input_ksize(const_op);
  op.UpdateInputDesc("ksize", const_desc);

  ge::TensorDesc const_desc1(ge::Shape({4}), ge::FORMAT_ND, ge::DT_INT32);
  int32_t const_value1[4] = {6, 7, 8, 9};
  auto const_op1 = ge::op::Constant().set_attr_value(
  ge::Tensor(const_desc1, (uint8_t *)const_value1, 4 * sizeof(int32_t)));
  op.set_input_strides(const_op1);
  op.UpdateInputDesc("strides", const_desc1);

  op.set_attr_data_format("NHWC");
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV2Test, MaxPoolV2_infer_shape_ksizeList_failed2) {
  ge::op::MaxPoolV2 op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}};
  auto tensor_desc1 = create_desc_shape_range({2,2},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {2,2},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x", tensor_desc1);

  ge::TensorDesc const_desc(ge::Shape({4}), ge::FORMAT_ND, ge::DT_INT32);
  int32_t const_value[4] = {6, 7, 8, 9};
  auto const_op = ge::op::Constant().set_attr_value(
  ge::Tensor(const_desc, (uint8_t *)const_value, 4 * sizeof(int32_t)));
  op.set_input_ksize(const_op);
  op.UpdateInputDesc("ksize", const_desc);

  ge::TensorDesc const_desc1(ge::Shape({4}), ge::FORMAT_ND, ge::DT_INT32);
  int32_t const_value1[4] = {6, 7, 8, 9};
  auto const_op1 = ge::op::Constant().set_attr_value(
  ge::Tensor(const_desc1, (uint8_t *)const_value1, 4 * sizeof(int32_t)));
  op.set_input_strides(const_op1);
  op.UpdateInputDesc("strides", const_desc1);

  op.set_attr_data_format("NCHW");
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV2Test, MaxPoolV2_infer_shape_padding_failed) {
  ge::op::MaxPoolV2 op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}};
  auto tensor_desc1 = create_desc_shape_range({2,2},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {2,2},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x", tensor_desc1);

  ge::TensorDesc const_desc(ge::Shape({4}), ge::FORMAT_ND, ge::DT_INT32);
  int32_t const_value[4] = {1, 1, 1, 1};
  auto const_op = ge::op::Constant().set_attr_value(
  ge::Tensor(const_desc, (uint8_t *)const_value, 4 * sizeof(int32_t)));
  op.set_input_ksize(const_op);
  op.UpdateInputDesc("ksize", const_desc);

  ge::TensorDesc const_desc1(ge::Shape({4}), ge::FORMAT_ND, ge::DT_INT32);
  int32_t const_value1[4] = {1, 1, 1, 1};
  auto const_op1 = ge::op::Constant().set_attr_value(
  ge::Tensor(const_desc1, (uint8_t *)const_value1, 4 * sizeof(int32_t)));
  op.set_input_strides(const_op1);
  op.UpdateInputDesc("strides", const_desc1);

  op.set_attr_data_format("NCHW");
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV2Test, MaxPoolV2_infer_shape_padding_failed2) {
  ge::op::MaxPoolV2 op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}};
  auto tensor_desc1 = create_desc_shape_range({2,2},
                                             ge::DT_INT64, ge::FORMAT_ND,
                                             {2,2},
                                             ge::FORMAT_ND, shape_range);
  op.UpdateInputDesc("x", tensor_desc1);

  ge::TensorDesc const_desc(ge::Shape({4}), ge::FORMAT_ND, ge::DT_INT32);
  int32_t const_value[4] = {1, 1, 1, 1};
  auto const_op = ge::op::Constant().set_attr_value(
  ge::Tensor(const_desc, (uint8_t *)const_value, 4 * sizeof(int32_t)));
  op.set_input_ksize(const_op);
  op.UpdateInputDesc("ksize", const_desc);

  ge::TensorDesc const_desc1(ge::Shape({4}), ge::FORMAT_ND, ge::DT_INT32);
  int32_t const_value1[4] = {1, 1, 1, 1};
  auto const_op1 = ge::op::Constant().set_attr_value(
  ge::Tensor(const_desc1, (uint8_t *)const_value1, 4 * sizeof(int32_t)));
  op.set_input_strides(const_op1);
  op.UpdateInputDesc("strides", const_desc1);

  op.set_attr_data_format("NCHW");
  op.set_attr_padding("AAAA");
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV2Test, InfershapeMaxPoolV2_001) {
  ge::op::MaxPoolV2 op;
  auto tensor_desc1 = create_desc_with_ori({2, 2, 2, 2}, ge::DT_INT64, ge::FORMAT_NHWC, {2, 2, 2, 2}, ge::FORMAT_NHWC);
  op.UpdateInputDesc("x", tensor_desc1);

  ge::TensorDesc const_desc(ge::Shape({4}), ge::FORMAT_ND, ge::DT_INT32);
  int32_t const_value[4] = {1, 1, 1, 1};
  auto const_op = ge::op::Constant().set_attr_value(ge::Tensor(const_desc, (uint8_t*)const_value, 4 * sizeof(int32_t)));
  op.set_input_ksize(const_op);
  op.UpdateInputDesc("ksize", const_desc);

  ge::TensorDesc const_desc1(ge::Shape({4}), ge::FORMAT_ND, ge::DT_INT32);
  int32_t const_value1[4] = {1, 1, 1, 1};
  auto const_op1 =
      ge::op::Constant().set_attr_value(ge::Tensor(const_desc1, (uint8_t*)const_value1, 4 * sizeof(int32_t)));
  op.set_input_strides(const_op1);
  op.UpdateInputDesc("strides", const_desc1);

  op.SetAttr("data_format", true);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaxPoolV2Test, InfershapeMaxPoolV2_002) {
  ge::op::MaxPoolV2 op;
  auto tensor_desc1 = create_desc_with_ori({2, 2, 2, 2}, ge::DT_INT64, ge::FORMAT_NHWC, {2, 2, 2, 2}, ge::FORMAT_NHWC);
  op.UpdateInputDesc("x", tensor_desc1);

  ge::TensorDesc const_desc(ge::Shape({4}), ge::FORMAT_NHWC, ge::DT_INT32);
  int32_t const_value[4] = {1, 1, 1, 1};
  auto const_op = ge::op::Constant().set_attr_value(ge::Tensor(const_desc, (uint8_t*)const_value, 4 * sizeof(int32_t)));
  op.set_input_ksize(const_op);
  op.UpdateInputDesc("ksize", const_desc);

  ge::TensorDesc const_desc1(ge::Shape({4}), ge::FORMAT_NHWC, ge::DT_INT32);
  int32_t const_value1[4] = {1, 1, 1, 1};
  auto const_op1 =
      ge::op::Constant().set_attr_value(ge::Tensor(const_desc1, (uint8_t*)const_value1, 4 * sizeof(int32_t)));
  op.set_input_strides(const_op1);
  op.UpdateInputDesc("strides", const_desc1);

  op.SetAttr("data_format", "NHWC");
  op.SetAttr("padding", "SAME");
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT64);
  std::vector<int64_t> expected_output_shape = {2, 2, 2, 2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(MaxPoolV2Test, InfershapeMaxPoolV2_003) {
  ge::op::MaxPoolV2 op;
  auto tensor_desc1 = create_desc_with_ori({2, 2, 2, 2}, ge::DT_INT64, ge::FORMAT_NHWC, {2, 2, 2, 2}, ge::FORMAT_NHWC);
  op.UpdateInputDesc("x", tensor_desc1);

  ge::TensorDesc const_desc(ge::Shape({4}), ge::FORMAT_NHWC, ge::DT_INT32);
  int32_t const_value[4] = {1, 1, 1, 1};
  auto const_op = ge::op::Constant().set_attr_value(ge::Tensor(const_desc, (uint8_t*)const_value, 4 * sizeof(int32_t)));
  op.set_input_ksize(const_op);
  op.UpdateInputDesc("ksize", const_desc);

  ge::TensorDesc const_desc1(ge::Shape({4}), ge::FORMAT_NHWC, ge::DT_INT32);
  int32_t const_value1[4] = {1, 1, 1, 1};
  auto const_op1 =
      ge::op::Constant().set_attr_value(ge::Tensor(const_desc1, (uint8_t*)const_value1, 4 * sizeof(int32_t)));
  op.set_input_strides(const_op1);
  op.UpdateInputDesc("strides", const_desc1);

  op.SetAttr("data_format", "NHWC");
  op.SetAttr("padding", "VALID");
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT64);
  std::vector<int64_t> expected_output_shape = {2, 2, 2, 2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(MaxPoolV2Test, InfershapeMaxPoolV2_004) {
  ge::op::MaxPoolV2 op;
  auto tensor_desc1 = create_desc_with_ori({2, 2, 2, 2}, ge::DT_INT64, ge::FORMAT_NHWC, {2, 2, 2, 2}, ge::FORMAT_NHWC);
  op.UpdateInputDesc("x", tensor_desc1);

  ge::TensorDesc const_desc(ge::Shape({4}), ge::FORMAT_NCHW, ge::DT_INT32);
  int32_t const_value[4] = {1, 1, 1, 1};
  auto const_op = ge::op::Constant().set_attr_value(ge::Tensor(const_desc, (uint8_t*)const_value, 4 * sizeof(int32_t)));
  op.set_input_ksize(const_op);
  op.UpdateInputDesc("ksize", const_desc);

  ge::TensorDesc const_desc1(ge::Shape({4}), ge::FORMAT_NCHW, ge::DT_INT32);
  int32_t const_value1[4] = {1, 1, 1, 1};
  auto const_op1 =
      ge::op::Constant().set_attr_value(ge::Tensor(const_desc1, (uint8_t*)const_value1, 4 * sizeof(int32_t)));
  op.set_input_strides(const_op1);
  op.UpdateInputDesc("strides", const_desc1);

  op.SetAttr("data_format", "NCHW");
  op.SetAttr("padding", "SAME");
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT64);
  std::vector<int64_t> expected_output_shape = {2, 2, 2, 2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(MaxPoolV2Test, InfershapeMaxPoolV2_005) {
  ge::op::MaxPoolV2 op;
  auto tensor_desc1 = create_desc_with_ori({2, 2, 2, 2}, ge::DT_INT64, ge::FORMAT_NHWC, {2, 2, 2, 2}, ge::FORMAT_NHWC);
  op.UpdateInputDesc("x", tensor_desc1);

  ge::TensorDesc const_desc(ge::Shape({4}), ge::FORMAT_NCHW, ge::DT_INT64);
  int64_t const_value[4] = {1, 1, 1, 1};
  auto const_op = ge::op::Constant().set_attr_value(ge::Tensor(const_desc, (uint8_t*)const_value, 4 * sizeof(int64_t)));
  op.set_input_ksize(const_op);
  op.UpdateInputDesc("ksize", const_desc);

  ge::TensorDesc const_desc1(ge::Shape({4}), ge::FORMAT_NCHW, ge::DT_INT32);
  int32_t const_value1[4] = {1, 1, 1, 1};
  auto const_op1 =
      ge::op::Constant().set_attr_value(ge::Tensor(const_desc1, (uint8_t*)const_value1, 4 * sizeof(int32_t)));
  op.set_input_strides(const_op1);
  op.UpdateInputDesc("strides", const_desc1);

  op.SetAttr("data_format", "NCHW");
  op.SetAttr("padding", "VALID");
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT64);
  std::vector<int64_t> expected_output_shape = {2, 2, 2, 2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}