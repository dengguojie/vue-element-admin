#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "image_ops.h"
#include "array_ops.h"
#include "op_proto_test_util.h"
#include "common/utils/ut_op_util.h"

using namespace ge;
using namespace op;
using namespace ut_util;

class resize_bilinear_v2_infer_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "resize_bilinear_v2_infer_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "resize_bilinear_v2_infer_test TearDown" << std::endl;
  }
};

TEST_F(resize_bilinear_v2_infer_test, resize_bilinear_v2_infer_test_1) {
  // input x shape {-1, -1, -1, 5}
  // input x range {{1, -1}, {2, 3}, {2, 3}, {5, 5}}
  // input size shape {2}
  // input size range {{2, 2}}
  // attr  NA
  // foramt NHWC
  // output shape  {5, 6}
  // output shape {{5, 5}, {6, 6}}

  // set input info
  auto shape_x1 = vector<int64_t>({-1, -1, -1, 5});
  std::vector<std::pair<int64_t, int64_t>> range_x1 = {{1, -1}, {2, 3}, {2, 3}, {5, 5}};
  auto shape_x2 = vector<int64_t>({2});
  std::vector<std::pair<int64_t, int64_t>> range_x2 = {{2, 2}};
  auto test_format = ge::FORMAT_NHWC;

  // expect result
  std::vector<int64_t> expected_shape = {-1, -1, -1, 5};
  std::vector<std::pair<int64_t, int64_t>> expected_range = {{1, -1}, {0, -1}, {0, -1}, {5, 5}};

  // create desc
  auto tensor_desc_x1 = create_desc_shape_range(shape_x1, ge::DT_FLOAT, test_format, shape_x1, test_format, range_x1);
  auto tensor_desc_x2 = create_desc_shape_range(shape_x2, ge::DT_INT32, test_format, shape_x2, test_format, range_x2);
  // new op and do infershape
  ge::op::ResizeBilinearV2 op;
  op.UpdateInputDesc("x", tensor_desc_x1);
  op.UpdateInputDesc("size", tensor_desc_x2);
  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc(0);
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_shape);
  std::vector<std::pair<int64_t, int64_t>> output_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_range), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_range, expected_range);
}

TEST_F(resize_bilinear_v2_infer_test, resize_bilinear_v2_infer_test_2) {
  // input x shape {-1, -1, -1, 5}
  // input x range {{1, -1}, {2, 3}, {2, 3}, {5, 5}}
  // input size shape {2}
  // input size range {{2, 2}}
  // attr  NA
  // foramt NCHW
  // output shape  {5, 6}
  // output shape {{5, 5}, {6, 6}}

  // set input info
  auto shape_x1 = vector<int64_t>({-1, -1, -1, -1});
  std::vector<std::pair<int64_t, int64_t>> range_x1 = {{1, -1}, {1, 2}, {1, 3}, {1, 4}};
  auto shape_x2 = vector<int64_t>({2});
  std::vector<std::pair<int64_t, int64_t>> range_x2 = {{2, 2}};
  auto test_format = ge::FORMAT_NCHW;

  // expect result
  std::vector<int64_t> expected_shape = {-1, -1, 1024, 1024};
  std::vector<std::pair<int64_t, int64_t>> expected_range = {{1, -1}, {1, 2}, {1024, 1024}, {1024, 1024}};

  // create desc
  auto tensor_desc_x1 = create_desc_shape_range(shape_x1, ge::DT_FLOAT, test_format, shape_x1, test_format, range_x1);

  // new a const for size
  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape(shape_x2), ge::FORMAT_ND, ge::DT_INT32);
  constDesc.SetSize(2 * sizeof(int32_t));
  constTensor.SetTensorDesc(constDesc);
  vector<int32_t> constData = {1024, 1024};
  constTensor.SetData((uint8_t*)constData.data(), 2 * sizeof(int32_t));
  auto const_op = ge::op::Constant().set_attr_value(constTensor);
  // delete [] constData;

  // new op and do infershape
  ge::op::ResizeBilinearV2 op;
  op.set_input_size(const_op);
  op.UpdateInputDesc("x", tensor_desc_x1);
  auto descBegin = op.GetInputDesc(0);
  descBegin.SetDataType(ge::DT_INT32);
  descBegin.SetShape(ge::Shape(shape_x2));
  op.UpdateInputDesc("size", descBegin);
  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc(0);
  // EXPECT_EQ(output_desc.GetShape().GetDims(), expected_shape);
  std::vector<std::pair<int64_t, int64_t>> output_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_range), ge::GRAPH_SUCCESS);
  // EXPECT_EQ(output_range, expected_range);
}

TEST_F(resize_bilinear_v2_infer_test, resize_bilinear_v2_static_shape_nchw) {
  // input x info
  auto input_x_shape = vector<int64_t>({3, 5, 16, 16});
  auto input_x_dtype = DT_FLOAT16;
  auto input_x_format = FORMAT_NCHW;
  std::vector<std::pair<int64_t, int64_t>> input_x_shape_range;
  // input axes info
  auto input_size_shape = vector<int64_t>({2});
  auto input_size_dtype = DT_INT32;
  std::vector<std::pair<int64_t, int64_t>> input_size_value_range = {{1, -1}, {2, 3}};
  vector<uint32_t> resize_value = {1, 2};
  // expect result info
  std::vector<int64_t> expected_output_shape = {3, 5, 1, 2};
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range;

  // gen ReduceSum op
  auto test_op = op::ResizeBilinearV2("ResizeBilinearV2");
  TENSOR_INPUT_WITH_SHAPE(test_op, x, input_x_shape, input_x_dtype, input_x_format, input_x_shape_range);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(test_op, size, input_size_shape, input_size_dtype, FORMAT_ND, resize_value);
  auto op_desc = OpDescUtils::GetOpDescFromOperator(test_op);
  auto size_desc = op_desc->MutableInputDesc(1);
  size_desc->SetValueRange(input_size_value_range);

  // run InferShapeAndType
  test_op.InferShapeAndType();

  // cmp the result
  auto output_desc = test_op.GetOutputDesc(0);
  EXPECT_EQ(output_desc.GetDataType(), DT_FLOAT);
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  (void)output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}

TEST_F(resize_bilinear_v2_infer_test, resize_bilinear_v2_static_shape_nhwc) {
  // input x info
  auto input_x_shape = vector<int64_t>({-1, 5, 16, 16});
  auto input_x_dtype = DT_FLOAT;
  auto input_x_format = FORMAT_NHWC;
  std::vector<std::pair<int64_t, int64_t>> input_x_shape_range = {{1, 22}, {5, 5}, {16, 16}, {16, 16}};
  // input axes info
  auto input_size_shape = vector<int64_t>({2});
  auto input_size_dtype = DT_INT32;
  std::vector<std::pair<int64_t, int64_t>> input_size_value_range = {{1, -1}, {2, 3}};
  vector<uint32_t> resize_value = {1, 2};
  // expect result info
  std::vector<int64_t> expected_output_shape = {-1, 1, 2, 16};
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{1, 22}, {1, 1}, {2, 2}, {16, 16}};

  // gen ReduceSum op
  auto test_op = op::ResizeBilinearV2("ResizeBilinearV2");
  TENSOR_INPUT_WITH_SHAPE(test_op, x, input_x_shape, input_x_dtype, input_x_format, input_x_shape_range);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(test_op, size, input_size_shape, input_size_dtype, FORMAT_ND, resize_value);
  auto op_desc = OpDescUtils::GetOpDescFromOperator(test_op);
  auto size_desc = op_desc->MutableInputDesc(1);
  size_desc->SetValueRange(input_size_value_range);

  // run InferShapeAndType
  test_op.InferShapeAndType();

  // cmp the result
  auto output_desc = test_op.GetOutputDesc(0);
  EXPECT_EQ(output_desc.GetDataType(), DT_FLOAT);
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  (void)output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}

TEST_F(resize_bilinear_v2_infer_test, resize_bilinear_v2_dy_shape_with_value_nhwc) {
  // input x info
  auto input_x_shape = vector<int64_t>({3, 5, 16, 16});
  auto input_x_dtype = DT_FLOAT16;
  auto input_x_format = FORMAT_NHWC;
  std::vector<std::pair<int64_t, int64_t>> input_x_shape_range;
  // input axes info
  auto input_size_shape = vector<int64_t>({2});
  auto input_size_dtype = DT_INT32;
  std::vector<std::pair<int64_t, int64_t>> input_size_value_range = {{1, -1}, {2, 3}};
  vector<uint32_t> resize_value = {1, 2};
  // expect result info
  std::vector<int64_t> expected_output_shape = {3, -1, -1, 16};
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{3, 3}, {1, -1}, {2, 3}, {16, 16}};

  // gen ReduceSum op
  auto test_op = op::ResizeBilinearV2("ResizeBilinearV2");
  TENSOR_INPUT_WITH_SHAPE(test_op, x, input_x_shape, input_x_dtype, input_x_format, input_x_shape_range);
  TENSOR_INPUT_WITH_SHAPE(test_op, size, input_size_shape, input_size_dtype, FORMAT_ND, {});
  auto op_desc = OpDescUtils::GetOpDescFromOperator(test_op);
  auto size_desc = op_desc->MutableInputDesc(1);
  size_desc->SetValueRange(input_size_value_range);

  // run InferShapeAndType
  test_op.InferShapeAndType();

  // cmp the result
  auto output_desc = test_op.GetOutputDesc(0);
  EXPECT_EQ(output_desc.GetDataType(), DT_FLOAT);
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  (void)output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}

TEST_F(resize_bilinear_v2_infer_test, resize_bilinear_v2_dy_shape_with_1value_nhwc) {
  // input x info
  auto input_x_shape = vector<int64_t>({3, 5, 16, 16});
  auto input_x_dtype = DT_FLOAT;
  auto input_x_format = FORMAT_NHWC;
  std::vector<std::pair<int64_t, int64_t>> input_x_shape_range;
  // input axes info
  auto input_size_shape = vector<int64_t>({2});
  auto input_size_dtype = DT_INT32;
  std::vector<std::pair<int64_t, int64_t>> input_size_value_range = {{2, 3}};
  vector<uint32_t> resize_value = {1, 2};
  // expect result info
  std::vector<int64_t> expected_output_shape = {3, -1, -1, 16};
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{3, 3}, {0, -1}, {0, -1}, {16, 16}};

  // gen ReduceSum op
  auto test_op = op::ResizeBilinearV2("ResizeBilinearV2");
  TENSOR_INPUT_WITH_SHAPE(test_op, x, input_x_shape, input_x_dtype, input_x_format, input_x_shape_range);
  TENSOR_INPUT_WITH_SHAPE(test_op, size, input_size_shape, input_size_dtype, FORMAT_ND, {});
  auto op_desc = OpDescUtils::GetOpDescFromOperator(test_op);
  auto size_desc = op_desc->MutableInputDesc(1);
  size_desc->SetValueRange(input_size_value_range);

  // run InferShapeAndType
  test_op.InferShapeAndType();

  // cmp the result
  auto output_desc = test_op.GetOutputDesc(0);
  EXPECT_EQ(output_desc.GetDataType(), DT_FLOAT);
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  (void)output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}

TEST_F(resize_bilinear_v2_infer_test, resize_bilinear_v2_unknownrangk_nhwc) {
  // input x info
  auto input_x_shape = vector<int64_t>({-2});
  auto input_x_dtype = DT_FLOAT16;
  auto input_x_format = FORMAT_NHWC;
  std::vector<std::pair<int64_t, int64_t>> input_x_shape_range;
  // input axes info
  auto input_size_shape = vector<int64_t>({2});
  auto input_size_dtype = DT_INT32;
  std::vector<std::pair<int64_t, int64_t>> input_size_value_range = {{2, 3}};
  vector<uint32_t> resize_value = {1, 2};
  // expect result info
  std::vector<int64_t> expected_output_shape = {-1, -1, -1, -1};
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{1, -1}, {0, -1}, {0, -1}, {1, -1}};

  // gen ReduceSum op
  auto test_op = op::ResizeBilinearV2("ResizeBilinearV2");
  TENSOR_INPUT_WITH_SHAPE(test_op, x, input_x_shape, input_x_dtype, input_x_format, input_x_shape_range);
  TENSOR_INPUT_WITH_SHAPE(test_op, size, input_size_shape, input_size_dtype, FORMAT_ND, {});
  auto op_desc = OpDescUtils::GetOpDescFromOperator(test_op);
  auto size_desc = op_desc->MutableInputDesc(1);
  size_desc->SetValueRange(input_size_value_range);

  // run InferShapeAndType
  test_op.InferShapeAndType();

  // cmp the result
  auto output_desc = test_op.GetOutputDesc(0);
  EXPECT_EQ(output_desc.GetDataType(), DT_FLOAT);
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  (void)output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);
}
