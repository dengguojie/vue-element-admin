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

class ResizeTest : public testing::Test {
    protected:
    static void SetUpTestCase() {
        std::cout << "ResizeTest SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "ResizeTest TearDown" << std::endl;
    }
};

TEST_F(ResizeTest, ResizeTest_test1) {
  ge::op::Resize resize_op;
  auto input_x_shape = vector<int64_t>({3, 5, 16, 16});
  auto input_x_dtype = DT_FLOAT16;
  auto input_x_format = FORMAT_NCHW;
  std::vector<std::pair<int64_t, int64_t>> input_x_shape_range;
  // input axes info
  auto input_size_shape = vector<int64_t>({4});
  auto input_size_dtype = DT_INT32;
  std::vector<std::pair<int64_t, int64_t>> input_size_value_range = {{3, 3}, {5, 5}, {1, -1}, {2, 3}};
  vector<int32_t> resize_value = {3, 5, 1, 2};
  // expect result info
  std::vector<int64_t> expected_output_shape = {3, 5, 1, 2};
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range;


  TENSOR_INPUT_WITH_SHAPE(resize_op, x, input_x_shape, input_x_dtype, input_x_format, input_x_shape_range);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(resize_op, sizes, input_size_shape, input_size_dtype, FORMAT_ND, resize_value);
  auto op_desc = OpDescUtils::GetOpDescFromOperator(resize_op);
  auto size_desc = op_desc->MutableInputDesc("sizes");
  size_desc->SetValueRange(input_size_value_range);

  // run InferShapeAndType
  resize_op.InferShapeAndType();

  // cmp the result
  auto output_desc = resize_op.GetOutputDesc(0);
  EXPECT_EQ(output_desc.GetDataType(), DT_FLOAT16);
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(ResizeTest, ResizeTest_test2) {
    // check NCDHW and scales
    ge::op::Resize resize_op;
    auto x_desc = create_desc_with_original_shape({1, 1, 4, 4, 4}, ge::DT_FLOAT, ge::FORMAT_ND,
                                                  {1, 1, 4, 4, 4}, ge::FORMAT_ND);
    resize_op.UpdateInputDesc("x", x_desc);

    ge::Tensor scalesConstTensor;
    ge::TensorDesc scalesConstDesc(ge::Shape(), ge::FORMAT_ND, ge::DT_FLOAT);
    scalesConstDesc.SetSize(5 * sizeof(float));
    scalesConstTensor.SetTensorDesc(scalesConstDesc);
    float scalesConstData[5] = {1.0, 1.0, 2.0, 2.0, 2.0};
    scalesConstTensor.SetData((uint8_t*)scalesConstData, 5 * sizeof(float));
    auto scales_input = ge::op::Constant().set_attr_value(scalesConstTensor);
    resize_op.set_input_scales(scales_input);
    auto ret = resize_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(ResizeTest, ResizeTest_test3) {
    // check static NCHW and scales
    ge::op::Resize resize_op;
    auto x_desc = create_desc_with_original_shape({1, 1, 4, 4}, ge::DT_FLOAT, ge::FORMAT_NCHW,
                                                  {1, 1, 4, 4}, ge::FORMAT_NCHW);
    resize_op.UpdateInputDesc("x", x_desc);

    ge::Tensor scalesConstTensor;
    ge::TensorDesc scalesConstDesc(ge::Shape(), ge::FORMAT_ND, ge::DT_FLOAT);
    scalesConstDesc.SetSize(4 * sizeof(float));
    scalesConstTensor.SetTensorDesc(scalesConstDesc);
    float scalesConstData[4] = {1.0, 1.0, 2.0, 2.0};
    scalesConstTensor.SetData((uint8_t*)scalesConstData, 4 * sizeof(float));
    auto scales_input = ge::op::Constant().set_attr_value(scalesConstTensor);
    resize_op.set_input_scales(scales_input);
    auto ret = resize_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(ResizeTest, ResizeTest_test4) {
    // check static NCHW and scales
    ge::op::Resize resize_op;
    auto x_desc = create_desc_with_original_shape({1, 1, 4, 4}, ge::DT_FLOAT, ge::FORMAT_ND,
                                                  {1, 1, 4, 4}, ge::FORMAT_ND);
    resize_op.UpdateInputDesc("x", x_desc);

    ge::Tensor scalesConstTensor;
    ge::TensorDesc scalesConstDesc(ge::Shape(), ge::FORMAT_ND, ge::DT_FLOAT);
    scalesConstDesc.SetSize(5 * sizeof(float));
    scalesConstTensor.SetTensorDesc(scalesConstDesc);
    float scalesConstData[5] = {1.0, 1.0, 2.0, 2.0, 2.0};
    scalesConstTensor.SetData((uint8_t*)scalesConstData, 5 * sizeof(float));
    auto scales_input = ge::op::Constant().set_attr_value(scalesConstTensor);
    resize_op.set_input_scales(scales_input);
    auto ret = resize_op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(ResizeTest, ResizeTest_test5) {
  ge::op::Resize resize_op;
  auto input_x_shape = vector<int64_t>({3, 5, 16, 16});
  auto input_x_dtype = DT_FLOAT16;
  auto input_x_format = FORMAT_NCHW;
  std::vector<std::pair<int64_t, int64_t>> input_x_shape_range;
  // input axes info
  auto input_size_shape = vector<int64_t>({5});
  auto input_size_dtype = DT_INT32;
  std::vector<std::pair<int64_t, int64_t>> input_size_value_range = {{3, 3}, {5, 5}, {1, -1}, {2, 3}};
  vector<int32_t> resize_value = {3, 5, 1, 2, 1};
  // expect result info
  std::vector<int64_t> expected_output_shape = {3, 5, 1, 2, 1};
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range;


  TENSOR_INPUT_WITH_SHAPE(resize_op, x, input_x_shape, input_x_dtype, input_x_format, input_x_shape_range);
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(resize_op, sizes, input_size_shape, input_size_dtype, FORMAT_ND, resize_value);
  auto op_desc = OpDescUtils::GetOpDescFromOperator(resize_op);
  auto size_desc = op_desc->MutableInputDesc("sizes");
  size_desc->SetValueRange(input_size_value_range);

  // run InferShapeAndType
  auto ret = resize_op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
