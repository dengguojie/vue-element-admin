#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "elewise_calculation_ops.h"
#include "array_ops.h"
#include "op_proto_test_util.h"

using namespace ge;
using namespace op;

class arg_min_infer_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "arg_min_infer_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "arg_min_infer_test TearDown" << std::endl;
  }
};

// input dimension is not const
TEST_F(arg_min_infer_test, arg_min_infer_test_1) {
  // set input info
  auto shape_x1 = vector<int64_t>({-1, -1, -1, 5});
  std::vector<std::pair<int64_t,int64_t>> range_x1 = {{1, -1}, {2, 3}, {2, 3}, {5, 5}};
  auto shape_x2 = vector<int64_t>({1});
  std::vector<std::pair<int64_t,int64_t>> range_x2 = {{1, 1}};
  auto test_format = ge::FORMAT_NHWC;

  // expect result
  std::vector<int64_t> expected_shape = {-1, -1, -1};
  std::vector<std::pair<int64_t,int64_t>> expected_range = {{0, -1}, {0, -1}, {0, -1}};

  // create desc
  auto tensor_desc_x1 = create_desc_shape_range(shape_x1, ge::DT_FLOAT, test_format,
                                                shape_x1, test_format, range_x1);
  auto tensor_desc_x2 = create_desc_shape_range(shape_x2, ge::DT_INT32, test_format,
                                                shape_x2, test_format, range_x2);
  // new op and do infershape
  ge::op::ArgMin op;
  op.UpdateInputDesc("x", tensor_desc_x1);
  op.UpdateInputDesc("dimension", tensor_desc_x2);
  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_shape);
  std::vector<std::pair<int64_t,int64_t>> output_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_range), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_range, expected_range);
}

// input dimension is const
TEST_F(arg_min_infer_test, arg_min_infer_test_2) {
  //new op
  ge::op::ArgMin op;
  // set input info
  ge::TensorDesc tensor_desc_x(ge::Shape({1, 5}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("x", tensor_desc_x);
  op.SetAttr("dtype", ge::DT_INT64);

  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape({1}), ge::FORMAT_ND, ge::DT_INT32);
  constDesc.SetSize(1 * sizeof(int32_t));
  constTensor.SetTensorDesc(constDesc);
  int32_t constData[1] = {3};
  constTensor.SetData((uint8_t*)constData, 1 * sizeof(int32_t));
  auto const0 = ge::op::Constant().set_attr_value(constTensor);
  op.set_input_dimension(const0);
  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}