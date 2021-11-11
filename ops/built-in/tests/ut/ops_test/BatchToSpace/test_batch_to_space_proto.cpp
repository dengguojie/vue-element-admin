#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "transformation_ops.h"
#include "array_ops.h"
#include "op_proto_test_util.h"

using namespace ge;
using namespace op;

class batch_to_space_infer_test : public testing::Test {
  protected:
  static void SetUpTestCase() {
    std::cout << "batch_to_space_infer_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "batch_to_space_infer_test TearDown" << std::endl;
  }
  template <typename T>
  static Operator GetConstNode(const std::vector<int64_t>& const_shape,
                              const std::vector<T>& const_value,
                              const std::string& const_name,
                              const ge::Format& const_format) {
    auto const_size = const_value.size();
    constexpr ge::DataType const_dtype = std::is_same<T, float>::value ? ge::DT_FLOAT : ge::DT_INT32;
    TensorDesc const_desc(ge::Shape(const_shape), const_format, const_dtype);
    Tensor const_tensor(const_desc);
    const_tensor.SetData(reinterpret_cast<const uint8_t*>(const_value.data()), const_size * sizeof(T));
    auto const_op = op::Const(const_name.c_str()).set_attr_value(const_tensor);
    return const_op;
  }
};

TEST_F(batch_to_space_infer_test, batch_to_space_infer_test_1) {
  // set input info
  auto input_shape = vector<int64_t>({10, 11, 20, 16});
  std::vector<std::pair<int64_t,int64_t>> input_range = {{10, 10}, {11, 11}, {20, 20}, {16, 16}};
  const std::vector<int64_t> v_scales = {1, 1, 2, 2};
  const std::vector<int64_t> dims = {4};
  ge::Shape shape(dims);
  auto const_format = ge::FORMAT_NHWC;
  auto crops = GetConstNode(dims, v_scales, "crops", const_format); 
  auto block_size = 2;
  auto test_format = ge::FORMAT_NHWC;
  // create desc
  auto input_x_desc = create_desc_shape_range(input_shape, ge::DT_FLOAT16, test_format,
  input_shape, test_format, input_range);
  // expect result
  std::vector<int64_t> expected_shape = {2, -1, -1, 16};
  // new op and do infershape
  ge::op::BatchToSpace op;
  op.UpdateInputDesc("x", input_x_desc);
  op.set_input_crops(crops);
  op.set_attr_block_size(block_size);
  auto ret = op.InferShapeAndType();
  // check result
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_shape);
  ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
TEST_F(batch_to_space_infer_test, batch_to_space_infer_test_2) {
  // set input info
  auto input_shape = vector<int64_t>({10, 11, 20, 16});
  std::vector<std::pair<int64_t,int64_t>> input_range = {{10, 10}, {11, 11}, {20, 20}, {16, 16}};
  const std::vector<int64_t> v_scales = {1, 1, 2, 2};
  const std::vector<int64_t> dims = {4};
  ge::Shape shape(dims);
  auto const_format = ge::FORMAT_NCHW;
  auto crops = GetConstNode(dims, v_scales, "crops", const_format); 
  auto block_size = 2;
  auto test_format = ge::FORMAT_NCHW;
  // create desc
  auto input_x_desc = create_desc_shape_range(input_shape, ge::DT_FLOAT16, test_format,
  input_shape, test_format, input_range);
  // expect result
  std::vector<int64_t> expected_shape = {2, 11, -1, -1};
  // new op and do infershape
  ge::op::BatchToSpace op;
  op.UpdateInputDesc("x", input_x_desc);
  op.set_input_crops(crops);
  op.set_attr_block_size(block_size);
  auto ret = op.InferShapeAndType();
  // check result
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_shape);
  ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
TEST_F(batch_to_space_infer_test, batch_to_space_infer_test_3) {
  // set input info
  auto input_shape = vector<int64_t>({10, 11, 20});
  std::vector<std::pair<int64_t,int64_t>> input_range = {{10, 10}, {11, 11}, {20, 20}};
  const std::vector<int64_t> v_scales = {1, 1, 2, 2};
  const std::vector<int64_t> dims = {4};
  ge::Shape shape(dims);
  auto const_format = ge::FORMAT_NCHW;
  auto crops = GetConstNode(dims, v_scales, "crops", const_format); 
  auto block_size = 2;
  auto test_format = ge::FORMAT_NCHW;
  // create desc
  auto input_x_desc = create_desc_shape_range(input_shape, ge::DT_FLOAT16, test_format,
  input_shape, test_format, input_range);
  // new op and do infershape
  ge::op::BatchToSpace op;
  op.UpdateInputDesc("x", input_x_desc);
  op.set_input_crops(crops);
  op.set_attr_block_size(block_size);
  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
TEST_F(batch_to_space_infer_test, batch_to_space_infer_test_4) {
  // set input info
  auto input_shape = vector<int64_t>({10, 11, 20, 16});
  std::vector<std::pair<int64_t,int64_t>> input_range = {{10, 10}, {11, 11}, {20, 20}, {16, 16}};
  const std::vector<int64_t> v_scales = {1, 1, 2, 2};
  const std::vector<int64_t> dims = {4};
  ge::Shape shape(dims);
  auto const_format = ge::FORMAT_NCHW;
  auto crops = GetConstNode(dims, v_scales, "crops", const_format); 
  auto test_format = ge::FORMAT_NCHW;
  // create desc
  auto input_x_desc = create_desc_shape_range(input_shape, ge::DT_FLOAT16, test_format,
  input_shape, test_format, input_range);
  // new op and do infershape
  ge::op::BatchToSpace op;
  op.UpdateInputDesc("x", input_x_desc);
  op.set_input_crops(crops);
  auto ret = op.InferShapeAndType();
  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
  ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}
TEST_F(batch_to_space_infer_test, batch_to_space_infer_test_5) {
  // set input info
  auto input_shape = vector<int64_t>({10, 11, 20, 16});
  std::vector<std::pair<int64_t,int64_t>> input_range = {{10, 10}, {11, 11}, {20, 20}, {16, 16}};
  const std::vector<int64_t> v_scales = {1, 1, 2, 2};
  const std::vector<int64_t> dims = {4};
  ge::Shape shape(dims);
  auto const_format = ge::FORMAT_NHWC;
  auto crops = GetConstNode(dims, v_scales, "crops", const_format); 
  auto block_size = 1;
  auto test_format = ge::FORMAT_NHWC;
  // create desc
  auto input_x_desc = create_desc_shape_range(input_shape, ge::DT_FLOAT16, test_format,
  input_shape, test_format, input_range);
  // new op and do infershape
  ge::op::BatchToSpace op;
  op.UpdateInputDesc("x", input_x_desc);
  op.set_input_crops(crops);
  op.set_attr_block_size(block_size);
  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}



