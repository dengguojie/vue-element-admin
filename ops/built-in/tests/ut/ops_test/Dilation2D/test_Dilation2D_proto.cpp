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

class dilation2d_infer_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "dilation2d_infer_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "dilation2d_infer_test TearDown" << std::endl;
  }
};

//REG_OP(Dilation2D)
//    .INPUT(x,TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64, DT_UINT8, DT_INT16, DT_INT8, DT_UINT16}))
//    .INPUT(filter,TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64, DT_UINT8, DT_INT16, DT_INT8, DT_UINT16}))
//    .OUTPUT(y,TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT32, DT_INT64, DT_UINT8, DT_INT16, DT_INT8, DT_UINT16}))
//    .REQUIRED_ATTR(strides, ListInt)
//    .REQUIRED_ATTR(rates, ListInt)
//    .ATTR(padding_mode, String, "SAME")
//    .ATTR(pads, ListInt, {0,0,0,0})
//    .ATTR(ceil_mode, Bool, false)
//    .ATTR(data_format, String, "NHWC")
//    .OP_END_FACTORY_REG(Dilation2D)

TEST_F(dilation2d_infer_test, dilation2d_infer_test_1) {
  std::vector<int64_t> stride_list = vector<int64_t>({1, 2, 3, 1});
  std::vector<int64_t> rate_list = vector<int64_t>({1, 3, 2, 1});
  std::vector<int64_t> pads = vector<int64_t>({0, 0, 0, 0});
  std::string padding_mode = "SAME";
  bool ceil_mode = false;
  std::string data_format = "NHWC";

  // expect result
  std::vector<int64_t> expected_shape = {3, 8, 6, 64};

  // new op and do infershape
  ge::op::Dilation2D op;
  op.UpdateInputDesc("x", create_desc_with_ori({3, 16, 16, 64}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {3, 16, 16, 64}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("filter", create_desc_with_ori({3, 3, 64}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {3, 3, 64}, ge::FORMAT_NHWC));
  op.set_attr_strides(stride_list);
  op.set_attr_rates(rate_list);
  op.set_attr_padding_mode(padding_mode);
  op.set_attr_pads(pads);
  op.set_attr_ceil_mode(ceil_mode);
  op.set_attr_data_format(data_format);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_shape);
}

TEST_F(dilation2d_infer_test, dilation2d_infer_test_2) {
  std::vector<int64_t> stride_list = vector<int64_t>({1, 1, 1, 1});
  std::vector<int64_t> rate_list = vector<int64_t>({1, 1, 1, 1});
  std::vector<int64_t> pads = vector<int64_t>({0, 0, 0, 0});
  std::string padding_mode = "VALID";
  bool ceil_mode = false;
  std::string data_format = "NHWC";

  // expect result
  std::vector<int64_t> expected_shape = {1, 1, 4, 16};

  // new op and do infershape
  ge::op::Dilation2D op;
  op.UpdateInputDesc("x", create_desc_with_ori({1, 3, 6, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {1, 3, 6, 16}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("filter", create_desc_with_ori({3, 3, 16}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {3, 3, 16}, ge::FORMAT_NHWC));
  op.set_attr_strides(stride_list);
  op.set_attr_rates(rate_list);
  op.set_attr_padding_mode(padding_mode);
  op.set_attr_pads(pads);
  op.set_attr_ceil_mode(ceil_mode);
  op.set_attr_data_format(data_format);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_shape);
}

TEST_F(dilation2d_infer_test, dilation2d_infer_test_3) {
  std::vector<int64_t> stride_list = vector<int64_t>({1, 1, 1, 1});
  std::vector<int64_t> rate_list = vector<int64_t>({1, 1, 1, 1});
  std::vector<int64_t> pads = vector<int64_t>({1, 1, 1, 1});
  std::string padding_mode = "CALCULATED";
  bool ceil_mode = false;
  std::string data_format = "NHWC";

  // expect result
  std::vector<int64_t> expected_shape = {1, 17, 17, 1024};

  // new op and do infershape
  ge::op::Dilation2D op;
  op.UpdateInputDesc("x", create_desc_with_ori({1, 17, 17, 1024}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {1, 17, 17, 1024}, ge::FORMAT_NHWC));
  op.UpdateInputDesc("filter", create_desc_with_ori({3, 3, 1024}, ge::DT_FLOAT16, ge::FORMAT_NHWC, {3, 3, 1024}, ge::FORMAT_NHWC));
  op.set_attr_strides(stride_list);
  op.set_attr_rates(rate_list);
  op.set_attr_padding_mode(padding_mode);
  op.set_attr_pads(pads);
  op.set_attr_ceil_mode(ceil_mode);
  op.set_attr_data_format(data_format);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_shape);
}

TEST_F(dilation2d_infer_test, dilation2d_infer_test_4) {
  std::vector<int64_t> stride_list = vector<int64_t>({1, 1, 1, 1});
  std::vector<int64_t> rate_list = vector<int64_t>({1, 1, 1, 1});
  std::vector<int64_t> pads = vector<int64_t>({0, 0, 0, 0});
  std::string padding_mode = "VALID";
  bool ceil_mode = false;
  std::string data_format = "NCHW";

  // expect result
  std::vector<int64_t> expected_shape = {1, 1024, 15, 15};

  // new op and do infershape
  ge::op::Dilation2D op;
  op.UpdateInputDesc("x", create_desc_with_ori({1, 1024, 17, 17}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1, 1024, 17, 17}, ge::FORMAT_NCHW));
  op.UpdateInputDesc("filter", create_desc_with_ori({1024, 3, 3}, ge::DT_FLOAT16, ge::FORMAT_NCHW, {1024, 3, 3}, ge::FORMAT_NCHW));
  op.set_attr_strides(stride_list);
  op.set_attr_rates(rate_list);
  op.set_attr_padding_mode(padding_mode);
  op.set_attr_pads(pads);
  op.set_attr_ceil_mode(ceil_mode);
  op.set_attr_data_format(data_format);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_shape);
}