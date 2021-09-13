#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "nn_pooling_ops.h"
#include "op_proto_test_util.h"

class MaxPoolGradWithArgmaxV1ProtoTest : public testing::Test {
    protected:
    static void SetUpTestCase() {
        std::cout << "max_pool_grad_with_argmax_v1 test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "max_pool_grad_with_argmax_v1 test TearDown" << std::endl;
    }
};

TEST_F(MaxPoolGradWithArgmaxV1ProtoTest, max_pool_grad_with_argmax_v1_test_case_1) {
    // [TODO] define your op here
    ge::op::MaxPoolGradWithArgmaxV1 op;
    ge::TensorDesc x;
    ge::Shape shape_x({32, 640, 20, 20});
    x.SetDataType(ge::DT_FLOAT16);
    x.SetShape(shape_x);
    x.SetOriginShape(shape_x);
    op.UpdateInputDesc("x", x);

    ge::TensorDesc grad;
    ge::Shape shape_grad({32, 640, 20, 20});
    grad.SetDataType(ge::DT_FLOAT16);
    grad.SetShape(shape_grad);
    grad.SetOriginShape(shape_grad);
    op.UpdateInputDesc("grad", grad);

    ge::TensorDesc argmax;
    ge::Shape shape_argmax({32, 640, 169, 26});
    argmax.SetDataType(ge::DT_UINT16);
    argmax.SetShape(shape_argmax);
    argmax.SetOriginShape(shape_argmax);
    op.UpdateInputDesc("argmax", argmax);

    op.SetAttr("ksize", (1, 13, 13, 1));
    op.SetAttr("strides", (1, 1, 1, 1));
    op.SetAttr("pads", (1, 6, 6, 1));
    op.SetAttr("dtype", 3);
    op.SetAttr("dilation", (1, 1, 1, 1));
    op.SetAttr("ceil_mode", false);

    // [TODO] call InferShapeAndType function here
    auto ret = op.InferShapeAndType();
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    // [TODO] compare dtype and shape of op output
    auto output_desc = op.GetOutputDesc("y");
    EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    std::vector<int64_t> expected_output_shape = {32, 640, 20, 20};
    EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(MaxPoolGradWithArgmaxV1ProtoTest, max_pool_grad_with_argmax_v1_test_case_2) {
  auto shape_x1 = vector<int64_t>({1, 64, 672, 672});
  std::vector<std::pair<int64_t,int64_t>> range_x1 = {{1, 1}, {64, 64}, {672, 672}, {672, 672}};

  // expect result
  std::vector<int64_t> expected_shape0 = {1, 64, 672, 672};
  std::vector<std::pair<int64_t,int64_t>> expected_range0 = {{1, 1}, {64, 64},{672, 672}, {672, 672}};
  // create desc
  auto tensor_desc_x1 = create_desc_shape_range(shape_x1, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                                shape_x1, ge::FORMAT_NCHW, range_x1);
  
  bool ceil_mode = false;
  // new op and do infershape
  ge::op::MaxPoolGradWithArgmaxV1 op;
  op.UpdateInputDesc("x", tensor_desc_x1);
  op.SetAttr("ksize", {1,2,2,1});
  op.SetAttr("strides", {1,1,1,1});
  op.SetAttr("pads", {1,0,0,1});
  op.SetAttr("dilation", {1,1,1,1});
  op.SetAttr("ceil_mode", ceil_mode);
  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc0 = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc0.GetShape().GetDims(), expected_shape0);
  std::vector<std::pair<int64_t,int64_t>> output_range0;
  EXPECT_EQ(output_desc0.GetShapeRange(output_range0), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_range0, expected_range0);
}

TEST_F(MaxPoolGradWithArgmaxV1ProtoTest, max_pool_grad_with_argmax_v1_test_case_3) {
  auto shape_x1 = vector<int64_t>({1, 64, -1, -1});
  std::vector<std::pair<int64_t,int64_t>> range_x1 = {{1, 1}, {64, 64}, {1, -1}, {1, -1}};

  // expect result
  std::vector<int64_t> expected_shape0 = {1, 64, -1, -1};
  std::vector<std::pair<int64_t,int64_t>> expected_range0 = {{1, 1}, {64, 64},{1, -1}, {1, -1}};
  // create desc
  auto tensor_desc_x1 = create_desc_shape_range(shape_x1, ge::DT_FLOAT16, ge::FORMAT_NCHW,
                                                shape_x1, ge::FORMAT_NCHW, range_x1);
  
  bool ceil_mode = false;
  // new op and do infershape
  ge::op::MaxPoolGradWithArgmaxV1 op;
  op.UpdateInputDesc("x", tensor_desc_x1);
  op.SetAttr("ksize", {1,2,2,1});
  op.SetAttr("strides", {1,1,1,1});
  op.SetAttr("pads", {1,0,0,1});
  op.SetAttr("dilation", {1,1,1,1});
  op.SetAttr("ceil_mode", ceil_mode);
  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc0 = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc0.GetShape().GetDims(), expected_shape0);
  std::vector<std::pair<int64_t,int64_t>> output_range0;
  EXPECT_EQ(output_desc0.GetShapeRange(output_range0), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_range0, expected_range0);
}