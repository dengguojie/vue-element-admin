#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "array_ops.h"
#include "pad_ops.h"

using namespace ge;
using namespace op;

class pad_v3_grad_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "pad_v3_grad_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "pad_v3_grad_test TearDown" << std::endl;
    }
};


TEST_F(pad_v3_grad_test, pad_v3_grad_infer_shape_01) {
  ge::op::PadV3Grad op;
  std::cout<< "pad_v3_grad test_0!!!"<<std::endl;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{1, -1}, {1, -1}, {1, -1}, {1, -1}};
  auto tensor_desc = create_desc_shape_range({-1, 64, -1, 20}, ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {-1, 64, -1, 20}, ge::FORMAT_ND, shape_range);

  auto paddings_desc = create_desc_shape_range({-1}, ge::DT_INT32, ge::FORMAT_ND,
                                               {-1}, ge::FORMAT_ND, {{1, -1}});

  op.UpdateInputDesc("x", tensor_desc);
  op.UpdateInputDesc("paddings", paddings_desc);
  op.SetAttr("mode", "reflect");
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {-1, -1, -1, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
}

TEST_F(pad_v3_grad_test, pad_v3_grad_infer_shape_02) {
  ge::op::PadV3Grad op;
  std::cout<< "pad_v3_grad test_1!!!"<<std::endl;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{1, -1}, {1, -1}, {1, -1}, {1, -1}};
  auto tensor_desc = create_desc_shape_range({-1, 20, -1, -1}, ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {-1, 20, -1, -1}, ge::FORMAT_ND, shape_range);

  auto paddings_desc = create_desc_shape_range({-1}, ge::DT_INT32, ge::FORMAT_ND,
                                               {-1}, ge::FORMAT_ND, {{1, -1}});
  
  op.UpdateInputDesc("x", tensor_desc);
  op.UpdateInputDesc("paddings", paddings_desc);
  op.SetAttr("mode", "reflect");
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {-1, -1, -1, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
}

TEST_F(pad_v3_grad_test, pad_v3_grad_infer_shape_03) {
  ge::op::PadV3Grad op;
  std::cout<< "pad_v3_grad test_2!!!"<<std::endl;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{1, -1}, {1, -1}, {1, -1}, {1, -1}};
  auto tensor_desc = create_desc_shape_range({-1, 20, -1, -1}, ge::DT_FLOAT16, ge::FORMAT_ND,
                                             {-1, 20, -1, -1}, ge::FORMAT_ND, shape_range);

  
  ge::Tensor constTensor;
  ge::TensorDesc constDesc(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  constDesc.SetSize(4 * sizeof(int32_t));
  constTensor.SetTensorDesc(constDesc);
  int32_t constData[4] = {1, 1, 1, 1};
  constTensor.SetData((uint8_t*)constData, 4 * sizeof(int32_t));
  auto const0 = ge::op::Constant().set_attr_value(constTensor);
  op.set_input_paddings(const0);
  auto descPaddings = op.GetInputDesc("paddings");
  descPaddings.SetDataType(ge::DT_INT32);
  descPaddings.SetShape(ge::Shape({static_cast<int64_t>(4)}));
  op.UpdateInputDesc("paddings", descPaddings);
  op.UpdateInputDesc("x", tensor_desc);
  op.SetAttr("mode", "reflect");
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {-1, 18, -1, -1};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t,int64_t>> output_shape_range;
  EXPECT_EQ(output_desc.GetShapeRange(output_shape_range), ge::GRAPH_SUCCESS);
}
