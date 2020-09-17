#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "array_ops.h"
#include "selection_ops.h"

class scatter_nd : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "scatter_nd SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "scatter_nd TearDown" << std::endl;
  }
};

// TODO fix me run failed
//TEST_F(scatter_nd, scatter_nd_infershape_diff_test_1) {
//  ge::op::ScatterNd op;
//  op.UpdateInputDesc("x", create_desc_with_ori({2, 2, 2}, ge::DT_INT32, ge::FORMAT_ND, {2, 2, 2}, ge::FORMAT_ND));
//  /*
//  ge::op::Constant shape;
//  shape.SetAttr("value", std::vector<int64_t>{1,2,3});
//  op.set_input_shape(shape);*/
//  ge::Tensor constTensor;
//  ge::TensorDesc constDesc(ge::Shape({3}), ge::FORMAT_ND, ge::DT_INT64);
//  constDesc.SetSize(3 * sizeof(int64_t));
//  constTensor.SetTensorDesc(constDesc);
//  int64_t* constData = new int64_t[3];
//  *(constData + 0) = -1;
//  *(constData + 1) = 2;
//  *(constData + 2) = 3;
//  constTensor.SetData((uint8_t*)constData, 3 * sizeof(int64_t));
//  auto const0 = ge::op::Constant().set_attr_value(constTensor);
//  op.set_input_shape(const0);
//
//
//  ge::TensorDesc tensor_shape = op.GetInputDesc("shape");
//  tensor_shape.SetDataType(ge::DT_INT64);
//
//  op.UpdateInputDesc("shape", tensor_shape);
//  auto ret = op.InferShapeAndType();
//  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
//  auto output_desc = op.GetOutputDesc("y");
//  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
//  std::vector<int64_t> expected_output_shape = {1,2,3};
//  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
//  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{1,-1},{2,2},{3,3}};
//  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
//  output_desc.GetShapeRange(output_shape_range);
//  EXPECT_EQ(output_shape_range, expected_output_shape_range);
//  delete []constData;
//}

/*TEST_F(scatter_nd, scatter_nd_infershape_diff_test_2) {
  ge::op::ScatterNd op;
  op.UpdateInputDesc("x", create_desc_with_ori({2, 2, 2}, ge::DT_INT32, ge::FORMAT_ND, {2, 2, 2}, ge::FORMAT_ND));


  auto data0 = ge::op::Data().set_attr_index(0);
  op.set_input_shape(data0);


  ge::TensorDesc tensor_shape = op.GetInputDesc("shape");
  tensor_shape.SetDataType(ge::DT_INT64);
  tensor_shape.SetShapeRange({{1,5}});

  op.UpdateInputDesc("shape", tensor_shape);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto output_desc = op.GetOutputDesc("y");
  EXPECT_EQ(output_desc.GetDataType(), ge::DT_INT32);
  std::vector<int64_t> expected_output_shape = {-2};
  EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
  std::vector<std::pair<int64_t, int64_t>> expected_output_shape_range = {{1,-1},{1,-1},{1,-1},{1,-1},{1,-1}};
  std::vector<std::pair<int64_t, int64_t>> output_shape_range;
  output_desc.GetShapeRange(output_shape_range);
  EXPECT_EQ(output_shape_range, expected_output_shape_range);

}
*/
