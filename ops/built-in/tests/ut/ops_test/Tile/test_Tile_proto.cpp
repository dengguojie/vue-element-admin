#include <gtest/gtest.h>
#include <iostream>
#include "array_ops.h"
#include "op_proto_test_util.h"
#include "elewise_calculation_ops.h"
#include "selection_ops.h"


class tile : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "tile SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "tile TearDown" << std::endl;
  }
};

TEST_F(tile, tile_infershape_test_0) {
ge::op::Tile op;
op.UpdateInputDesc("x", create_desc_shape_range({2, 1}, ge::DT_FLOAT,
ge::FORMAT_ND, {2, 1}, ge::FORMAT_ND, {{2,2}, {1,1}}));

ge::Tensor consttensor;
ge::TensorDesc constdesc(ge::Shape({2}), ge::FORMAT_ND, ge::DT_INT32);
constdesc.SetSize(2 * sizeof(int32_t));
consttensor.SetTensorDesc(constdesc);
int32_t constdata[2] = {8,9};
consttensor.SetData((uint8_t*)constdata, 2 * sizeof(int32_t));
auto multiples = ge::op::Constant().set_attr_value(consttensor);
op.set_input_multiples(multiples);
op.UpdateInputDesc("multiples", constdesc);

auto ret = op.InferShapeAndType();
EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
auto output_desc = op.GetOutputDesc("y");
EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
std::vector<int64_t> expected_output_shape = {16, 9};
EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(tile, tile_infershape_test_1) {
ge::op::Tile op;
op.UpdateInputDesc("x", create_desc_shape_range({1}, ge::DT_FLOAT,
ge::FORMAT_ND, {1}, ge::FORMAT_ND, {{1,1}}));
ge::Tensor consttensor;
ge::TensorDesc constdesc(ge::Shape({1}), ge::FORMAT_ND, ge::DT_INT32);
constdesc.SetSize(1 * sizeof(int32_t));
consttensor.SetTensorDesc(constdesc);
int32_t constdata[1] = {9,};
consttensor.SetData((uint8_t*)constdata, 1 * sizeof(int32_t));
auto multiples = ge::op::Constant().set_attr_value(consttensor);
op.set_input_multiples(multiples);
op.UpdateInputDesc("multiples", constdesc);

auto ret = op.InferShapeAndType();
EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
auto output_desc = op.GetOutputDesc("y");
EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
std::vector<int64_t> expected_output_shape = {9};
EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
