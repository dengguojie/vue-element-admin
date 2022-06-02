#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "array_ops.h"
#include "selection_ops.h"
#include "graph/debug/ge_attr_define.h"
#include "utils/attr_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "common/utils/ut_op_common.h"

// ----------------MovingSumWithSigmoid-------------------
class MovingSumWithSigmoidProtoTest : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "MovingSumWithSigmoid Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MovingSumWithSigmoid Proto Test TearDown" << std::endl;
  }
};


TEST_F(MovingSumWithSigmoidProtoTest, MovingSumWithSigmoidProtoTest_0) {
  ge::op::MovingSumWithSigmoid op;

  ge::TensorDesc alpha_desc;
  ge::Shape xShape({51200});
  alpha_desc.SetDataType(ge::DT_FLOAT);
  alpha_desc.SetShape(xShape);
  alpha_desc.SetOriginShape(xShape);
  
  ge::TensorDesc offset_desc;
  ge::Shape yShape({2});
  offset_desc.SetDataType(ge::DT_INT32);
  offset_desc.SetShape(yShape);
  offset_desc.SetOriginShape(yShape);

  op.UpdateInputDesc("alpha", alpha_desc);
  op.UpdateInputDesc("energy", alpha_desc);
  op.UpdateInputDesc("offset", offset_desc);
  op.SetAttr("ksize", 0);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
  CommonInferShapeOperatorFail(op, {"ksize"});
}

TEST_F(MovingSumWithSigmoidProtoTest, MovingSumWithSigmoidProtoTest_1) {
  ge::op::MovingSumWithSigmoid op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{1, -1}};

  ge::TensorDesc alpha_desc;
  ge::Shape xShape({-1});
  alpha_desc.SetDataType(ge::DT_FLOAT);
  alpha_desc.SetShape(xShape);
  alpha_desc.SetOriginShape(xShape);
  alpha_desc.SetShapeRange(shape_range);

  ge::TensorDesc offset_desc;
  ge::Shape yShape({-1});
  offset_desc.SetDataType(ge::DT_INT32);
  offset_desc.SetShape(yShape);
  offset_desc.SetOriginShape(yShape);
  offset_desc.SetShapeRange(shape_range);

  op.UpdateInputDesc("alpha", alpha_desc);
  op.UpdateInputDesc("energy", alpha_desc);
  op.UpdateInputDesc("offset", offset_desc);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

using namespace ut_util;
TEST_F(MovingSumWithSigmoidProtoTest, MovingSumWithSigmoidProtoTest_2) {
  using namespace ge;
  ge::op::MovingSumWithSigmoid op;
  auto input_alpha_shape = vector<int64_t>({2});
  auto input_alpha_dtype = DT_FLOAT;
  auto input_energy_shape = vector<int64_t>({2});
  auto input_energy_dtype = DT_FLOAT;
  // input axes info
  vector<int64_t> input_offset_shape = {4};
  auto offset_dtype = DT_INT32;

  vector<int32_t> offset_value = {4,3,2,1};
  TENSOR_INPUT_WITH_SHAPE(op, alpha, input_alpha_shape, input_alpha_dtype, FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE(op, energy, input_energy_shape, input_energy_dtype, FORMAT_ND, {});
  TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(op, offset, input_offset_shape, offset_dtype, FORMAT_ND, offset_value);
  op.SetAttr("ksize", 0);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  std::vector<bool> input_const = {false, false, true};
  CommonInferShapeOperatorWithConst(op, input_const, {}, {{7,3}});
}
