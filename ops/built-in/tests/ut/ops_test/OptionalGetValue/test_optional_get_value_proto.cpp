/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "data_flow_ops.h"

// ----------------optionalGetValue-------------------
class OptionalGetValueProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "OptionalGetValue Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "OptionalGetValue Proto Test TearDown" << std::endl;
  }
};


TEST_F(OptionalGetValueProtoTest, optionalGetValue_infershape_success_test) {
  ge::op::OptionalGetValue op;
  op.UpdateInputDesc("optional", create_desc({}, ge::DT_VARIANT));
  std::vector<ge::DataType> output_types{ge::DT_FLOAT16, ge::DT_FLOAT};
  op.SetAttr("output_types", output_types);
  std::vector<int64_t> shape0{2,2};
  std::vector<int64_t> shape1{3,3};
  ge::Operator::OpListListInt output_shapes{shape0, shape1};
  op.SetAttr("output_shapes", output_shapes);
  op.create_dynamic_output_components(2);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
//  get output shapetype
  auto output_desc0 = op.GetDynamicOutputDesc("components", 0);
  EXPECT_EQ(output_desc0.GetDataType(), ge::DT_FLOAT16);
  auto output_desc1 = op.GetDynamicOutputDesc("components", 1);
  EXPECT_EQ(output_desc1.GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(output_desc0.GetShape().GetDims(), shape0);
  EXPECT_EQ(output_desc1.GetShape().GetDims(), shape1);
}

TEST_F(OptionalGetValueProtoTest, optionalGetValue_infershape_optional_test) {
  ge::op::OptionalGetValue op;
  op.UpdateInputDesc("optional", create_desc({4,3,1}, ge::DT_VARIANT));
  std::vector<ge::DataType> output_types{ ge::DT_FLOAT16, ge::DT_FLOAT};
  op.SetAttr("output_types", output_types);
  std::vector<int64_t> shape0{2,2};
  std::vector<int64_t> shape1{3,3};
  ge::Operator::OpListListInt output_shapes{shape0, shape1};
  op.SetAttr("output_shapes", output_shapes);
  op.create_dynamic_output_components(2);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(OptionalGetValueProtoTest, optionalGetValue_infershape_attrShape_test) {
  ge::op::OptionalGetValue op;
  op.set_attr_output_types({ge::DT_FLOAT16});
  op.set_attr_output_shapes({{4, 4, 3}, {1}});
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(OptionalGetValueProtoTest, optionalGetValue_infershape_attr_outputType_test) {
  ge::op::OptionalGetValue op;
  op.set_attr_output_types({ge::DT_DOUBLE});
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(OptionalGetValueProtoTest, optionalGetValue_infershape_attr_outputShape_test) {
  ge::op::OptionalGetValue op;
  op.set_attr_output_shapes({{4, 4, 3}});
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(OptionalGetValueProtoTest, optionalGetValue_infershape_attr_type_size_test) {
  ge::op::OptionalGetValue op;
  std::vector<ge::DataType> output_types;
  op.SetAttr("output_types", output_types);
  std::vector<int64_t> shape0{2,2};
  std::vector<int64_t> shape1{3,3};
  ge::Operator::OpListListInt output_shapes{shape0, shape1};
  op.SetAttr("output_shapes", output_shapes);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(OptionalGetValueProtoTest, optionalGetValue_infershape_attr_shape_size_test) {
  ge::op::OptionalGetValue op;
  std::vector<ge::DataType> output_types{ge::DT_FLOAT16, ge::DT_FLOAT16};
  op.SetAttr("output_types", output_types);
  ge::Operator::OpListListInt output_shapes;
  op.SetAttr("output_shapes", output_shapes);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(OptionalGetValueProtoTest, optionalGetValue_infershape_fail_test) {
  ge::op::OptionalGetValue op;
  op.UpdateInputDesc("optional", create_desc({}, ge::DT_VARIANT));
  std::vector<ge::DataType> output_types{ge::DT_FLOAT16, ge::DT_FLOAT};
  op.SetAttr("output_types", output_types);
  std::vector<int64_t> shape0{2,2};
  std::vector<int64_t> shape1{3,3};
  ge::Operator::OpListListInt output_shapes{shape0, shape1};
  op.SetAttr("output_shapes", output_shapes);
  op.create_dynamic_output_components(1);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}