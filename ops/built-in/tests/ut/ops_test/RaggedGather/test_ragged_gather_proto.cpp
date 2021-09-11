#include <iostream>
#include <gtest/gtest.h>
#include "op_proto_test_util.h"
#include "ragged_array_ops.h"

class RaggedGather : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "RaggedGather SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "RaggedGather TearDown" << std::endl;
  }
};

TEST_F(RaggedGather, RaggedGather_infer_shape_0) {
  ge::op::RaggedGather op;
  op.UpdateInputDesc("params_nested_splits", create_desc({}, ge::DT_STRING));
  op.UpdateInputDesc("params_dense_values", create_desc({}, ge::DT_STRING)); 
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
} 

TEST_F(RaggedGather, RaggedGather_infer_shape_1) {
  ge::op::RaggedGather op;
  op.UpdateInputDesc("params_nested_splits", create_desc({1}, ge::DT_STRING));
  op.UpdateInputDesc("params_dense_values", create_desc({1}, ge::DT_STRING)); 
  op.SetAttr("PARAMS_RAGGED_RANK", 1);
  op.SetAttr("OUTPUT_RAGGED_RANK", 0);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
} 

TEST_F(RaggedGather, RaggedGather_infer_shape_2) {
  ge::op::RaggedGather op;
  op.UpdateInputDesc("params_nested_splits", create_desc({1}, ge::DT_STRING));
  op.UpdateInputDesc("params_dense_values", create_desc({1}, ge::DT_STRING)); 
  op.SetAttr("PARAMS_RAGGED_RANK", 0);
  op.SetAttr("OUTPUT_RAGGED_RANK", 0);
  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
} 

TEST_F(RaggedGather, RaggedGather_infer_shape_3) {
  //new op
  ge::op::RaggedGather op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}};
  auto tensor_desc = create_desc_shape_range({2},
                                             ge::DT_INT32, ge::FORMAT_ND,
                                             {2},
                                             ge::FORMAT_ND, shape_range);
  op.create_dynamic_input_params_nested_splits(2);
  op.UpdateDynamicInputDesc("params_nested_splits", 0, tensor_desc);
  op.UpdateDynamicInputDesc("params_nested_splits", 1, tensor_desc);

  ge::TensorDesc tensor_desc_values(ge::Shape({4}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("params_dense_values", tensor_desc_values);

  ge::TensorDesc tensor_desc_indices(ge::Shape({4,2}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("indices", tensor_desc_indices);

  op.SetAttr("Tsplits", ge::DT_INT32);
  op.SetAttr("PARAMS_RAGGED_RANK", 0);
  op.SetAttr("OUTPUT_RAGGED_RANK", 1);
  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(RaggedGather, RaggedGather_infer_shape_4) {
  //new op
  ge::op::RaggedGather op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}};
  auto tensor_desc = create_desc_shape_range({2},
                                             ge::DT_INT32, ge::FORMAT_ND,
                                             {2},
                                             ge::FORMAT_ND, shape_range);
  op.create_dynamic_input_params_nested_splits(2);
  op.UpdateDynamicInputDesc("params_nested_splits", 0, tensor_desc);
  op.UpdateDynamicInputDesc("params_nested_splits", 1, tensor_desc);

  ge::TensorDesc tensor_desc_values(ge::Shape(), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("params_dense_values", tensor_desc_values);

  ge::TensorDesc tensor_desc_indices(ge::Shape({4,2}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("indices", tensor_desc_indices);

  op.SetAttr("Tsplits", ge::DT_INT32);
  op.SetAttr("PARAMS_RAGGED_RANK", 0);
  op.SetAttr("OUTPUT_RAGGED_RANK", 1);
  auto ret = op.InferShapeAndType();

  // check result
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(RaggedGather, RaggedGather_infer_shape_5) {
  //new op
  ge::op::RaggedGather op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}};
  auto tensor_desc = create_desc_shape_range({2},
                                             ge::DT_INT32, ge::FORMAT_ND,
                                             {2},
                                             ge::FORMAT_ND, shape_range);
  op.create_dynamic_input_params_nested_splits(2);
  op.UpdateDynamicInputDesc("params_nested_splits", 0, tensor_desc);
  op.UpdateDynamicInputDesc("params_nested_splits", 1, tensor_desc);

  ge::TensorDesc tensor_desc_values(ge::Shape({4}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("params_dense_values", tensor_desc_values);

  ge::TensorDesc tensor_desc_indices(ge::Shape({4,2}), ge::FORMAT_ND, ge::DT_INT32);
  op.UpdateInputDesc("indices", tensor_desc_indices);

  op.SetAttr("PARAMS_RAGGED_RANK", 0);
  op.SetAttr("OUTPUT_RAGGED_RANK", 1);
  auto ret = op.InferShapeAndType();

  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(RaggedGather, RaggedGather_infer_shape_6) {
  //new op
  ge::op::RaggedGather op;
  std::vector<std::pair<int64_t,int64_t>> shape_range = {{2, 2}};
  auto tensor_desc = create_desc_shape_range({2},
                                             ge::DT_INT32, ge::FORMAT_ND,
                                             {2},
                                             ge::FORMAT_ND, shape_range);
  op.create_dynamic_input_params_nested_splits(2);
  op.UpdateDynamicInputDesc("params_nested_splits", 0, tensor_desc);
  op.UpdateDynamicInputDesc("params_nested_splits", 1, tensor_desc);
  op.SetAttr("Tsplits", ge::DT_INT32);
  ge::TensorDesc tensor_desc_values(ge::Shape({4}), ge::FORMAT_ND, ge::DT_INT32);
  tensor_desc_values.SetOriginShape(ge::Shape({4}));
  op.UpdateInputDesc("params_dense_values", tensor_desc_values);

  ge::TensorDesc tensor_desc_indices(ge::Shape({4,2}), ge::FORMAT_ND, ge::DT_INT32);
  tensor_desc_indices.SetOriginShape(ge::Shape({4,2}));
  op.UpdateInputDesc("indices", tensor_desc_indices);
  op.SetAttr("PARAMS_RAGGED_RANK", 0);
  op.SetAttr("OUTPUT_RAGGED_RANK", 1);
  op.create_dynamic_output_output_nested_splits(2);

  auto ret = op.InferShapeAndType();
  auto output_nested_splits_desc = op.GetDynamicOutputDesc("output_nested_splits",0);
  EXPECT_EQ(output_nested_splits_desc.GetDataType(),ge::DT_INT32);
  // check result
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}
