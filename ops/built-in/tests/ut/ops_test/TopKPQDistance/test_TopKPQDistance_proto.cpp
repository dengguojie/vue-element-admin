#include <gtest/gtest.h>
#include <iostream>
#include "op_proto_test_util.h"
#include "vector_search.h"

// ----------------TopK-------------------
class TopKPQDistanceProtoTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "TopKPQDistanceProtoTest Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "TopKPQDistanceProtoTest Proto Test TearDown" << std::endl;
  }
};

TEST_F(TopKPQDistanceProtoTest, TopKPQDistance_infer_shape_suc0) {
  ge::op::TopKPQDistance op;
  op.SetAttr("group_size", 2);
  op.SetAttr("k", 6);

  op.create_dynamic_input_actual_count(1);
  op.create_dynamic_input_pq_distance(1);
  op.create_dynamic_input_pq_index(1);
  op.create_dynamic_input_pq_ivf(1);
  op.create_dynamic_input_grouped_extreme_distance(1);

  ge::TensorDesc tensor_desc0 = create_desc({1}, ge::DT_INT32);
  op.UpdateDynamicInputDesc("actual_count", 0, tensor_desc0);

  ge::TensorDesc tensor_desc1 = create_desc({12}, ge::DT_INT32);
  op.UpdateDynamicInputDesc("pq_distance", 0, tensor_desc1);
  op.UpdateDynamicInputDesc("pq_index", 0, tensor_desc1);
  op.UpdateDynamicInputDesc("pq_ivf", 0, tensor_desc1);

  ge::TensorDesc tensor_desc2 = create_desc({6}, ge::DT_INT32);
  op.UpdateDynamicInputDesc("grouped_extreme_distance", 0, tensor_desc2);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

TEST_F(TopKPQDistanceProtoTest, TopKPQDistance_infer_shape_suc1) {
  ge::op::TopKPQDistance op;
  op.SetAttr("group_size", 2);
  op.SetAttr("k", 12);

  op.create_dynamic_input_actual_count(2);
  op.create_dynamic_input_pq_distance(2);
  op.create_dynamic_input_pq_index(2);
  op.create_dynamic_input_pq_ivf(2);
  op.create_dynamic_input_grouped_extreme_distance(2);

  ge::TensorDesc tensor_desc0 = create_desc({1}, ge::DT_INT32);
  op.UpdateDynamicInputDesc("actual_count", 0, tensor_desc0);
  op.UpdateDynamicInputDesc("actual_count", 1, tensor_desc0);

  ge::TensorDesc tensor_desc1 = create_desc({12}, ge::DT_INT32);
  op.UpdateDynamicInputDesc("pq_distance", 0, tensor_desc1);
  op.UpdateDynamicInputDesc("pq_index", 0, tensor_desc1);
  op.UpdateDynamicInputDesc("pq_ivf", 0, tensor_desc1);
  op.UpdateDynamicInputDesc("pq_distance", 1, tensor_desc1);
  op.UpdateDynamicInputDesc("pq_index", 1, tensor_desc1);
  op.UpdateDynamicInputDesc("pq_ivf", 1, tensor_desc1);

  ge::TensorDesc tensor_desc2 = create_desc({6}, ge::DT_INT32);
  op.UpdateDynamicInputDesc("grouped_extreme_distance", 0, tensor_desc2);
  op.UpdateDynamicInputDesc("grouped_extreme_distance", 1, tensor_desc2);

  auto status = op.VerifyAllAttr(true);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

TEST_F(TopKPQDistanceProtoTest, TopKPQDistance_infer_shape_suc2) {
  ge::op::TopKPQDistance op;
  op.SetAttr("k", 5);

  auto status = op.InferShapeAndType();
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  std::vector<int64_t> expectedOutputShape = {5};
  auto outputDesc1 = op.GetOutputDescByName("topk_distance");
  EXPECT_EQ(outputDesc1.GetShape().GetDims(), expectedOutputShape);
}
