#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "split_combination_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class concat_offset_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "concat_offset_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "concat_offset_test TearDown" << std::endl;
    }
};
TEST_F(concat_offset_test, concat_offset_test_1){
  ge::Graph graph("concat_offset_1");
  auto shape_x1 = vector<int64_t>({-1,});
  TensorDesc desc_x1(ge::Shape(shape_x1), FORMAT_ND, DT_INT32);
  std::vector<std::pair<int64_t, int64_t>>x1_range;
  x1_range.push_back(std::pair<int64_t, int64_t>{1,9});
  desc_x1.SetShapeRange(x1_range);

  auto data_x1 = op::Data("x1");
  data_x1.update_input_desc_x(desc_x1);
  data_x1.update_output_desc_y(desc_x1);

  auto shape_x2 = vector<int64_t>({-1,});
  TensorDesc desc_x2(ge::Shape(shape_x2), FORMAT_ND, DT_INT32);
  std::vector<std::pair<int64_t, int64_t>>x2_range;
  x2_range.push_back(std::pair<int64_t, int64_t>{2,10});
  desc_x2.SetShapeRange(x2_range);

  auto data_x2 = op::Data("x2");
  data_x2.update_input_desc_x(desc_x2);
  data_x2.update_output_desc_y(desc_x2);

  auto shape_x3 = vector<int64_t>({-1,});
  TensorDesc desc_x3(ge::Shape(shape_x3), FORMAT_ND, DT_INT32);
  std::vector<std::pair<int64_t, int64_t>>x3_range;
  x3_range.push_back(std::pair<int64_t, int64_t>{3,12});
  desc_x3.SetShapeRange(x3_range);

  auto data_x3 = op::Data("x3");
  data_x3.update_input_desc_x(desc_x3);
  data_x3.update_output_desc_y(desc_x3);

  //init concat_dim
  Tensor input_concat_dim_tensor;
  input_concat_dim_tensor.SetTensorDesc(TensorDesc(ge::Shape({1}), ge::FORMAT_ND, ge::DT_INT32));
  uint32_t* input_concat_dim_value = new uint32_t[1]{1};
  input_concat_dim_tensor.SetData((uint8_t*)input_concat_dim_value, 1*sizeof(uint32_t));
  auto input_concat_dim_data = op::Const("concat_dim").set_attr_value(input_concat_dim_tensor);
  delete []input_concat_dim_value;

  //new op
  auto test_layer = op::ConcatOffset("ConcatOffset_1");
  test_layer.set_input_concat_dim(input_concat_dim_data)
            .create_dynamic_input_x(3)
            .set_dynamic_input_x(0,data_x1)
            .set_dynamic_input_x(1,data_x2)
            .set_dynamic_input_x(2,data_x3)
            .set_attr_N(3)
            .create_dynamic_output_y(3);
  std::vector<Operator> inputs{input_concat_dim_data, data_x1, data_x2,data_x3};
  std::vector<Operator> outputs{test_layer};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);

  bool findOp = false;
  bool shapeMatch1 = false;
  bool shapeMatch2 = false;
  bool shapeMatch3 = false;

  //except_shape
  auto expect_shape1 = vector<int64_t>({-1,});
  auto expect_shape2 = vector<int64_t>({-1,});
  auto expect_shape3 = vector<int64_t>({-1,});

  for(auto node: compute_graph_ptr->GetAllNodes()){
      if(node->GetType() == "ConcatOffset"){
          std::cout<< "ConcatOffset test_1!!!"<<std::endl;
          findOp = true;
          auto outputDesc1 = node->GetOpDesc()->GetOutputDesc(0);
          auto outputDesc2 = node->GetOpDesc()->GetOutputDesc(1);
          auto outputDesc3 = node->GetOpDesc()->GetOutputDesc(2);
          std::vector<int64_t> output_shape1 = outputDesc1.GetShape().GetDims();
          std::vector<int64_t> output_shape2 = outputDesc2.GetShape().GetDims();
          std::vector<int64_t> output_shape3 = outputDesc3.GetShape().GetDims();
          if(output_shape1 == expect_shape1){
              shapeMatch1 = true;
          }
          if(output_shape2 == expect_shape2){
              shapeMatch1 = true;
          }
          if(output_shape3 == expect_shape3){
              shapeMatch1 = true;
          }
      }
  }
  EXPECT_EQ(findOp, true);
  EXPECT_EQ(ShapeMatch1, true);
  EXPECT_EQ(ShapeMatch2, true);
  EXPECT_EQ(ShapeMatch3, true);
}

