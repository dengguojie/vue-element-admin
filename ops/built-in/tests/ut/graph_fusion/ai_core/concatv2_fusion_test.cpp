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

class concatv2_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "concatv2_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "concatv2_fusion_test TearDown" << std::endl;
    }
};

TEST_F(concatv2_fusion_test, concatv2_fusion_test_1) {
    ge::Graph graph("concatv2_fusion_test_1");

    ge::Tensor inputx0DataTensor;
    std::vector<int64_t> crops_vec0{1};
    ge::Shape shape_x0(crops_vec0);
    ge::TensorDesc tensorDescX0(shape_x0, FORMAT_ND, DT_INT64);
    int64_t inputx0_size = tensorDescX0.GetShape().GetShapeSize();
    tensorDescX0.SetSize(inputx0_size * sizeof(int64_t));
    inputx0DataTensor.SetTensorDesc(tensorDescX0);
    int64_t* inputx0_data = nullptr;
    inputx0_data = new int64_t[inputx0_size];
    *(inputx0_data + 0) = 0;
    inputx0DataTensor.SetData((uint8_t*)inputx0_data, inputx0_size * sizeof(int64_t));
    delete[] inputx0_data;
    auto inputx0Data = op::Constant("inputx0Data").set_attr_value(inputx0DataTensor);

    ge::Tensor inputx1DataTensor;
    std::vector<int64_t> crops_vec1{1};
    ge::Shape shape_x1(crops_vec1);
    ge::TensorDesc tensorDescX1(shape_x1, FORMAT_ND, DT_INT64);
    int64_t inputx1_size = tensorDescX1.GetShape().GetShapeSize();
    tensorDescX1.SetSize(inputx1_size * sizeof(int64_t));
    inputx1DataTensor.SetTensorDesc(tensorDescX1);
    int64_t* inputx1_data = nullptr;
    inputx1_data = new int64_t[inputx1_size];
    *(inputx1_data + 0) = 0;
    inputx1DataTensor.SetData((uint8_t*)inputx1_data, inputx1_size * sizeof(int64_t));
    delete[] inputx1_data;
    auto inputx1Data = op::Constant("inputx1Data").set_attr_value(inputx1DataTensor);

    ge::Tensor crops_tensor;
    std::vector<int64_t> crops_vec2{1};
    ge::Shape crops_shape(crops_vec2);
    ge::TensorDesc crops_desc(crops_shape, FORMAT_ND, DT_INT64);
    int64_t crops_size = crops_desc.GetShape().GetShapeSize();
    crops_desc.SetSize(crops_size * sizeof(int64_t));
    crops_tensor.SetTensorDesc(crops_desc);
    int64_t* crops_data = nullptr;
    crops_data = new int64_t[crops_size];
    *(crops_data + 0) = 0;
    crops_tensor.SetData((uint8_t*)crops_data, crops_size * sizeof(int64_t));
    delete[] crops_data;
    auto concat_dim = op::Constant("concat_dim").set_attr_value(crops_tensor);

    auto concat_layer = op::ConcatV2("concatv2");
    concat_layer.create_dynamic_input_x(69);
    for (int64_t n = 0; n < 69; n++) {
        concat_layer.set_dynamic_input_x(n, inputx0Data);
    }
    concat_layer.set_input_concat_dim(concat_dim);
    concat_layer.set_attr_N(69);

    std::vector<Operator> inputs{inputx0Data, inputx1Data, concat_dim};
    std::vector<Operator> outputs{concat_layer};
    graph.SetInputs(inputs).SetOutputs(outputs);
    ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
    fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
    fe::FusionPassTestUtils::RunGraphFusionPass("ZConcatExt2FusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

    bool findConcatV2 = false;
    int concat_v2_d_node_count = 0;
    for (auto node: compute_graph_ptr->GetAllNodes()) {
        if (node->GetType() == "ConcatV2D") {
            findConcatV2 = true;
            concat_v2_d_node_count++;
        }
    }
    EXPECT_EQ(findConcatV2, true);
    EXPECT_GT(concat_v2_d_node_count, 1);
}

TEST_F(concatv2_fusion_test, concatv2_fusion_test_2) {
  ge::Graph graph("concatv2_fusion_test_2");

  ge::Tensor inputx0DataTensor;
  std::vector<int64_t> crops_vec0{-1};
  ge::Shape shape_x0(crops_vec0);
  ge::TensorDesc tensorDescX0(shape_x0, FORMAT_ND, DT_INT64);
  int64_t inputx0_size = 1;
  tensorDescX0.SetSize(inputx0_size * sizeof(int64_t));
  inputx0DataTensor.SetTensorDesc(tensorDescX0);
  int64_t* inputx0_data = nullptr;
  inputx0_data = new int64_t[inputx0_size];
  *(inputx0_data + 0) = 0;
  inputx0DataTensor.SetData((uint8_t*)inputx0_data, inputx0_size * sizeof(int64_t));
  delete[] inputx0_data;
  auto inputx0Data = op::Constant("inputx0Data").set_attr_value(inputx0DataTensor);

  ge::Tensor inputx1DataTensor;
  std::vector<int64_t> crops_vec1{1};
  ge::Shape shape_x1(crops_vec1);
  ge::TensorDesc tensorDescX1(shape_x1, FORMAT_ND, DT_INT64);
  int64_t inputx1_size = tensorDescX1.GetShape().GetShapeSize();
  tensorDescX1.SetSize(inputx1_size * sizeof(int64_t));
  inputx1DataTensor.SetTensorDesc(tensorDescX1);
  int64_t* inputx1_data = nullptr;
  inputx1_data = new int64_t[inputx1_size];
  *(inputx1_data + 0) = 0;
  inputx1DataTensor.SetData((uint8_t*)inputx1_data, inputx1_size * sizeof(int64_t));
  delete[] inputx1_data;
  auto inputx1Data = op::Constant("inputx1Data").set_attr_value(inputx1DataTensor);

  ge::Tensor crops_tensor;
  std::vector<int64_t> crops_vec2{1};
  ge::Shape crops_shape(crops_vec2);
  ge::TensorDesc crops_desc(crops_shape, FORMAT_ND, DT_INT64);
  int64_t crops_size = crops_desc.GetShape().GetShapeSize();
  crops_desc.SetSize(crops_size * sizeof(int64_t));
  crops_tensor.SetTensorDesc(crops_desc);
  int64_t* crops_data = nullptr;
  crops_data = new int64_t[crops_size];
  *(crops_data + 0) = 0;
  crops_tensor.SetData((uint8_t*)crops_data, crops_size * sizeof(int64_t));
  delete[] crops_data;
  auto concat_dim = op::Constant("concat_dim").set_attr_value(crops_tensor);

  auto concat_layer = op::ConcatV2("concatv2");
  concat_layer.create_dynamic_input_x(69);
  for (int64_t n = 0; n < 69; n++) {
    concat_layer.set_dynamic_input_x(n, inputx0Data);
  }
  concat_layer.set_input_concat_dim(concat_dim);
  concat_layer.set_attr_N(69);

  std::vector<Operator> inputs{inputx0Data, inputx1Data, concat_dim};
  std::vector<Operator> outputs{concat_layer};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("ZConcatExt2FusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findConcatV2 = false;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "ConcatV2D") {
      findConcatV2 = true;
    }
  }
  EXPECT_EQ(findConcatV2, true);
}

TEST_F(concatv2_fusion_test, concatv2_fusion_test_3) {
  ge::Graph graph("concatv2_fusion_test_3");

  ge::Tensor inputx0DataTensor;
  std::vector<int64_t> crops_vec0{-1};
  ge::Shape shape_x0(crops_vec0);
  ge::TensorDesc tensorDescX0(shape_x0, FORMAT_ND, DT_INT64);
  int64_t inputx0_size = 1;
  tensorDescX0.SetSize(inputx0_size * sizeof(int64_t));
  inputx0DataTensor.SetTensorDesc(tensorDescX0);
  int64_t* inputx0_data = nullptr;
  inputx0_data = new int64_t[inputx0_size];
  *(inputx0_data + 0) = 0;
  inputx0DataTensor.SetData((uint8_t*)inputx0_data, inputx0_size * sizeof(int64_t));
  delete[] inputx0_data;
  auto inputx0Data = op::Constant("inputx0Data").set_attr_value(inputx0DataTensor);

  ge::Tensor inputx1DataTensor;
  std::vector<int64_t> crops_vec1{1};
  ge::Shape shape_x1(crops_vec1);
  ge::TensorDesc tensorDescX1(shape_x1, FORMAT_ND, DT_INT64);
  int64_t inputx1_size = tensorDescX1.GetShape().GetShapeSize();
  tensorDescX1.SetSize(inputx1_size * sizeof(int64_t));
  inputx1DataTensor.SetTensorDesc(tensorDescX1);
  int64_t* inputx1_data = nullptr;
  inputx1_data = new int64_t[inputx1_size];
  *(inputx1_data + 0) = 0;
  inputx1DataTensor.SetData((uint8_t*)inputx1_data, inputx1_size * sizeof(int64_t));
  delete[] inputx1_data;
  auto inputx1Data = op::Constant("inputx1Data").set_attr_value(inputx1DataTensor);

  ge::Tensor crops_tensor;
  std::vector<int64_t> crops_vec2{1};
  ge::Shape crops_shape(crops_vec2);
  ge::TensorDesc crops_desc(crops_shape, FORMAT_ND, DT_INT64);
  int64_t crops_size = crops_desc.GetShape().GetShapeSize();
  crops_desc.SetSize(crops_size * sizeof(int64_t));
  crops_tensor.SetTensorDesc(crops_desc);
  int64_t* crops_data = nullptr;
  crops_data = new int64_t[crops_size];
  *(crops_data + 0) = 0;
  crops_tensor.SetData((uint8_t*)crops_data, crops_size * sizeof(int64_t));
  delete[] crops_data;
  auto concat_dim = op::Constant("concat_dim").set_attr_value(crops_tensor);

  auto concat_layer = op::ConcatV2("concatv2");
  const int64_t input_count = 48;
  concat_layer.create_dynamic_input_x(input_count);
  for (int64_t n = 0; n < input_count; n++) {
    concat_layer.set_dynamic_input_x(n, inputx0Data);
  }
  concat_layer.set_input_concat_dim(concat_dim);
  concat_layer.set_attr_N(input_count);

  std::vector<Operator> inputs{inputx0Data, inputx1Data, concat_dim};
  std::vector<Operator> outputs{concat_layer};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("ZConcatExt2FusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findConcatV2 = false;
  int concat_v2_d_node_count = 0;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "ConcatV2D") {
      findConcatV2 = true;
      concat_v2_d_node_count++;
    }
  }
  EXPECT_EQ(findConcatV2, true);
  EXPECT_EQ(concat_v2_d_node_count, 1);
}

TEST_F(concatv2_fusion_test, concatv2_fusion_test_4) {
  ge::Graph graph("concatv2_fusion_test_4");

  ge::Tensor inputx0DataTensor;
  std::vector<int64_t> crops_vec0{-1};
  ge::Shape shape_x0(crops_vec0);
  ge::TensorDesc tensorDescX0(shape_x0, FORMAT_ND, DT_INT64);
  int64_t inputx0_size = 1;
  tensorDescX0.SetSize(inputx0_size * sizeof(int64_t));
  inputx0DataTensor.SetTensorDesc(tensorDescX0);
  int64_t* inputx0_data = nullptr;
  inputx0_data = new int64_t[inputx0_size];
  *(inputx0_data + 0) = 0;
  inputx0DataTensor.SetData((uint8_t*)inputx0_data, inputx0_size * sizeof(int64_t));
  delete[] inputx0_data;
  auto inputx0Data = op::Constant("inputx0Data").set_attr_value(inputx0DataTensor);

  ge::Tensor inputx1DataTensor;
  std::vector<int64_t> crops_vec1{1};
  ge::Shape shape_x1(crops_vec1);
  ge::TensorDesc tensorDescX1(shape_x1, FORMAT_ND, DT_INT64);
  int64_t inputx1_size = tensorDescX1.GetShape().GetShapeSize();
  tensorDescX1.SetSize(inputx1_size * sizeof(int64_t));
  inputx1DataTensor.SetTensorDesc(tensorDescX1);
  int64_t* inputx1_data = nullptr;
  inputx1_data = new int64_t[inputx1_size];
  *(inputx1_data + 0) = 0;
  inputx1DataTensor.SetData((uint8_t*)inputx1_data, inputx1_size * sizeof(int64_t));
  delete[] inputx1_data;
  auto inputx1Data = op::Constant("inputx1Data").set_attr_value(inputx1DataTensor);

  ge::Tensor crops_tensor;
  std::vector<int64_t> crops_vec2{1};
  ge::Shape crops_shape(crops_vec2);
  ge::TensorDesc crops_desc(crops_shape, FORMAT_ND, DT_INT64);
  int64_t crops_size = crops_desc.GetShape().GetShapeSize();
  crops_desc.SetSize(crops_size * sizeof(int64_t));
  crops_tensor.SetTensorDesc(crops_desc);
  int64_t* crops_data = nullptr;
  crops_data = new int64_t[crops_size];
  *(crops_data + 0) = 0;
  crops_tensor.SetData((uint8_t*)crops_data, crops_size * sizeof(int64_t));
  delete[] crops_data;
  auto concat_dim = op::Constant("concat_dim").set_attr_value(crops_tensor);

  auto concat_layer = op::ConcatV2("concatv2");
  const int64_t input_count = 49;
  concat_layer.create_dynamic_input_x(input_count);
  for (int64_t n = 0; n < input_count; n++) {
    concat_layer.set_dynamic_input_x(n, inputx0Data);
  }
  concat_layer.set_input_concat_dim(concat_dim);
  concat_layer.set_attr_N(input_count);

  std::vector<Operator> inputs{inputx0Data, inputx1Data, concat_dim};
  std::vector<Operator> outputs{concat_layer};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("ZConcatExt2FusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findConcatV2 = false;
  int concat_v2_d_node_count = 0;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "ConcatV2D") {
      findConcatV2 = true;
      concat_v2_d_node_count++;
    }
  }
  EXPECT_EQ(findConcatV2, true);
  EXPECT_GT(concat_v2_d_node_count, 1);
}

TEST_F(concatv2_fusion_test, concatv2_fusion_test_5) {
  ge::Graph graph("concatv2_fusion_test_1");

  ge::Tensor inputx0DataTensor;
  std::vector<int64_t> crops_vec0{1};
  ge::Shape shape_x0(crops_vec0);
  ge::TensorDesc tensorDescX0(shape_x0, FORMAT_ND, DT_INT64);
  int64_t inputx0_size = tensorDescX0.GetShape().GetShapeSize();
  tensorDescX0.SetSize(inputx0_size * sizeof(int64_t));
  inputx0DataTensor.SetTensorDesc(tensorDescX0);
  int64_t* inputx0_data = nullptr;
  inputx0_data = new int64_t[inputx0_size];
  *(inputx0_data + 0) = 0;
  inputx0DataTensor.SetData((uint8_t*)inputx0_data, inputx0_size * sizeof(int64_t));
  delete[] inputx0_data;
  auto inputx0Data = op::Constant("inputx0Data").set_attr_value(inputx0DataTensor);

  ge::Tensor inputx1DataTensor;
  std::vector<int64_t> crops_vec1{1};
  ge::Shape shape_x1(crops_vec1);
  ge::TensorDesc tensorDescX1(shape_x1, FORMAT_ND, DT_INT64);
  int64_t inputx1_size = tensorDescX1.GetShape().GetShapeSize();
  tensorDescX1.SetSize(inputx1_size * sizeof(int64_t));
  inputx1DataTensor.SetTensorDesc(tensorDescX1);
  int64_t* inputx1_data = nullptr;
  inputx1_data = new int64_t[inputx1_size];
  *(inputx1_data + 0) = 0;
  inputx1DataTensor.SetData((uint8_t*)inputx1_data, inputx1_size * sizeof(int64_t));
  delete[] inputx1_data;
  auto inputx1Data = op::Constant("inputx1Data").set_attr_value(inputx1DataTensor);

  ge::Tensor crops_tensor;
  std::vector<int64_t> crops_vec2{1};
  ge::Shape crops_shape(crops_vec2);
  ge::TensorDesc crops_desc(crops_shape, FORMAT_ND, DT_INT64);
  int64_t crops_size = crops_desc.GetShape().GetShapeSize();
  crops_desc.SetSize(crops_size * sizeof(int64_t));
  crops_tensor.SetTensorDesc(crops_desc);
  int64_t* crops_data = nullptr;
  crops_data = new int64_t[crops_size];
  *(crops_data + 0) = 0;
  crops_tensor.SetData((uint8_t*)crops_data, crops_size * sizeof(int64_t));
  delete[] crops_data;
  auto concat_dim = op::Constant("concat_dim").set_attr_value(crops_tensor);

  auto concat_layer = op::ConcatV2("concatv2");
  const int64_t input_count = 10000;
  concat_layer.create_dynamic_input_x(input_count);
  for (int64_t n = 0; n < input_count; n++) {
    concat_layer.set_dynamic_input_x(n, inputx0Data);
  }
  concat_layer.set_input_concat_dim(concat_dim);
  concat_layer.set_attr_N(input_count);

  std::vector<Operator> inputs{inputx0Data, inputx1Data, concat_dim};
  std::vector<Operator> outputs{concat_layer};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("ZConcatExt2FusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findConcatV2 = false;
  int concat_v2_d_node_count = 0;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "ConcatV2D") {
      findConcatV2 = true;
      concat_v2_d_node_count++;
    }
  }
  EXPECT_EQ(findConcatV2, true);
  EXPECT_EQ(concat_v2_d_node_count, 163);
}

TEST_F(concatv2_fusion_test, concatv2_fusion_test_6) {
  ge::Graph graph("concatv2_fusion_test_6");

  ge::Tensor inputx0DataTensor;
  std::vector<int64_t> crops_vec0{0};
  ge::Shape shape_x0(crops_vec0);
  TensorDesc tensorDescX0(shape_x0, FORMAT_ND, DT_INT64);
  int64_t inputx0_size = tensorDescX0.GetShape().GetShapeSize();
  tensorDescX0.SetSize(inputx0_size * sizeof(int64_t));
  inputx0DataTensor.SetTensorDesc(tensorDescX0);
  int64_t* inputx0_data = nullptr;
  inputx0_data = new int64_t[inputx0_size];
  *(inputx0_data + 0) = 0;
  inputx0DataTensor.SetData((uint8_t*)inputx0_data, inputx0_size * sizeof(int64_t));
  delete[] inputx0_data;
  auto inputx0Data = op::Constant("inputx0Data").set_attr_value(inputx0DataTensor);

  ge::Tensor inputx1DataTensor;
  std::vector<int64_t> crops_vec1{1};
  ge::Shape shape_x1(crops_vec1);
  ge::TensorDesc tensorDescX1(shape_x1, FORMAT_ND, DT_INT64);
  int64_t inputx1_size = tensorDescX1.GetShape().GetShapeSize();
  tensorDescX1.SetSize(inputx1_size * sizeof(int64_t));
  inputx1DataTensor.SetTensorDesc(tensorDescX1);
  int64_t* inputx1_data = nullptr;
  inputx1_data = new int64_t[inputx1_size];
  *(inputx1_data + 0) = 0;
  inputx1DataTensor.SetData((uint8_t*)inputx1_data, inputx1_size * sizeof(int64_t));
  delete[] inputx1_data;
  auto inputx1Data = op::Constant("inputx1Data").set_attr_value(inputx1DataTensor);

  ge::Tensor crops_tensor;
  std::vector<int64_t> crops_vec2{1};
  ge::Shape crops_shape(crops_vec2);
  ge::TensorDesc crops_desc(crops_shape, FORMAT_ND, DT_INT64);
  int64_t crops_size = crops_desc.GetShape().GetShapeSize();
  crops_desc.SetSize(crops_size * sizeof(int64_t));
  crops_tensor.SetTensorDesc(crops_desc);
  int64_t* crops_data = nullptr;
  crops_data = new int64_t[crops_size];
  *(crops_data + 0) = 0;
  crops_tensor.SetData((uint8_t*)crops_data, crops_size * sizeof(int64_t));
  delete[] crops_data;									     
  auto concat_dim = op::Constant("concat_dim").set_attr_value(crops_tensor);												  
  auto concat_layer = op::ConcatV2("concatv2");
  concat_layer.create_dynamic_input_x(18);
  for (int64_t n = 0; n < 18; n++) {
    concat_layer.set_dynamic_input_x(n, inputx0Data);
  }
  concat_layer.set_input_concat_dim(concat_dim);
  concat_layer.set_attr_N(18);

  std::vector<Operator> inputs{inputx0Data, inputx1Data, concat_dim};
  std::vector<Operator> outputs{concat_layer};
  graph.SetInputs(inputs).SetOutputs(outputs);
  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::InferShapeAndType(compute_graph_ptr);
  fe::FusionPassTestUtils::RunGraphFusionPass("ZConcatExt2FusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  bool findConcatV2 = false;
  int concat_v2_d_node_count = 0;
  for (auto node: compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "ConcatV2D") {
      findConcatV2 = true;
      concat_v2_d_node_count++;
    }
  }
  EXPECT_EQ(findConcatV2, true);
  EXPECT_EQ(concat_v2_d_node_count, 1);
}
																						
