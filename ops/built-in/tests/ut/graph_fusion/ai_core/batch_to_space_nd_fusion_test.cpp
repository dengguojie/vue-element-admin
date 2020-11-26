#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "transformation_ops.h"
#include "array_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class batch_to_space_nd_fusion_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "batch_to_space_nd_fusion_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "batch_to_space_nd_fusion_test TearDown" << std::endl;
  }
};

/*NHWC-4D OK
input shape = [1,2,3,4] float
block shape = [2,] int64 .... [1,1]
crops shape = [2,2] int64 .... [[0,0],[0,0]]*/
TEST_F(batch_to_space_nd_fusion_test, batch_to_space_nd_fusion_test_1_1) {
  ge::Graph graph("batch_to_space_nd_fusion_test_1_1");

  ge::Tensor block_tensor;
  std::vector<int64_t> block_vec{2};
  ge::Shape block_shape(block_vec);
  ge::TensorDesc block_desc(block_shape, FORMAT_ND, DT_INT64);
  int64_t block_size = block_desc.GetShape().GetShapeSize();
  block_desc.SetSize(block_size * sizeof(int64_t));
  block_tensor.SetTensorDesc(block_desc);
  int64_t* block_data = nullptr;
  block_data = new int64_t[block_size];
  *(block_data + 0) = 1;
  *(block_data + 1) = 1;
  block_tensor.SetData((uint8_t*)block_data, block_size * sizeof(int64_t));
  delete[] block_data;

  ge::Tensor crops_tensor;
  std::vector<int64_t> crops_vec{4};
  ge::Shape crops_shape(crops_vec);
  ge::TensorDesc crops_desc(crops_shape, FORMAT_ND, DT_INT64);
  int64_t crops_size = crops_desc.GetShape().GetShapeSize();
  crops_desc.SetSize(crops_size * sizeof(int64_t));
  crops_tensor.SetTensorDesc(crops_desc);
  int64_t* crops_data = nullptr;
  crops_data = new int64_t[crops_size];
  *(crops_data + 0) = 0;
  *(crops_data + 1) = 0;
  *(crops_data + 2) = 0;
  *(crops_data + 3) = 0;
  crops_tensor.SetData((uint8_t*)crops_data, crops_size * sizeof(int64_t));
  delete[] crops_data;

  auto block = op::Constant().set_attr_value(block_tensor);
  auto crops = op::Constant().set_attr_value(crops_tensor);
  auto data0 = op::Data().set_attr_index(0);
  auto batch_op = op::BatchToSpaceND("batch_op").set_input_x(data0).set_input_block_shape(block).set_input_crops(crops);

  block.update_output_desc_y(block_desc);
  crops.update_output_desc_y(crops_desc);

  std::vector<int64_t> data0_vec{1, 2, 3, 4};
  ge::Shape data0_shape(data0_vec);
  ge::TensorDesc data0_desc(data0_shape, FORMAT_NHWC, DT_FLOAT);
  data0.update_input_desc_x(data0_desc);
  data0.update_output_desc_y(data0_desc);
  batch_op.update_input_desc_x(data0_desc);
  batch_op.update_output_desc_y(data0_desc);
  std::vector<Operator> inputs{data0};
  std::vector<Operator> outputs{batch_op};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("ConstToAttrBatchToSpaceNdPass", fe::BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);

  bool findOp = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "BatchToSpaceNDD") {
      findOp = true;
    }
  }
  EXPECT_EQ(findOp, true);
}
/*NHWC(NWC)-3D OK
input shape = [1,2,3] float
block shape = [1,] int64 .... [1]
crops shape = [1,2] int64 .... [[0,0]]*/
TEST_F(batch_to_space_nd_fusion_test, batch_to_space_nd_fusion_test_1_2) {
  ge::Graph graph("batch_to_space_nd_fusion_test_1_2");

  ge::Tensor block_tensor;
  std::vector<int64_t> block_vec{1};
  ge::Shape block_shape(block_vec);
  ge::TensorDesc block_desc(block_shape, FORMAT_ND, DT_INT64);
  int64_t block_size = block_desc.GetShape().GetShapeSize();
  block_desc.SetSize(block_size * sizeof(int64_t));
  block_tensor.SetTensorDesc(block_desc);
  int64_t* block_data = nullptr;
  block_data = new int64_t[block_size];
  *(block_data + 0) = 1;
  block_tensor.SetData((uint8_t*)block_data, block_size * sizeof(int64_t));
  delete[] block_data;

  ge::Tensor crops_tensor;
  std::vector<int64_t> crops_vec{2};
  ge::Shape crops_shape(crops_vec);
  ge::TensorDesc crops_desc(crops_shape, FORMAT_ND, DT_INT64);
  int64_t crops_size = crops_desc.GetShape().GetShapeSize();
  crops_desc.SetSize(crops_size * sizeof(int64_t));
  crops_tensor.SetTensorDesc(crops_desc);
  int64_t* crops_data = nullptr;
  crops_data = new int64_t[crops_size];
  *(crops_data + 0) = 0;
  *(crops_data + 1) = 0;
  crops_tensor.SetData((uint8_t*)crops_data, crops_size * sizeof(int64_t));
  delete[] crops_data;

  auto block = op::Constant().set_attr_value(block_tensor);
  auto crops = op::Constant().set_attr_value(crops_tensor);
  auto data0 = op::Data().set_attr_index(0);
  auto batch_op = op::BatchToSpaceND("batch_op").set_input_x(data0).set_input_block_shape(block).set_input_crops(crops);

  block.update_output_desc_y(block_desc);
  crops.update_output_desc_y(crops_desc);

  std::vector<int64_t> data0_vec{1, 2, 3};
  ge::Shape data0_shape(data0_vec);
  ge::TensorDesc data0_desc(data0_shape, FORMAT_NHWC, DT_FLOAT);
  data0.update_input_desc_x(data0_desc);
  data0.update_output_desc_y(data0_desc);
  batch_op.update_input_desc_x(data0_desc);
  batch_op.update_output_desc_y(data0_desc);
  std::vector<Operator> inputs{data0};
  std::vector<Operator> outputs{batch_op};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("ConstToAttrBatchToSpaceNdPass", fe::BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);

  bool findOp = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "BatchToSpaceNDD") {
      findOp = true;
    }
  }
  EXPECT_EQ(findOp, true);
}
/*NCHW OK
input shape = [1,2,3,4] float
block shape = [3,] int32 .... [1,1,1]
crops shape = [3,2] int32 .... [[0,0],[0,0],[0,0]]*/
TEST_F(batch_to_space_nd_fusion_test, batch_to_space_nd_fusion_test_2) {
  ge::Graph graph("batch_to_space_nd_fusion_test_2");

  ge::Tensor block_tensor;
  std::vector<int64_t> block_vec{3};
  ge::Shape block_shape(block_vec);
  ge::TensorDesc block_desc(block_shape, FORMAT_ND, DT_INT32);
  int64_t block_size = block_desc.GetShape().GetShapeSize();
  block_desc.SetSize(block_size * sizeof(int32_t));
  block_tensor.SetTensorDesc(block_desc);
  int32_t* block_data = nullptr;
  block_data = new int32_t[block_size];
  *(block_data + 0) = 1;
  *(block_data + 1) = 1;
  *(block_data + 2) = 1;
  block_tensor.SetData((uint8_t*)block_data, block_size * sizeof(int32_t));
  delete[] block_data;

  ge::Tensor crops_tensor;
  std::vector<int64_t> crops_vec{6};
  ge::Shape crops_shape(crops_vec);
  ge::TensorDesc crops_desc(crops_shape, FORMAT_ND, DT_INT32);
  int64_t crops_size = crops_desc.GetShape().GetShapeSize();
  crops_desc.SetSize(crops_size * sizeof(int32_t));
  crops_tensor.SetTensorDesc(crops_desc);
  int32_t* crops_data = nullptr;
  crops_data = new int32_t[crops_size];
  *(crops_data + 0) = 0;
  *(crops_data + 1) = 0;
  *(crops_data + 2) = 0;
  *(crops_data + 3) = 0;
  *(crops_data + 4) = 0;
  *(crops_data + 5) = 0;
  crops_tensor.SetData((uint8_t*)crops_data, crops_size * sizeof(int32_t));
  delete[] crops_data;

  auto block = op::Constant().set_attr_value(block_tensor);
  auto crops = op::Constant().set_attr_value(crops_tensor);
  auto data0 = op::Data().set_attr_index(0);
  auto batch_op = op::BatchToSpaceND("batch_op").set_input_x(data0).set_input_block_shape(block).set_input_crops(crops);

  block.update_output_desc_y(block_desc);
  crops.update_output_desc_y(crops_desc);

  std::vector<int64_t> data0_vec{1, 2, 3, 4};
  ge::Shape data0_shape(data0_vec);
  ge::TensorDesc data0_desc(data0_shape, FORMAT_NCHW, DT_FLOAT);
  data0.update_input_desc_x(data0_desc);
  data0.update_output_desc_y(data0_desc);
  batch_op.update_input_desc_x(data0_desc);
  batch_op.update_output_desc_y(data0_desc);
  std::vector<Operator> inputs{data0};
  std::vector<Operator> outputs{batch_op};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("ConstToAttrBatchToSpaceNdPass", fe::BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);

  bool findOp = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "BatchToSpaceNDD") {
      findOp = true;
    }
  }
  EXPECT_EQ(findOp, true);
}
/*NDHWC OK
input shape = [1,2,3,4,5] float
block shape = [3,] int64 .... [1,1,1]
crops shape = [3,2] int64 .... [[0,0],[0,0],[0,0]]*/
TEST_F(batch_to_space_nd_fusion_test, batch_to_space_nd_fusion_test_3) {
  ge::Graph graph("batch_to_space_nd_fusion_test_3");

  ge::Tensor block_tensor;
  std::vector<int64_t> block_vec{3};
  ge::Shape block_shape(block_vec);
  ge::TensorDesc block_desc(block_shape, FORMAT_ND, DT_INT64);
  int64_t block_size = block_desc.GetShape().GetShapeSize();
  block_desc.SetSize(block_size * sizeof(int64_t));
  block_tensor.SetTensorDesc(block_desc);
  int64_t* block_data = nullptr;
  block_data = new int64_t[block_size];
  *(block_data + 0) = 1;
  *(block_data + 1) = 1;
  *(block_data + 2) = 1;
  block_tensor.SetData((uint8_t*)block_data, block_size * sizeof(int64_t));
  delete[] block_data;

  ge::Tensor crops_tensor;
  std::vector<int64_t> crops_vec{6};
  ge::Shape crops_shape(crops_vec);
  ge::TensorDesc crops_desc(crops_shape, FORMAT_ND, DT_INT64);
  int64_t crops_size = crops_desc.GetShape().GetShapeSize();
  crops_desc.SetSize(crops_size * sizeof(int64_t));
  crops_tensor.SetTensorDesc(crops_desc);
  int64_t* crops_data = nullptr;
  crops_data = new int64_t[crops_size];
  *(crops_data + 0) = 0;
  *(crops_data + 1) = 0;
  *(crops_data + 2) = 0;
  *(crops_data + 3) = 0;
  *(crops_data + 4) = 0;
  *(crops_data + 5) = 0;
  crops_tensor.SetData((uint8_t*)crops_data, crops_size * sizeof(int64_t));
  delete[] crops_data;

  auto block = op::Constant().set_attr_value(block_tensor);
  auto crops = op::Constant().set_attr_value(crops_tensor);
  auto data0 = op::Data().set_attr_index(0);
  auto batch_op = op::BatchToSpaceND("batch_op").set_input_x(data0).set_input_block_shape(block).set_input_crops(crops);

  block.update_output_desc_y(block_desc);
  crops.update_output_desc_y(crops_desc);

  std::vector<int64_t> data0_vec{1, 2, 3, 4, 5};
  ge::Shape data0_shape(data0_vec);
  ge::TensorDesc data0_desc(data0_shape, FORMAT_NDHWC, DT_FLOAT);
  data0.update_input_desc_x(data0_desc);
  data0.update_output_desc_y(data0_desc);
  batch_op.update_input_desc_x(data0_desc);
  batch_op.update_output_desc_y(data0_desc);
  std::vector<Operator> inputs{data0};
  std::vector<Operator> outputs{batch_op};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("ConstToAttrBatchToSpaceNdPass", fe::BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);

  bool findOp = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "BatchToSpaceNDD") {
      findOp = true;
    }
  }
  EXPECT_EQ(findOp, true);
}
/*NCDHW OK
input shape = [1,2,3,4,5] float
block shape = [4,] int32 .... [1,1,1,1]
crops shape = [4,2] int32 .... [[0,0],[0,0],[0,0],[0,0]]*/
TEST_F(batch_to_space_nd_fusion_test, batch_to_space_nd_fusion_test_4) {
  ge::Graph graph("batch_to_space_nd_fusion_test_4");

  ge::Tensor block_tensor;
  std::vector<int64_t> block_vec{4};
  ge::Shape block_shape(block_vec);
  ge::TensorDesc block_desc(block_shape, FORMAT_ND, DT_INT32);
  int64_t block_size = block_desc.GetShape().GetShapeSize();
  block_desc.SetSize(block_size * sizeof(int32_t));
  block_tensor.SetTensorDesc(block_desc);
  int32_t* block_data = nullptr;
  block_data = new int32_t[block_size];
  *(block_data + 0) = 1;
  *(block_data + 1) = 1;
  *(block_data + 2) = 1;
  *(block_data + 3) = 1;
  block_tensor.SetData((uint8_t*)block_data, block_size * sizeof(int32_t));
  delete[] block_data;

  ge::Tensor crops_tensor;
  std::vector<int64_t> crops_vec{8};
  ge::Shape crops_shape(crops_vec);
  ge::TensorDesc crops_desc(crops_shape, FORMAT_ND, DT_INT32);
  int64_t crops_size = crops_desc.GetShape().GetShapeSize();
  crops_desc.SetSize(crops_size * sizeof(int32_t));
  crops_tensor.SetTensorDesc(crops_desc);
  int32_t* crops_data = nullptr;
  crops_data = new int32_t[crops_size];
  *(crops_data + 0) = 0;
  *(crops_data + 1) = 0;
  *(crops_data + 2) = 0;
  *(crops_data + 3) = 0;
  *(crops_data + 4) = 0;
  *(crops_data + 5) = 0;
  *(crops_data + 6) = 0;
  *(crops_data + 7) = 0;
  crops_tensor.SetData((uint8_t*)crops_data, crops_size * sizeof(int32_t));
  delete[] crops_data;

  auto block = op::Constant().set_attr_value(block_tensor);
  auto crops = op::Constant().set_attr_value(crops_tensor);
  auto data0 = op::Data().set_attr_index(0);
  auto batch_op = op::BatchToSpaceND("batch_op").set_input_x(data0).set_input_block_shape(block).set_input_crops(crops);

  block.update_output_desc_y(block_desc);
  crops.update_output_desc_y(crops_desc);

  std::vector<int64_t> data0_vec{1, 2, 3, 4, 5};
  ge::Shape data0_shape(data0_vec);
  ge::TensorDesc data0_desc(data0_shape, FORMAT_NCDHW, DT_FLOAT);
  data0.update_input_desc_x(data0_desc);
  data0.update_output_desc_y(data0_desc);
  batch_op.update_input_desc_x(data0_desc);
  batch_op.update_output_desc_y(data0_desc);
  std::vector<Operator> inputs{data0};
  std::vector<Operator> outputs{batch_op};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("ConstToAttrBatchToSpaceNdPass", fe::BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);

  bool findOp = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "BatchToSpaceNDD") {
      findOp = true;
    }
  }
  EXPECT_EQ(findOp, true);
}
/*ND NOK*/
TEST_F(batch_to_space_nd_fusion_test, batch_to_space_nd_fusion_test_5) {
  ge::Graph graph("batch_to_space_nd_fusion_test_5");

  ge::Tensor block_tensor;
  std::vector<int64_t> block_vec{2};
  ge::Shape block_shape(block_vec);
  ge::TensorDesc block_desc(block_shape, FORMAT_ND, DT_INT64);
  int64_t block_size = block_desc.GetShape().GetShapeSize();
  block_desc.SetSize(block_size * sizeof(int64_t));
  block_tensor.SetTensorDesc(block_desc);
  int64_t* block_data = nullptr;
  block_data = new int64_t[block_size];
  *(block_data + 0) = 1;
  *(block_data + 1) = 1;
  block_tensor.SetData((uint8_t*)block_data, block_size * sizeof(int64_t));
  delete[] block_data;

  ge::Tensor crops_tensor;
  std::vector<int64_t> crops_vec{4};
  ge::Shape crops_shape(crops_vec);
  ge::TensorDesc crops_desc(crops_shape, FORMAT_ND, DT_INT64);
  int64_t crops_size = crops_desc.GetShape().GetShapeSize();
  crops_desc.SetSize(crops_size * sizeof(int64_t));
  crops_tensor.SetTensorDesc(crops_desc);
  int64_t* crops_data = nullptr;
  crops_data = new int64_t[crops_size];
  *(crops_data + 0) = 0;
  *(crops_data + 1) = 0;
  *(crops_data + 2) = 0;
  *(crops_data + 3) = 0;
  crops_tensor.SetData((uint8_t*)crops_data, crops_size * sizeof(int64_t));
  delete[] crops_data;

  auto block = op::Constant().set_attr_value(block_tensor);
  auto crops = op::Constant().set_attr_value(crops_tensor);
  auto data0 = op::Data().set_attr_index(0);
  auto batch_op = op::BatchToSpaceND("batch_op").set_input_x(data0).set_input_block_shape(block).set_input_crops(crops);

  block.update_output_desc_y(block_desc);
  crops.update_output_desc_y(crops_desc);

  std::vector<int64_t> data0_vec{1, 2, 3, 4};
  ge::Shape data0_shape(data0_vec);
  ge::TensorDesc data0_desc(data0_shape, FORMAT_ND, DT_FLOAT);
  data0.update_input_desc_x(data0_desc);
  data0.update_output_desc_y(data0_desc);
  batch_op.update_input_desc_x(data0_desc);
  batch_op.update_output_desc_y(data0_desc);
  std::vector<Operator> inputs{data0};
  std::vector<Operator> outputs{batch_op};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("ConstToAttrBatchToSpaceNdPass", fe::BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);

  bool findOp = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "BatchToSpaceNDD") {
      findOp = true;
    }
  }
  EXPECT_EQ(findOp, false);
}
/*NHWC NOK:
1) input shape is 5D
2) input shape is 4D block_shape not 1D [2,] and crops not 2D [2,2]
3) input shape is 3D block_shape not 1D [1,] and crops not 2D [1,2]*/
TEST_F(batch_to_space_nd_fusion_test, batch_to_space_nd_fusion_test_6_1) {
  ge::Graph graph("batch_to_space_nd_fusion_test_6_1");

  ge::Tensor block_tensor;
  std::vector<int64_t> block_vec{2};
  ge::Shape block_shape(block_vec);
  ge::TensorDesc block_desc(block_shape, FORMAT_ND, DT_INT64);
  int64_t block_size = block_desc.GetShape().GetShapeSize();
  block_desc.SetSize(block_size * sizeof(int64_t));
  block_tensor.SetTensorDesc(block_desc);
  int64_t* block_data = nullptr;
  block_data = new int64_t[block_size];
  *(block_data + 0) = 1;
  *(block_data + 1) = 1;
  block_tensor.SetData((uint8_t*)block_data, block_size * sizeof(int64_t));
  delete[] block_data;

  ge::Tensor crops_tensor;
  std::vector<int64_t> crops_vec{4};
  ge::Shape crops_shape(crops_vec);
  ge::TensorDesc crops_desc(crops_shape, FORMAT_ND, DT_INT64);
  int64_t crops_size = crops_desc.GetShape().GetShapeSize();
  crops_desc.SetSize(crops_size * sizeof(int64_t));
  crops_tensor.SetTensorDesc(crops_desc);
  int64_t* crops_data = nullptr;
  crops_data = new int64_t[crops_size];
  *(crops_data + 0) = 0;
  *(crops_data + 1) = 0;
  *(crops_data + 2) = 0;
  *(crops_data + 3) = 0;
  crops_tensor.SetData((uint8_t*)crops_data, crops_size * sizeof(int64_t));
  delete[] crops_data;

  auto block = op::Constant().set_attr_value(block_tensor);
  auto crops = op::Constant().set_attr_value(crops_tensor);
  auto data0 = op::Data().set_attr_index(0);
  auto batch_op = op::BatchToSpaceND("batch_op").set_input_x(data0).set_input_block_shape(block).set_input_crops(crops);

  block.update_output_desc_y(block_desc);
  crops.update_output_desc_y(crops_desc);

  std::vector<int64_t> data0_vec{1, 2, 3, 4, 5};
  ge::Shape data0_shape(data0_vec);
  ge::TensorDesc data0_desc(data0_shape, FORMAT_NHWC, DT_FLOAT);
  data0.update_input_desc_x(data0_desc);
  data0.update_output_desc_y(data0_desc);
  batch_op.update_input_desc_x(data0_desc);
  batch_op.update_output_desc_y(data0_desc);
  std::vector<Operator> inputs{data0};
  std::vector<Operator> outputs{batch_op};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("ConstToAttrBatchToSpaceNdPass", fe::BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);

  bool findOp = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "BatchToSpaceNDD") {
      findOp = true;
    }
  }
  EXPECT_EQ(findOp, false);
}
TEST_F(batch_to_space_nd_fusion_test, batch_to_space_nd_fusion_test_6_2) {
  ge::Graph graph("batch_to_space_nd_fusion_test_6_2");

  ge::Tensor block_tensor;
  std::vector<int64_t> block_vec{1};
  ge::Shape block_shape(block_vec);
  ge::TensorDesc block_desc(block_shape, FORMAT_ND, DT_INT64);
  int64_t block_size = block_desc.GetShape().GetShapeSize();
  block_desc.SetSize(block_size * sizeof(int64_t));
  block_tensor.SetTensorDesc(block_desc);
  int64_t* block_data = nullptr;
  block_data = new int64_t[block_size];
  *(block_data + 0) = 1;
  block_tensor.SetData((uint8_t*)block_data, block_size * sizeof(int64_t));
  delete[] block_data;

  ge::Tensor crops_tensor;
  std::vector<int64_t> crops_vec{2};
  ge::Shape crops_shape(crops_vec);
  ge::TensorDesc crops_desc(crops_shape, FORMAT_ND, DT_INT64);
  int64_t crops_size = crops_desc.GetShape().GetShapeSize();
  crops_desc.SetSize(crops_size * sizeof(int64_t));
  crops_tensor.SetTensorDesc(crops_desc);
  int64_t* crops_data = nullptr;
  crops_data = new int64_t[crops_size];
  *(crops_data + 0) = 0;
  *(crops_data + 1) = 0;
  crops_tensor.SetData((uint8_t*)crops_data, crops_size * sizeof(int64_t));
  delete[] crops_data;

  auto block = op::Constant().set_attr_value(block_tensor);
  auto crops = op::Constant().set_attr_value(crops_tensor);
  auto data0 = op::Data().set_attr_index(0);
  auto batch_op = op::BatchToSpaceND("batch_op").set_input_x(data0).set_input_block_shape(block).set_input_crops(crops);

  block.update_output_desc_y(block_desc);
  crops.update_output_desc_y(crops_desc);

  std::vector<int64_t> data0_vec{1, 2, 3, 4};
  ge::Shape data0_shape(data0_vec);
  ge::TensorDesc data0_desc(data0_shape, FORMAT_NHWC, DT_FLOAT);
  data0.update_input_desc_x(data0_desc);
  data0.update_output_desc_y(data0_desc);
  batch_op.update_input_desc_x(data0_desc);
  batch_op.update_output_desc_y(data0_desc);
  std::vector<Operator> inputs{data0};
  std::vector<Operator> outputs{batch_op};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("ConstToAttrBatchToSpaceNdPass", fe::BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);

  bool findOp = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "BatchToSpaceNDD") {
      findOp = true;
    }
  }
  EXPECT_EQ(findOp, false);
}
TEST_F(batch_to_space_nd_fusion_test, batch_to_space_nd_fusion_test_6_3) {
  ge::Graph graph("batch_to_space_nd_fusion_test_6_3");

  ge::Tensor block_tensor;
  std::vector<int64_t> block_vec{2};
  ge::Shape block_shape(block_vec);
  ge::TensorDesc block_desc(block_shape, FORMAT_ND, DT_INT64);
  int64_t block_size = block_desc.GetShape().GetShapeSize();
  block_desc.SetSize(block_size * sizeof(int64_t));
  block_tensor.SetTensorDesc(block_desc);
  int64_t* block_data = nullptr;
  block_data = new int64_t[block_size];
  *(block_data + 0) = 1;
  *(block_data + 0) = 2;
  block_tensor.SetData((uint8_t*)block_data, block_size * sizeof(int64_t));
  delete[] block_data;

  ge::Tensor crops_tensor;
  std::vector<int64_t> crops_vec{4};
  ge::Shape crops_shape(crops_vec);
  ge::TensorDesc crops_desc(crops_shape, FORMAT_ND, DT_INT64);
  int64_t crops_size = crops_desc.GetShape().GetShapeSize();
  crops_desc.SetSize(crops_size * sizeof(int64_t));
  crops_tensor.SetTensorDesc(crops_desc);
  int64_t* crops_data = nullptr;
  crops_data = new int64_t[crops_size];
  *(crops_data + 0) = 0;
  *(crops_data + 1) = 0;
  *(crops_data + 2) = 0;
  *(crops_data + 3) = 0;
  crops_tensor.SetData((uint8_t*)crops_data, crops_size * sizeof(int64_t));
  delete[] crops_data;

  auto block = op::Constant().set_attr_value(block_tensor);
  auto crops = op::Constant().set_attr_value(crops_tensor);
  auto data0 = op::Data().set_attr_index(0);
  auto batch_op = op::BatchToSpaceND("batch_op").set_input_x(data0).set_input_block_shape(block).set_input_crops(crops);

  block.update_output_desc_y(block_desc);
  crops.update_output_desc_y(crops_desc);

  std::vector<int64_t> data0_vec{1, 2, 3};
  ge::Shape data0_shape(data0_vec);
  ge::TensorDesc data0_desc(data0_shape, FORMAT_NHWC, DT_FLOAT);
  data0.update_input_desc_x(data0_desc);
  data0.update_output_desc_y(data0_desc);
  batch_op.update_input_desc_x(data0_desc);
  batch_op.update_output_desc_y(data0_desc);
  std::vector<Operator> inputs{data0};
  std::vector<Operator> outputs{batch_op};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("ConstToAttrBatchToSpaceNdPass", fe::BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);

  bool findOp = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "BatchToSpaceNDD") {
      findOp = true;
    }
  }
  EXPECT_EQ(findOp, false);
}
/*NCHW NOK:
1) input shape is 5D
2) input shape is 4D block_shape not 1D [2,] and crops not 2D [2,2]*/
TEST_F(batch_to_space_nd_fusion_test, batch_to_space_nd_fusion_test_7_1) {
  ge::Graph graph("batch_to_space_nd_fusion_test_7_1");

  ge::Tensor block_tensor;
  std::vector<int64_t> block_vec{3};
  ge::Shape block_shape(block_vec);
  ge::TensorDesc block_desc(block_shape, FORMAT_ND, DT_INT32);
  int64_t block_size = block_desc.GetShape().GetShapeSize();
  block_desc.SetSize(block_size * sizeof(int32_t));
  block_tensor.SetTensorDesc(block_desc);
  int32_t* block_data = nullptr;
  block_data = new int32_t[block_size];
  *(block_data + 0) = 1;
  *(block_data + 1) = 1;
  *(block_data + 2) = 1;
  block_tensor.SetData((uint8_t*)block_data, block_size * sizeof(int32_t));
  delete[] block_data;

  ge::Tensor crops_tensor;
  std::vector<int64_t> crops_vec{6};
  ge::Shape crops_shape(crops_vec);
  ge::TensorDesc crops_desc(crops_shape, FORMAT_ND, DT_INT32);
  int64_t crops_size = crops_desc.GetShape().GetShapeSize();
  crops_desc.SetSize(crops_size * sizeof(int32_t));
  crops_tensor.SetTensorDesc(crops_desc);
  int32_t* crops_data = nullptr;
  crops_data = new int32_t[crops_size];
  *(crops_data + 0) = 0;
  *(crops_data + 1) = 0;
  *(crops_data + 2) = 0;
  *(crops_data + 3) = 0;
  *(crops_data + 4) = 0;
  *(crops_data + 5) = 0;
  crops_tensor.SetData((uint8_t*)crops_data, crops_size * sizeof(int32_t));
  delete[] crops_data;

  auto block = op::Constant().set_attr_value(block_tensor);
  auto crops = op::Constant().set_attr_value(crops_tensor);
  auto data0 = op::Data().set_attr_index(0);
  auto batch_op = op::BatchToSpaceND("batch_op").set_input_x(data0).set_input_block_shape(block).set_input_crops(crops);

  block.update_output_desc_y(block_desc);
  crops.update_output_desc_y(crops_desc);

  std::vector<int64_t> data0_vec{1, 2, 3, 4, 5};
  ge::Shape data0_shape(data0_vec);
  ge::TensorDesc data0_desc(data0_shape, FORMAT_NCHW, DT_FLOAT);
  data0.update_input_desc_x(data0_desc);
  data0.update_output_desc_y(data0_desc);
  batch_op.update_input_desc_x(data0_desc);
  batch_op.update_output_desc_y(data0_desc);
  std::vector<Operator> inputs{data0};
  std::vector<Operator> outputs{batch_op};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("ConstToAttrBatchToSpaceNdPass", fe::BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);

  bool findOp = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "BatchToSpaceNDD") {
      findOp = true;
    }
  }
  EXPECT_EQ(findOp, false);
}
TEST_F(batch_to_space_nd_fusion_test, batch_to_space_nd_fusion_test_7_2) {
  ge::Graph graph("batch_to_space_nd_fusion_test_7_2");

  ge::Tensor block_tensor;
  std::vector<int64_t> block_vec{2};
  ge::Shape block_shape(block_vec);
  ge::TensorDesc block_desc(block_shape, FORMAT_ND, DT_INT32);
  int64_t block_size = block_desc.GetShape().GetShapeSize();
  block_desc.SetSize(block_size * sizeof(int32_t));
  block_tensor.SetTensorDesc(block_desc);
  int32_t* block_data = nullptr;
  block_data = new int32_t[block_size];
  *(block_data + 0) = 1;
  *(block_data + 1) = 1;
  block_tensor.SetData((uint8_t*)block_data, block_size * sizeof(int32_t));
  delete[] block_data;

  ge::Tensor crops_tensor;
  std::vector<int64_t> crops_vec{4};
  ge::Shape crops_shape(crops_vec);
  ge::TensorDesc crops_desc(crops_shape, FORMAT_ND, DT_INT32);
  int64_t crops_size = crops_desc.GetShape().GetShapeSize();
  crops_desc.SetSize(crops_size * sizeof(int32_t));
  crops_tensor.SetTensorDesc(crops_desc);
  int32_t* crops_data = nullptr;
  crops_data = new int32_t[crops_size];
  *(crops_data + 0) = 0;
  *(crops_data + 1) = 0;
  *(crops_data + 2) = 0;
  *(crops_data + 3) = 0;
  crops_tensor.SetData((uint8_t*)crops_data, crops_size * sizeof(int32_t));
  delete[] crops_data;

  auto block = op::Constant().set_attr_value(block_tensor);
  auto crops = op::Constant().set_attr_value(crops_tensor);
  auto data0 = op::Data().set_attr_index(0);
  auto batch_op = op::BatchToSpaceND("batch_op").set_input_x(data0).set_input_block_shape(block).set_input_crops(crops);

  block.update_output_desc_y(block_desc);
  crops.update_output_desc_y(crops_desc);

  std::vector<int64_t> data0_vec{1, 2, 3, 4};
  ge::Shape data0_shape(data0_vec);
  ge::TensorDesc data0_desc(data0_shape, FORMAT_NCHW, DT_FLOAT);
  data0.update_input_desc_x(data0_desc);
  data0.update_output_desc_y(data0_desc);
  batch_op.update_input_desc_x(data0_desc);
  batch_op.update_output_desc_y(data0_desc);
  std::vector<Operator> inputs{data0};
  std::vector<Operator> outputs{batch_op};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("ConstToAttrBatchToSpaceNdPass", fe::BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);

  bool findOp = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "BatchToSpaceNDD") {
      findOp = true;
    }
  }
  EXPECT_EQ(findOp, false);
}
/*NDHWC NOK:
1) input shape is 6D
2) input shape is 5D block_shape not 1D [3,] and crops not 2D [3,2]*/
TEST_F(batch_to_space_nd_fusion_test, batch_to_space_nd_fusion_test_8_1) {
  ge::Graph graph("batch_to_space_nd_fusion_test_8_1");

  ge::Tensor block_tensor;
  std::vector<int64_t> block_vec{3};
  ge::Shape block_shape(block_vec);
  ge::TensorDesc block_desc(block_shape, FORMAT_ND, DT_INT64);
  int64_t block_size = block_desc.GetShape().GetShapeSize();
  block_desc.SetSize(block_size * sizeof(int64_t));
  block_tensor.SetTensorDesc(block_desc);
  int64_t* block_data = nullptr;
  block_data = new int64_t[block_size];
  *(block_data + 0) = 1;
  *(block_data + 1) = 1;
  *(block_data + 2) = 1;
  block_tensor.SetData((uint8_t*)block_data, block_size * sizeof(int64_t));
  delete[] block_data;

  ge::Tensor crops_tensor;
  std::vector<int64_t> crops_vec{6};
  ge::Shape crops_shape(crops_vec);
  ge::TensorDesc crops_desc(crops_shape, FORMAT_ND, DT_INT64);
  int64_t crops_size = crops_desc.GetShape().GetShapeSize();
  crops_desc.SetSize(crops_size * sizeof(int64_t));
  crops_tensor.SetTensorDesc(crops_desc);
  int64_t* crops_data = nullptr;
  crops_data = new int64_t[crops_size];
  *(crops_data + 0) = 0;
  *(crops_data + 1) = 0;
  *(crops_data + 2) = 0;
  *(crops_data + 3) = 0;
  *(crops_data + 4) = 0;
  *(crops_data + 5) = 0;
  crops_tensor.SetData((uint8_t*)crops_data, crops_size * sizeof(int64_t));
  delete[] crops_data;

  auto block = op::Constant().set_attr_value(block_tensor);
  auto crops = op::Constant().set_attr_value(crops_tensor);
  auto data0 = op::Data().set_attr_index(0);
  auto batch_op = op::BatchToSpaceND("batch_op").set_input_x(data0).set_input_block_shape(block).set_input_crops(crops);

  block.update_output_desc_y(block_desc);
  crops.update_output_desc_y(crops_desc);

  std::vector<int64_t> data0_vec{1, 2, 3, 4, 5, 6};
  ge::Shape data0_shape(data0_vec);
  ge::TensorDesc data0_desc(data0_shape, FORMAT_NDHWC, DT_FLOAT);
  data0.update_input_desc_x(data0_desc);
  data0.update_output_desc_y(data0_desc);
  batch_op.update_input_desc_x(data0_desc);
  batch_op.update_output_desc_y(data0_desc);
  std::vector<Operator> inputs{data0};
  std::vector<Operator> outputs{batch_op};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("ConstToAttrBatchToSpaceNdPass", fe::BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);

  bool findOp = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "BatchToSpaceNDD") {
      findOp = true;
    }
  }
  EXPECT_EQ(findOp, false);
}
TEST_F(batch_to_space_nd_fusion_test, batch_to_space_nd_fusion_test_8_2) {
  ge::Graph graph("batch_to_space_nd_fusion_test_8_2");

  ge::Tensor block_tensor;
  std::vector<int64_t> block_vec{2};
  ge::Shape block_shape(block_vec);
  ge::TensorDesc block_desc(block_shape, FORMAT_ND, DT_INT64);
  int64_t block_size = block_desc.GetShape().GetShapeSize();
  block_desc.SetSize(block_size * sizeof(int64_t));
  block_tensor.SetTensorDesc(block_desc);
  int64_t* block_data = nullptr;
  block_data = new int64_t[block_size];
  *(block_data + 0) = 1;
  *(block_data + 1) = 1;
  block_tensor.SetData((uint8_t*)block_data, block_size * sizeof(int64_t));
  delete[] block_data;

  ge::Tensor crops_tensor;
  std::vector<int64_t> crops_vec{4};
  ge::Shape crops_shape(crops_vec);
  ge::TensorDesc crops_desc(crops_shape, FORMAT_ND, DT_INT64);
  int64_t crops_size = crops_desc.GetShape().GetShapeSize();
  crops_desc.SetSize(crops_size * sizeof(int64_t));
  crops_tensor.SetTensorDesc(crops_desc);
  int64_t* crops_data = nullptr;
  crops_data = new int64_t[crops_size];
  *(crops_data + 0) = 0;
  *(crops_data + 1) = 0;
  *(crops_data + 2) = 0;
  *(crops_data + 3) = 0;
  crops_tensor.SetData((uint8_t*)crops_data, crops_size * sizeof(int64_t));
  delete[] crops_data;

  auto block = op::Constant().set_attr_value(block_tensor);
  auto crops = op::Constant().set_attr_value(crops_tensor);
  auto data0 = op::Data().set_attr_index(0);
  auto batch_op = op::BatchToSpaceND("batch_op").set_input_x(data0).set_input_block_shape(block).set_input_crops(crops);

  block.update_output_desc_y(block_desc);
  crops.update_output_desc_y(crops_desc);

  std::vector<int64_t> data0_vec{1, 2, 3, 4, 5};
  ge::Shape data0_shape(data0_vec);
  ge::TensorDesc data0_desc(data0_shape, FORMAT_NDHWC, DT_FLOAT);
  data0.update_input_desc_x(data0_desc);
  data0.update_output_desc_y(data0_desc);
  batch_op.update_input_desc_x(data0_desc);
  batch_op.update_output_desc_y(data0_desc);
  std::vector<Operator> inputs{data0};
  std::vector<Operator> outputs{batch_op};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("ConstToAttrBatchToSpaceNdPass", fe::BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);

  bool findOp = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "BatchToSpaceNDD") {
      findOp = true;
    }
  }
  EXPECT_EQ(findOp, false);
}
/*NCDHW NOK:
1) input shape is 6D
2) input shape is 5D block_shape not 1D [4,] and crops not 2D [4,2]*/
TEST_F(batch_to_space_nd_fusion_test, batch_to_space_nd_fusion_test_9_1) {
  ge::Graph graph("batch_to_space_nd_fusion_test_9_1");

  ge::Tensor block_tensor;
  std::vector<int64_t> block_vec{4};
  ge::Shape block_shape(block_vec);
  ge::TensorDesc block_desc(block_shape, FORMAT_ND, DT_INT32);
  int64_t block_size = block_desc.GetShape().GetShapeSize();
  block_desc.SetSize(block_size * sizeof(int32_t));
  block_tensor.SetTensorDesc(block_desc);
  int32_t* block_data = nullptr;
  block_data = new int32_t[block_size];
  *(block_data + 0) = 1;
  *(block_data + 1) = 1;
  *(block_data + 2) = 1;
  *(block_data + 3) = 1;
  block_tensor.SetData((uint8_t*)block_data, block_size * sizeof(int32_t));
  delete[] block_data;

  ge::Tensor crops_tensor;
  std::vector<int64_t> crops_vec{8};
  ge::Shape crops_shape(crops_vec);
  ge::TensorDesc crops_desc(crops_shape, FORMAT_ND, DT_INT32);
  int64_t crops_size = crops_desc.GetShape().GetShapeSize();
  crops_desc.SetSize(crops_size * sizeof(int32_t));
  crops_tensor.SetTensorDesc(crops_desc);
  int32_t* crops_data = nullptr;
  crops_data = new int32_t[crops_size];
  *(crops_data + 0) = 0;
  *(crops_data + 1) = 0;
  *(crops_data + 2) = 0;
  *(crops_data + 3) = 0;
  *(crops_data + 4) = 0;
  *(crops_data + 5) = 0;
  *(crops_data + 6) = 0;
  *(crops_data + 7) = 0;
  crops_tensor.SetData((uint8_t*)crops_data, crops_size * sizeof(int32_t));
  delete[] crops_data;

  auto block = op::Constant().set_attr_value(block_tensor);
  auto crops = op::Constant().set_attr_value(crops_tensor);
  auto data0 = op::Data().set_attr_index(0);
  auto batch_op = op::BatchToSpaceND("batch_op").set_input_x(data0).set_input_block_shape(block).set_input_crops(crops);

  block.update_output_desc_y(block_desc);
  crops.update_output_desc_y(crops_desc);

  std::vector<int64_t> data0_vec{1, 2, 3, 4, 5, 6};
  ge::Shape data0_shape(data0_vec);
  ge::TensorDesc data0_desc(data0_shape, FORMAT_NCDHW, DT_FLOAT);
  data0.update_input_desc_x(data0_desc);
  data0.update_output_desc_y(data0_desc);
  batch_op.update_input_desc_x(data0_desc);
  batch_op.update_output_desc_y(data0_desc);
  std::vector<Operator> inputs{data0};
  std::vector<Operator> outputs{batch_op};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("ConstToAttrBatchToSpaceNdPass", fe::BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);

  bool findOp = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "BatchToSpaceNDD") {
      findOp = true;
    }
  }
  EXPECT_EQ(findOp, false);
}
TEST_F(batch_to_space_nd_fusion_test, batch_to_space_nd_fusion_test_9_2) {
  ge::Graph graph("batch_to_space_nd_fusion_test_9_2");

  ge::Tensor block_tensor;
  std::vector<int64_t> block_vec{3};
  ge::Shape block_shape(block_vec);
  ge::TensorDesc block_desc(block_shape, FORMAT_ND, DT_INT32);
  int64_t block_size = block_desc.GetShape().GetShapeSize();
  block_desc.SetSize(block_size * sizeof(int32_t));
  block_tensor.SetTensorDesc(block_desc);
  int32_t* block_data = nullptr;
  block_data = new int32_t[block_size];
  *(block_data + 0) = 1;
  *(block_data + 1) = 1;
  *(block_data + 2) = 1;
  block_tensor.SetData((uint8_t*)block_data, block_size * sizeof(int32_t));
  delete[] block_data;

  ge::Tensor crops_tensor;
  std::vector<int64_t> crops_vec{6};
  ge::Shape crops_shape(crops_vec);
  ge::TensorDesc crops_desc(crops_shape, FORMAT_ND, DT_INT32);
  int64_t crops_size = crops_desc.GetShape().GetShapeSize();
  crops_desc.SetSize(crops_size * sizeof(int32_t));
  crops_tensor.SetTensorDesc(crops_desc);
  int32_t* crops_data = nullptr;
  crops_data = new int32_t[crops_size];
  *(crops_data + 0) = 0;
  *(crops_data + 1) = 0;
  *(crops_data + 2) = 0;
  *(crops_data + 3) = 0;
  *(crops_data + 4) = 0;
  *(crops_data + 5) = 0;
  crops_tensor.SetData((uint8_t*)crops_data, crops_size * sizeof(int32_t));
  delete[] crops_data;

  auto block = op::Constant().set_attr_value(block_tensor);
  auto crops = op::Constant().set_attr_value(crops_tensor);
  auto data0 = op::Data().set_attr_index(0);
  auto batch_op = op::BatchToSpaceND("batch_op").set_input_x(data0).set_input_block_shape(block).set_input_crops(crops);

  block.update_output_desc_y(block_desc);
  crops.update_output_desc_y(crops_desc);

  std::vector<int64_t> data0_vec{1, 2, 3, 4, 5};
  ge::Shape data0_shape(data0_vec);
  ge::TensorDesc data0_desc(data0_shape, FORMAT_NCDHW, DT_FLOAT);
  data0.update_input_desc_x(data0_desc);
  data0.update_output_desc_y(data0_desc);
  batch_op.update_input_desc_x(data0_desc);
  batch_op.update_output_desc_y(data0_desc);
  std::vector<Operator> inputs{data0};
  std::vector<Operator> outputs{batch_op};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);
  fe::FusionPassTestUtils::RunGraphFusionPass("ConstToAttrBatchToSpaceNdPass", fe::BUILT_IN_GRAPH_PASS,
                                              *compute_graph_ptr);

  bool findOp = false;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "BatchToSpaceNDD") {
      findOp = true;
    }
  }
  EXPECT_EQ(findOp, false);
}