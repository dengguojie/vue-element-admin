#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "all_ops.h"
#include "fusion_pass_test_utils.h"

using namespace ge;
using namespace op;

class pad_conv2d_fusion_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "pad_conv2d_fusion_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "pad_conv2d_fusion_test TearDown" << std::endl;
  }
};

TEST_F(pad_conv2d_fusion_test, one_pad_one_conv2d) {
  std::cout << "enter pad_conv2d_fusion_test.one_pad_one_conv2d" << std::endl;
  /*
   * data    padding_const
   *   \         /
   *    \       /
   *     \     /
   *      \   /
   *       pad    conv2d_filter_const
   *        |     /
   *        |    /
   *        conv2d
   * */

  ge::Graph graph("one_pad_one_conv2d");

  // data
  auto data = op::Data("data");
  ge::Shape data_shape({1, 224, 224, 3});
  ge::TensorDesc data_tensor_desc(data_shape, FORMAT_NHWC, DT_FLOAT);
  data_tensor_desc.SetOriginFormat(FORMAT_NHWC);
  data.update_input_desc_x(data_tensor_desc);
  data.update_output_desc_y(data_tensor_desc);

  // padding_const
  std::vector<vector<int32_t>> paddings = {{0,0}, {3,3}, {3,3}, {0,0}};
  vector<int64_t> paddings_dims = {4, 2};
  TensorDesc padding_const_tensor_desc(ge::Shape(paddings_dims), FORMAT_NHWC, DT_INT32);
  Tensor padding_const_tensor(padding_const_tensor_desc);
  uint32_t* padding_const_tensor_value = new uint32_t[8];
  for (size_t dim = 0; dim < 8; dim++) {
    *(padding_const_tensor_value + dim) = paddings[dim / 2][dim % 2];
  }
  padding_const_tensor.SetData((uint8_t*)padding_const_tensor_value, 8 * sizeof(uint32_t));
  auto paddings_const = op::Const("paddings").set_attr_value(padding_const_tensor);

  // pad
  auto pad = op::Pad("pad").set_input_x(data).set_input_paddings(paddings_const);
  pad.update_input_desc_x(data_tensor_desc);
  pad.update_input_desc_paddings(padding_const_tensor_desc);
  ge::Shape pad_output_shape({1, 230, 230, 3});
  ge::TensorDesc pad_output_tensor_desc(pad_output_shape, FORMAT_NHWC, DT_FLOAT);
  pad.update_output_desc_y(pad_output_tensor_desc);

  // conv2d_filter_const
  TensorDesc filterDesc(ge::Shape({7,7,3,64}), FORMAT_HWCN, DT_FLOAT);
  filterDesc.SetOriginFormat(FORMAT_HWCN);
  auto conv2d_filter_const = op::Const("conv2d_filter_const");
  Tensor filter;
  float * filterValue = new float[7*7*3*64];
  filter.SetTensorDesc(filterDesc);
  filter.SetData((uint8_t*)filterValue, 4*7*7*3*64);
  conv2d_filter_const.set_attr_value(filter);

  // conv2d
  auto conv2d = op::Conv2D("conv2d");
  conv2d.set_input_x(pad)
  .set_input_filter(conv2d_filter_const)
  .set_attr_strides({1,2,2,1})
  .set_attr_pads({0,0,0,0});
  conv2d.update_input_desc_x(pad_output_tensor_desc);
  conv2d.update_input_desc_filter(filterDesc);
  ge::Shape conv2d_output_shape({1, 112, 112, 64});
  ge::TensorDesc conv2d_output_tensor_desc(conv2d_output_shape, FORMAT_NHWC, DT_FLOAT);
  conv2d.update_output_desc_y(conv2d_output_tensor_desc);


  std::vector<Operator> inputs = {data,paddings_const,conv2d_filter_const};
  std::vector<Operator> outputs = {conv2d};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  ge::NodePtr pad_node = nullptr;
  ge::NodePtr conv_node = nullptr;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Pad") {
       pad_node = node;
    }
    if (node->GetType() == "Conv2D") {
      conv_node = node;
    }
  }
  ge::GraphUtils::AddEdge(pad_node->GetOutControlAnchor(), conv_node->GetInControlAnchor());


  // check before fusion
  EXPECT_EQ(compute_graph_ptr->GetAllNodesSize(), 5);

  // do fusion
  fe::FusionPassTestUtils::RunGraphFusionPass("PadConv2dFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  // check after fusion
  EXPECT_EQ(compute_graph_ptr->GetAllNodesSize(), 3);

  bool has_pad_node = false;
  vector<int64_t> pads;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "Pad") {
      has_pad_node = true;
    }
    if (node->GetType() == "Conv2D") {
      (void)ge::AttrUtils::GetListInt(node->GetOpDesc(), "pads", pads);
    }
  }
  EXPECT_EQ(has_pad_node, false);
  for (auto pad : pads) {
    EXPECT_EQ(pad, 3);
  }
}

TEST_F(pad_conv2d_fusion_test, one_pad_two_conv2d) {
  std::cout << "enter pad_conv2d_fusion_test.one_pad_two_conv2d" << std::endl;

  /*
   *                                data    padding_const
   *                                  \         /
   *                                   \       /
   *                                    \     /
   *                                     \   /
   *                                      pad
   *                                       |
   *              conv2d_filter_const_1    |       conv2d_filter_const_2
   *                           \          / \          /
   *                            \        /   \        /
   *                             \      /     \      /
   *                              \    /       \    /
   *                               \  /         \  /
   *                              conv2d_1   conv2d_2
   * */

  ge::Graph graph("one_pad_two_conv2d");

  // data
  auto data = op::Data("data");
  ge::Shape data_shape({1, 224, 224, 3});
  ge::TensorDesc data_tensor_desc(data_shape, FORMAT_NHWC, DT_FLOAT);
  data_tensor_desc.SetOriginFormat(FORMAT_NHWC);
  data.update_input_desc_x(data_tensor_desc);
  data.update_output_desc_y(data_tensor_desc);

  // padding_const
  std::vector<vector<int32_t>> paddings = {{0,0}, {3,3}, {3,3}, {0,0}};
  vector<int64_t> paddings_dims = {4, 2};
  TensorDesc padding_const_tensor_desc(ge::Shape(paddings_dims), FORMAT_NHWC, DT_INT32);
  Tensor padding_const_tensor(padding_const_tensor_desc);
  uint32_t* padding_const_tensor_value = new uint32_t[8];
  for (size_t dim = 0; dim < 8; dim++) {
    *(padding_const_tensor_value + dim) = paddings[dim / 2][dim % 2];
  }
  padding_const_tensor.SetData((uint8_t*)padding_const_tensor_value, 8 * sizeof(uint32_t));
  auto paddings_const = op::Const("paddings").set_attr_value(padding_const_tensor);

  // pad
  auto pad = op::Pad("pad").set_input_x(data).set_input_paddings(paddings_const);
  pad.update_input_desc_x(data_tensor_desc);
  pad.update_input_desc_paddings(padding_const_tensor_desc);
  ge::Shape pad_output_shape({1, 230, 230, 3});
  ge::TensorDesc pad_output_tensor_desc(pad_output_shape, FORMAT_NHWC, DT_FLOAT);
  pad.update_output_desc_y(pad_output_tensor_desc);

  // conv2d_filter_const_1
  TensorDesc filterDesc_1(ge::Shape({7,7,3,64}), FORMAT_HWCN, DT_FLOAT);
  filterDesc_1.SetOriginFormat(FORMAT_HWCN);
  auto conv2d_filter_const_1 = op::Const("conv2d_filter_const_1");
  Tensor filter_1;
  float * filterValue_1 = new float[7*7*3*64];
  filter_1.SetTensorDesc(filterDesc_1);
  filter_1.SetData((uint8_t*)filterValue_1, 4*7*7*3*64);
  conv2d_filter_const_1.set_attr_value(filter_1);

  // conv2d_filter_const_2
  TensorDesc filterDesc_2(ge::Shape({7,7,3,64}), FORMAT_HWCN, DT_FLOAT);
  filterDesc_2.SetOriginFormat(FORMAT_HWCN);
  auto conv2d_filter_const_2 = op::Const("conv2d_filter_const_2");
  Tensor filter_2;
  float * filterValue_2 = new float[7*7*3*64];
  filter_2.SetTensorDesc(filterDesc_2);
  filter_2.SetData((uint8_t*)filterValue_2, 4*7*7*3*64);
  conv2d_filter_const_2.set_attr_value(filter_2);

  // conv2d_1
  auto conv2d_1 = op::Conv2D("conv2d_1");
  conv2d_1.set_input_x(pad)
  .set_input_filter(conv2d_filter_const_1)
  .set_attr_strides({1,2,2,1})
  .set_attr_pads({0,0,0,0});
  conv2d_1.update_input_desc_x(pad_output_tensor_desc);
  conv2d_1.update_input_desc_filter(filterDesc_1);
  ge::Shape conv2d_output_shape_1({1, 112, 112, 64});
  ge::TensorDesc conv2d_output_tensor_desc_1(conv2d_output_shape_1, FORMAT_NHWC, DT_FLOAT);
  conv2d_1.update_output_desc_y(conv2d_output_tensor_desc_1);

  // conv2d_2
  auto conv2d_2 = op::Conv2D("conv2d_2");
  conv2d_2.set_input_x(pad)
  .set_input_filter(conv2d_filter_const_2)
  .set_attr_strides({1,2,2,1})
  .set_attr_pads({0,0,0,0});
  conv2d_2.update_input_desc_x(pad_output_tensor_desc);
  conv2d_2.update_input_desc_filter(filterDesc_2);
  ge::Shape conv2d_output_shape_2({1, 112, 112, 64});
  ge::TensorDesc conv2d_output_tensor_desc_2(conv2d_output_shape_2, FORMAT_NHWC, DT_FLOAT);
  conv2d_2.update_output_desc_y(conv2d_output_tensor_desc_2);


  std::vector<Operator> inputs = {data,paddings_const,conv2d_filter_const_1, conv2d_filter_const_2};
  std::vector<Operator> outputs = {conv2d_1, conv2d_2};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  // check before fusion
  EXPECT_EQ(compute_graph_ptr->GetAllNodesSize(), 7);

  // do fusion
  fe::Status status = fe::FusionPassTestUtils::RunGraphFusionPass("PadConv2dFusionPass",
      fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  // check after fusion
  EXPECT_EQ(status, fe::NOT_CHANGED);
//  EXPECT_EQ(compute_graph_ptr->GetAllNodesSize(), 7);
//
//  bool has_pad_node = false;
//  vector<int64_t> pads;
//  for (auto node : compute_graph_ptr->GetAllNodes()) {
//    if (node->GetType() == "Pad") {
//      has_pad_node = true;
//    }
//    if (node->GetType() == "Conv2D") {
//      (void)ge::AttrUtils::GetListInt(node->GetOpDesc(), "pads", pads);
//    }
//  }
//  EXPECT_EQ(has_pad_node, true);
}

TEST_F(pad_conv2d_fusion_test, one_padv3_one_conv2d) {
  std::cout << "enter pad_conv2d_fusion_test.one_pad_one_conv2d" << std::endl;
  /*
   * data    padding_const
   *   \         /
   *    \       /
   *     \     /
   *      \   /
   *       padv3    conv2d_filter_const
   *        |     /
   *        |    /
   *        conv2d
   * */

  ge::Graph graph("one_padv3_one_conv2d");

  // data
  auto data = op::Data("data");
  ge::Shape data_shape({1, 224, 224, 3});
  ge::TensorDesc data_tensor_desc(data_shape, FORMAT_NHWC, DT_FLOAT);
  data_tensor_desc.SetOriginFormat(FORMAT_NHWC);
  data.update_input_desc_x(data_tensor_desc);
  data.update_output_desc_y(data_tensor_desc);

  // padding_const
  std::vector<int32_t> paddings = {0,3,3,0,0,3,3,0};
  vector<int64_t> paddings_dims = {1, 8};
  TensorDesc padding_const_tensor_desc(ge::Shape(paddings_dims), FORMAT_NHWC, DT_INT32);
  Tensor padding_const_tensor(padding_const_tensor_desc);
  uint32_t* padding_const_tensor_value = new uint32_t[8];
  for (size_t dim = 0; dim < 8; dim++) {
    *(padding_const_tensor_value + dim) = paddings[dim];
  }
  padding_const_tensor.SetData((uint8_t*)padding_const_tensor_value, 8 * sizeof(uint32_t));
  auto paddings_const = op::Const("paddings").set_attr_value(padding_const_tensor);

  // pad
  auto pad = op::PadV3("padv3").set_input_x(data).set_input_paddings(paddings_const);
  pad.update_input_desc_x(data_tensor_desc);
  pad.update_input_desc_paddings(padding_const_tensor_desc);
  ge::Shape pad_output_shape({1, 230, 230, 3});
  ge::TensorDesc pad_output_tensor_desc(pad_output_shape, FORMAT_NHWC, DT_FLOAT);
  pad.update_output_desc_y(pad_output_tensor_desc);

  // conv2d_filter_const
  TensorDesc filterDesc(ge::Shape({7,7,3,64}), FORMAT_HWCN, DT_FLOAT);
  filterDesc.SetOriginFormat(FORMAT_HWCN);
  auto conv2d_filter_const = op::Const("conv2d_filter_const");
  Tensor filter;
  float * filterValue = new float[7*7*3*64];
  filter.SetTensorDesc(filterDesc);
  filter.SetData((uint8_t*)filterValue, 4*7*7*3*64);
  conv2d_filter_const.set_attr_value(filter);

  // conv2d
  auto conv2d = op::Conv2D("conv2d");
  conv2d.set_input_x(pad)
  .set_input_filter(conv2d_filter_const)
  .set_attr_strides({1,2,2,1})
  .set_attr_pads({0,0,0,0});
  conv2d.update_input_desc_x(pad_output_tensor_desc);
  conv2d.update_input_desc_filter(filterDesc);
  ge::Shape conv2d_output_shape({1, 112, 112, 64});
  ge::TensorDesc conv2d_output_tensor_desc(conv2d_output_shape, FORMAT_NHWC, DT_FLOAT);
  conv2d.update_output_desc_y(conv2d_output_tensor_desc);


  std::vector<Operator> inputs = {data,paddings_const,conv2d_filter_const};
  std::vector<Operator> outputs = {conv2d};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  ge::NodePtr pad_node = nullptr;
  ge::NodePtr conv_node = nullptr;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "PadV3") {
       pad_node = node;
    }
    if (node->GetType() == "Conv2D") {
      conv_node = node;
    }
  }
  ge::GraphUtils::AddEdge(pad_node->GetOutControlAnchor(), conv_node->GetInControlAnchor());


  // check before fusion
  EXPECT_EQ(compute_graph_ptr->GetAllNodesSize(), 5);

  // do fusion
  fe::FusionPassTestUtils::RunGraphFusionPass("PadConv2dFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  // check after fusion
  EXPECT_EQ(compute_graph_ptr->GetAllNodesSize(), 3);

  bool has_pad_node = false;
  vector<int64_t> pads;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "PadV3") {
      has_pad_node = true;
    }
    if (node->GetType() == "Conv2D") {
      (void)ge::AttrUtils::GetListInt(node->GetOpDesc(), "pads", pads);
    }
  }
  EXPECT_EQ(has_pad_node, false);
  for (auto pad : pads) {
    EXPECT_EQ(pad, 3);
  }
}

TEST_F(pad_conv2d_fusion_test, one_padv3_constant_value_one_conv2d) {
  std::cout << "enter pad_conv2d_fusion_test.one_pad_one_conv2d" << std::endl;
  /*
   * data    padding_const
   *   \        /  constant_value(1)
   *    \      /  /
   *     \    /  /
   *      \  /  /
   *       pad    conv2d_filter_const
   *        |     /
   *        |    /
   *        conv2d
   * */

ge::Graph graph("one_padv3_constant_value_one_conv2d");

  // data
  auto data = op::Data("data");
  ge::Shape data_shape({1, 224, 224, 3});
  ge::TensorDesc data_tensor_desc(data_shape, FORMAT_NHWC, DT_FLOAT);
  data_tensor_desc.SetOriginFormat(FORMAT_NHWC);
  data.update_input_desc_x(data_tensor_desc);
  data.update_output_desc_y(data_tensor_desc);

  // padding_const
  std::vector<int32_t> paddings = {0,3,3,0,0,3,3,0};
  vector<int64_t> paddings_dims = {1, 8};
  TensorDesc padding_const_tensor_desc(ge::Shape(paddings_dims), FORMAT_NHWC, DT_INT32);
  Tensor padding_const_tensor(padding_const_tensor_desc);
  uint32_t* padding_const_tensor_value = new uint32_t[8];
  for (size_t dim = 0; dim < 8; dim++) {
    *(padding_const_tensor_value + dim) = paddings[dim];
  }
  padding_const_tensor.SetData((uint8_t*)padding_const_tensor_value, 8 * sizeof(uint32_t));
  auto paddings_const = op::Const("paddings").set_attr_value(padding_const_tensor);

  ge::Shape constant_value_shape({1,1});
  ge::TensorDesc constant_value_desc(data_shape, FORMAT_NHWC, DT_FLOAT);
  Tensor constant_value_tensor(constant_value_desc);
  float* constant_value_tensor_value = new float[1];
  *constant_value_tensor_value = 1;
  constant_value_tensor.SetData((uint8_t*)constant_value_tensor_value, 1 * sizeof(float));
  auto constant_const = op::Const("constant_value").set_attr_value(constant_value_tensor);

  // pad
  auto pad = op::PadV3("Padv3").set_input_x(data).set_input_paddings(paddings_const).set_input_constant_values(constant_const);
  pad.update_input_desc_x(data_tensor_desc);
  pad.update_input_desc_paddings(padding_const_tensor_desc);
  ge::Shape pad_output_shape({1, 230, 230, 3});
  ge::TensorDesc pad_output_tensor_desc(pad_output_shape, FORMAT_NHWC, DT_FLOAT);
  pad.update_output_desc_y(pad_output_tensor_desc);

  // conv2d_filter_const
  TensorDesc filterDesc(ge::Shape({7,7,3,64}), FORMAT_HWCN, DT_FLOAT);
  filterDesc.SetOriginFormat(FORMAT_HWCN);
  auto conv2d_filter_const = op::Const("conv2d_filter_const");
  Tensor filter;
  float * filterValue = new float[7*7*3*64];
  filter.SetTensorDesc(filterDesc);
  filter.SetData((uint8_t*)filterValue, 4*7*7*3*64);
  conv2d_filter_const.set_attr_value(filter);

  // conv2d
  auto conv2d = op::Conv2D("conv2d");
  conv2d.set_input_x(pad)
  .set_input_filter(conv2d_filter_const)
  .set_attr_strides({1,2,2,1})
  .set_attr_pads({0,0,0,0});
  conv2d.update_input_desc_x(pad_output_tensor_desc);
  conv2d.update_input_desc_filter(filterDesc);
  ge::Shape conv2d_output_shape({1, 112, 112, 64});
  ge::TensorDesc conv2d_output_tensor_desc(conv2d_output_shape, FORMAT_NHWC, DT_FLOAT);
  conv2d.update_output_desc_y(conv2d_output_tensor_desc);


  std::vector<Operator> inputs = {data,paddings_const,conv2d_filter_const};
  std::vector<Operator> outputs = {conv2d};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  ge::NodePtr pad_node = nullptr;
  ge::NodePtr conv_node = nullptr;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "PadV3") {
       pad_node = node;
    }
    if (node->GetType() == "Conv2D") {
      conv_node = node;
    }
  }
  ge::GraphUtils::AddEdge(pad_node->GetOutControlAnchor(), conv_node->GetInControlAnchor());


  // check before fusion
  EXPECT_EQ(compute_graph_ptr->GetAllNodesSize(), 6);

  // do fusion
  fe::FusionPassTestUtils::RunGraphFusionPass("PadConv2dFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);

  // check after fusion
  EXPECT_EQ(compute_graph_ptr->GetAllNodesSize(), 6);

  bool has_pad_node = false;
  vector<int64_t> pads;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "PadV3") {
      has_pad_node = true;
    }
    if (node->GetType() == "Conv2D") {
      (void)ge::AttrUtils::GetListInt(node->GetOpDesc(), "pads", pads);
    }
  }
  EXPECT_EQ(has_pad_node, true);
}

TEST_F(pad_conv2d_fusion_test, one_padv3_one_conv2d_one_slice) {
  std::cout << "enter pad_conv2d_fusion_test.one_padv3_one_conv2d_one_slice" << std::endl;
  /*
   *            data    padding_const
   *                \         /
   *                 \       /
   *                  \     /
   *                   padv3
   *                  /     \
   *                 /       \
   *                /         \
   *               /       dbn \
   *              /         | \ \
   *             /  filter  |  \ \
   *            /   /    \  |   dw
   *           conv2d      dx
   *                        |
   *                      slice
   **/
  ge::Graph graph("one_padv3_one_conv2d");

  // data
  auto data = op::Data("data");
  ge::Shape data_shape({1, 224, 224, 3});
  ge::TensorDesc data_tensor_desc(data_shape, FORMAT_NHWC, DT_FLOAT);
  data_tensor_desc.SetOriginFormat(FORMAT_NHWC);
  data.update_input_desc_x(data_tensor_desc);
  data.update_output_desc_y(data_tensor_desc);

  // padding_const
  std::vector<int32_t> paddings = {0,3,3,0,0,3,3,0};
  vector<int64_t> paddings_dims = {1, 8};
  TensorDesc padding_const_tensor_desc(ge::Shape(paddings_dims), FORMAT_NHWC, DT_INT32);
  Tensor padding_const_tensor(padding_const_tensor_desc);
  uint32_t* padding_const_tensor_value = new uint32_t[8];
  for (size_t dim = 0; dim < 8; dim++) {
    *(padding_const_tensor_value + dim) = paddings[dim];
  }
  padding_const_tensor.SetData((uint8_t*)padding_const_tensor_value, 8 * sizeof(uint32_t));
  auto paddings_const = op::Const("paddings").set_attr_value(padding_const_tensor);

  // pad
  auto pad = op::PadV3("padv3").set_input_x(data).set_input_paddings(paddings_const);
  pad.update_input_desc_x(data_tensor_desc);
  pad.update_input_desc_paddings(padding_const_tensor_desc);
  ge::Shape pad_output_shape({1, 230, 230, 3});
  ge::TensorDesc pad_output_tensor_desc(pad_output_shape, FORMAT_NHWC, DT_FLOAT);
  pad.update_output_desc_y(pad_output_tensor_desc);


  // conv2d_filter_const
  TensorDesc filterDesc(ge::Shape({7,7,3,64}), FORMAT_HWCN, DT_FLOAT);
  filterDesc.SetOriginFormat(FORMAT_HWCN);
  auto conv2d_filter_const = op::Const("conv2d_filter_const");
  Tensor filter;
  float * filterValue = new float[7*7*3*64];
  filter.SetTensorDesc(filterDesc);
  filter.SetData((uint8_t*)filterValue, 4*7*7*3*64);
  conv2d_filter_const.set_attr_value(filter);

  // conv2d
  auto conv2d = op::Conv2D("conv2d");
  conv2d.set_input_x(pad)
  .set_input_filter(conv2d_filter_const)
  .set_attr_strides({1,2,2,1})
  .set_attr_pads({0,0,0,0});
  conv2d.update_input_desc_x(pad_output_tensor_desc);
  conv2d.update_input_desc_filter(filterDesc);
  ge::Shape conv2d_output_shape({1, 112, 112, 64});
  ge::TensorDesc conv2d_output_tensor_desc(conv2d_output_shape, FORMAT_NHWC, DT_FLOAT);
  conv2d.update_output_desc_y(conv2d_output_tensor_desc);

  // dbn inputs
  TensorDesc gradDesc(ge::Shape({1, 112, 112, 64}), FORMAT_NHWC, DT_FLOAT);
  gradDesc.SetOriginFormat(FORMAT_NHWC);
  TensorDesc channelDesc(ge::Shape({1, 1, 1, 64}), FORMAT_NHWC, DT_FLOAT);
  channelDesc.SetOriginFormat(FORMAT_NHWC);
  TensorDesc offsetDesc(ge::Shape({4}), FORMAT_ND, DT_FLOAT);
  offsetDesc.SetOriginFormat(FORMAT_ND);

  auto grad_data = op::Data("data1");
  grad_data.update_input_desc_x(gradDesc);
  grad_data.update_output_desc_y(gradDesc);
  auto grad_x_data = op::Data("data2");
  grad_x_data.update_input_desc_x(gradDesc);
  grad_x_data.update_output_desc_y(gradDesc);
  auto diff_scaler_data = op::Data("data3");
  diff_scaler_data.update_input_desc_x(channelDesc);
  diff_scaler_data.update_output_desc_y(channelDesc);
  auto diff_offset_data = op::Data("data4");
  diff_offset_data.update_input_desc_x(channelDesc);
  diff_offset_data.update_output_desc_y(channelDesc);
  auto scalar_data = op::Data("data5");
  scalar_data.update_input_desc_x(channelDesc);
  scalar_data.update_output_desc_y(channelDesc);
  auto mean_data = op::Data("data6");
  mean_data.update_input_desc_x(channelDesc);
  mean_data.update_output_desc_y(channelDesc);
  auto variance_data = op::Data("data7");
  variance_data.update_input_desc_x(channelDesc);
  variance_data.update_output_desc_y(channelDesc);

  auto dbn = op::BNTrainingReduceGrad("dbn");
  dbn.set_input_grads(grad_data)
  .set_input_x(grad_x_data)
  .set_input_diff_scale(diff_scaler_data)
  .set_input_diff_offset(diff_offset_data)
  .set_input_scale(scalar_data)
  .set_input_batch_mean(mean_data)
  .set_input_batch_variance(variance_data)
  .set_attr_epsilon(0.0001);
  dbn.update_output_desc_y(gradDesc);

  // dx
  auto dx = op::Conv2DBackpropInputD("dx");
  dx.set_input_filter(conv2d_filter_const)
  .set_input_out_backprop(dbn)
  .set_attr_strides({1,2,2,1})
  .set_attr_pads({0,0,0,0});
  dx.update_output_desc_y(pad_output_tensor_desc);

  // Slice
  auto offsets_data = op::Data("data9");
  offsets_data.update_input_desc_x(offsetDesc);
  offsets_data.update_output_desc_y(offsetDesc);
  auto size_data = op::Data("data10");
  size_data.update_input_desc_x(offsetDesc);
  size_data.update_output_desc_y(offsetDesc);
  auto slice = op::Slice("slice");
  slice.set_input_x(dx)
  .set_input_offsets(offsets_data)
  .set_input_size(size_data);
  slice.update_output_desc_y(data_tensor_desc);

  //dw
  auto dw = op::Conv2DBackpropFilterD("dw");
  dw.set_input_x(pad)
  .set_input_out_backprop(dbn)
  .set_attr_strides({1,2,2,1})
  .set_attr_pads({0,0,0,0});
  dx.update_output_desc_y(filterDesc);

  auto conv2d1 = op::Conv2D("conv2d1");
  conv2d1.set_input_x(slice)
  .set_input_filter(dw)
  .set_attr_strides({1,2,2,1})
  .set_attr_pads({3, 3,3,3});
  conv2d1.update_input_desc_x(data_tensor_desc);
  conv2d1.update_input_desc_filter(filterDesc);
  conv2d1.update_output_desc_y(conv2d_output_tensor_desc);

  auto add1 = op::Add("add");
  add1.set_input_x1(conv2d1)
  .set_input_x2(conv2d);
  add1.update_output_desc_y(conv2d_output_tensor_desc);

  std::vector<Operator> inputs = {data,paddings_const,conv2d_filter_const, grad_data, grad_x_data, diff_scaler_data,
                                  diff_offset_data,scalar_data, mean_data, variance_data, offsets_data, size_data};
  std::vector<Operator> outputs = {add1};
  graph.SetInputs(inputs).SetOutputs(outputs);

  ge::ComputeGraphPtr compute_graph_ptr = ge::GraphUtils::GetComputeGraph(graph);

  ge::NodePtr pad_node = nullptr;
  ge::NodePtr conv_node = nullptr;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "PadV3") {
       pad_node = node;
    }
    if (node->GetType() == "Conv2D") {
      conv_node = node;
    }
  }

  // do fusion
  fe::FusionPassTestUtils::RunGraphFusionPass("PadConv2dFusionPass", fe::BUILT_IN_GRAPH_PASS, *compute_graph_ptr);
  // check after fusion

  bool has_pad_node = false;
  bool has_slice_node = false;
  vector<int64_t> pads;
  for (auto node : compute_graph_ptr->GetAllNodes()) {
    if (node->GetType() == "PadV3") {
      has_pad_node = true;
    }
    if (node->GetType() == "Conv2D") {
      (void)ge::AttrUtils::GetListInt(node->GetOpDesc(), "pads", pads);
    }
    if (node->GetType() == "Slice") {
      has_slice_node = true;
    }

  }
  EXPECT_EQ(has_pad_node, false);
  EXPECT_EQ(has_slice_node, false);
  for (auto pad : pads) {
    EXPECT_EQ(pad, 3);
  }
}