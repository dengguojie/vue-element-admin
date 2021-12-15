#include "gtest/gtest.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "array_ops.h"
#include "selection_ops.h"
#include "all_ops.h"
#include "fusion_pass_test_utils.h"
#include "graph_builder_utils.h"

using namespace ge;
using namespace op;

class ArgmaxOneHotFusionTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ArgmaxOneHotFusionTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ArgmaxOneHotFusionTest TearDown" << std::endl;
  }
};

TEST_F(ArgmaxOneHotFusionTest, ArgmaxOneHotFusionTest_001) {
  ut::GraphBuilder builder = ut::GraphBuilder(this->test_info_->name());
  int32_t depth = 2;
  int32_t first_dim = 1024;
  auto data = builder.AddNode("Data", "Data", 0, 1, {first_dim}, ge::FORMAT_ND, ge::DT_FLOAT);
  auto data1 = builder.AddNode("Data1", "Data", 0, 1, {1}, ge::FORMAT_ND, ge::DT_INT32);
  auto data2 = builder.AddNode("Data2", "Data", 0, 1, {1}, ge::FORMAT_ND, ge::DT_INT32);
  auto data3 = builder.AddNode("Data3", "Data", 0, 1, {1}, ge::FORMAT_ND, ge::DT_INT32);
  auto argmax = builder.AddNode("ArgMaxV2", "ArgMaxV2", {
    {Format::FORMAT_ND, ge::DT_FLOAT, {1024}}
                                },
                                {
                                    {Format::FORMAT_ND, ge::DT_INT64, {first_dim}}
                                });
  auto one_hot = builder.AddNode("OneHot", "OneHot", {
                                            {Format::FORMAT_ND, ge::DT_INT64, {1}},
                                            {Format::FORMAT_ND, ge::DT_INT32, {1}},
                                            {Format::FORMAT_ND, ge::DT_INT32, {1}},
                                            {Format::FORMAT_ND, ge::DT_INT32, {1}},
                                        },
                                 {
                                            {Format::FORMAT_ND, ge::DT_FLOAT, {first_dim, depth}},
                                        });
  auto net_output = builder.AddNode("NetOutput", "NetOutput", 1, 0, {first_dim, depth}, Format::FORMAT_ND, ge::DT_FLOAT);
  builder.AddDataEdge(data, 0, argmax, 0);
  builder.AddDataEdge(argmax, 0, one_hot, 0);
  builder.AddDataEdge(data1, 0, one_hot, 1);
  builder.AddDataEdge(data2, 0, one_hot, 2);
  builder.AddDataEdge(data3, 0, one_hot, 3);
  builder.AddDataEdge(one_hot, 0, net_output, 0);
  ge::AttrUtils::SetInt(one_hot->GetOpDesc(), "axis", -1);
  ge::GeTensorDesc tensor_desc;
  tensor_desc.SetDataType(ge::DT_INT32);
  tensor_desc.SetFormat(ge::FORMAT_ND);
  tensor_desc.SetShape(GeShape({1}));
  int32_t dimension = 0;
  auto weight_ptr = std::make_shared<ge::GeTensor>(tensor_desc, reinterpret_cast<uint8_t *>(&dimension),
                                                   sizeof(dimension));
  OpDescUtils::SetWeights(argmax, {weight_ptr});
  argmax->GetOpDesc()->UpdateInputName({{"x", 0}, {"dimension", 1}});
  auto graph = builder.GetGraph();

  GraphUtils::DumpGEGraphToOnnx(*graph, "0");
  auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("ArgmaxOneHotFusionPass",
                                                         fe::BUILT_IN_GRAPH_PASS,
                                                         *graph);
  int op_count = 0;
  GraphUtils::DumpGEGraphToOnnx(*graph, "1");
  for (auto node: graph->GetAllNodes()) {
    if (node->GetType() == "ArgMaxV2") {
      if (node->GetOpDesc()->GetOutputDesc(0).GetDataType() == ge::DT_INT32) {
        op_count++;
      }
    }

    if (node->GetType() == "OneHot") {
      if (node->GetOpDesc()->GetInputDesc(0).GetDataType() == ge::DT_INT32) {
        op_count++;
      }
    }
  }
  EXPECT_EQ(op_count, 2);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(ArgmaxOneHotFusionTest, ArgmaxOneHotFusionTest_002) {
  ut::GraphBuilder builder = ut::GraphBuilder(this->test_info_->name());
  int32_t depth = 2;
  int32_t first_dim = 1024;
  auto data = builder.AddNode("Data", "Data", 0, 1, {first_dim}, ge::FORMAT_ND, ge::DT_INT64);
  auto data1 = builder.AddNode("Data1", "Data", 0, 1, {1}, ge::FORMAT_ND, ge::DT_INT32);
  auto data2 = builder.AddNode("Data2", "Data", 0, 1, {1}, ge::FORMAT_ND, ge::DT_INT32);
  auto data3 = builder.AddNode("Data3", "Data", 0, 1, {1}, ge::FORMAT_ND, ge::DT_INT32);
  auto argmax = builder.AddNode("ArgMaxV2", "ArgMaxV2", {
    {Format::FORMAT_ND, ge::DT_INT32, {1024}}
    },
                                {
    {Format::FORMAT_ND, ge::DT_INT64, {first_dim}}
                                });
  auto one_hot = builder.AddNode("OneHot", "OneHot", {
    {Format::FORMAT_ND, ge::DT_INT64, {1}},
    {Format::FORMAT_ND, ge::DT_INT32, {1}},
    {Format::FORMAT_ND, ge::DT_INT32, {1}},
    {Format::FORMAT_ND, ge::DT_INT32, {1}},
    },
                                 {
    {Format::FORMAT_ND, ge::DT_FLOAT, {first_dim, depth}},
    });
  auto net_output = builder.AddNode("NetOutput", "NetOutput", 1, 0, {first_dim, depth}, Format::FORMAT_ND, ge::DT_FLOAT);
  builder.AddDataEdge(data, 0, one_hot, 0);
  builder.AddDataEdge(argmax, 0, one_hot, 1);
  builder.AddDataEdge(data1, 0, argmax, 0);
  builder.AddDataEdge(data2, 0, one_hot, 2);
  builder.AddDataEdge(data3, 0, one_hot, 3);
  builder.AddDataEdge(one_hot, 0, net_output, 0);
  ge::AttrUtils::SetInt(one_hot->GetOpDesc(), "axis", -1);
  ge::GeTensorDesc tensor_desc;
  tensor_desc.SetDataType(ge::DT_INT32);
  tensor_desc.SetFormat(ge::FORMAT_ND);
  tensor_desc.SetShape(GeShape({1}));
  int32_t dimension = 0;
  auto weight_ptr = std::make_shared<ge::GeTensor>(tensor_desc, reinterpret_cast<uint8_t *>(&dimension),
                                                   sizeof(dimension));
  OpDescUtils::SetWeights(argmax, {weight_ptr});
  argmax->GetOpDesc()->UpdateInputName({{"x", 0}, {"dimension", 1}});
  auto graph = builder.GetGraph();

  GraphUtils::DumpGEGraphToOnnx(*graph, "0");
  auto ret = fe::FusionPassTestUtils::RunGraphFusionPass("ArgmaxOneHotFusionPass",
                                                         fe::BUILT_IN_GRAPH_PASS,
                                                         *graph);
  int op_count = 0;
  GraphUtils::DumpGEGraphToOnnx(*graph, "1");
  for (auto node: graph->GetAllNodes()) {
    if (node->GetType() == "ArgMaxV2") {
      if (node->GetOpDesc()->GetOutputDesc(0).GetDataType() == ge::DT_INT32) {
        op_count++;
      }
    }

    if (node->GetType() == "OneHot") {
      if (node->GetOpDesc()->GetInputDesc(0).GetDataType() == ge::DT_INT32) {
        op_count++;
      }
    }
  }
  EXPECT_EQ(op_count, 0);
  EXPECT_NE(ret, GRAPH_SUCCESS);
}
