#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "onnx_parser.h"

using namespace ge;

class flatten_onnx_plugin_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "flatten_onnx_plugin_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "flatten_onnx_plugin_test TearDown" << std::endl;
  }
};

TEST_F(flatten_onnx_plugin_test, test_flatten_input_4d_axis_neg1) {
  ge::Graph graph;

  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/test_flatten_input_4d_axis_neg1.onnx";
  std::map<ge::AscendString, ge::AscendString> parser_params;

  auto status = aclgrphParseONNX(modelFile.c_str(), parser_params, graph);
  EXPECT_TRUE(status != ge::GRAPH_SUCCESS);
}

TEST_F(flatten_onnx_plugin_test, test_flatten_input_4d_axis_0) {
  ge::Graph graph;

  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/test_flatten_input_4d_axis_0.onnx";
  std::map<ge::AscendString, ge::AscendString> parser_params;

  // check parse
  auto status = aclgrphParseONNX(modelFile.c_str(), parser_params, graph);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  // check graph node num
  std::vector<ge::GNode> nodes = graph.GetAllNodes();
  EXPECT_EQ(nodes.size(), 4);

  // check node type
  bool findFlattenV2 = false;
  bool findExpandDims = false;
  ge::AscendString type;
  for (auto node : nodes) {
    EXPECT_EQ(node.GetType(type), ge::GRAPH_SUCCESS);
    const string tmpType = type.GetString();
    if (tmpType == "FlattenV2") {
      findFlattenV2 = true;
    }
    if (tmpType == "ExpandDims") {
      findExpandDims = true;
    }
  }
  EXPECT_TRUE(findFlattenV2);
  EXPECT_TRUE(findExpandDims);
}

TEST_F(flatten_onnx_plugin_test, test_flatten_input_4d_axis_2) {
  ge::Graph graph;

  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/test_flatten_input_4d_axis_2.onnx";
  std::map<ge::AscendString, ge::AscendString> parser_params;

  // check parse
  auto status = aclgrphParseONNX(modelFile.c_str(), parser_params, graph);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  // check graph node num
  std::vector<ge::GNode> nodes = graph.GetAllNodes();
  EXPECT_EQ(nodes.size(), 3);

  // check node type
  int numFlattenV2 = 0;
  ge::AscendString type;
  for (auto node : nodes) {
    EXPECT_EQ(node.GetType(type), ge::GRAPH_SUCCESS);
    const string tmpType = type.GetString();
    if (tmpType == "FlattenV2") {
      numFlattenV2++;
    }
  }
  EXPECT_TRUE(numFlattenV2 == 2);
}

TEST_F(flatten_onnx_plugin_test, test_flatten_input_4d_axis_3) {
  ge::Graph graph;

  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/test_flatten_input_4d_axis_3.onnx";
  std::map<ge::AscendString, ge::AscendString> parser_params;

  // check parse
  auto status = aclgrphParseONNX(modelFile.c_str(), parser_params, graph);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  // check graph node num
  std::vector<ge::GNode> nodes = graph.GetAllNodes();
  EXPECT_EQ(nodes.size(), 3);

  // check node type
  int numFlattenV2 = 0;
  ge::AscendString type;
  for (auto node : nodes) {
    EXPECT_EQ(node.GetType(type), ge::GRAPH_SUCCESS);
    const string tmpType = type.GetString();
    if (tmpType == "FlattenV2") {
      numFlattenV2++;
    }
  }
  EXPECT_TRUE(numFlattenV2 == 2);
}

TEST_F(flatten_onnx_plugin_test, test_flatten_input_4d_axis_4) {
  ge::Graph graph;

  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/test_flatten_input_4d_axis_4.onnx";
  std::map<ge::AscendString, ge::AscendString> parser_params;

  // check parse
  auto status = aclgrphParseONNX(modelFile.c_str(), parser_params, graph);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  // check graph node num
  std::vector<ge::GNode> nodes = graph.GetAllNodes();
  EXPECT_EQ(nodes.size(), 3);

  // check node type
  int numFlattenV2 = 0;
  ge::AscendString type;
  for (auto node : nodes) {
    EXPECT_EQ(node.GetType(type), ge::GRAPH_SUCCESS);
    const string tmpType = type.GetString();
    if (tmpType == "FlattenV2") {
      numFlattenV2++;
    }
  }
  EXPECT_TRUE(numFlattenV2 == 2);
}

TEST_F(flatten_onnx_plugin_test, test_flatten_v12) {
  ge::Graph graph;

  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/test_flatten_v12.onnx";
  std::map<ge::AscendString, ge::AscendString> parser_params;

  // check parse
  auto status = aclgrphParseONNX(modelFile.c_str(), parser_params, graph);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  // check graph node num
  std::vector<ge::GNode> nodes = graph.GetAllNodes();
  EXPECT_EQ(nodes.size(), 3);

  // check node type
  int numFlattenV2 = 0;
  ge::AscendString type;
  for (auto node : nodes) {
    EXPECT_EQ(node.GetType(type), ge::GRAPH_SUCCESS);
    const string tmpType = type.GetString();
    if (tmpType == "FlattenV2") {
      numFlattenV2++;
    }
  }
  EXPECT_TRUE(numFlattenV2 == 2);
}

TEST_F(flatten_onnx_plugin_test, test_flatten_v13) {
  ge::Graph graph;

  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/test_flatten_v13.onnx";
  std::map<ge::AscendString, ge::AscendString> parser_params;

  // check parse
  auto status = aclgrphParseONNX(modelFile.c_str(), parser_params, graph);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  // check graph node num
  std::vector<ge::GNode> nodes = graph.GetAllNodes();
  EXPECT_EQ(nodes.size(), 3);

  // check node type
  int numFlattenV2 = 0;
  ge::AscendString type;
  for (auto node : nodes) {
    EXPECT_EQ(node.GetType(type), ge::GRAPH_SUCCESS);
    const string tmpType = type.GetString();
    if (tmpType == "FlattenV2") {
      numFlattenV2++;
    }
  }
  EXPECT_TRUE(numFlattenV2 == 2);
}

TEST_F(flatten_onnx_plugin_test, test_flatten_v9) {
  ge::Graph graph;

  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/test_flatten_v9.onnx";
  std::map<ge::AscendString, ge::AscendString> parser_params;

  // check parse
  auto status = aclgrphParseONNX(modelFile.c_str(), parser_params, graph);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

  // check graph node num
  std::vector<ge::GNode> nodes = graph.GetAllNodes();
  EXPECT_EQ(nodes.size(), 3);

  // check node type
  int numFlattenV2 = 0;
  ge::AscendString type;
  for (auto node : nodes) {
    EXPECT_EQ(node.GetType(type), ge::GRAPH_SUCCESS);
    const string tmpType = type.GetString();
    if (tmpType == "FlattenV2") {
      numFlattenV2++;
    }
  }
  EXPECT_TRUE(numFlattenV2 == 2);
}
