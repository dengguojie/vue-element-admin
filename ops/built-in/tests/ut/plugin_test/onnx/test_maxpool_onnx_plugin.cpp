#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "onnx_parser.h"

using namespace ge;

class maxpool_onnx_plugin_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "maxpool_onnx_plugin_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "maxpool_onnx_plugin_test TearDown" << std::endl;
  }
};

TEST_F(maxpool_onnx_plugin_test, maxpool_onnx_plugin_test_case_1) {
  ge::Graph graph;

  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/test_maxpool_case_1.onnx";
  std::map<ge::AscendString, ge::AscendString> parser_params;

  auto status = aclgrphParseONNX(modelFile.c_str(), parser_params, graph);

  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  // check op count, some op need check op attr, op input count.
  std::vector<ge::GNode> nodes = graph.GetAllNodes();
  EXPECT_EQ(nodes.size(), 2);
}

TEST_F(maxpool_onnx_plugin_test, maxpool_onnx_plugin_test_case_2) {
  ge::Graph graph;

  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/test_maxpool_case_2.onnx";
  std::map<ge::AscendString, ge::AscendString> parser_params;

  auto status = aclgrphParseONNX(modelFile.c_str(), parser_params, graph);

  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  // check op count, some op need check op attr, op input count.
  std::vector<ge::GNode> nodes = graph.GetAllNodes();
  EXPECT_EQ(nodes.size(), 2);
}

TEST_F(maxpool_onnx_plugin_test, maxpool_onnx_plugin_test_case_3) {
  ge::Graph graph;

  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/test_maxpool_case_3.onnx";
  std::map<ge::AscendString, ge::AscendString> parser_params;

  auto status = aclgrphParseONNX(modelFile.c_str(), parser_params, graph);

  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  // check op count, some op need check op attr, op input count.
  std::vector<ge::GNode> nodes = graph.GetAllNodes();
  EXPECT_EQ(nodes.size(), 2);
}

TEST_F(maxpool_onnx_plugin_test, maxpool_onnx_plugin_test_case_4) {
  ge::Graph graph;

  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/test_maxpool_case_4.onnx";
  std::map<ge::AscendString, ge::AscendString> parser_params;

  auto status = aclgrphParseONNX(modelFile.c_str(), parser_params, graph);

  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  // check op count, some op need check op attr, op input count.
  std::vector<ge::GNode> nodes = graph.GetAllNodes();
  EXPECT_EQ(nodes.size(), 2);
}

TEST_F(maxpool_onnx_plugin_test, maxpool_onnx_plugin_test_case_5) {
  ge::Graph graph;

  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/test_maxpool_case_5.onnx";
  std::map<ge::AscendString, ge::AscendString> parser_params;

  auto status = aclgrphParseONNX(modelFile.c_str(), parser_params, graph);

  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  // check op count, some op need check op attr, op input count.
  std::vector<ge::GNode> nodes = graph.GetAllNodes();
  EXPECT_EQ(nodes.size(), 2);
}

TEST_F(maxpool_onnx_plugin_test, maxpool_onnx_plugin_test_case_6) {
  ge::Graph graph;

  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/test_maxpool_case_v12.onnx";
  std::map<ge::AscendString, ge::AscendString> parser_params;

  auto status = aclgrphParseONNX(modelFile.c_str(), parser_params, graph);

  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  // check op count, some op need check op attr, op input count.
  std::vector<ge::GNode> nodes = graph.GetAllNodes();
  EXPECT_EQ(nodes.size(), 2);
}

TEST_F(maxpool_onnx_plugin_test, maxpool_onnx_plugin_test_case_7) {
  ge::Graph graph;

  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/test_maxpool_case_v13.onnx";
  std::map<ge::AscendString, ge::AscendString> parser_params;

  auto status = aclgrphParseONNX(modelFile.c_str(), parser_params, graph);

  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  // check op count, some op need check op attr, op input count.
  std::vector<ge::GNode> nodes = graph.GetAllNodes();
  EXPECT_EQ(nodes.size(), 2);
}

TEST_F(maxpool_onnx_plugin_test, maxpool_onnx_plugin_test_case_8) {
  ge::Graph graph;

  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/test_maxpool_case_v9.onnx";
  std::map<ge::AscendString, ge::AscendString> parser_params;

  auto status = aclgrphParseONNX(modelFile.c_str(), parser_params, graph);

  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  // check op count, some op need check op attr, op input count.
  std::vector<ge::GNode> nodes = graph.GetAllNodes();
  EXPECT_EQ(nodes.size(), 2);
}

TEST_F(maxpool_onnx_plugin_test, maxpool_onnx_plugin_test_case_9) {
  ge::Graph graph;

  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/test_maxpool_case_1d_input.onnx";
  std::map<ge::AscendString, ge::AscendString> parser_params;

  auto status = aclgrphParseONNX(modelFile.c_str(), parser_params, graph);

  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  // check op count, some op need check op attr, op input count.
  std::vector<ge::GNode> nodes = graph.GetAllNodes();
  EXPECT_EQ(nodes.size(), 4);
}

TEST_F(maxpool_onnx_plugin_test, maxpool_onnx_plugin_test_case_10) {
  ge::Graph graph;

  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/test_maxpool_case_aicpu.onnx";
  std::map<ge::AscendString, ge::AscendString> parser_params;

  auto status = aclgrphParseONNX(modelFile.c_str(), parser_params, graph);

  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  // check op count, some op need check op attr, op input count.
  std::vector<ge::GNode> nodes = graph.GetAllNodes();
  EXPECT_EQ(nodes.size(), 7);
}