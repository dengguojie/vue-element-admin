#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "onnx_parser.h"

using namespace ge;

class split_onnx_plugin_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "split_onnx_plugin_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "split_onnx_plugin_test TearDown" << std::endl;
  }
};

TEST_F(split_onnx_plugin_test, test_split_v11_no_default) {
  ge::Graph graph;
  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/test_split_v11_no_default.onnx";
  std::map<ge::AscendString, ge::AscendString> parser_params;
  auto status = aclgrphParseONNX(modelFile.c_str(), parser_params, graph);
  EXPECT_EQ(status, ge::GRAPH_FAILED);
}

TEST_F(split_onnx_plugin_test, test_split_v11_default_split) {
  ge::Graph graph;
  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/test_split_v11_default_split.onnx";
  std::map<ge::AscendString, ge::AscendString> parser_params;
  auto status = aclgrphParseONNX(modelFile.c_str(), parser_params, graph);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

TEST_F(split_onnx_plugin_test, test_split_v13_no_default) {
  ge::Graph graph;
  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/test_split_v13_no_default.onnx";
  std::map<ge::AscendString, ge::AscendString> parser_params;
  auto status = aclgrphParseONNX(modelFile.c_str(), parser_params, graph);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

TEST_F(split_onnx_plugin_test, test_split_v13_default_split) {
  ge::Graph graph;
  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/test_split_v13_default_split.onnx";
  std::map<ge::AscendString, ge::AscendString> parser_params;
  auto status = aclgrphParseONNX(modelFile.c_str(), parser_params, graph);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}

TEST_F(split_onnx_plugin_test, test_split_v11_split) {
  ge::Graph graph;
  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/test_split_v11_split.onnx";
  std::map<ge::AscendString, ge::AscendString> parser_params;
  auto status = aclgrphParseONNX(modelFile.c_str(), parser_params, graph);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}
