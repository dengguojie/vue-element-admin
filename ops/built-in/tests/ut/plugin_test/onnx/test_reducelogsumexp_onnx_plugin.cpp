#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "onnx_parser.h"

using namespace ge;

class reducelogsumexp_onnx_plugin_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "reducelogsumexp_onnx_plugin_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "reducelogsumexp_onnx_plugin_test TearDown" << std::endl;
  }
};

TEST_F(reducelogsumexp_onnx_plugin_test, test_reducelogsumexp_case_V11) {
  ge::Graph graph;

  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/test_reducelogsumexp_case_v11.onnx";
  std::map<ge::AscendString, ge::AscendString> parser_params;

  auto status = aclgrphParseONNX(modelFile.c_str(), parser_params, graph);

  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  // check op count, some op need check op attr, op input count.
  std::vector<ge::GNode> nodes = graph.GetAllNodes();
  EXPECT_EQ(nodes.size(), 5);
}

TEST_F(reducelogsumexp_onnx_plugin_test, test_reducelogsumexp_case1_V11) {
  ge::Graph graph;

  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/test_reducelogsumexp_case1_v11.onnx";
  std::map<ge::AscendString, ge::AscendString> parser_params;

  auto status = aclgrphParseONNX(modelFile.c_str(), parser_params, graph);

  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  // check op count, some op need check op attr, op input count.
  std::vector<ge::GNode> nodes = graph.GetAllNodes();
  EXPECT_EQ(nodes.size(), 5);
}

