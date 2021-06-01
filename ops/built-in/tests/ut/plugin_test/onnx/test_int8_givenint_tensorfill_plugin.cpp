#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "onnx_parser.h"

using namespace ge;

class int8_givenint_tensorfill_onnx_plugin_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "int8_givenint_tensorfill_onnx_plugin_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "int8_givenint_tensorfill_onnx_plugin_test TearDown" << std::endl;
  }
};

TEST_F(int8_givenint_tensorfill_onnx_plugin_test, int8_givenint_tensorfill_onnx_plugin_test_case_1) {
  ge::Graph graph;

  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/test_int8_givenint_tensorfill_case_1.onnx";
  std::map<ge::AscendString, ge::AscendString> parser_params;

  auto status = aclgrphParseONNX(modelFile.c_str(), parser_params, graph);

  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  // check op count, some op need check op attr, op input count.
  std::vector<ge::GNode> nodes = graph.GetAllNodes();
  EXPECT_EQ(nodes.size(), 3);
}

