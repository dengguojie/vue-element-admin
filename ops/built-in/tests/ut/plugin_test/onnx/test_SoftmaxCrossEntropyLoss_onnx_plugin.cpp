#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "onnx_parser.h"
#include "parser_common.h"

using namespace ge;

class SoftmaxCrossEntropyLoss_onnx_plugin_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SoftmaxCrossEntropyLoss_onnx_plugin_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SoftmaxCrossEntropyLoss_onnx_plugin_test TearDown" << std::endl;
  }
};

TEST_F(SoftmaxCrossEntropyLoss_onnx_plugin_test, SoftmaxCrossEntropyLoss_onnx_plugin_test_case_1) {
  CleanGlobal();
  ge::Graph graph;

  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/test_SoftmaxCrossEntropyLoss_case_v12.onnx";
  std::map<ge::AscendString, ge::AscendString> parser_params;
  auto status = aclgrphParseONNX(modelFile.c_str(), parser_params, graph);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

}


TEST_F(SoftmaxCrossEntropyLoss_onnx_plugin_test, SoftmaxCrossEntropyLoss_onnx_plugin_test_case_2) {
  CleanGlobal();
  ge::Graph graph;

  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/test_SoftmaxCrossEntropyLoss_case_v13.onnx";
  std::map<ge::AscendString, ge::AscendString> parser_params;
  auto status = aclgrphParseONNX(modelFile.c_str(), parser_params, graph);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);

}
