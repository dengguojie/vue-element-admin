#include <string>
#include <vector>
#include <stdio.h>

#include "gtest/gtest.h"
#include "tensorflow_parser.h"

using namespace ge;

class ascend_weight_quant_plugin_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ascend_weight_quant_plugin_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ascend_weight_quant_plugin_test TearDown" << std::endl;
  }
};

TEST_F(ascend_weight_quant_plugin_test, ascend_weight_quant_plugin_test) {
  ge::Graph graph;

  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/ascendweightquant_case.pb";

  auto status = aclgrphParseTensorFlow(modelFile.c_str(), graph);
}
