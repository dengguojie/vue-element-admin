#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "tensorflow_parser.h"

using namespace ge;

class dilation2d_backprop_input_plugin_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "dilation2d_backprop_input_plugin_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "dilation2d_backprop_input_plugin_test TearDown" << std::endl;
  }
};

TEST_F(dilation2d_backprop_input_plugin_test, dilation2d_backprop_input_plugin_test_1) {
  ge::Graph graph;

  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/dilation2DBackpropInput_case_1.pb";

  auto status = aclgrphParseTensorFlow(modelFile.c_str(), graph);
}
