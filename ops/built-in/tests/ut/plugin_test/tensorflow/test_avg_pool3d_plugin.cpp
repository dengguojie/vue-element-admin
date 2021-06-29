#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "tensorflow_parser.h"

using namespace ge;

class avgpool3d_plugin_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "avgpool3d_plugin_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "avgpool3d_plugin_test TearDown" << std::endl;
  }
};

TEST_F(avgpool3d_plugin_test, avgpool3d_plugin_test_format_NDHWC) {
  ge::Graph graph;

  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/avgpool3d_case_1.pb";

  auto status = aclgrphParseTensorFlow(modelFile.c_str(), graph);
}
