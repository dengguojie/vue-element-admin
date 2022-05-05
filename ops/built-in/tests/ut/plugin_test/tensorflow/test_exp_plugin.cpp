#include <string>
#include <vector>
#include "gtest/gtest.h"
#include "tensorflow_parser.h"
#include "plugin_test_utils.h"

using namespace ge;

class elu_plugin_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "elu_plugin_test SetUp" << std::endl;
    CleanGlobal();
  }

  static void TearDownTestCase() {
    std::cout << "elu_plugin_test TearDown" << std::endl;
  }
};

TEST_F(elu_plugin_test, elu_plugin_test_1) {
  ge::Graph graph;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/elu_case_1.pb";
  auto status = aclgrphParseTensorFlow(modelFile.c_str(), graph);
}