#include <string>
#include <vector>
#include "gtest/gtest.h"
#include "tensorflow_parser.h"
#include "plugin_test_utils.h"

using namespace ge;

class log_softmax_plugin_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "log_softmax_plugin_test SetUp" << std::endl;
    CleanGlobal();
  }

  static void TearDownTestCase() {
    std::cout << "log_softmax_plugin_test TearDown" << std::endl;
  }
};

TEST_F(log_softmax_plugin_test, log_softmax_plugin_test_1) {
  ge::Graph graph;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/log_softmax.pb";
  auto status = aclgrphParseTensorFlow(modelFile.c_str(), graph);
}