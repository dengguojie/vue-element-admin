#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "tensorflow_parser.h"

using namespace ge;

class concatv2_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "xxxxxxconcatv2_plugin_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "xxxxconcatv2_plugin_test TearDown" << std::endl;
  }
};

TEST_F(concatv2_test, concatv2_test_case_1) {
  ge::Graph graph;

  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/concatv2.pb";

  auto status = aclgrphParseTensorFlow(modelFile.c_str(), graph);
}
