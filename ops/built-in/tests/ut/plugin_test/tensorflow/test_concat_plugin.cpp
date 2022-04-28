#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "tensorflow_parser.h"

using namespace ge;

class concat_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "xxxxxxconcat_plugin_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "xxxxconcat_plugin_test TearDown" << std::endl;
  }
};

TEST_F(concat_test, concat_test_case_1) {
  ge::Graph graph;

  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/concat.pb";

  auto status = aclgrphParseTensorFlow(modelFile.c_str(), graph);
}
