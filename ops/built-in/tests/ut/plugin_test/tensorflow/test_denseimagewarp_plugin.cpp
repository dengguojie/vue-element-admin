#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "tensorflow_parser.h"

using namespace ge;

class denseimagewarp_plugin_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "denseimagewarp_plugin_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "denseimagewarp_plugin_test TearDown" << std::endl;
  }
};

TEST_F(denseimagewarp_plugin_test, denseimagewarp_plugin_testcase_1) {
  ge::Graph graph;

  std::cout << __FILE__ << std::endl;
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/denseimagewarp_case_1.pb";

  auto status = aclgrphParseTensorFlow(modelFile.c_str(), graph);
  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
}
