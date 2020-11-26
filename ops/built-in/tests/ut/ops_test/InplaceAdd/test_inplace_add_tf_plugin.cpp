#include <fstream>
#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#include "common/debug/log.h"
#include "common/debug/memory_dumper.h"
#include "common/types.h"

#include "omg/parser/parser_factory.h"

#define private public
#define protected public
#include "parser/tensorflow/tensorflow_parser.h"
#undef private

#include "parser/tensorflow/tensorflow_data_parser.h"
#include "parser/tensorflow/tensorflow_op_parser.h"
#include "parser/tensorflow/tensorflow_fusion_op_parser.h"
#include "proto/tensorflow/graph.pb.h"
#include "proto/tensorflow/node_def.pb.h"
#include "proto/tensorflow/attr_value.pb.h"
#include "omg/omg.h"
#include "framework/common/types.h"
#include "framework/omg/omg_inner_types.h"
#include "register/op_registry.h"
#include "parser/common/register_tbe.h"
#include "parser/common/op_parser_factory.h"
#include "parser/common/pre_checker.h"
#include <google/protobuf/util/json_util.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include "parser/tensorflow/tensorflow_constant_parser.h"
#include "stdlib.h"
#include "common/op_map.h"
#include "external/register/scope/scope_fusion_pass_register.h"

using google::protobuf::Message;

using namespace domi::tensorflow;
using namespace domi;
using namespace testing;
using namespace std;
using namespace google::protobuf;
using domi::tensorflow::AttrValue;
using ge::TENSORFLOW_ATTR_T;
using ge::TENSORFLOWF_TENSOR_NHWC;
using ge::TENSORFLOW_ATTR_DATA_FORMAT;
using ge::OpRegistrationTbe;
using ge::ReadProtoFromText;
using ge::TensorFlowUtil;
using ge::TensorFlowOpParser;
using ge::L2LOSS;

class UTEST_omg_parser_tensorflow_parser : public testing::Test
{
protected:
    void SetUp()
    {
      std::cout << "INPLACEADD_UT SetUp" << std::endl;
    }

    void TearDown()
    {
      std::cout << "INPLACEADD_UT TearDown" << std::endl; 
    }

    void register_tbe_op()
    {
        std::vector<OpRegistrationData> registrationDatas = OpRegistry::Instance()->registrationDatas;
        for(OpRegistrationData reg_data : registrationDatas)
        {
            OpRegistrationTbe::Instance()->Finalize(reg_data);
            OpRegistry::Instance()->Register(reg_data);
        }
        OpRegistry::Instance()->registrationDatas.clear();
    }

};

static const string GRAPH_DEFAULT_NAME = "default";

TEST_F(UTEST_omg_parser_tensorflow_parser, nodedef_IplaceAdd_Op_ret_success) {
    const char *tensorflow_pb_txt_file = "llt/ops/llt_new/ut/ops_test/InplaceAdd/inplace_add_pb_tf/inplace_add.txt";
    tensorflow::GraphDef graphDef;
    ReadProtoFromText(tensorflow_pb_txt_file, &graphDef);

    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>(GRAPH_DEFAULT_NAME);
    ModelParserFactory* factory = ModelParserFactory::Instance();

    shared_ptr<ModelParser> model_parser = factory->CreateModelParser(TENSORFLOW);
    ASSERT_TRUE(NULL != model_parser);

    shared_ptr<TensorFlowModelParser> tensorflow_parser = static_pointer_cast<TensorFlowModelParser>(model_parser);
 
    Status ret = tensorflow_parser->ParseProto(&graphDef, graph);

    EXPECT_EQ(SUCCESS, ret);
}
