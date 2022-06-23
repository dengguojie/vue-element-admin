#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "tf_plugin/tensorflow_fusion_op_parser_util.h"

using namespace ge;
using AttrValueMap = ::google::protobuf::Map<string, domi::tensorflow::AttrValue>;

class tensor_fusion_op_parser_test : public testing::Test {
    protected:
    static void SetUpTestCase() {
        std::cout << "tensor_fusion_op_parser_test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "tensor_fusion_op_parser_test TearDown" << std::endl;
    }
};

TEST_F(tensor_fusion_op_parser_test, tensor_fusion_op_parser_test_1) {
    std::cout << __FILE__ << std::endl;
    int32_t param = 0;
    int index = 0;

    domi::tensorflow::AttrValue attr_value;
    domi::tensorflow::NodeDef node_def;
    domi::tensorflow::TensorProto* tensor = attr_value.mutable_tensor();

    tensor->add_int_val(10);
    node_def.mutable_attr()->insert(AttrValueMap::value_type("value", attr_value));
    domi::ParseParamFromConst(&node_def, param, index);
}

TEST_F(tensor_fusion_op_parser_test, tensor_fusion_op_parser_test_2) {
    std::cout << __FILE__ << std::endl;
    float param = 0;
    int index = 0;

    domi::tensorflow::AttrValue attr_value;
    domi::tensorflow::NodeDef node_def;
    domi::tensorflow::TensorProto* tensor = attr_value.mutable_tensor();
    tensor->add_float_val(10.0);
    node_def.mutable_attr()->insert(AttrValueMap::value_type("value", attr_value));
    domi::ParseParamFromConst(&node_def, param, index);
}

TEST_F(tensor_fusion_op_parser_test, tensor_fusion_op_parser_test_3) {
    std::cout << __FILE__ << std::endl;
    int32_t param = 0;
    int index = 1;

    domi::tensorflow::AttrValue attr_value;
    domi::tensorflow::NodeDef node_def;
    domi::tensorflow::TensorProto* tensor = attr_value.mutable_tensor();
    domi::tensorflow::TensorShapeProto* tf_shape = tensor->mutable_tensor_shape();
    for (int i = 0; i < 5; i++) {
        domi::tensorflow::TensorShapeProto_Dim* tf_dims = tf_shape->add_dim();
        tf_dims->set_size(1);
    }

    node_def.mutable_attr()->insert(AttrValueMap::value_type("value", attr_value));
    domi::ParseParamFromConst(&node_def, param, index);
}

TEST_F(tensor_fusion_op_parser_test, tensor_fusion_op_parser_test_4) {
    std::cout << __FILE__ << std::endl;
    float param = 0;
    int index = 1;

    domi::tensorflow::AttrValue attr_value;
    domi::tensorflow::NodeDef node_def;
    domi::tensorflow::TensorProto* tensor = attr_value.mutable_tensor();
    domi::tensorflow::TensorShapeProto* tf_shape = tensor->mutable_tensor_shape();
    for (int i = 0; i < 5; i++) {
        domi::tensorflow::TensorShapeProto_Dim* tf_dims = tf_shape->add_dim();
        tf_dims->set_size(1);
    }

    node_def.mutable_attr()->insert(AttrValueMap::value_type("value", attr_value));
    domi::ParseParamFromConst(&node_def, param, index);
}