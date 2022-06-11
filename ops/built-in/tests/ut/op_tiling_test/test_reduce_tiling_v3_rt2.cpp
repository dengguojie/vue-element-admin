//
// Created by xukaiwei on 7/1/21.
//
#include <iostream>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#include "graph/utils/attr_utils.h"
#define private public
#define private public
#include "register/op_tiling_registry.h"

#include "graph/graph.h"
#include "graph/compute_graph.h"
#include "graph/utils/graph_utils.h"
#include "op_tiling/reduce_tiling_v3.h"
#include "op_tiling/tiling_handler.h"
//#include "op_tiling/vector_tiling_rt2.h"
#include "common_autotiling_util.h"

#include "reduce_ops.h"
#include "array_ops.h"
#include "test_common.h"


using namespace std;
using namespace ge;
using namespace optiling;

class ReduceTilingV3_RT2 : public testing::Test {
protected:
   static void SetUpTestCase() {
     std::cout << "ReduceTilingV3_RT2 SetUp" << std::endl;
   }

   static void TearDownTestCase() {
     std::cout << "ReduceTilingV3_RT2 TearDown" << std::endl;
   }
};

static string to_string(const std::stringstream &tiling_data) {
  auto data = tiling_data.str();
  string result;
  int32_t tmp = 0;
  for (size_t i = 0; i < data.length(); i += sizeof(int32_t)) {
    memcpy(&tmp, data.c_str() + i, sizeof(tmp));
    result += std::to_string(tmp);
    result += " ";
  }
  return result;
}

static void contruct_tensor(ge::OpDescPtr& op_desc, const std::vector<int64_t>& shape, const ge::DataType dtype,
                            bool is_input=true, ge::Format format=ge::FORMAT_ND) {
  ge::GeTensorDesc tensor;
  tensor.SetShape(ge::GeShape(shape));
  tensor.SetFormat(format);
  tensor.SetDataType(dtype);
  if (is_input) {
    op_desc->AddInputDesc(tensor);
  } else {
    op_desc->AddOutputDesc(tensor);
  }
}

template<typename T1, typename T2>
static bool compare_map(const std::unordered_map<T1, T2>& map1, const std::unordered_map<T1, T2>& map2) {
  if (map1.size() != map2.size()) {
    return false;
  }
  for (const auto& it: map1) {
    if (map2.count(it.first) == 0) {
      return false;
    }
    if (map1.at(it.first) != map2.at(it.first)) {
      return false;
    }
  }
  return true;
}

template<typename T1, typename T2>
static bool compare_var_attr_map(const std::unordered_map<T1, T2>& map1, const std::unordered_map<T1, T2>& map2) {
  if (map1.size() != map2.size()) {
     std::cout << "map1.size is :" << map1.size() << endl;
     std::cout << "map2.size is :" << map2.size() << endl;
     std::cout << "map1.size";
    return false;
  }
  for (const auto& it: map1) {
    if (map2.count(it.first) == 0) {
      std::cout << "map2.count";
      return false;
    }

    std::vector<VarAttr> var_attr_1_list = map1.at(it.first);
    std::vector<VarAttr> var_attr_2_list = map2.at(it.first);

    if (var_attr_1_list.size() != var_attr_2_list.size()) {
     std::cout << "var_attr_1_list.size()";
      return false;
    }

    for (int i = 0; i < var_attr_1_list.size(); i ++){
      VarAttr var_attr_1 = var_attr_1_list[i];
      VarAttr var_attr_2 = var_attr_2_list[i];
      if (var_attr_1.name != var_attr_2.name) {
        std::cout << "var_attr_1.name";
        return false;
      }
      if (var_attr_1.length != var_attr_2.length) {
        std::cout << "ar_attr_1.length";
        return false;
      }
      if (var_attr_1.type != var_attr_2.type) {
       std::cout << "var_attr_1.type";
        return false;
      }
      if (var_attr_1.src_type != var_attr_2.src_type) {
      std::cout << "var_attr_1.src_type";
        return false;
      }
    }
  }
  return true;
}
static bool compare_reduce_struct(const optiling::v3::ReduceCompileInfo ptr1,
                                  const optiling::v3::ReduceCompileInfo ptr2) {
  if (ptr1.is_const != ptr2.is_const) {
   std::cout << "ERROR1";
    return false;
  }
  if (ptr1.is_const_post != ptr2.is_const_post) {
  std::cout << "ERROR2";
    return false;
  }
  if (ptr1.atomic != ptr2.atomic) {
  std::cout << "ERROR3" <<std::endl;
  std::cout << ptr1.atomic <<std::endl;
  std::cout << ptr2.atomic <<std::endl;
    return false;
  }
  if (ptr1.is_keep_dims != ptr2.is_keep_dims) {
  std::cout << "ERROR4";
    return false;
  }
  if (ptr1.idx_before_reduce != ptr2.idx_before_reduce) {
  std::cout << "ERROR5";
    return false;
  }
  if (ptr1.zero_ub_factor != ptr2.zero_ub_factor) {
   std::cout << "ERROR6";
    return false;
  }
  if (ptr1.core_num != ptr2.core_num) {
  std::cout << "ERROR7";
    return false;
  }
  if (ptr1.min_block_size != ptr2.min_block_size) {
  std::cout << "ERROR8";
    return false;
  }
  if (ptr1.coef != ptr2.coef) {
  std::cout << "ERROR9";
    return false;
  }
  if (ptr1.pattern_info != ptr2.pattern_info) {
  std::cout << "ERROR10";
    return false;
  }
  if (ptr1.ub_info_rf != ptr2.ub_info_rf) {
  std::cout << "ERROR11";
    return false;
  }
  if (ptr1.ub_info != ptr2.ub_info) {
  std::cout << "ERROR12";
    return false;
  }
  if (ptr1.ori_axis.first != ptr1.ori_axis.first) {
  std::cout << "ERROR13";
    return false;
  }
  if (ptr1.ori_axis.second != ptr1.ori_axis.second) {
  std::cout << "ERROR14";
    return false;
  }
  if (ptr1.axes_idx.first != ptr1.axes_idx.first) {
  std::cout << "ERROR15";
    return false;
  }
  if (ptr1.axes_idx.second != ptr1.axes_idx.second) {
  std::cout << "ERROR16";
    return false;
  }
  if (ptr1.compile_pattern.first != ptr1.compile_pattern.first) {
  std::cout << "ERROR17";
    return false;
  }
  if (ptr1.compile_pattern.second != ptr1.compile_pattern.second) {
  std::cout << "ERROR18";
    return false;
  }
  if (!compare_map(ptr1.block_dim_map, ptr2.block_dim_map)) {
  std::cout << "ERROR19";
    return false;
  }
  if (!compare_map(ptr1.atomic_flags_map, ptr2.atomic_flags_map)) {
  std::cout << "ERROR20";
    return false;
  }

  return true;
}

TEST_F(ReduceTilingV3_RT2, ReduceParseTest1) {
  std::string compile_info = R"({"_ori_axis": [0],
                                "_pattern": "CommReduce",
                                "_common_info": [32, 1, 8, 1, 1],
                                "_pattern_info": [5],
                                "axes_idx":0,
                                "_compile_pattern":0,
                                "_ub_info": [16256],
                                "_ub_info_rf": [16256],
                                "_block_dims": {"40000400": 0},
                                "_atomic_flags": {"40000400": false}})";
  v3::ReduceCompileInfo expectcompile_info;
  expectcompile_info.is_const = false;
  expectcompile_info.is_const_post = false;
  expectcompile_info.idx_before_reduce = 0;
  expectcompile_info.zero_ub_factor = -1;
  expectcompile_info.core_num = 32;
  expectcompile_info.is_keep_dims = true;
  expectcompile_info.min_block_size = 8;
  expectcompile_info.atomic = true;
  expectcompile_info.coef = 1;
  expectcompile_info.pattern_info = {5};
  expectcompile_info.ub_info_rf = {16256};
  expectcompile_info.ub_info = {16256};
  expectcompile_info.ori_axis.first = true;
  expectcompile_info.ori_axis.second = {0};
  expectcompile_info.axes_idx.first = false;
  expectcompile_info.compile_pattern.first = false;
  expectcompile_info.block_dim_map = {{"40000400", 0}};
  expectcompile_info.atomic_flags_map = {{"40000400",false}};

  std::vector<std::vector<int64_t>> inputs {
    {1}
  };
  std::vector<std::vector<int64_t>> outputs {
    {1}
  };
  ge::DataType dtype = ge::DT_FLOAT;
  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::v3::ReduceCompileInfo reduce_info;
  test.SetCompileInfo(compile_info, &reduce_info);

  ASSERT_TRUE(compare_reduce_struct(expectcompile_info, reduce_info));
}


TEST_F(ReduceTilingV3_RT2, ReduceTiling1) {
  using namespace optiling;

  std::vector<std::vector<int64_t>> inputs {
    {1}
  };
  std::vector<std::vector<int64_t>> outputs {
    {1}
  };

  std::string compile_info = R"({ "_ori_axis": [0], "_pattern": "CommReduce","push_status": 0,"_common_info": [32, 1, 8,
   1, 1], "_pattern_info": [5], "_ub_info": [16256], "_ub_info_rf": [16256],"_reduce_vars": {"4293966796": [20000, 30000, 40000]}, "_vars": {"-1000500": ["_dim_1_0", "_block_factor", "_ub_factor"]}})";

  ge::DataType dtype = ge::DT_FLOAT;
  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::v3::ReduceCompileInfo reduce_info;
  test.SetCompileInfo(compile_info, &reduce_info);
  EXPECT_EQ(test.Test(), true);
  EXPECT_EQ(test.GetBlockDims(), 1);
  EXPECT_EQ(test.GetInt32TilingData(), "1, 1, 1");
}

TEST_F(ReduceTilingV3_RT2, ReduceTiling2) {
  using namespace optiling;

  std::vector<std::vector<int64_t>> inputs {
    {2, 39, 0}
  };
  std::vector<std::vector<int64_t>> outputs {
    {2, 39, 1}
  };

  std::string compile_info = R"({ "_ori_axis": [2], "_pattern": "CommReduce", "push_status": 0,
                                               "_zero_ub_factor": 25600, "_vars": {"10": ["_dim_1", "_ub_factor"]}})";


  ge::DataType dtype = ge::DT_FLOAT;
  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::v3::ReduceCompileInfo reduce_info;
  test.SetCompileInfo(compile_info, &reduce_info);

  EXPECT_EQ(test.Test(), true);
  EXPECT_EQ(test.GetBlockDims(), 1);
  EXPECT_EQ(test.GetInt32TilingData(), "78, 25600");
}

TEST_F(ReduceTilingV3_RT2, ReduceTiling3) {
  using namespace optiling;

    std::vector<std::vector<int64_t>> inputs {
    {2, 39, 0}
  };
  std::vector<std::vector<int64_t>> outputs {
    {}
  };

  std::string compile_info = R"({ "_ori_axis": [1], "_pattern": "CommReduce", "push_status": 0, "_zero_ub_factor": 32128, "_vars": {"110": ["_dim_2", "_ub_factor"]}})";

  ge::DataType dtype = ge::DT_FLOAT;
  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::v3::ReduceCompileInfo reduce_info;
  test.SetCompileInfo(compile_info, &reduce_info);
  EXPECT_EQ(test.Test(), true);
  EXPECT_EQ(test.GetBlockDims(), 1);
  EXPECT_EQ(test.GetInt32TilingData(), "2, 128");

}

TEST_F(ReduceTilingV3_RT2, ReduceTiling4) {
  using namespace optiling;

  std::string compile_info = R"({"_ori_axis": [0],"_pattern": "CommReduce", "push_status": 0, "_zero_ub_factor": 32512, "_common_info": [32,1,8,1,1], "_pattern_info": [1], "_ub_info":[32512], "_ub_info_rf": [32512], "_reduce_shape_known": true, "_const_shape_post": true, "_compile_pattern": 1, "_block_dims":{"1":32},
     "_atomic_flags":{"1": true},
     "_vars": {"1": []}})";


  std::vector<std::vector<int64_t>> inputs {
    {64, 64}
  };
  std::vector<std::vector<int64_t>> outputs {
    {1, 64}
  };

  ge::DataType dtype = ge::DT_FLOAT;
  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::v3::ReduceCompileInfo reduce_info;
  test.SetCompileInfo(compile_info, &reduce_info);
  EXPECT_EQ(test.Test(), true);
}

TEST_F(ReduceTilingV3_RT2, ReduceTiling5) {
  using namespace optiling;

  std::string compile_info = R"({ "_ori_axis": [0], "_pattern": "CommReduce","push_status": 0,"common_info": [32, 1, 8, 1, 1], "pattern_info": [20000], "ub_info": [16256], "_ub_info_rf": [16256], "_vars": {"-1000500": ["dim_1_0", "block_factor", "ub_factor"]}})";

  std::vector<std::vector<int64_t>> inputs {
    {1}
  };
  std::vector<std::vector<int64_t>> outputs {
    {1}
  };

  ge::DataType dtype = ge::DT_FLOAT;
  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::v3::ReduceCompileInfo reduce_info;
  test.SetCompileInfo(compile_info, &reduce_info);
  EXPECT_EQ(test.Test(), false);
}

TEST_F(ReduceTilingV3_RT2, ReduceTiling6) {
  using namespace optiling;
  std::string compile_info = R"({ "axes_idx": 0, "_pattern": "CommReduce","push_status": 0,"common_info": [32, 1, 8, 1, 1], "pattern_info": [20000], "ub_info": [16256], "_ub_info_rf": [16256], "_vars": {"-1000500": ["dim_1_0", "block_factor", "ub_factor"]}})";

  std::vector<std::vector<int64_t>> inputs {
    {1}
  };
  std::vector<std::vector<int64_t>> outputs {
    {1}
  };

  ge::DataType dtype = ge::DT_FLOAT;
  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::v3::ReduceCompileInfo reduce_info;
  test.SetCompileInfo(compile_info, &reduce_info);
  EXPECT_EQ(test.Test(), false);
}

// ReduceTiling7 const_tensor

// FineTuning tune0
TEST_F(ReduceTilingV3_RT2, ReduceTiling8) {
  using namespace optiling;


  std::string compile_info = R"({"_ori_axis": [0],"_pattern": "CommReduce", "push_status": 0,
                               "_zero_ub_factor": 32512, "_common_info": [32,1,16,0,1],
                               "_pattern_info": [5,4,9], "_ub_info":[21632, 21376, 21632],
                               "_ub_info_rf": [21632,16000,21632],
                               "_pattern": "CommReduce",
                               "_reduce_vars": {"4292866396": [20000,20001,20002, 30000, 40000]},
                               "_vars": {"1": []}})";

  std::vector<std::vector<int64_t>> inputs {
    {10000, 9, 80}
  };
  std::vector<std::vector<int64_t>> outputs {
    {1, 9, 80}
  };

  ge::DataType dtype = ge::DT_FLOAT;
  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::v3::ReduceCompileInfo reduce_info;
  test.SetCompileInfo(compile_info, &reduce_info);

  EXPECT_EQ(test.Test(), true);
}

// FineTuning tune1
TEST_F(ReduceTilingV3_RT2, ReduceTiling9) {
  using namespace optiling;

  std::string compile_info = R"({"_ori_axis": [0,2,3,4,5],"_pattern": "CommReduce",
                               "_common_info": [32,1,8,1,1],
                               "_pattern_info": [5,4,9], "_ub_info":[32512, 32128, 16128],
                               "_ub_info_rf": [32512, 21376, 32512],
                               "_pattern": "CommReduce",
                               "_reduce_vars": {"1100900": [20000,20001,20002,20003, 30000, 40000]},
                               "_vars": {"1": []}})";
  std::vector<std::vector<int64_t>> inputs {
    {16, 1, 8, 38, 1, 16, 16}
  };
  std::vector<std::vector<int64_t>> outputs {
    {1, 16}
  };

  optiling::utils::OpRunInfo runInfo;

  ge::DataType dtype = ge::DT_FLOAT;
  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::v3::ReduceCompileInfo reduce_info;
  test.SetCompileInfo(compile_info, &reduce_info);
  EXPECT_EQ(test.Test(), true);
}

// for new interface
TEST_F(ReduceTilingV3_RT2, ReduceTiling10) {
  using namespace optiling;

  std::string compile_info = R"({"_ori_axis": [0],"_pattern": "CommReduce", "push_status": 0, "_zero_ub_factor": 32512, "_common_info": [32,1,8,1,1], "_pattern_info": [1], "_ub_info":[32512], "_ub_info_rf": [32512], "_reduce_shape_known": true, "_const_shape_post": true, "_compile_pattern": 1, "_block_dims":{"1":32},
       "_atomic_flags":{"1": true},
       "_vars": {"1": []}})";
  std::vector<std::vector<int64_t>> inputs {
    {64,64}
  };
  std::vector<std::vector<int64_t>> outputs {
    {1, 16}
  };
  // new interface
  std::vector<int64_t> ori_axis{{0}};
  ge::DataType dtype = ge::DT_FLOAT;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::v3::ReduceCompileInfo reduce_info;
  test.SetCompileInfo(compile_info, &reduce_info);
//  OpInfo opInfo(inputs, dtype, ori_axis, &reduce_info);
  OpInfo opInfo(&reduce_info);
  opInfo.SetAxes(&ori_axis);
  EXPECT_EQ(test.Test(&opInfo), true);
}


TEST_F(ReduceTilingV3_RT2, ReduceTiling11) {
  using namespace optiling;

  std::vector<std::vector<int64_t>> inputs {
    {1}
  };
  std::vector<std::vector<int64_t>> outputs {
    {1}
  };

  std::string compile_info = R"({"_ori_axis": [0],
                                "_pattern": "CommReduce",
                                "_common_info": [32, 1, 8, 1, 1],
                                "_pattern_info": [5],
                                "_ub_info": [16256],
                                "_ub_info_rf": [16256],
                                "_reduce_vars": {"4293966796": [20000,20001,20002, 30000, 40000]},
                                "_vars": {"-1000500": ["_dim_1_0", "_block_factor", "_ub_factor"]}})";

  // new interface
  std::vector<int64_t> ori_axis{{0}};
  ge::DataType dtype = ge::DT_FLOAT;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::v3::ReduceCompileInfo reduce_info;
  test.SetCompileInfo(compile_info, &reduce_info);
//  OpInfo opInfo(inputs, dtype, ori_axis, &reduce_info);
  OpInfo opInfo(&reduce_info);
  opInfo.SetAxes(&ori_axis);
  EXPECT_EQ(test.Test(&opInfo), true);
}

TEST_F(ReduceTilingV3_RT2, ReduceTiling12) {
  using namespace optiling;

  std::vector<std::vector<int64_t>> inputs {
    {64,64}
  };
  std::vector<std::vector<int64_t>> outputs {
    {1,64}
  };

  std::string compile_info = R"({"_ori_axis": [-2],"_pattern": "CommReduce", "_zero_ub_factor": 32512, "_common_info": [32,1,8,1,1], "_pattern_info": [1], "_ub_info":[32512], "_ub_info_rf": [32512], "_reduce_shape_known": true, "_compile_pattern": 1, "_block_dims":{"1":32},
         "_atomic_flags":{"1": true},
         "_vars": {"1": []}})";

  // new interface
  std::vector<int64_t> ori_axis{{-2}};
  ge::DataType dtype = ge::DT_FLOAT;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::v3::ReduceCompileInfo reduce_info;
  test.SetCompileInfo(compile_info, &reduce_info);
//  OpInfo opInfo(inputs, dtype, ori_axis, &reduce_info);
  OpInfo opInfo(&reduce_info);
  opInfo.SetAxes(&ori_axis);
  EXPECT_EQ(test.Test(&opInfo), true);
}

TEST_F(ReduceTilingV3_RT2, ReduceTiling13) {
  using namespace optiling;

  std::vector<std::vector<int64_t>> inputs {
    {12456,15}
  };
  std::vector<std::vector<int64_t>> outputs {
    {12456,1}
  };

  std::string compile_info = R"({"_ori_axis": [-1],"_pattern": "CommReduce", "_zero_ub_factor": 32512, "_common_info":
  [32,1,8,1,1,256], "_pattern_info": [1], "_ub_info":[32512], "_ub_info_rf": [32512],"_ub_info_pad": [20462],
  "_reduce_shape_known": true,
  "_compile_pattern": 1, "_block_dims":{"1":32},
         "_atomic_flags":{"1": true},
         "_vars": {"1": []}})";

  // new interface
  std::vector<int64_t> ori_axis{{-1}};
  ge::DataType dtype = ge::DT_FLOAT;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::v3::ReduceCompileInfo reduce_info;
  test.SetCompileInfo(compile_info, &reduce_info);
//  OpInfo opInfo(inputs, dtype, ori_axis, &reduce_info);
  OpInfo opInfo(&reduce_info);
  opInfo.SetAxes(&ori_axis);
  EXPECT_EQ(test.Test(&opInfo), true);
}


TEST_F(ReduceTilingV3_RT2, ReduceTiling13_1) {
  using namespace optiling;
  std::vector<std::vector<int64_t>> inputs {
    {12456,15,45}
  };
  std::vector<std::vector<int64_t>> outputs {
    {12456,1,45}
  };

  std::string compile_info = R"({"_ori_axis": [1],"_pattern": "CommReduce", "_zero_ub_factor": 32512, "_common_info":
  [32,1,8,1,1,256], "_pattern_info": [1], "_ub_info":[32512], "_ub_info_rf": [32512],"_ub_info_pad": [20462],
  "_reduce_shape_known": true,
  "_compile_pattern": 1, "_block_dims":{"1":32},
         "_atomic_flags":{"1": true},
         "_vars": {"1": []}})";

  // new interface
  std::vector<int64_t> ori_axis{{1}};

  ge::DataType dtype = ge::DT_FLOAT;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::v3::ReduceCompileInfo reduce_info;
  test.SetCompileInfo(compile_info, &reduce_info);
//  OpInfo opInfo(inputs, dtype, ori_axis, &reduce_info);
  OpInfo opInfo(&reduce_info);
  opInfo.SetAxes(&ori_axis);
  EXPECT_EQ(test.Test(&opInfo), true);
}

TEST_F(ReduceTilingV3_RT2, ReduceTiling13_2) {
  using namespace optiling;

  std::vector<std::vector<int64_t>> inputs {
    {12,4444,4}
  };
  std::vector<std::vector<int64_t>> outputs {
    {12,1,4}
  };

  std::string compile_info = R"({"_ori_axis": [1],"_pattern": "CommReduce", "_zero_ub_factor": 32512, "_common_info":
  [32,1,8,0,1,256], "_pattern_info": [1], "_ub_info":[32512], "_ub_info_rf": [32512],"_ub_info_pad": [20462],
  "_reduce_shape_known": true,
  "_compile_pattern": 1, "_block_dims":{"1":32},
         "_atomic_flags":{"1": true},
         "_vars": {"1": []}})";

  // new interface
  std::vector<int64_t> ori_axis{{1}};
  ge::DataType dtype = ge::DT_FLOAT;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::v3::ReduceCompileInfo reduce_info;
  test.SetCompileInfo(compile_info, &reduce_info);
//  OpInfo opInfo(inputs, dtype, ori_axis, &reduce_info);
  OpInfo opInfo(&reduce_info);
  opInfo.SetAxes(&ori_axis);
  EXPECT_EQ(test.Test(&opInfo), true);
}


TEST_F(ReduceTilingV3_RT2, ReduceTiling13_3) {
  using namespace optiling;
  std::vector<std::vector<int64_t>> inputs {
    {3, 81920,2}
  };
  std::vector<std::vector<int64_t>> outputs {
    {3,1,2}
  };

  std::string compile_info = R"({"_ori_axis": [-1],"_pattern": "CommReduce", "_zero_ub_factor": 32512, "_common_info":
  [32,1,8,1,1,256], "_pattern_info": [1], "_ub_info":[32512], "_ub_info_rf": [32512],"_ub_info_pad": [20462],
  "_reduce_shape_known": true,
  "_compile_pattern": 1, "_block_dims":{"1":32},
         "_atomic_flags":{"1": true},
         "_vars": {"1": []}})";

  // new interface
  std::vector<int64_t> ori_axis{{-1}};
  ge::DataType dtype = ge::DT_FLOAT;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::v3::ReduceCompileInfo reduce_info;
  test.SetCompileInfo(compile_info, &reduce_info);
//  OpInfo opInfo(inputs, dtype, ori_axis, &reduce_info);
  OpInfo opInfo(&reduce_info);
  opInfo.SetAxes(&ori_axis);
  EXPECT_EQ(test.Test(&opInfo), true);
}

TEST_F(ReduceTilingV3_RT2, ReduceTiling14) {
  using namespace optiling;

  std::vector<std::vector<int64_t>> inputs {
    {123456,15}
  };
  std::vector<std::vector<int64_t>> outputs {
    {123456,1}
  };

  std::string compile_info = R"({"_ori_axis": [-1],"_pattern": "CommReduce", "_zero_ub_factor": 32512, "_common_info":
  [32,1,8,1,1,256,1], "_pattern_info": [1], "_ub_info":[32512], "_ub_info_rf": [32512],"_ub_info_pad": [20462],
  "_ub_info_transpose": [32512], "_reduce_shape_known": true, "_compile_pattern": 1, "_block_dims":{"1":32},
   "_atomic_flags":{"1": true}, "_vars": {"1": []}})";

  // new interface
  std::vector<int64_t> ori_axis{{-1}};
  ge::DataType dtype = ge::DT_FLOAT;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::v3::ReduceCompileInfo reduce_info;
  test.SetCompileInfo(compile_info, &reduce_info);
//  OpInfo opInfo(inputs, dtype, ori_axis, &reduce_info);
  OpInfo opInfo(&reduce_info);
  opInfo.SetAxes(&ori_axis);
  EXPECT_EQ(test.Test(&opInfo), true);
}

static void ReduceSumCompute(std::vector<int64_t> inputA, std::vector<int64_t> inputB, int32_t* axes,
                             std::vector<int64_t> output, ge::DataType dtypeA, ge::DataType dtypeB,
                             ge::DataType dtypeOutput, std::string compile_info, bool isCustom, std::string caseName) {
  using namespace optiling;
  std::vector<std::vector<int64_t>> input_shapes{inputA, inputB};
  std::vector<std::vector<int64_t>> output_shapes{output};

  AutoTilingTest test(input_shapes, output_shapes, dtypeA, dtypeB);
  optiling::v3::ReduceCompileInfo reduce_info;
  test.SetCompileInfo(compile_info, &reduce_info);
  test.SetInt32ConstInput(1, axes, 1);

  if (!isCustom) {
    EXPECT_EQ(test.Test(), true);
  } else {
//    OpInfo opInfo(input_shapes, dtypeA, &reduce_info);
  OpInfo opInfo(&reduce_info);
    EXPECT_EQ(test.Test(&opInfo), true);
  }
}

static void ReduceSumComputeInt64(std::vector<int64_t> inputA, std::vector<int64_t> inputB, int64_t* axes,
                             std::vector<int64_t> output, ge::DataType dtypeA, ge::DataType dtypeB,
                             ge::DataType dtypeOutput, std::string compile_info, bool isCustom, std::string caseName) {
  using namespace optiling;
  std::vector<std::vector<int64_t>> input_shapes{inputA, inputB};
  std::vector<std::vector<int64_t>> output_shapes{output};

  AutoTilingTest test(input_shapes, output_shapes, dtypeA, dtypeB);
  optiling::v3::ReduceCompileInfo reduce_info;
  test.SetCompileInfo(compile_info, &reduce_info);
  test.SetInt64ConstInput(1, axes, 1);

  if (!isCustom) {
    EXPECT_EQ(test.Test(), true);
  } else {
//    OpInfo opInfo(input_shapes, dtypeA);
  OpInfo opInfo(&reduce_info);
    EXPECT_EQ(test.Test(&opInfo), true);
  }
}

TEST_F(ReduceTilingV3_RT2, ReduceSumTiling1) {
  std::string caseName = "ReduceSumTiling1";
  std::string compile_info = R"({"_pattern": "CommReduce", "_common_info": [32,1,8,1,1], "_pattern_info": [5, 4, 9],
  "axes_idx":1, "_ub_info_rf": [32512, 21376, 32512],"_reduce_vars": {"1000900": [20000,20001, 30000, 40000]},
  "_ub_info": [32512, 21376, 16128], "_idx_before_reduce": 0, "_vars": {"4293966796": ["_dim_1", "_block_factor", "_ub_factor"], "4293866796": ["_dim_1", "_block_factor", "_ub_factor"], "500": ["_dim_1", "_block_factor", "_ub_factor"], "100500": ["_dim_1", "_block_factor", "_ub_factor"], "2147483647": ["_dim_1"], "4294966896": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "4294866896": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "1000400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "1100400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "4294966396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4294866396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4294766396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4292866396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4292766396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "1000900": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "1100900": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "1200900": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"]}, "_normal_vars": {"4293966796": ["_dim_1", "_block_factor", "_ub_factor"], "4293866796": ["_dim_1", "_block_factor", "_ub_factor"], "500": ["_dim_1", "_block_factor", "_ub_factor"], "100500": ["_dim_1", "_block_factor", "_ub_factor"], "2147483647": ["_dim_1"], "4294966896": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "4294866896": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "1000400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "1100400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "4294966396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4294866396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4294766396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4292866396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4292766396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "1000900": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "1100900": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "1200900": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"]}, "_attr_vars": {"4293966796": [], "4293866796": [], "500": [], "100500": [], "2147483647": [], "4294966896": [], "4294866896": [], "1000400": [], "1100400": [], "4294966396": [], "4294866396": [], "4294766396": [], "4292866396": [], "4292766396": [], "1000900": [], "1100900": [], "1200900": []}, "_custom_vars": {"4293966796": [], "4293866796": [], "500": [], "100500": [], "2147483647": [], "4294966896": [], "4294866896": [], "1000400": [], "1100400": [], "4294966396": [], "4294866396": [], "4294766396": [], "4292866396": [], "4292766396": [], "1000900": [], "1100900": [], "1200900": []}})";
  std::vector<int64_t> inputA{32,256};
  std::vector<int64_t> inputB{1};
  int32_t axes[1] = {0};
  std::vector<int64_t> output{1,256};
  ge::DataType dtypeA = ge::DT_FLOAT;
  ge::DataType dtypeB = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  bool isCustom = true;
  ReduceSumCompute(inputA, inputB, axes, output, dtypeA, dtypeB, dtypeOutput, compile_info, isCustom, caseName);
}

TEST_F(ReduceTilingV3_RT2, ReduceSumTiling2) {
  std::string caseName = "ReduceSumTiling2";
  std::string compile_info = R"({"_pattern": "CommReduce", "_common_info": [32,1,8,1,1], "axes_idx":1, "_pattern_info":
  [5, 4, 9], "_ub_info_rf": [32512, 21376, 32512],"_reduce_vars": {"1000900": [20000,20001, 30000, 40000]}, "_ub_info": [32512, 21376, 16128], "_idx_before_reduce": 0, "_vars": {"4293966796": ["_dim_1", "_block_factor", "_ub_factor"], "4293866796": ["_dim_1", "_block_factor", "_ub_factor"], "500": ["_dim_1", "_block_factor", "_ub_factor"], "100500": ["_dim_1", "_block_factor", "_ub_factor"], "2147483647": ["_dim_1"], "4294966896": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "4294866896": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "1000400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "1100400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "4294966396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4294866396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4294766396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4292866396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4292766396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "1000900": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "1100900": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "1200900": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"]}, "_normal_vars": {"4293966796": ["_dim_1", "_block_factor", "_ub_factor"], "4293866796": ["_dim_1", "_block_factor", "_ub_factor"], "500": ["_dim_1", "_block_factor", "_ub_factor"], "100500": ["_dim_1", "_block_factor", "_ub_factor"], "2147483647": ["_dim_1"], "4294966896": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "4294866896": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "1000400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "1100400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "4294966396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4294866396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4294766396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4292866396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4292766396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "1000900": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "1100900": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "1200900": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"]}, "_attr_vars": {"4293966796": [], "4293866796": [], "500": [], "100500": [], "2147483647": [], "4294966896": [], "4294866896": [], "1000400": [], "1100400": [], "4294966396": [], "4294866396": [], "4294766396": [], "4292866396": [], "4292766396": [], "1000900": [], "1100900": [], "1200900": []}, "_custom_vars": {"4293966796": [], "4293866796": [], "500": [], "100500": [], "2147483647": [], "4294966896": [], "4294866896": [], "1000400": [], "1100400": [], "4294966396": [], "4294866396": [], "4294766396": [], "4292866396": [], "4292766396": [], "1000900": [], "1100900": [], "1200900": []}})";
  std::vector<int64_t> inputA{{32,256}};
  std::vector<int64_t> inputB{1};
  int32_t axes[1] = {0};
  std::vector<int64_t> output{1,256};
  ge::DataType dtypeA = ge::DT_FLOAT;
  ge::DataType dtypeB = ge::DT_INT32;
  ge::DataType dtypeOutput = dtypeA;
  bool isCustom = false;
  ReduceSumCompute(inputA, inputB, axes, output, dtypeA, dtypeB, dtypeOutput, compile_info, isCustom, caseName);
}

TEST_F(ReduceTilingV3_RT2, ReduceSumTiling3) {
  std::string caseName = "ReduceSumTiling3";
  std::string compile_info = R"({"_pattern": "CommReduce", "_common_info": [32,1,8,1,1], "axes_idx":1, "_pattern_info":
  [5, 4, 9], "_ub_info_rf": [32512, 21376, 32512], "_reduce_vars": {"1000900": [20000,20001, 30000, 40000]},"_ub_info": [32512, 21376, 16128], "_idx_before_reduce": 0, "_vars": {"4293966796": ["_dim_1", "_block_factor", "_ub_factor"], "4293866796": ["_dim_1", "_block_factor", "_ub_factor"], "500": ["_dim_1", "_block_factor", "_ub_factor"], "100500": ["_dim_1", "_block_factor", "_ub_factor"], "2147483647": ["_dim_1"], "4294966896": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "4294866896": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "1000400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "1100400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "4294966396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4294866396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4294766396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4292866396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4292766396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "1000900": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "1100900": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "1200900": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"]}, "_normal_vars": {"4293966796": ["_dim_1", "_block_factor", "_ub_factor"], "4293866796": ["_dim_1", "_block_factor", "_ub_factor"], "500": ["_dim_1", "_block_factor", "_ub_factor"], "100500": ["_dim_1", "_block_factor", "_ub_factor"], "2147483647": ["_dim_1"], "4294966896": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "4294866896": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "1000400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "1100400": ["_dim_0", "_dim_1", "_block_factor", "_ub_factor"], "4294966396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4294866396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4294766396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4292866396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4292766396": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "1000900": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "1100900": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "1200900": ["_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"]}, "_attr_vars": {"4293966796": [], "4293866796": [], "500": [], "100500": [], "2147483647": [], "4294966896": [], "4294866896": [], "1000400": [], "1100400": [], "4294966396": [], "4294866396": [], "4294766396": [], "4292866396": [], "4292766396": [], "1000900": [], "1100900": [], "1200900": []}, "_custom_vars": {"4293966796": [], "4293866796": [], "500": [], "100500": [], "2147483647": [], "4294966896": [], "4294866896": [], "1000400": [], "1100400": [], "4294966396": [], "4294866396": [], "4294766396": [], "4292866396": [], "4292766396": [], "1000900": [], "1100900": [], "1200900": []}})";
  std::vector<int64_t> inputA{32,256};
  std::vector<int64_t> inputB{1};
  int64_t axes[1] = {0};
  std::vector<int64_t> output{1,256};
  ge::DataType dtypeA = ge::DT_FLOAT;
  ge::DataType dtypeB = ge::DT_INT64;
  ge::DataType dtypeOutput = dtypeA;
  bool isCustom = false;
  ReduceSumComputeInt64(inputA, inputB, axes, output, dtypeA, dtypeB, dtypeOutput, compile_info, isCustom, caseName);
}

TEST_F(ReduceTilingV3_RT2, ReduceTiling_var_attr) {
  using namespace optiling;

  std::vector<std::vector<int64_t>> inputs {
    {1}
  };
  std::vector<std::vector<int64_t>> outputs {
    {1}
  };
  ge::DataType dtype = ge::DT_FLOAT;

  std::string compile_info = R"({ "_ori_axis": [0],"_var_attr_mode":1,
                                 "_var_attrs": {"4293966796":[{"length":1,"name":"alpha","type":"int32", "index": 0,
                                               "src_type":"int64"}]},
                                 "_pattern": "CommReduce",
                                 "push_status": 0,
                                 "_common_info": [32, 1, 8, 1, 1],
                                 "_pattern_info": [5],
                                 "_ub_info": [16256], "_ub_info_rf": [16256],
                                 "_reduce_vars": {"4293966796": [20000, 30000, 40000]},
                                 "_vars": {"-1000500": ["_dim_1_0", "_block_factor", "_ub_factor"]}})";

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  std::vector<std::pair<std::string, int64_t>> reduce_attr = {{"alpha", 12345}};
  test.SetAttrs<int64_t>(reduce_attr);

  optiling::v3::ReduceCompileInfo reduce_info;
  test.SetCompileInfo(compile_info, &reduce_info);
  EXPECT_EQ(test.Test(), true);

  EXPECT_EQ(test.GetBlockDims(), 1);
  EXPECT_EQ(test.GetInt32TilingData(), "1, 1, 1, 12345");
}


TEST_F(ReduceTilingV3_RT2, ReduceTiling_var_attr_2) {
  using namespace optiling;

  std::vector<std::vector<int64_t>> inputs {
    {64,64}
  };
  std::vector<std::vector<int64_t>> outputs {
    {1,64}
  };
  ge::DataType dtype = ge::DT_FLOAT;

  std::string compile_info = R"({"_ori_axis": [0],"_var_attr_mode":0,"_var_attrs": [{"length":1,"index": 0,"name":"alpha","type":"int32","src_type":"int64"}],
                            "_pattern": "CommReduce", "push_status": 0, "_zero_ub_factor": 32512, "_common_info": [32,1,8,1,1], "_pattern_info": [1], "_ub_info":[32512], "_ub_info_rf": [32512], "_reduce_shape_known": true, "_const_shape_post": true, "_compile_pattern": 1, "_block_dims":{"1":32},
                            "_atomic_flags":{"1": true}, "_vars": {"1": []}})";

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  std::vector<std::pair<std::string, int64_t>> reduce_attr = {{"alpha", 12345}};
  test.SetAttrs<int64_t>(reduce_attr);

  optiling::v3::ReduceCompileInfo reduce_info;
  test.SetCompileInfo(compile_info, &reduce_info);
  EXPECT_EQ(test.Test(), true);
  EXPECT_EQ(test.GetInt32TilingData(), "12345");
}

TEST_F(ReduceTilingV3_RT2, ReduceTiling_reduce_all_rt3) {
  using namespace optiling;
  std::vector<std::vector<int64_t>> inputs {
    {500,64,64}
  };
  std::vector<std::vector<int64_t>> outputs {
    {500,1,64}
  };
  ge::DataType dtype = ge::DT_FLOAT;

  std::string compile_info = R"({"_reduce_axes_type": 0, "_pattern": "CommReduce", "_zero_ub_factor": 32512,
  "_common_info":[32,1,8,1,1,256], "_pattern_info": [5,4,9], "_ub_info":[32512, 32128, 16128],"_reduce_vars": {"1100400": [20000, 30000, 40000]},
  "_ub_info_rf": [32512, 21376, 32512]})";

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::v3::ReduceCompileInfo reduce_info;
  test.SetCompileInfo(compile_info, &reduce_info);
  EXPECT_EQ(test.Test(), true);
  EXPECT_EQ(test.GetTilingKey(), 1100400);
}


TEST_F(ReduceTilingV3_RT2, ReduceTilingGroupReduce) {
  using namespace optiling;
  std::vector<std::vector<int64_t>> inputs {
    {32, 544, 512}
  };
  std::vector<std::vector<int64_t>> outputs {
    {1, 1, 512}
  };
  std::string compile_info = R"({"_ori_axis": [0, 1],"_pattern": "CommReduce",
                               "_zero_ub_factor": 32512, "_common_info": [32,1,16,0,1,256,1,4,1],
                               "_pattern_info": [5,4,9], "_ub_info":[21632, 15488, 21632],
                               "_ub_info_rf": [21632, 15488, 21632],
                               "_ub_info_pad": [0, 17664, 19560],
                               "_workspace_size": 2,
                               "_reduce_vars": {"4293866396": [20000, 20001,20002,30000, 40000]},
                               "_vars": {"4293866396": []}})";

  std::vector<ge::DataType> dtype = {ge::DT_FLOAT16};

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::v3::ReduceCompileInfo reduce_info;
  test.SetCompileInfo(compile_info, &reduce_info);
  EXPECT_EQ(test.Test(), true);
  EXPECT_EQ(test.GetTilingKey(), 4293866396);
  EXPECT_EQ(test.GetInt32TilingData(), "1, 17408, 512, 544, 42");
}

TEST_F(ReduceTilingV3_RT2, ReduceTiling_reduce_5HD_rt2) {
  using namespace optiling;
  std::vector<std::vector<int64_t>> inputs {
    {85, 2, 11, 7, 16}
  };
  std::vector<std::vector<int64_t>> outputs {
    {85, 2, 11, 7, 16}
  };
  std::vector<std::vector<int64_t>> ori_inputs {
    {85, 11, 7, 22}
  };
  std::vector<std::vector<int64_t>> ori_outputs {
    {85, 11, 7, 22}
  };

  std::vector<ge::DataType> dtype = {ge::DT_FLOAT16};
  std::vector<ge::Format> format = {ge::FORMAT_ND};

  std::string compile_info = R"({"_disable_fuse_axes": [1, 4], "_ori_axis": [1, 4], "_ori_dim_index": 3, "_pattern":
  "CommReduce", "_common_info": [32, 1, 32, 0, 1, 1024, 1, 2, true], "_pattern_info": [20], "_ub_info_rf": [27648],
  "_ub_info": [35456], "_ub_info_pad": [0], "_ub_info_transpose": [0], "_idx_before_reduce": 0, "_vars": {"4294965296":
  ["_ori_dim_3", "_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4294765296": ["_ori_dim_3", "_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4294865296": ["_ori_dim_3", "_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4294665296": ["_ori_dim_3", "_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4292765296": ["_ori_dim_3", "_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4292865296": ["_ori_dim_3", "_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4292665296": ["_ori_dim_3", "_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "2147483647": ["_ori_dim_3", "_dim_0", "_dim_1", "_dim_2"]}, "_normal_vars": {"4294965296": ["_ori_dim_3", "_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4294765296": ["_ori_dim_3", "_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4294865296": ["_ori_dim_3", "_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4294665296": ["_ori_dim_3", "_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4292765296": ["_ori_dim_3", "_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4292865296": ["_ori_dim_3", "_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "4292665296": ["_ori_dim_3", "_dim_0", "_dim_1", "_dim_2", "_block_factor", "_ub_factor"], "2147483647": ["_ori_dim_3", "_dim_0", "_dim_1", "_dim_2"]}, "_custom_vars": {"4294965296": [], "4294765296": [], "4294865296": [], "4294665296": [], "4292765296": [], "4292865296": [], "4292665296": [], "2147483647": []}, "_reduce_vars": {"4294965296": [10003, 20000, 20001, 20002, 30000, 40000], "4294765296": [10003, 20000, 20001, 20002, 30000, 40000], "4294865296": [10003, 20000, 20001, 20002, 30000, 40000], "4294665296": [10003, 20000, 20001, 20002, 30000, 40000], "4292765296": [10003, 20000, 20001, 20002, 30000, 40000], "4292865296": [10003, 20000, 20001, 20002, 30000, 40000], "4292665296": [10003, 20000, 20001, 20002, 30000, 40000], "2147483647": [10003, 20000, 20001, 20002]}})";

  AutoTilingTest test(ori_inputs, inputs, ori_outputs, outputs, dtype, dtype, format, format, format, format);
  optiling::v3::ReduceCompileInfo reduce_info;
  test.SetCompileInfo(compile_info, &reduce_info);
  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "22, 85, 2, 77, 3, 3";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);

}
