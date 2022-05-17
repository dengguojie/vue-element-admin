/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "dynamic_atomic_addr_clean.h"
#include "register/op_impl_registry.h"
#include <gtest/gtest.h>
#include "kernel_run_context_facker.h"
#include "runtime/storage_shape.h"
#include "runtime/infer_shape_context.h"
#include "runtime/tiling_context.h"
#include "runtime/kernel_context.h"
#include "runtime/tiling_data.h"

namespace gert_test {
class DynamicAtomicAddrCleanUT : public testing::Test {};
static string to_string(int32_t *tiling_data, int32_t tiling_len) {
  string result;
  for (size_t i = 0; i < tiling_len; i ++) {
    result += std::to_string(tiling_data[i]);
    result += " ";
  }
  return result;
}

TEST_F(DynamicAtomicAddrCleanUT, TilingOk) {
  // compile info
  int32_t ub_size = 126976;
  int32_t core_num = 2;
  int32_t workspace_num = 1;
  std::vector<int64_t> _workspace_size_list = {32};
  optiling::DynamicAtomicAddrCleanCompileInfo compile_info;
  compile_info.core_num = core_num;
  compile_info.ub_size = ub_size;
  compile_info.workspace_num = workspace_num;
  compile_info._workspace_size_list = _workspace_size_list;

  // tiling data
  auto param = gert::TilingData::CreateCap(2048);
  auto holder = gert::TilingContextFaker()
                    .NodeIoNum(0, 0)
                    .CompileInfo(&compile_info)
                    .TilingData(param.get())
                    .Build();

  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicAtomicAddrClean"), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicAtomicAddrClean")->tiling;
  ASSERT_NE(tiling_func, nullptr);

  EXPECT_EQ(tiling_func(holder.GetContext<gert::TilingContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(DynamicAtomicAddrCleanUT, TilingParseOk) {
  char *json_str = "{\"_workspace_size_list\": [32], \"vars\": {\"ub_size\": 126976, \"core_num\": 2, \"workspace_num\": 1}}";
  optiling::DynamicAtomicAddrCleanCompileInfo compile_info;
  auto holder = gert::KernelRunContextFaker()
                    .KernelIONum(1, 1)
                    .Inputs({json_str})
                    .Outputs({&compile_info})
                    .Build();

  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicAtomicAddrClean"), nullptr);
  auto tiling_prepare_func = gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicAtomicAddrClean")->tiling_parse;
  ASSERT_NE(tiling_prepare_func, nullptr);

  EXPECT_EQ(tiling_prepare_func(holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
  int32_t ub_size = 126976;
  int32_t core_num = 2;
  int32_t workspace_num = 1;
  std::vector<int64_t> _workspace_size_list = {32};
  EXPECT_EQ(compile_info.ub_size, ub_size);
  EXPECT_EQ(compile_info.core_num, core_num);
  EXPECT_EQ(compile_info.workspace_num, workspace_num);
  EXPECT_EQ(compile_info._workspace_size_list[0], _workspace_size_list[0]);
}

TEST_F(DynamicAtomicAddrCleanUT, DetaDepencyOk) {
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicAtomicAddrClean"), nullptr);
  EXPECT_FALSE(gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicAtomicAddrClean")->IsInputDataDependency(0));
  EXPECT_FALSE(gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicAtomicAddrClean")->IsInputDataDependency(1));
  EXPECT_FALSE(gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicAtomicAddrClean")->IsInputDataDependency(2));
  EXPECT_FALSE(gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicAtomicAddrClean")->IsInputDataDependency(3));
}
}  // namespace gert_test
