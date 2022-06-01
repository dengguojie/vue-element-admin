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
#include "runtime/continuous_vector.h"
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
  size_t clean_size = 4 * 56 * 56 * 16 * 2; // float16

  // compile info
  optiling::DynamicAtomicAddrCleanCompileInfo compile_info;
  compile_info.core_num = 2;
  compile_info.ub_size = 126976;
  compile_info.workspace_num = 1;
  compile_info._workspace_index_list = {0};

  auto workspace_sizes_holder = gert::ContinuousVector::Create<size_t>(8);
  auto workspace_sizes = reinterpret_cast<gert::ContinuousVector *>(workspace_sizes_holder.get());
  workspace_sizes->SetSize(1);
  reinterpret_cast<size_t *>(workspace_sizes->MutableData())[0] = 32;

  // tiling data
  auto param = gert::TilingData::CreateCap(2048);
  auto self_workspace_sizes = gert::ContinuousVector::Create<gert::TensorAddress>(8);


  auto holder = gert::KernelRunContextFaker()
                    // 输入信息：一个workspace size，一个需要清空的shape；后面跟着CompileInfo，TilingFunc
                    .KernelIONum(2 + 2, 5)
                    .IrInputNum(2)
                    .NodeIoNum(2, 0)  // 一个workspace size，一个需要清空的shape；
                    .Inputs({workspace_sizes_holder.get(), (void*)clean_size, &compile_info, nullptr})
                    .Outputs({nullptr, nullptr, nullptr, param.get(), self_workspace_sizes.get()})
                    .Build();

  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicAtomicAddrClean"), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicAtomicAddrClean")->tiling;
  ASSERT_NE(tiling_func, nullptr);

  EXPECT_EQ(tiling_func(holder.GetContext<gert::TilingContext>()), ge::GRAPH_SUCCESS);
  // todo check tiling result
}

TEST_F(DynamicAtomicAddrCleanUT, DynamicAtomicAddrClean_tiling_1) {
  size_t clean_size = 4 * 56 * 56 * 16 * 2; // float16

  // compile info
  optiling::DynamicAtomicAddrCleanCompileInfo compile_info;
  compile_info.core_num = 2;
  compile_info.ub_size = 126976;
  compile_info.workspace_num = 1;
  compile_info._workspace_index_list = {32};

  auto workspace_sizes_holder = gert::ContinuousVector::Create<size_t>(8);
  auto workspace_sizes = reinterpret_cast<gert::ContinuousVector *>(workspace_sizes_holder.get());
  workspace_sizes->SetSize(8);
  reinterpret_cast<size_t *>(workspace_sizes->MutableData())[0] = 1;
  reinterpret_cast<size_t *>(workspace_sizes->MutableData())[1] = 2;
  reinterpret_cast<size_t *>(workspace_sizes->MutableData())[2] = 3;
  reinterpret_cast<size_t *>(workspace_sizes->MutableData())[3] = 4;
  reinterpret_cast<size_t *>(workspace_sizes->MutableData())[4] = 5;
  reinterpret_cast<size_t *>(workspace_sizes->MutableData())[5] = 6;
  reinterpret_cast<size_t *>(workspace_sizes->MutableData())[6] = 7;
  reinterpret_cast<size_t *>(workspace_sizes->MutableData())[7] = 8;
  // tiling data
  auto param = gert::TilingData::CreateCap(2048);
  auto self_workspace_sizes = gert::ContinuousVector::Create<gert::TensorAddress>(8);


  auto holder = gert::KernelRunContextFaker()
                    // 输入信息：一个workspace size，一个需要清空的shape；后面跟着CompileInfo，TilingFunc
                    .KernelIONum(2 + 2, 5)
                    .IrInputNum(2)
                    .NodeIoNum(2, 0)  // 一个workspace size，一个需要清空的shape；
                    .Inputs({workspace_sizes_holder.get(), (void*)clean_size, &compile_info, nullptr})
                    .Outputs({nullptr, nullptr, nullptr, param.get(), self_workspace_sizes.get()})
                    .Build();

  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicAtomicAddrClean"), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicAtomicAddrClean")->tiling;
  ASSERT_NE(tiling_func, nullptr);

  EXPECT_EQ(tiling_func(holder.GetContext<gert::TilingContext>()), ge::GRAPH_SUCCESS);
  // todo check tiling result
}

TEST_F(DynamicAtomicAddrCleanUT, TilingParseOk) {
  char *json_str = "{\"_workspace_index_list\": [32], \"vars\": {\"ub_size\": 126976, \"core_num\": 2, \"workspace_num\": 1}}";
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
  std::vector<int64_t> _workspace_index_list = {32};
  EXPECT_EQ(compile_info.ub_size, ub_size);
  EXPECT_EQ(compile_info.core_num, core_num);
  EXPECT_EQ(compile_info.workspace_num, workspace_num);
  EXPECT_EQ(compile_info._workspace_index_list[0], _workspace_index_list[0]);
}

TEST_F(DynamicAtomicAddrCleanUT, DetaDepencyOk) {
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicAtomicAddrClean"), nullptr);
  EXPECT_FALSE(gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicAtomicAddrClean")->IsInputDataDependency(0));
  EXPECT_FALSE(gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicAtomicAddrClean")->IsInputDataDependency(1));
  EXPECT_FALSE(gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicAtomicAddrClean")->IsInputDataDependency(2));
  EXPECT_FALSE(gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicAtomicAddrClean")->IsInputDataDependency(3));
}
}  // namespace gert_test
