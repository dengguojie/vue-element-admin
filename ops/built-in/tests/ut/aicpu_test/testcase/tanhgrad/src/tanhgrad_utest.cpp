/**
 * Copyright 2021 Huawei Technologies Co., Ltd.
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

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <complex>
#include <iostream>
#include "aicpu_read_file.h"
#include "aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"

class TEST_TANHGRAD_UT : public testing::Test {
 protected:
  std::float_t* float_null_{nullptr};
  std::float_t float_0_[0];
  std::float_t float_12_[12]{1.0f};
  std::float_t float_16_[16]{1.0f};
  std::int64_t int64_22_[22]{0L};
  std::double_t double_16_[16]{0.0f};
  std::int32_t int32_16_[16]{1};
  std::int32_t int32_22_[22]{1};
  std::float_t float_empty_[0]{};
  bool bool_22_[22]{true};
};

template <typename T>
inline aicpu::DataType ToDataType() {
  return aicpu::DataType::DT_UNDEFINED;
}

template <>
inline aicpu::DataType ToDataType<bool>() {
  return aicpu::DataType::DT_BOOL;
}

template <>
inline aicpu::DataType ToDataType<Eigen::half>() {
  return aicpu::DataType::DT_FLOAT16;
}

template <>
inline aicpu::DataType ToDataType<std::float_t>() {
  return aicpu::DataType::DT_FLOAT;
}

template <>
inline aicpu::DataType ToDataType<std::double_t>() {
  return aicpu::DataType::DT_DOUBLE;
}

template <>
inline aicpu::DataType ToDataType<std::int8_t>() {
  return aicpu::DataType::DT_INT8;
}

template <>
inline aicpu::DataType ToDataType<std::int16_t>() {
  return aicpu::DataType::DT_INT16;
}

template <>
inline aicpu::DataType ToDataType<std::int32_t>() {
  return aicpu::DataType::DT_INT32;
}

template <>
inline aicpu::DataType ToDataType<std::int64_t>() {
  return aicpu::DataType::DT_INT64;
}

template <>
inline aicpu::DataType ToDataType<std::uint8_t>() {
  return aicpu::DataType::DT_UINT8;
}

template <>
inline aicpu::DataType ToDataType<std::uint16_t>() {
  return aicpu::DataType::DT_UINT16;
}

template <>
inline aicpu::DataType ToDataType<std::uint32_t>() {
  return aicpu::DataType::DT_UINT32;
}

template <>
inline aicpu::DataType ToDataType<std::uint64_t>() {
  return aicpu::DataType::DT_UINT64;
}

template <>
inline aicpu::DataType ToDataType<std::complex<std::float_t> >() {
  return aicpu::DataType::DT_COMPLEX64;
}
template <>
inline aicpu::DataType ToDataType<std::complex<std::double_t> >() {
  return aicpu::DataType::DT_COMPLEX128;
}

template <typename T>
inline const char* ToDataName() {
  return typeid(T).name();
}

template <>
inline const char* ToDataName<Eigen::half>() {
  return "float16";
}

template <>
inline const char* ToDataName<std::float_t>() {
  return "float32";
}

template <>
inline const char* ToDataName<std::double_t>() {
  return "float64";
}

template <>
inline const char* ToDataName<std::int8_t>() {
  return "int8";
}

template <>
inline const char* ToDataName<std::int16_t>() {
  return "int16";
}

template <>
inline const char* ToDataName<std::int32_t>() {
  return "int32";
}

template <>
inline const char* ToDataName<std::int64_t>() {
  return "int64";
}

template <>
inline const char* ToDataName<std::uint8_t>() {
  return "uint8";
}

template <>
inline const char* ToDataName<std::uint16_t>() {
  return "uint16";
}

template <>
inline const char* ToDataName<std::uint32_t>() {
  return "uint32";
}

template <>
inline const char* ToDataName<std::uint64_t>() {
  return "uint64";
}

template <>
inline const char* ToDataName<std::complex<std::float_t> >() {
  return "complex64";
}
template <>
inline const char* ToDataName<std::complex<std::double_t> >() {
  return "complex128";
}
inline std::uint64_t SizeOf(std::vector<std::int64_t>& shape) {
  return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
}

template <std::shared_ptr<aicpu::Device> aicpu::CpuKernelContext::*DEVICE_PTR>
struct Friend {
  friend void SetDeviceNull(aicpu::CpuKernelContext& ctx) {
    ctx.*DEVICE_PTR = nullptr;
  }
};

template struct Friend<&aicpu::CpuKernelContext::device_>;
void SetDeviceNull(aicpu::CpuKernelContext& ctx);

inline void RunKernelTanhGrad(std::shared_ptr<aicpu::NodeDef> node_def,
                              aicpu::DeviceType device_type, uint32_t expect,
                              bool bad_kernel = false) {
  std::string node_def_str;
  node_def->SerializeToString(node_def_str);
  aicpu::CpuKernelContext ctx(device_type);
  EXPECT_EQ(ctx.Init(node_def.get()), aicpu::KernelStatus::KERNEL_STATUS_OK);
  if (bad_kernel) {
    SetDeviceNull(ctx);
  }
  std::uint32_t ret{aicpu::CpuKernelRegister::Instance().RunCpuKernel(ctx)};
  EXPECT_EQ(ret, expect);
}

template <typename Tin1, typename Tin2, typename Tout>
void CreateAndRunKernelTanhGrad(
    const std::vector<std::int64_t>& dims_in_y,
    const std::vector<std::int64_t>& dims_in_dy,
    const std::vector<std::int64_t>& dims_out, Tin1* input_y, Tin2* input_dy,
    Tout* output,
    aicpu::KernelStatus status = aicpu::KernelStatus::KERNEL_STATUS_OK,
    bool bad_kernel = false) {
  const auto data_type_in_y{ToDataType<Tin1>()};
  const auto data_type_in_dy{ToDataType<Tin2>()};
  const auto data_type_out{ToDataType<Tout>()};
  EXPECT_NE(data_type_in_y, aicpu::DataType::DT_UNDEFINED);
  EXPECT_NE(data_type_in_dy, aicpu::DataType::DT_UNDEFINED);
  EXPECT_NE(data_type_out, aicpu::DataType::DT_UNDEFINED);
  auto node_def{aicpu::CpuKernelUtils::CreateNodeDef()};
  aicpu::NodeDefBuilder(node_def.get(), "TanhGrad", "TanhGrad")
      .Input({"y", data_type_in_y, dims_in_y, input_y})
      .Input({"dy", data_type_in_dy, dims_in_dy, input_dy})
      .Output({"z", data_type_out, dims_out, output});
  RunKernelTanhGrad(node_def, aicpu::DeviceType::HOST, status, bad_kernel);
}

template <typename Tin1, typename Tin2, typename Tout>
void CreateAndRunKernelNullptrTanhGrad(
    const std::vector<std::int64_t>& dims_in_y,
    const std::vector<std::int64_t>& dims_in_dy,
    const std::vector<std::int64_t>& dims_out, Tin1 input_y, Tin2 input_dy,
    Tout output,
    aicpu::KernelStatus status = aicpu::KernelStatus::KERNEL_STATUS_OK,
    bool bad_kernel = false) {
  const auto data_type_in_y{ToDataType<Tin1>()};
  const auto data_type_in_dy{ToDataType<Tin2>()};
  const auto data_type_out{ToDataType<Tout>()};
  EXPECT_NE(data_type_in_y, aicpu::DataType::DT_UNDEFINED);
  EXPECT_NE(data_type_in_dy, aicpu::DataType::DT_UNDEFINED);
  EXPECT_NE(data_type_out, aicpu::DataType::DT_UNDEFINED);
  auto node_def{aicpu::CpuKernelUtils::CreateNodeDef()};
  aicpu::NodeDefBuilder(node_def.get(), "TanhGrad", "TanhGrad")
      .Input({"y", data_type_in_y, dims_in_y, input_y})
      .Input({"dy", data_type_in_dy, dims_in_dy, input_dy})
      .Output({"z", data_type_out, dims_out, output});
  RunKernelTanhGrad(node_def, aicpu::DeviceType::HOST, status, bad_kernel);
}

template <typename Tin, typename Tout>
void CreateAndRunKernelTanhGrad(
    const std::vector<std::int64_t>& dims, Tin* input_y, Tin* input_dy,
    Tout* output,
    aicpu::KernelStatus status = aicpu::KernelStatus::KERNEL_STATUS_OK,
    bool bad_kernel = false) {
  CreateAndRunKernelTanhGrad(dims, dims, dims, input_y, input_dy, output,
                             status, bad_kernel);
}

template <typename Tin, typename Tout>
void CreateAndRunKernelTanhGradParamInvalid(
    const std::vector<std::int64_t>& dims_in_y,
    const std::vector<std::int64_t>& dims_in_dy,
    const std::vector<std::int64_t>& dims_out, Tin* input_y, Tin* input_dy,
    Tout* output) {
  CreateAndRunKernelTanhGrad(dims_in_y, dims_in_dy, dims_out, input_y, input_dy,
                             output,
                             aicpu::KernelStatus::KERNEL_STATUS_PARAM_INVALID);
}

template <typename Tin, typename Tout>
void CreateAndRunKernelTanhGradParamInvalid(
    const std::vector<std::int64_t>& dims, Tin* input_y, Tin* input_dy,
    Tout* output) {
  CreateAndRunKernelTanhGradParamInvalid(dims, dims, dims, input_y, input_dy,
                                         output);
}

template <typename Tin1, typename Tin2, typename Tout>
void CreateAndRunKernelTanhGradParamInvalidV2(
    const std::vector<std::int64_t>& dims, Tin1* input_y, Tin2* input_dy,
    Tout* output) {
  CreateAndRunKernelTanhGrad(dims, dims, dims, input_y, input_dy, output,
                             aicpu::KernelStatus::KERNEL_STATUS_PARAM_INVALID);
}

template <typename Tin, typename Tout>
void CreateAndRunKernelTanhGradInnerError(
    const std::vector<std::int64_t>& dims_in_y,
    const std::vector<std::int64_t>& dims_in_dy,
    const std::vector<std::int64_t>& dims_out, Tin* input_y, Tin* input_dy,
    Tout* output) {
  CreateAndRunKernelTanhGrad(
      dims_in_y, dims_in_dy, dims_out, input_y, input_dy, output,
      aicpu::KernelStatus::KERNEL_STATUS_INNER_ERROR, true);
}

template <typename Tin, typename Tout>
void CreateAndRunKernelTanhGradInnerError(const std::vector<std::int64_t>& dims,
                                          Tin* input_y, Tin* input_dy,
                                          Tout* output) {
  CreateAndRunKernelTanhGradInnerError(dims, dims, dims, input_y, input_dy,
                                       output);
}

template <typename T>
bool ReadBinFile(std::string file_name, T buf[], std::size_t size) {
  try {
    std::ifstream in_file{file_name};
    if (!in_file.is_open()) {
      std::cout << "open file: " << file_name << " failed." << std::endl;
      return false;
    }
    in_file.read(reinterpret_cast<char*>(buf), size * sizeof(buf[0]));
    in_file.close();
  } catch (std::exception& e) {
    std::cout << "read file " << file_name << " failed, " << e.what()
              << std::endl;
    return false;
  }
  return true;
}

template <typename T>
bool WriteBinFile(std::string file_name, T buf[], std::size_t size) {
  try {
    std::ofstream out_file{file_name};
    if (!out_file.is_open()) {
      std::cout << "open file: " << file_name << " failed." << std::endl;
      return false;
    }
    out_file.write(reinterpret_cast<char*>(buf), size * sizeof(buf[0]));
    out_file.close();
  } catch (std::exception& e) {
    std::cout << "write file " << file_name << " failed, " << e.what()
              << std::endl;
    return false;
  }
  return true;
}

template <typename T>
bool WriteFile(std::string file_name, T buf[], std::size_t size) {
  try {
    std::ofstream out_file{file_name};
    if (!out_file.is_open()) {
      std::cout << "open file: " << file_name << " failed." << std::endl;
      return false;
    }
    out_file << std::setprecision(std::numeric_limits<T>::digits10 + 1);
    for (auto index{0}; index < size; index++) {
      out_file << buf[index] << '\n';
    }
    out_file.close();
  } catch (std::exception& e) {
    std::cout << "write file " << file_name << " failed, " << e.what()
              << std::endl;
    return false;
  }
  return true;
}

template <typename T>
void RunTestTanhGrad(std::uint32_t flag) {
  const auto data_name{ToDataName<T>()};
  if (flag == 0) {
    const auto data_name{ToDataName<T>()};
    // read input y
    std::uint64_t dim_y[1];
    std::string data_dim_y_path{ktestcaseFilePath +
                                "tanhgrad/data/tanhgrad_bigdata_y_" +
                                data_name + "_dim.txt"};
    EXPECT_EQ(ReadFile(data_dim_y_path, dim_y, 1), true);

    std::uint64_t shape_y[dim_y[0]];
    std::string data_shape_y_path{ktestcaseFilePath +
                                  "tanhgrad/data/tanhgrad_bigdata_y_" +
                                  data_name + "_shape.txt"};
    EXPECT_EQ(ReadFile(data_shape_y_path, shape_y, dim_y[0]), true);

    std::vector<std::int64_t> dims_y(shape_y, shape_y + dim_y[0]);
    auto input_y_size{SizeOf(dims_y)};

    T data_y[input_y_size];
    std::string data_y_path{ktestcaseFilePath +
                            "tanhgrad/data/tanhgrad_bigdata_input_y_" +
                            data_name + ".bin"};
    EXPECT_EQ(ReadBinFile(data_y_path, data_y, input_y_size), true);
    // read input dy
    std::uint64_t dim_dy[1];
    std::string data_dim_dy_path{ktestcaseFilePath +
                                 "tanhgrad/data/tanhgrad_bigdata_dy_" +
                                 data_name + "_dim.txt"};
    EXPECT_EQ(ReadFile(data_dim_dy_path, dim_dy, 1), true);

    std::uint64_t shape_dy[dim_dy[0]];
    std::string data_shape_dy_path{ktestcaseFilePath +
                                   "tanhgrad/data/tanhgrad_bigdata_dy_" +
                                   data_name + "_shape.txt"};
    EXPECT_EQ(ReadFile(data_shape_dy_path, shape_dy, dim_dy[0]), true);

    std::vector<std::int64_t> dims_dy(shape_dy, shape_dy + dim_dy[0]);
    auto input_dy_size{SizeOf(dims_dy)};
    T data_dy[input_dy_size];
    std::string data_dy_path{ktestcaseFilePath +
                             "tanhgrad/data/tanhgrad_bigdata_input_dy_" +
                             data_name + ".bin"};
    EXPECT_EQ(ReadBinFile(data_dy_path, data_dy, input_dy_size), true);

    // read output z
    std::uint64_t dim_z[1];
    std::string data_dim_z_path{ktestcaseFilePath +
                                "tanhgrad/data/tanhgrad_bigdata_output_" +
                                data_name + "_dim.txt"};
    EXPECT_EQ(ReadFile(data_dim_z_path, dim_z, 1), true);

    std::uint64_t shape_z[dim_z[0]];
    std::string data_shape_z_path{ktestcaseFilePath +
                                  "tanhgrad/data/tanhgrad_bigdata_output_" +
                                  data_name + "_shape.txt"};
    EXPECT_EQ(ReadFile(data_shape_z_path, shape_z, dim_z[0]), true);

    std::vector<std::int64_t> dims_z(shape_z, shape_z + dim_z[0]);
    auto input_z_size{SizeOf(dims_z)};
    T output[input_z_size];
    CreateAndRunKernelTanhGrad(dims_y, dims_dy, dims_z, data_y, data_dy,
                               output);
    std::string out_data_actual_path{ktestcaseFilePath +
                                     "tanhgrad/data/tanhgrad_bigdata_output_" +
                                     data_name + "_actual.txt"};
    EXPECT_EQ(WriteFile(out_data_actual_path, output, input_z_size), true);

    T expect_out[input_z_size];
    std::string out_data_path{ktestcaseFilePath +
                              "tanhgrad/data/tanhgrad_bigdata_output_" +
                              data_name + ".bin"};
    EXPECT_EQ(ReadBinFile(out_data_path, expect_out, input_z_size), true);
    EXPECT_EQ(CompareResult(output, expect_out, input_z_size), true);
  } else if (flag == 1) {
    const auto data_name{ToDataName<T>()};
    // read input y
    std::uint64_t dim_y[1];
    std::string data_dim_y_path{ktestcaseFilePath +
                                "tanhgrad/data/tanhgrad_data_y_" + data_name +
                                "_dim.txt"};
    EXPECT_EQ(ReadFile(data_dim_y_path, dim_y, 1), true);

    std::uint64_t shape_y[dim_y[0]];
    std::string data_shape_y_path{ktestcaseFilePath +
                                  "tanhgrad/data/tanhgrad_data_y_" + data_name +
                                  "_shape.txt"};
    EXPECT_EQ(ReadFile(data_shape_y_path, shape_y, dim_y[0]), true);

    std::vector<std::int64_t> dims_y(shape_y, shape_y + dim_y[0]);
    auto input_y_size{SizeOf(dims_y)};

    T data_y[input_y_size];
    std::string data_y_path{ktestcaseFilePath +
                            "tanhgrad/data/tanhgrad_data_input_y_" + data_name +
                            ".bin"};
    EXPECT_EQ(ReadBinFile(data_y_path, data_y, input_y_size), true);
    // read input dy
    std::uint64_t dim_dy[1];
    std::string data_dim_dy_path{ktestcaseFilePath +
                                 "tanhgrad/data/tanhgrad_data_dy_" + data_name +
                                 "_dim.txt"};
    EXPECT_EQ(ReadFile(data_dim_dy_path, dim_dy, 1), true);

    std::uint64_t shape_dy[dim_dy[0]];
    std::string data_shape_dy_path{ktestcaseFilePath +
                                   "tanhgrad/data/tanhgrad_data_dy_" +
                                   data_name + "_shape.txt"};
    EXPECT_EQ(ReadFile(data_shape_dy_path, shape_dy, dim_dy[0]), true);

    std::vector<std::int64_t> dims_dy(shape_dy, shape_dy + dim_dy[0]);
    auto input_dy_size{SizeOf(dims_dy)};

    T data_dy[input_dy_size];
    std::string data_dy_path{ktestcaseFilePath +
                             "tanhgrad/data/tanhgrad_data_input_dy_" +
                             data_name + ".bin"};
    EXPECT_EQ(ReadBinFile(data_dy_path, data_dy, input_dy_size), true);
    // read output z
    std::uint64_t dim_z[1];
    std::string data_dim_z_path{ktestcaseFilePath +
                                "tanhgrad/data/tanhgrad_data_output_" +
                                data_name + "_dim.txt"};
    EXPECT_EQ(ReadFile(data_dim_z_path, dim_z, 1), true);

    std::uint64_t shape_z[dim_z[0]];
    std::string data_shape_z_path{ktestcaseFilePath +
                                  "tanhgrad/data/tanhgrad_data_output_" +
                                  data_name + "_shape.txt"};
    EXPECT_EQ(ReadFile(data_shape_z_path, shape_z, dim_z[0]), true);

    std::vector<std::int64_t> dims_z(shape_z, shape_z + dim_z[0]);
    auto input_z_size{SizeOf(dims_z)};
    T output[input_z_size];
    CreateAndRunKernelTanhGrad(dims_y, dims_dy, dims_z, data_y, data_dy,
                               output);
    std::string out_data_actual_path{ktestcaseFilePath +
                                     "tanhgrad/data/tanhgrad_data_output_" +
                                     data_name + "_actual.txt"};
    EXPECT_EQ(WriteFile(out_data_actual_path, output, input_z_size), true);

    T expect_out[input_z_size];
    std::string out_data_path{ktestcaseFilePath +
                              "tanhgrad/data/tanhgrad_data_output_" +
                              data_name + ".bin"};
    EXPECT_EQ(ReadBinFile(out_data_path, expect_out, input_z_size), true);
    EXPECT_EQ(CompareResult(output, expect_out, input_z_size), true);
  }
}

template <typename T>
void RunTestTanhGradDiff(std::uint32_t flag) {
  const auto data_name{ToDataName<T>()};
  if (flag == 0) {
    const auto data_name{ToDataName<T>()};
    // read input y
    std::uint64_t dim_y[1];
    std::string data_dim_y_path{ktestcaseFilePath +
                                "tanhgrad/data/tanhgrad_diff_bigdata_y_" +
                                data_name + "_dim.txt"};
    EXPECT_EQ(ReadFile(data_dim_y_path, dim_y, 1), true);

    std::uint64_t shape_y[dim_y[0]];
    std::string data_shape_y_path{ktestcaseFilePath +
                                  "tanhgrad/data/tanhgrad_diff_bigdata_y_" +
                                  data_name + "_shape.txt"};
    EXPECT_EQ(ReadFile(data_shape_y_path, shape_y, dim_y[0]), true);

    std::vector<std::int64_t> dims_y(shape_y, shape_y + dim_y[0]);
    auto input_y_size{SizeOf(dims_y)};

    T data_y[input_y_size];
    std::string data_y_path{ktestcaseFilePath +
                            "tanhgrad/data/tanhgrad_diff_bigdata_input_y_" +
                            data_name + ".bin"};
    EXPECT_EQ(ReadBinFile(data_y_path, data_y, input_y_size), true);
    // read input dy
    std::uint64_t dim_dy[1];
    std::string data_dim_dy_path{ktestcaseFilePath +
                                 "tanhgrad/data/tanhgrad_diff_bigdata_dy_" +
                                 data_name + "_dim.txt"};
    EXPECT_EQ(ReadFile(data_dim_dy_path, dim_dy, 1), true);

    std::uint64_t shape_dy[dim_dy[0]];
    std::string data_shape_dy_path{ktestcaseFilePath +
                                   "tanhgrad/data/tanhgrad_diff_bigdata_dy_" +
                                   data_name + "_shape.txt"};
    EXPECT_EQ(ReadFile(data_shape_dy_path, shape_dy, dim_dy[0]), true);

    std::vector<std::int64_t> dims_dy(shape_dy, shape_dy + dim_dy[0]);
    auto input_dy_size{SizeOf(dims_dy)};
    T data_dy[input_dy_size];
    std::string data_dy_path{ktestcaseFilePath +
                             "tanhgrad/data/tanhgrad_diff_bigdata_input_dy_" +
                             data_name + ".bin"};
    EXPECT_EQ(ReadBinFile(data_dy_path, data_dy, input_dy_size), true);

    // read output z
    std::uint64_t dim_z[1];
    std::string data_dim_z_path{ktestcaseFilePath +
                                "tanhgrad/data/tanhgrad_diff_bigdata_output_" +
                                data_name + "_dim.txt"};
    EXPECT_EQ(ReadFile(data_dim_z_path, dim_z, 1), true);

    std::uint64_t shape_z[dim_z[0]];
    std::string data_shape_z_path{
        ktestcaseFilePath + "tanhgrad/data/tanhgrad_diff_bigdata_output_" +
        data_name + "_shape.txt"};
    EXPECT_EQ(ReadFile(data_shape_z_path, shape_z, dim_z[0]), true);

    std::vector<std::int64_t> dims_z(shape_z, shape_z + dim_z[0]);
    auto input_z_size{SizeOf(dims_z)};
    T output[input_z_size];
    CreateAndRunKernelTanhGrad(dims_y, dims_dy, dims_z, data_y, data_dy,
                               output);
    std::string out_data_actual_path{
        ktestcaseFilePath + "tanhgrad/data/tanhgrad_diff_bigdata_output_" +
        data_name + "_actual.txt"};
    EXPECT_EQ(WriteFile(out_data_actual_path, output, input_z_size), true);

    T expect_out[input_z_size];
    std::string out_data_path{ktestcaseFilePath +
                              "tanhgrad/data/tanhgrad_diff_bigdata_output_" +
                              data_name + ".bin"};
    EXPECT_EQ(ReadBinFile(out_data_path, expect_out, input_z_size), true);
    EXPECT_EQ(CompareResult(output, expect_out, input_z_size), true);
  } else if (flag == 1) {
    const auto data_name{ToDataName<T>()};
    // read input y
    std::uint64_t dim_y[1];
    std::string data_dim_y_path{ktestcaseFilePath +
                                "tanhgrad/data/tanhgrad_diff_data_y_" +
                                data_name + "_dim.txt"};
    EXPECT_EQ(ReadFile(data_dim_y_path, dim_y, 1), true);

    std::uint64_t shape_y[dim_y[0]];
    std::string data_shape_y_path{ktestcaseFilePath +
                                  "tanhgrad/data/tanhgrad_diff_data_y_" +
                                  data_name + "_shape.txt"};
    EXPECT_EQ(ReadFile(data_shape_y_path, shape_y, dim_y[0]), true);

    std::vector<std::int64_t> dims_y(shape_y, shape_y + dim_y[0]);
    auto input_y_size{SizeOf(dims_y)};

    T data_y[input_y_size];
    std::string data_y_path{ktestcaseFilePath +
                            "tanhgrad/data/tanhgrad_diff_data_input_y_" +
                            data_name + ".bin"};
    EXPECT_EQ(ReadBinFile(data_y_path, data_y, input_y_size), true);
    // read input dy
    std::uint64_t dim_dy[1];
    std::string data_dim_dy_path{ktestcaseFilePath +
                                 "tanhgrad/data/tanhgrad_diff_data_dy_" +
                                 data_name + "_dim.txt"};
    EXPECT_EQ(ReadFile(data_dim_dy_path, dim_dy, 1), true);

    std::uint64_t shape_dy[dim_dy[0]];
    std::string data_shape_dy_path{ktestcaseFilePath +
                                   "tanhgrad/data/tanhgrad_diff_data_dy_" +
                                   data_name + "_shape.txt"};
    EXPECT_EQ(ReadFile(data_shape_dy_path, shape_dy, dim_dy[0]), true);

    std::vector<std::int64_t> dims_dy(shape_dy, shape_dy + dim_dy[0]);
    auto input_dy_size{SizeOf(dims_dy)};

    T data_dy[input_dy_size];
    std::string data_dy_path{ktestcaseFilePath +
                             "tanhgrad/data/tanhgrad_diff_data_input_dy_" +
                             data_name + ".bin"};
    EXPECT_EQ(ReadBinFile(data_dy_path, data_dy, input_dy_size), true);
    // read output z
    std::uint64_t dim_z[1];
    std::string data_dim_z_path{ktestcaseFilePath +
                                "tanhgrad/data/tanhgrad_diff_data_output_" +
                                data_name + "_dim.txt"};
    EXPECT_EQ(ReadFile(data_dim_z_path, dim_z, 1), true);

    std::uint64_t shape_z[dim_z[0]];
    std::string data_shape_z_path{ktestcaseFilePath +
                                  "tanhgrad/data/tanhgrad_diff_data_output_" +
                                  data_name + "_shape.txt"};
    EXPECT_EQ(ReadFile(data_shape_z_path, shape_z, dim_z[0]), true);

    std::vector<std::int64_t> dims_z(shape_z, shape_z + dim_z[0]);
    auto input_z_size{SizeOf(dims_z)};
    T output[input_z_size];
    CreateAndRunKernelTanhGrad(dims_y, dims_dy, dims_z, data_y, data_dy,
                               output);
    std::string out_data_actual_path{
        ktestcaseFilePath + "tanhgrad/data/tanhgrad_diff_data_output_" +
        data_name + "_actual.txt"};
    EXPECT_EQ(WriteFile(out_data_actual_path, output, input_z_size), true);

    T expect_out[input_z_size];
    std::string out_data_path{ktestcaseFilePath +
                              "tanhgrad/data/tanhgrad_diff_data_output_" +
                              data_name + ".bin"};
    EXPECT_EQ(ReadBinFile(out_data_path, expect_out, input_z_size), true);
    EXPECT_EQ(CompareResult(output, expect_out, input_z_size), true);
  }
}

#define ADD_SAME_BIGDATA_CASE(base_type, aicpu_type)              \
  TEST_F(TEST_TANHGRAD_UT, DATA_TYPE_SAME_BIGDATA_##aicpu_type) { \
    RunTestTanhGrad<base_type>(0);                                \
  }
#define ADD_SAME_CASE(base_type, aicpu_type)              \
  TEST_F(TEST_TANHGRAD_UT, DATA_TYPE_SAME_##aicpu_type) { \
    RunTestTanhGrad<base_type>(1);                        \
  }

#define ADD_DIFF_CASE(base_type, aicpu_type)              \
  TEST_F(TEST_TANHGRAD_UT, DATA_TYPE_DIFF_##aicpu_type) { \
    RunTestTanhGradDiff<base_type>(1);                    \
  }

// exception instance
TEST_F(TEST_TANHGRAD_UT, BAD_KERNEL_EXCEPTION) {
  CreateAndRunKernelTanhGrad(
      {2, 8}, {2, 8}, {2, 8}, float_16_, float_16_, float_16_,
      aicpu::KernelStatus::KERNEL_STATUS_INNER_ERROR, true);
}

TEST_F(TEST_TANHGRAD_UT, INPUT1_DTYPE_EXCEPTION) {
  CreateAndRunKernelTanhGradParamInvalid({2, 8}, float_16_, float_16_,
                                         double_16_);
}

TEST_F(TEST_TANHGRAD_UT, INPUT2_DTYPE_EXCEPTION) {
  CreateAndRunKernelTanhGradParamInvalidV2({2, 8}, float_16_, double_16_,
                                           float_16_);
}

TEST_F(TEST_TANHGRAD_UT, INPUT3_DTYPE_EXCEPTION) {
  CreateAndRunKernelTanhGradParamInvalidV2({2, 8}, double_16_, float_16_,
                                           float_16_);
}

TEST_F(TEST_TANHGRAD_UT, INPUT_NULL_EXCEPTION) {
  CreateAndRunKernelTanhGradParamInvalid({2, 11}, float_null_, float_null_,
                                         float_null_);
}

TEST_F(TEST_TANHGRAD_UT, OUTPUT_DATA_NULL_EXCEPTION) {
  CreateAndRunKernelTanhGradParamInvalid({0, 0}, float_0_, float_0_,
                                         float_null_);
}

TEST_F(TEST_TANHGRAD_UT, INPUT1_DATA_NULL_EXCEPTION) {
  CreateAndRunKernelTanhGradParamInvalid({0, 0}, float_0_, float_null_,
                                         float_0_);
}

TEST_F(TEST_TANHGRAD_UT, INPUT0_DATA_NULL_EXCEPTION) {
  CreateAndRunKernelTanhGradParamInvalid({0, 0}, float_null_, float_0_,
                                         float_0_);
}
TEST_F(TEST_TANHGRAD_UT, INPUT_EMPTY_EXCEPTION) {
  CreateAndRunKernelTanhGrad({0, 0}, float_0_, float_0_, float_0_);
}
TEST_F(TEST_TANHGRAD_UT, NO_OUTPUT_EXCEPTION) {
  const auto data_type_in{ToDataType<std::float_t>()};
  auto node_def{aicpu::CpuKernelUtils::CreateNodeDef()};
  aicpu::NodeDefBuilder(node_def.get(), "TanhGrad", "TanhGrad")
      .Input({"y", data_type_in, {2, 6}, float_12_})
      .Input({"dy", data_type_in, {2, 6}, float_12_});
  RunKernelTanhGrad(node_def, aicpu::DeviceType::HOST,
                    aicpu::KernelStatus::KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_TANHGRAD_UT, INPUT_BOOL_UNSUPPORT) {
  CreateAndRunKernelTanhGradParamInvalid({2, 11}, bool_22_, bool_22_, bool_22_);
}

ADD_SAME_CASE(Eigen::half, DT_FLOAT16)
ADD_SAME_CASE(std::float_t, DT_FLOAT)
ADD_SAME_CASE(std::double_t, DT_DOUBLE)
ADD_SAME_CASE(std::complex<std::float_t>, DT_COMPLEX64)
ADD_SAME_CASE(std::complex<std::double_t>, DT_COMPLEX128)

ADD_DIFF_CASE(Eigen::half, DT_FLOAT16)
ADD_DIFF_CASE(std::float_t, DT_FLOAT)
ADD_DIFF_CASE(std::double_t, DT_DOUBLE)

ADD_SAME_BIGDATA_CASE(Eigen::half, DT_FLOAT16)
