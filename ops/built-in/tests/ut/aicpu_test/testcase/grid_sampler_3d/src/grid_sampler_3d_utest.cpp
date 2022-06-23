/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "gtest/gtest.h" 
#ifndef private
#define private public
#define protected public
#endif      
#include "aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "aicpu_read_file.h"
#include "node_def_builder.h"
#undef private
#undef protected
#include "Eigen/Core"
#include <iostream>

using namespace std;
using namespace aicpu;

class TEST_GRIDSAMPLER3D_UT : public testing::Test{};

#define CREATE_NODEDEF(shapes, data_types, datas, attr1, attr2, attr3)     \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();         \
  NodeDefBuilder(node_def.get(),"GridSampler3D", "GridSampler3D")          \
      .Input({"x", data_types[0], shapes[0], datas[0]})                    \
      .Input({"grid", data_types[1], shapes[1], datas[1]})                 \
      .Output({"y", data_types[2], shapes[2], datas[2]})                   \
      .Attr("interpolation_mode", attr1)                                   \
      .Attr("padding_mode", attr2)                                         \
      .Attr("align_corners", attr3);

float reflect_coordinates(float coord, int64_t twice_low, int64_t twice_high) {
  if (twice_low == twice_high) {
    return static_cast<float>(0);
  }
  float min = static_cast<float>(twice_low) / 2;
  float span = static_cast<float>(twice_high - twice_low) / 2;
  coord = fabs(coord - min);
  float extra = fmod(coord, span);
  int64_t flips = static_cast<int64_t>(floor(coord / span));
  if (flips % 2 == 0) {
    return extra + min;
  } else {
    return span - extra + min;
  }
}

void grid_sampler_compute_source_index(
    float &coord, int64_t size, string padding_mode, bool align_corners) {
  if (align_corners) {
    coord = ((coord + 1) / 2) * (size - 1);
  } else {
    coord = ((coord + 1) * size - 1) / 2;
  }
  if (padding_mode == "border") {
    coord = min(static_cast<float>(size - 1), max(coord, static_cast<float>(0)));
  } else if (padding_mode == "reflection") {
    if (align_corners) {
      coord = reflect_coordinates(coord, 0, 2 * (size - 1));
    } else {
      coord = reflect_coordinates(coord, -1, 2 * size - 1);
    }
    coord = min(static_cast<float>(size - 1), max(coord, static_cast<float>(0)));
  }
}

bool within_bounds_3d_forward(int64_t d, int64_t h, int64_t w, int64_t D, int64_t H, int64_t W) {
  return d >= 0 && d < D && h >= 0 && h < H && w >= 0 && w < W;
}

void CalcExpectWithHalfData(const NodeDef &node_def, Eigen::half expect_out[]){
  auto x_data = node_def.MutableInputs(0);
  auto x_data_addr = reinterpret_cast<Eigen::half *>(x_data->GetData());
  auto x_shape = x_data->GetTensorShape();
  auto x_dims = x_shape->GetDimSizes();
  int64_t x_stride[5];
  int64_t stride_tmp = 1;
  for (int32_t i = 4; i > -1; i--) {
    x_stride[i] = stride_tmp;
    stride_tmp *= x_dims[i];
  }
  auto grid_data = node_def.MutableInputs(1);
  auto grid_data_addr = reinterpret_cast<Eigen::half *>(grid_data->GetData());
  auto grid_shape = grid_data->GetTensorShape();
  auto grid_dims = grid_shape->GetDimSizes();
  int64_t grid_stride[5];
  stride_tmp = 1;
  for (int32_t i = 4; i > -1; i--) {
    grid_stride[i] = stride_tmp;
    stride_tmp *= grid_dims[i];
  }
  auto y_data = node_def.MutableOutputs(0);
  auto y_shape = y_data->GetTensorShape();
  auto y_dims = y_shape->GetDimSizes();
  int64_t y_stride[5];
  stride_tmp = 1;
  for (int32_t i = 4; i > -1; i--) {
    y_stride[i] = stride_tmp;
    stride_tmp *= y_dims[i];
  }
  int64_t y_data_num = y_data->NumElements();
  auto attrs = node_def.Attrs();
  auto attr1 = attrs["interpolation_mode"];
  auto attr2 = attrs["padding_mode"];
  auto attr3 = attrs["align_corners"];
  string interpolation_mode = attr1->GetString();
  string padding_mode = attr2->GetString();
  bool align_corners = attr3->GetBool();
  int64_t y_iter[5] = {0};
  const int64_t y_c = y_dims[1];
  auto NextIndex = [&]() {
    int64_t carry = 1;
    for (int32_t id = 4; id > -1; id--) {
      if (id == 1) {
        continue;
      }
      int64_t iter_val = y_iter[id] + carry;
      if (iter_val == y_dims[id]) {
        y_iter[id] = 0;
      } else {
        y_iter[id] = iter_val;
        carry = 0;
        break;
      }
    }
    return (carry == 0);
  };
  do {
    int64_t grid_offset = y_iter[0] * grid_stride[0] +
                          y_iter[2] * grid_stride[1] +
                          y_iter[3] * grid_stride[2] +
                          y_iter[4] * grid_stride[3] ;
    float x = static_cast<float>(grid_data_addr[grid_offset]);
    float y = static_cast<float>(grid_data_addr[grid_offset + grid_stride[4]]);
    float z = static_cast<float>(grid_data_addr[grid_offset + 2 * grid_stride[4]]);
    grid_sampler_compute_source_index(x, x_dims[4], padding_mode, align_corners);
    grid_sampler_compute_source_index(y, x_dims[3], padding_mode, align_corners);
    grid_sampler_compute_source_index(z, x_dims[2], padding_mode, align_corners);
    auto x_ptr_NC = y_iter[0] * x_stride[0];
    auto y_ptr_NCDHW = y_iter[0] * y_stride[0] +
                        y_iter[2] * y_stride[2] +
                        y_iter[3] * y_stride[3] +
                        y_iter[4] * y_stride[4] ;
    if (interpolation_mode == "bilinear") {
      int64_t x_tnw = static_cast<int64_t>(floor(x));
      int64_t y_tnw = static_cast<int64_t>(floor(y));
      int64_t z_tnw = static_cast<int64_t>(floor(z));
      int64_t x_tne = x_tnw + 1;
      int64_t y_tne = y_tnw;
      int64_t z_tne = z_tnw;
      int64_t x_tsw = x_tnw;
      int64_t y_tsw = y_tnw + 1;
      int64_t z_tsw = z_tnw;
      int64_t x_tse = x_tnw + 1;
      int64_t y_tse = y_tnw + 1;
      int64_t z_tse = z_tnw;
      int64_t x_bnw = x_tnw;
      int64_t y_bnw = y_tnw;
      int64_t z_bnw = z_tnw + 1;
      int64_t x_bne = x_tnw + 1;
      int64_t y_bne = y_tnw;
      int64_t z_bne = z_tnw + 1;
      int64_t x_bsw = x_tnw;
      int64_t y_bsw = y_tnw + 1;
      int64_t z_bsw = z_tnw + 1;
      int64_t x_bse = x_tnw + 1;
      int64_t y_bse = y_tnw + 1;
      int64_t z_bse = z_tnw + 1;
      Eigen::half tnw = static_cast<Eigen::half>((x_bse - x) * (y_bse - y) * (z_bse - z));
      Eigen::half tne = static_cast<Eigen::half>((x - x_bsw) * (y_bsw - y) * (z_bsw - z));
      Eigen::half tsw = static_cast<Eigen::half>((x_bne - x) * (y - y_bne) * (z_bne - z));
      Eigen::half tse = static_cast<Eigen::half>((x - x_bnw) * (y - y_bnw) * (z_bnw - z));
      Eigen::half bnw = static_cast<Eigen::half>((x_tse - x) * (y_tse - y) * (z - z_tse));
      Eigen::half bne = static_cast<Eigen::half>((x - x_tsw) * (y_tsw - y) * (z - z_tsw));
      Eigen::half bsw = static_cast<Eigen::half>((x_tne - x) * (y - y_tne) * (z - z_tne));
      Eigen::half bse = static_cast<Eigen::half>((x - x_tnw) * (y - y_tnw) * (z - z_tnw));
      for (int64_t c = 0; c < y_c; c++, x_ptr_NC += x_stride[1], y_ptr_NCDHW += y_stride[1]) {
        expect_out[y_ptr_NCDHW] = static_cast<Eigen::half>(0);
        if (within_bounds_3d_forward(z_tnw, y_tnw, x_tnw, x_dims[2], x_dims[3], x_dims[4])) {
          auto x_index = x_ptr_NC + z_tnw * x_stride[2] + y_tnw * x_stride[3] + x_tnw * x_stride[4];
          expect_out[y_ptr_NCDHW] += x_data_addr[x_index] * tnw;
        }
        if (within_bounds_3d_forward(z_tne, y_tne, x_tne, x_dims[2], x_dims[3], x_dims[4])) {
          auto x_index = x_ptr_NC + z_tne * x_stride[2] + y_tne * x_stride[3] + x_tne * x_stride[4];
          expect_out[y_ptr_NCDHW] += x_data_addr[x_index] * tne;
        }
        if (within_bounds_3d_forward(z_tsw, y_tsw, x_tsw, x_dims[2], x_dims[3], x_dims[4])) {
          auto x_index = x_ptr_NC + z_tsw * x_stride[2] + y_tsw * x_stride[3] + x_tsw * x_stride[4];
          expect_out[y_ptr_NCDHW] += x_data_addr[x_index] * tsw;
        }
        if (within_bounds_3d_forward(z_tse, y_tse, x_tse, x_dims[2], x_dims[3], x_dims[4])) {
          auto x_index = x_ptr_NC + z_tse * x_stride[2] + y_tse * x_stride[3] + x_tse * x_stride[4];
          expect_out[y_ptr_NCDHW] += x_data_addr[x_index] * tse;
        }
        if (within_bounds_3d_forward(z_bnw, y_bnw, x_bnw, x_dims[2], x_dims[3], x_dims[4])) {
          auto x_index = x_ptr_NC + z_bnw * x_stride[2] + y_bnw * x_stride[3] + x_bnw * x_stride[4];
          expect_out[y_ptr_NCDHW] += x_data_addr[x_index] * bnw;
        }
        if (within_bounds_3d_forward(z_bne, y_bne, x_bne, x_dims[2], x_dims[3], x_dims[4])) {
          auto x_index = x_ptr_NC + z_bne * x_stride[2] + y_bne * x_stride[3] + x_bne * x_stride[4];
          expect_out[y_ptr_NCDHW] += x_data_addr[x_index] * bne;
        }
        if (within_bounds_3d_forward(z_bsw, y_bsw, x_bsw, x_dims[2], x_dims[3], x_dims[4])) {
          auto x_index = x_ptr_NC + z_bsw * x_stride[2] + y_bsw * x_stride[3] + x_bsw * x_stride[4];
          expect_out[y_ptr_NCDHW] += x_data_addr[x_index] * bsw;
        }
        if (within_bounds_3d_forward(z_bse, y_bse, x_bse, x_dims[2], x_dims[3], x_dims[4])) {
          auto x_index = x_ptr_NC + z_bse * x_stride[2] + y_bse * x_stride[3] + x_bse * x_stride[4];
          expect_out[y_ptr_NCDHW] += x_data_addr[x_index] * bse;
        }
      }
    } else if (interpolation_mode == "nearest") {
      int64_t x_nearest = static_cast<int64_t>(round(x));
      int64_t y_nearest = static_cast<int64_t>(round(y));
      int64_t z_nearest = static_cast<int64_t>(round(z));
      for (int64_t c = 0; c < y_c; c++, x_ptr_NC += x_stride[1], y_ptr_NCDHW += y_stride[1]) {
        if (within_bounds_3d_forward(z_nearest, y_nearest, x_nearest, x_dims[2], x_dims[3], x_dims[4])) {
          auto x_index = x_ptr_NC + z_nearest * x_stride[2] + y_nearest * x_stride[3] + x_nearest * x_stride[4];
          expect_out[y_ptr_NCDHW] = x_data_addr[x_index];
        } else {
          expect_out[y_ptr_NCDHW] = static_cast<Eigen::half>(0);
        }
      }
    }
  } while (NextIndex());
}

// read input and output data from files which generate by your python file
template<typename T>
void RunGridSampler3DKernel1(vector<string> data_files,
                   vector<DataType> data_types,
                   vector<vector<int64_t>> &shapes,
                   string attr1 = "bilinear",
                   string attr2 = "zeros",
                   bool attr3 = false) {
  // read data from file for input1
  string data_path = ktestcaseFilePath + data_files[0];
  uint64_t input1_size = CalTotalElements(shapes, 0);
  T *input1 = new T[input1_size];
  bool status = ReadFile(data_path, input1, input1_size);
  EXPECT_EQ(status, true);

  // read data from file for input2
  data_path = ktestcaseFilePath + data_files[1];
  uint64_t input2_size = CalTotalElements(shapes, 1);
  T *input2 = new T[input2_size];
  status = ReadFile(data_path, input2, input2_size);
  EXPECT_EQ(status, true);

  uint64_t output_size = CalTotalElements(shapes, 2);
  T *output = new T[output_size];
  vector<void *> datas = {(void *)input1,
                          (void *)input2,
                          (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas, attr1, attr2, attr3);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // read data from file for expect ouput
  data_path = ktestcaseFilePath + data_files[2];
  T *output_exp = new T[output_size];
  status = ReadFile(data_path, output_exp, output_size);
  EXPECT_EQ(status, true);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
  delete [] input1;
  delete [] input2;
  delete [] output;
  delete [] output_exp;
}

void RunGridSampler3DKernel2(vector<DataType> data_types,
                             vector<vector<int64_t>> &shapes,
                             string attr1 = "bilinear",
                             string attr2 = "zeros",
                             bool attr3 = false){
  // gen data use SetRandomValue for input1
  uint64_t input1_size = CalTotalElements(shapes, 0);
  Eigen::half *input1 = new Eigen::half[input1_size];
  SetRandomValue<Eigen::half>(input1, input1_size);

  // gen data use SetRandomValue for input2
  uint64_t input2_size = CalTotalElements(shapes, 1);
  Eigen::half *input2 = new Eigen::half[input2_size];
  SetRandomValue<Eigen::half>(input2, input2_size, static_cast<float>(-1.0), static_cast<float>(1.0));

  uint64_t output_size = CalTotalElements(shapes, 2);
  Eigen::half *output = new Eigen::half[output_size];
  vector<void *> datas = {(void *)input1,
                          (void *)input2,
                          (void *)output};

  CREATE_NODEDEF(shapes, data_types, datas, attr1, attr2, attr3);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // calculate output_exp
  Eigen::half *output_exp = new Eigen::half[output_size];
  CalcExpectWithHalfData(*node_def.get(), output_exp);

  bool compare = CompareResult(output, output_exp, output_size);
  EXPECT_EQ(compare, true);
  delete [] input1;
  delete [] input2;
  delete [] output;
  delete [] output_exp;    
}

TEST_F(TEST_GRIDSAMPLER3D_UT, GRIDSAMPLER3D1_SUCC){
  vector<DataType>data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>>shapes = {{5, 2, 4, 4, 4}, {5, 1, 3, 2, 3}, {5, 2, 1, 3, 2}};
  vector<string>files{"grid_sampler_3d/data/gridsampler3d_data_input1_1.txt",
                      "grid_sampler_3d/data/gridsampler3d_data_input2_1.txt",
                      "grid_sampler_3d/data/gridsampler3d_data_output1_1.txt"};
  RunGridSampler3DKernel1<double>(files, data_types, shapes, "bilinear", "zeros", true);
}

TEST_F(TEST_GRIDSAMPLER3D_UT, GRIDSAMPLER3D2_SUCC){
  vector<DataType>data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>>shapes = {{5, 3, 3, 4, 4}, {5, 1, 3, 2, 3}, {5, 3, 1, 3, 2}};
  vector<string>files{"grid_sampler_3d/data/gridsampler3d_data_input1_2.txt",
                      "grid_sampler_3d/data/gridsampler3d_data_input2_2.txt",
                      "grid_sampler_3d/data/gridsampler3d_data_output1_2.txt"};
  RunGridSampler3DKernel1<float>(files, data_types, shapes, "bilinear", "zeros", true);
}

TEST_F(TEST_GRIDSAMPLER3D_UT, GRIDSAMPLER3D3_SUCC){
  vector<DataType>data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>>shapes = {{5, 3, 2, 4, 4}, {5, 1, 3, 2, 3}, {5, 3, 1, 3, 2}};
  vector<string>files{"grid_sampler_3d/data/gridsampler3d_data_input1_3.txt",
                      "grid_sampler_3d/data/gridsampler3d_data_input2_3.txt",
                      "grid_sampler_3d/data/gridsampler3d_data_output1_3.txt"};
  RunGridSampler3DKernel1<double>(files, data_types, shapes, "bilinear", "border", true);
}

TEST_F(TEST_GRIDSAMPLER3D_UT, GRIDSAMPLER3D4_SUCC){
  vector<DataType>data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>>shapes = {{32, 2, 4, 112, 112}, {32, 2, 36, 36, 3}, {32, 2, 2, 36, 36}};
  vector<string>files{"grid_sampler_3d/data/gridsampler3d_data_input1_4.txt",
                      "grid_sampler_3d/data/gridsampler3d_data_input2_4.txt",
                      "grid_sampler_3d/data/gridsampler3d_data_output1_4.txt"};
  RunGridSampler3DKernel1<float>(files, data_types, shapes, "bilinear", "border", true);
}

TEST_F(TEST_GRIDSAMPLER3D_UT, GRIDSAMPLER3D5_SUCC){
  vector<DataType>data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>>shapes = {{5, 3, 2, 4, 4}, {5, 1, 3, 2, 3}, {5, 3, 1, 3, 2}};
  vector<string>files{"grid_sampler_3d/data/gridsampler3d_data_input1_5.txt",
                      "grid_sampler_3d/data/gridsampler3d_data_input2_5.txt",
                      "grid_sampler_3d/data/gridsampler3d_data_output1_5.txt"};
  RunGridSampler3DKernel1<double>(files, data_types, shapes, "bilinear", "reflection", true);
}

TEST_F(TEST_GRIDSAMPLER3D_UT, GRIDSAMPLER3D6_SUCC){
  vector<DataType>data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>>shapes = {{5, 3, 2, 4, 4}, {5, 1, 3, 2, 3}, {5, 3, 1, 3, 2}};
  vector<string>files{"grid_sampler_3d/data/gridsampler3d_data_input1_6.txt",
                      "grid_sampler_3d/data/gridsampler3d_data_input2_6.txt",
                      "grid_sampler_3d/data/gridsampler3d_data_output1_6.txt"};
  RunGridSampler3DKernel1<float>(files, data_types, shapes, "bilinear", "reflection", true);
}

TEST_F(TEST_GRIDSAMPLER3D_UT, GRIDSAMPLER3D7_SUCC){
  vector<DataType>data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>>shapes = {{5, 3, 2, 4, 4}, {5, 1, 3, 2, 3}, {5, 3, 1, 3, 2}};
  vector<string>files{"grid_sampler_3d/data/gridsampler3d_data_input1_7.txt",
                      "grid_sampler_3d/data/gridsampler3d_data_input2_7.txt",
                      "grid_sampler_3d/data/gridsampler3d_data_output1_7.txt"};
  RunGridSampler3DKernel1<double>(files, data_types, shapes, "bilinear", "zeros", false);
}

TEST_F(TEST_GRIDSAMPLER3D_UT, GRIDSAMPLER3D8_SUCC){
  vector<DataType>data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>>shapes = {{5, 3, 2, 4, 4}, {5, 1, 3, 2, 3}, {5, 3, 1, 3, 2}};
  vector<string>files{"grid_sampler_3d/data/gridsampler3d_data_input1_8.txt",
                      "grid_sampler_3d/data/gridsampler3d_data_input2_8.txt",
                      "grid_sampler_3d/data/gridsampler3d_data_output1_8.txt"};
  RunGridSampler3DKernel1<float>(files, data_types, shapes, "bilinear", "zeros", false);
}

TEST_F(TEST_GRIDSAMPLER3D_UT, GRIDSAMPLER3D9_SUCC){
  vector<DataType>data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>>shapes = {{5, 3, 2, 4, 4}, {5, 1, 3, 2, 3}, {5, 3, 1, 3, 2}};
  vector<string>files{"grid_sampler_3d/data/gridsampler3d_data_input1_9.txt",
                      "grid_sampler_3d/data/gridsampler3d_data_input2_9.txt",
                      "grid_sampler_3d/data/gridsampler3d_data_output1_9.txt"};
  RunGridSampler3DKernel1<double>(files, data_types, shapes, "bilinear", "border", false);
}

TEST_F(TEST_GRIDSAMPLER3D_UT, GRIDSAMPLER3D10_SUCC){
  vector<DataType>data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>>shapes = {{5, 3, 2, 4, 4}, {5, 1, 3, 2, 3}, {5, 3, 1, 3, 2}};
  vector<string>files{"grid_sampler_3d/data/gridsampler3d_data_input1_10.txt",
                      "grid_sampler_3d/data/gridsampler3d_data_input2_10.txt",
                      "grid_sampler_3d/data/gridsampler3d_data_output1_10.txt"};
  RunGridSampler3DKernel1<float>(files, data_types, shapes, "bilinear", "border", false);
}

TEST_F(TEST_GRIDSAMPLER3D_UT, GRIDSAMPLER3D11_SUCC){
  vector<DataType>data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>>shapes = {{5, 3, 2, 4, 4}, {5, 1, 3, 2, 3}, {5, 3, 1, 3, 2}};
  vector<string>files{"grid_sampler_3d/data/gridsampler3d_data_input1_11.txt",
                      "grid_sampler_3d/data/gridsampler3d_data_input2_11.txt",
                      "grid_sampler_3d/data/gridsampler3d_data_output1_11.txt"};
  RunGridSampler3DKernel1<double>(files, data_types, shapes, "bilinear", "reflection", false);
}

TEST_F(TEST_GRIDSAMPLER3D_UT, GRIDSAMPLER3D12_SUCC){
  vector<DataType>data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>>shapes = {{5, 3, 2, 4, 4}, {5, 1, 3, 2, 3}, {5, 3, 1, 3, 2}};
  vector<string>files{"grid_sampler_3d/data/gridsampler3d_data_input1_12.txt",
                      "grid_sampler_3d/data/gridsampler3d_data_input2_12.txt",
                      "grid_sampler_3d/data/gridsampler3d_data_output1_12.txt"};
  RunGridSampler3DKernel1<float>(files, data_types, shapes, "bilinear", "reflection", false);
}

TEST_F(TEST_GRIDSAMPLER3D_UT, GRIDSAMPLER3D13_SUCC){
  vector<DataType>data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>>shapes = {{5, 3, 2, 4, 4}, {5, 1, 3, 2, 3}, {5, 3, 1, 3, 2}};
  vector<string>files{"grid_sampler_3d/data/gridsampler3d_data_input1_13.txt",
                      "grid_sampler_3d/data/gridsampler3d_data_input2_13.txt",
                      "grid_sampler_3d/data/gridsampler3d_data_output1_13.txt"};
  RunGridSampler3DKernel1<double>(files, data_types, shapes, "nearest", "zeros", true);
}

TEST_F(TEST_GRIDSAMPLER3D_UT, GRIDSAMPLER3D14_SUCC){
  vector<DataType>data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>>shapes = {{5, 7, 21, 4, 4}, {5, 2, 3, 2, 3}, {5, 7, 2, 3, 2}};
  vector<string>files{"grid_sampler_3d/data/gridsampler3d_data_input1_14.txt",
                      "grid_sampler_3d/data/gridsampler3d_data_input2_14.txt",
                      "grid_sampler_3d/data/gridsampler3d_data_output1_14.txt"};
  RunGridSampler3DKernel1<float>(files, data_types, shapes, "nearest", "zeros", true);
}

TEST_F(TEST_GRIDSAMPLER3D_UT, GRIDSAMPLER3D15_SUCC){
  vector<DataType>data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>>shapes = {{5, 1, 5, 4, 4}, {5, 1, 3, 2, 3}, {5, 1, 1, 3, 2}};
  vector<string>files{"grid_sampler_3d/data/gridsampler3d_data_input1_15.txt",
                      "grid_sampler_3d/data/gridsampler3d_data_input2_15.txt",
                      "grid_sampler_3d/data/gridsampler3d_data_output1_15.txt"};
  RunGridSampler3DKernel1<double>(files, data_types, shapes, "nearest", "border", true);
}

TEST_F(TEST_GRIDSAMPLER3D_UT, GRIDSAMPLER3D16_SUCC){
  vector<DataType>data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>>shapes = {{5, 2, 4, 4, 4}, {5, 1, 3, 2, 3}, {5, 2, 1, 3, 2}};
  vector<string>files{"grid_sampler_3d/data/gridsampler3d_data_input1_16.txt",
                      "grid_sampler_3d/data/gridsampler3d_data_input2_16.txt",
                      "grid_sampler_3d/data/gridsampler3d_data_output1_16.txt"};
  RunGridSampler3DKernel1<float>(files, data_types, shapes, "nearest", "border", true);
}

TEST_F(TEST_GRIDSAMPLER3D_UT, GRIDSAMPLER3D17_SUCC){
  vector<DataType>data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>>shapes = {{5, 2, 4, 11, 7}, {5, 1, 3, 2, 3}, {5, 2, 1, 3, 2}};
  vector<string>files{"grid_sampler_3d/data/gridsampler3d_data_input1_17.txt",
                      "grid_sampler_3d/data/gridsampler3d_data_input2_17.txt",
                      "grid_sampler_3d/data/gridsampler3d_data_output1_17.txt"};
  RunGridSampler3DKernel1<double>(files, data_types, shapes, "nearest", "reflection", true);
}

TEST_F(TEST_GRIDSAMPLER3D_UT, GRIDSAMPLER3D18_SUCC){
  vector<DataType>data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>>shapes = {{1, 3, 4, 4, 4}, {1, 1, 3, 2, 3}, {1, 3, 1, 3, 2}};
  vector<string>files{"grid_sampler_3d/data/gridsampler3d_data_input1_18.txt",
                      "grid_sampler_3d/data/gridsampler3d_data_input2_18.txt",
                      "grid_sampler_3d/data/gridsampler3d_data_output1_18.txt"};
  RunGridSampler3DKernel1<float>(files, data_types, shapes, "nearest", "reflection", true);
}

TEST_F(TEST_GRIDSAMPLER3D_UT, GRIDSAMPLER3D19_SUCC){
  vector<DataType>data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>>shapes = {{5, 2, 4, 4, 4}, {5, 1, 3, 2, 3}, {5, 2, 1, 3, 2}};
  vector<string>files{"grid_sampler_3d/data/gridsampler3d_data_input1_19.txt",
                      "grid_sampler_3d/data/gridsampler3d_data_input2_19.txt",
                      "grid_sampler_3d/data/gridsampler3d_data_output1_19.txt"};
  RunGridSampler3DKernel1<double>(files, data_types, shapes, "nearest", "zeros", false);
}

TEST_F(TEST_GRIDSAMPLER3D_UT, GRIDSAMPLER3D20_SUCC){
  vector<DataType>data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>>shapes = {{5, 2, 4, 4, 4}, {5, 1, 3, 2, 3}, {5, 2, 1, 3, 2}};
  vector<string>files{"grid_sampler_3d/data/gridsampler3d_data_input1_20.txt",
                      "grid_sampler_3d/data/gridsampler3d_data_input2_20.txt",
                      "grid_sampler_3d/data/gridsampler3d_data_output1_20.txt"};
  RunGridSampler3DKernel1<float>(files, data_types, shapes, "nearest", "zeros", false);
}

TEST_F(TEST_GRIDSAMPLER3D_UT, GRIDSAMPLER3D21_SUCC){
  vector<DataType>data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>>shapes = {{5, 2, 4, 4, 4}, {5, 1, 3, 2, 3}, {5, 2, 1, 3, 2}};
  vector<string>files{"grid_sampler_3d/data/gridsampler3d_data_input1_21.txt",
                      "grid_sampler_3d/data/gridsampler3d_data_input2_21.txt",
                      "grid_sampler_3d/data/gridsampler3d_data_output1_21.txt"};
  RunGridSampler3DKernel1<double>(files, data_types, shapes, "nearest", "border", false);
}

TEST_F(TEST_GRIDSAMPLER3D_UT, GRIDSAMPLER3D22_SUCC){
  vector<DataType>data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>>shapes = {{5, 2, 4, 4, 4}, {5, 1, 3, 2, 3}, {5, 2, 1, 3, 2}};
  vector<string>files{"grid_sampler_3d/data/gridsampler3d_data_input1_22.txt",
                      "grid_sampler_3d/data/gridsampler3d_data_input2_22.txt",
                      "grid_sampler_3d/data/gridsampler3d_data_output1_22.txt"};
  RunGridSampler3DKernel1<float>(files, data_types, shapes, "nearest", "border", false);
}

TEST_F(TEST_GRIDSAMPLER3D_UT, GRIDSAMPLER3D23_SUCC){
  vector<DataType>data_types = {DT_DOUBLE, DT_DOUBLE, DT_DOUBLE};
  vector<vector<int64_t>>shapes = {{32, 2, 4, 224, 224}, {32, 2, 36, 36, 3}, {32, 2, 2, 36, 36}};
  vector<string>files{"grid_sampler_3d/data/gridsampler3d_data_input1_23.txt",
                      "grid_sampler_3d/data/gridsampler3d_data_input2_23.txt",
                      "grid_sampler_3d/data/gridsampler3d_data_output1_23.txt"};
  RunGridSampler3DKernel1<double>(files, data_types, shapes, "nearest", "reflection", false);
}

TEST_F(TEST_GRIDSAMPLER3D_UT, GRIDSAMPLER3D24_SUCC){
  vector<DataType>data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>>shapes = {{5, 2, 4, 4, 4}, {5, 1, 3, 2, 3}, {5, 2, 1, 3, 2}};
  vector<string>files{"grid_sampler_3d/data/gridsampler3d_data_input1_24.txt",
                      "grid_sampler_3d/data/gridsampler3d_data_input2_24.txt",
                      "grid_sampler_3d/data/gridsampler3d_data_output1_24.txt"};
  RunGridSampler3DKernel1<float>(files, data_types, shapes, "nearest", "reflection", false);
}

TEST_F(TEST_GRIDSAMPLER3D_UT, GRIDSAMPLER3D25_SUCC){
  vector<DataType>data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>>shapes = {{5, 3, 2, 4, 4}, {5, 1, 3, 2, 3}, {5, 3, 1, 3, 2}};
  RunGridSampler3DKernel2(data_types, shapes, "nearest", "reflection", false);
}

TEST_F(TEST_GRIDSAMPLER3D_UT, GRIDSAMPLER3D26_SUCC){
  vector<DataType>data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>>shapes = {{5, 3, 2, 4, 4}, {5, 1, 3, 2, 3}, {5, 3, 1, 3, 2}};
  RunGridSampler3DKernel2(data_types, shapes, "nearest", "border", false);
}

TEST_F(TEST_GRIDSAMPLER3D_UT, GRIDSAMPLER3D27_SUCC){
  vector<DataType>data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>>shapes = {{5, 3, 2, 4, 4}, {5, 1, 3, 2, 3}, {5, 3, 1, 3, 2}};
  RunGridSampler3DKernel2(data_types, shapes, "nearest", "zeros", false);
}

TEST_F(TEST_GRIDSAMPLER3D_UT, GRIDSAMPLER3D28_SUCC){
  vector<DataType>data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>>shapes = {{32, 2, 4, 224, 224}, {32, 2, 36, 36, 3}, {32, 2, 2, 36, 36}};
  RunGridSampler3DKernel2(data_types, shapes, "nearest", "reflection", true);
}

TEST_F(TEST_GRIDSAMPLER3D_UT, GRIDSAMPLER3D29_SUCC){
  vector<DataType>data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>>shapes = {{5, 3, 2, 4, 4}, {5, 1, 3, 2, 3}, {5, 3, 1, 3, 2}};
  RunGridSampler3DKernel2(data_types, shapes, "nearest", "border", true);
}

TEST_F(TEST_GRIDSAMPLER3D_UT, GRIDSAMPLER3D30_SUCC){
  vector<DataType>data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>>shapes = {{5, 3, 2, 4, 4}, {5, 1, 3, 2, 3}, {5, 3, 1, 3, 2}};
  RunGridSampler3DKernel2(data_types, shapes, "nearest", "zeros", true);
}

TEST_F(TEST_GRIDSAMPLER3D_UT, GRIDSAMPLER3D31_SUCC){
  vector<DataType>data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>>shapes = {{5, 3, 2, 4, 4}, {5, 1, 3, 2, 3}, {5, 3, 1, 3, 2}};
  RunGridSampler3DKernel2(data_types, shapes, "bilinear", "reflection", false);
}

TEST_F(TEST_GRIDSAMPLER3D_UT, GRIDSAMPLER3D32_SUCC){
  vector<DataType>data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>>shapes = {{5, 3, 2, 4, 4}, {5, 1, 3, 2, 3}, {5, 3, 1, 3, 2}};
  RunGridSampler3DKernel2(data_types, shapes, "bilinear", "border", false);
}

TEST_F(TEST_GRIDSAMPLER3D_UT, GRIDSAMPLER3D33_SUCC){
  vector<DataType>data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>>shapes = {{5, 3, 2, 4, 4}, {5, 1, 3, 2, 3}, {5, 3, 1, 3, 2}};
  RunGridSampler3DKernel2(data_types, shapes, "bilinear", "zeros", false);
}

TEST_F(TEST_GRIDSAMPLER3D_UT, GRIDSAMPLER3D34_SUCC){
  vector<DataType>data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>>shapes = {{5, 3, 2, 4, 4}, {5, 1, 3, 2, 3}, {5, 3, 1, 3, 2}};
  RunGridSampler3DKernel2(data_types, shapes, "bilinear", "reflection", true);
}

TEST_F(TEST_GRIDSAMPLER3D_UT, GRIDSAMPLER3D35_SUCC){
  vector<DataType>data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>>shapes = {{5, 3, 2, 4, 4}, {5, 1, 3, 2, 3}, {5, 3, 1, 3, 2}};
  RunGridSampler3DKernel2(data_types, shapes, "bilinear", "border", true);
}

TEST_F(TEST_GRIDSAMPLER3D_UT, GRIDSAMPLER3D36_SUCC){
  vector<DataType>data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};
  vector<vector<int64_t>>shapes = {{32, 2, 4, 224, 224}, {32, 2, 36, 36, 3}, {32, 2, 2, 36, 36}};
  RunGridSampler3DKernel2(data_types, shapes, "bilinear", "zeros", true);
}

// exception instance
TEST_F(TEST_GRIDSAMPLER3D_UT, INPUT_NULL_EXCEPTION) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{5, 2, 4, 4, 4}, {5, 1, 3, 2, 3}, {5, 2, 1, 3, 2}};
  float output[60] = {(float)0};
  vector<void *> datas = {(void *)nullptr, (void *)nullptr, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas,  "bilinear", "zeros", true);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_GRIDSAMPLER3D_UT, INPUT_TYPE_EXCEPTION) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT16, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{5, 2, 4, 4, 4}, {5, 1, 3, 2, 3}, {5, 2, 1, 3, 2}};
  float input1[640] = {(float)0};
  Eigen::half input2[90] = {static_cast<Eigen::half>(0)};
  float output[60] = {(float)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas,  "bilinear", "border", false);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_GRIDSAMPLER3D_UT, INPUT_TYPE_UNSUPPORT) {
  vector<DataType> data_types = {DT_INT32, DT_INT32, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{5, 2, 4, 4, 4}, {5, 1, 3, 2, 3}, {5, 2, 1, 3, 2}};
  int32_t input1[640] = {0};
  int32_t input2[90] = {0};
  float output[60] = {(float)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas,  "nearest", "zeros", true);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_GRIDSAMPLER3D_UT, ATTR_ERROR_EXCEPTION) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{5, 2, 4, 4, 4}, {5, 1, 3, 2, 3}, {5, 2, 1, 3, 2}};
  float input1[640] = {(float)0};
  float input2[90] = {(float)0};
  float output[60] = {(float)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas,  "error", "border", false);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_GRIDSAMPLER3D_UT, SHAPE_ERROR_EXCEPTION) {
  vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
  vector<vector<int64_t>> shapes = {{5, 2, 4, 4, 4}, {1, 1, 3, 2, 4}, {5, 2, 1, 3, 2}};
  float input1[640] = {(float)0};
  float input2[24] = {(float)0};
  float output[60] = {(float)0};
  vector<void *> datas = {(void *)input1, (void *)input2, (void *)output};
  CREATE_NODEDEF(shapes, data_types, datas,  "error", "border", false);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}