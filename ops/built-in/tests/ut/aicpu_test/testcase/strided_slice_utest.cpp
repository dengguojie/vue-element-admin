#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include "aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#undef private
#undef protected
#include "Eigen/Core"

using namespace std;
using namespace aicpu;

class STRIDED_SLICE_UT : public testing::Test {};

#define CREATE_NODEDEF(dtype1, dtype2)                              \
  auto node_def = CpuKernelUtils::CreateNodeDef();                  \
  NodeDefBuilder(node_def.get(), "StridedSlice", "StridedSlice")    \
      .Input({"x", dtype1, { 3, 2, 3 }, x.data()})                  \
      .Input({"begin", dtype2, {3}, begin.data()})                  \
      .Input({"end", dtype2, {3}, end.data()})                      \
      .Input({"strides", dtype2, {3}, strides.data()})              \
      .Output({"y", dtype1, { 1, 2, 3 }, y.data()});

#define ADD_CASE(dtype1, T1, dtype2, T2)                            \
  TEST_F(STRIDED_SLICE_UT, Test##dtype1##dtype2)                    \
  {                                                                 \
    vector<T1> x{ 1, 1, 1, 2, 2, 2, 3, 3, 3,                        \
                  4, 4, 4, 5, 5, 5, 6, 6, 6 };                      \
    vector<T2> begin{ 1, 0, 0 };                                    \
    vector<T2> end{ 2, 2, 3 };                                      \
    vector<T2> strides{ 1, 1, 1 };                                  \
    vector<T1> y(6);                                                \
    CREATE_NODEDEF(dtype1, dtype2)                                  \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK)                    \
    vector<T1> expectY{ 3, 3, 3, 4, 4, 4 };                         \
    EXPECT_EQ(y, expectY);                                          \
  }

TEST_F(STRIDED_SLICE_UT, ExpMasks)
{
  auto node_def = CpuKernelUtils::CreateNodeDef();

  vector<int32_t> x{ 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6 };
  vector<int32_t> begin{ 1, 0, 0 };
  vector<int32_t> end{ 2, 2, 3 };
  vector<int32_t> strides{ 1, 1, 1 };
  vector<int32_t> y(6);

  NodeDefBuilder(node_def.get(), "StridedSlice", "StridedSlice")
      .Input({"x", DT_INT32, { 3, 2, 3 }, x.data()})
      .Input({"begin", DT_INT32, {3}, begin.data()})
      .Input({"end", DT_INT32, {3}, end.data()})
      .Input({"strides", DT_INT32, {3}, strides.data()})
      .Output({"y", DT_INT32, { 1, 2, 3 }, y.data()})
      .Attr("begin_mask", 0)
      .Attr("end_mask", 0)
      .Attr("ellipsis_mask", 0)
      .Attr("new_axis_mask", 0)
      .Attr("shrink_axis_mask", 0);

  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  vector<int32_t> expectY{ 3, 3, 3, 4, 4, 4 };
  EXPECT_EQ(y, expectY);
}

TEST_F(STRIDED_SLICE_UT, ExpNegative1)
{
  auto node_def = CpuKernelUtils::CreateNodeDef();

  vector<int32_t> x{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                     16, 17, 18 };
  vector<int32_t> begin{ 1, -1, 0 };
  vector<int32_t> end{ 2, -3, 3 };
  vector<int32_t> strides{ 1, -1, 2 };
  vector<int32_t> y(4);

  NodeDefBuilder(node_def.get(), "StridedSlice", "StridedSlice")
      .Input({"x", DT_INT32, { 3, 2, 3 }, x.data()})
      .Input({"begin", DT_INT32, {3}, begin.data()})
      .Input({"end", DT_INT32, {3}, end.data()})
      .Input({"strides", DT_INT32, {3}, strides.data()})
      .Output({"y", DT_INT32, { 1, 2, 2 }, y.data()});

  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  vector<int32_t> expectY{ 10, 12, 7, 9 };
  EXPECT_EQ(y, expectY);
}

TEST_F(STRIDED_SLICE_UT, ExpNegative2)
{
  auto node_def = CpuKernelUtils::CreateNodeDef();

  vector<int32_t> x{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                     16, 17, 18 };
  vector<int32_t> begin{ 1, 0, -1 };
  vector<int32_t> end{ 2, 2, -4 };
  vector<int32_t> strides{ 1, 1, -1 };
  vector<int32_t> y(6);

  NodeDefBuilder(node_def.get(), "StridedSlice", "StridedSlice")
      .Input({"x", DT_INT32, { 3, 2, 3 }, x.data()})
      .Input({"begin", DT_INT32, {3}, begin.data()})
      .Input({"end", DT_INT32, {3}, end.data()})
      .Input({"strides", DT_INT32, {3}, strides.data()})
      .Output({"y", DT_INT32, { 1, 2, 3 }, y.data()});

  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  vector<int32_t> expectY{ 9, 8, 7, 12, 11, 10 };
  EXPECT_EQ(y, expectY);
}

TEST_F(STRIDED_SLICE_UT, ExpNegative3)
{
  auto node_def = CpuKernelUtils::CreateNodeDef();

  vector<int32_t> x{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                     16, 17, 18 };
  vector<int32_t> begin{ 1, 0, -3 };
  vector<int32_t> end{ 2, 2, -1 };
  vector<int32_t> strides{ 1, 1, 1 };
  vector<int32_t> y(4);

  NodeDefBuilder(node_def.get(), "StridedSlice", "StridedSlice")
      .Input({"x", DT_INT32, { 3, 2, 3 }, x.data()})
      .Input({"begin", DT_INT32, {3}, begin.data()})
      .Input({"end", DT_INT32, {3}, end.data()})
      .Input({"strides", DT_INT32, {3}, strides.data()})
      .Output({"y", DT_INT32, { 1, 2, 2 }, y.data()});

  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  vector<int32_t> expectY{ 7, 8, 10, 11 };
  EXPECT_EQ(y, expectY);
}

TEST_F(STRIDED_SLICE_UT, ExpBool)
{
  auto node_def = CpuKernelUtils::CreateNodeDef();

  array<bool, 6> x{ true, false, false, true, true, true };
  vector<int32_t> begin{ 1, 0 };
  vector<int32_t> end{ 2, 2 };
  vector<int32_t> strides{ 1, 1 };
  array<bool, 2> y;

  NodeDefBuilder(node_def.get(), "StridedSlice", "StridedSlice")
      .Input({"x", DT_BOOL, { 3, 2 }, x.data()})
      .Input({"begin", DT_INT32, {2}, begin.data()})
      .Input({"end", DT_INT32, {2}, end.data()})
      .Input({"strides", DT_INT32, {2}, strides.data()})
      .Output({"y", DT_BOOL, { 1, 2 }, y.data()});

  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  array<bool, 2> expectY{ false, true };
  EXPECT_EQ(y, expectY);
}

TEST_F(STRIDED_SLICE_UT, ExpHalf)
{
  auto node_def = CpuKernelUtils::CreateNodeDef();

  Eigen::half x[] = { Eigen::half(1), Eigen::half(1), Eigen::half(1),
                      Eigen::half(2), Eigen::half(2), Eigen::half(2),
                      Eigen::half(3), Eigen::half(3), Eigen::half(3),
                      Eigen::half(4), Eigen::half(4), Eigen::half(4),
                      Eigen::half(5), Eigen::half(5), Eigen::half(5),
                      Eigen::half(6), Eigen::half(6), Eigen::half(6) };
  vector<int32_t> begin{ 1, 0, 0 };
  vector<int32_t> end{ 2, 2, 3 };
  vector<int32_t> strides{ 1, 1, 1 };
  Eigen::half y[6];

  NodeDefBuilder(node_def.get(), "StridedSlice", "StridedSlice")
      .Input({"x", DT_FLOAT16, { 3, 2, 3 }, x})
      .Input({"begin", DT_INT32, {3}, begin.data()})
      .Input({"end", DT_INT32, {3}, end.data()})
      .Input({"strides", DT_INT32, {3}, strides.data()})
      .Output({"y", DT_FLOAT16, { 1, 2, 3 }, y});

  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  Eigen::half expectY[] = { Eigen::half(3), Eigen::half(3), Eigen::half(3),
                            Eigen::half(4), Eigen::half(4), Eigen::half(4) };
  EXPECT_EQ(y[0], expectY[0]);
  EXPECT_EQ(y[1], expectY[1]);
  EXPECT_EQ(y[2], expectY[2]);
  EXPECT_EQ(y[3], expectY[3]);
  EXPECT_EQ(y[4], expectY[4]);
  EXPECT_EQ(y[5], expectY[5]);
}

ADD_CASE(DT_INT8, int8_t, DT_INT32, int32_t)
ADD_CASE(DT_INT8, int8_t, DT_INT64, int64_t)
ADD_CASE(DT_INT16, int16_t, DT_INT32, int32_t)
ADD_CASE(DT_INT16, int16_t, DT_INT64, int64_t)
ADD_CASE(DT_INT32, int32_t, DT_INT32, int32_t)
ADD_CASE(DT_INT32, int32_t, DT_INT64, int64_t)
ADD_CASE(DT_INT64, int64_t, DT_INT32, int32_t)
ADD_CASE(DT_INT64, int64_t, DT_INT64, int64_t)
ADD_CASE(DT_UINT8, uint8_t, DT_INT32, int32_t)
ADD_CASE(DT_UINT8, uint8_t, DT_INT64, int64_t)
ADD_CASE(DT_UINT16, uint16_t, DT_INT32, int32_t)
ADD_CASE(DT_UINT16, uint16_t, DT_INT64, int64_t)
ADD_CASE(DT_UINT32, uint32_t, DT_INT32, int32_t)
ADD_CASE(DT_UINT32, uint32_t, DT_INT64, int64_t)
ADD_CASE(DT_UINT64, uint64_t, DT_INT32, int32_t)
ADD_CASE(DT_UINT64, uint64_t, DT_INT64, int64_t)
ADD_CASE(DT_FLOAT, float, DT_INT32, int32_t)
ADD_CASE(DT_FLOAT, float, DT_INT64, int64_t)
ADD_CASE(DT_DOUBLE, double, DT_INT32, int32_t)
ADD_CASE(DT_DOUBLE, double, DT_INT64, int64_t)
