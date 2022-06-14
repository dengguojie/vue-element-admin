#include "gtest/gtest.h"
#include "fp16_t.hpp"

namespace fe {

class fp16_t_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "fp16_t_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "fp16_t_test TearDown" << std::endl;
  }
};

TEST_F(fp16_t_test, fp16_t_test_1) {
  fp16_t val = 8.1;
  fp16_t val2 = 9.5;
  fp16_t midVal = 35500;
  fp16_t largeVal = 65500.0;
  double doubleVal = val.toDouble();
  int8_t int8Val = val.toInt8();
  int8_t midInt8Val = midVal.toInt8();
  int8_t largeInt8Val = largeVal.toInt8();
  uint8_t uint8Val = val.toUInt8();
  int16_t int16Val = val.toInt16();
  int16_t midInt16Val = midVal.toInt16();
  int16_t largeInt16Val = largeVal.toInt16();
  uint16_t uint16Val = val.toUInt16();
  int32_t int32Val = val.toInt32();
  uint32_t uint32Val = val.toUInt32();
  uint32_t largeUint32Val = largeVal.toUInt32();
  fp16_t addVal = val + val;
  fp16_t mulVal = val * val;
  fp16_t divVal = val / val;
  divVal = val / val2;
  divVal = val2 / val;
  fp16_t int8ToFp16 = int8Val;
  fp16_t uin8ToFp16 = uint8Val;
  fp16_t int16ToFp16 = int16Val;
  fp16_t largeInt16ToFp16 = largeInt16Val;
  fp16_t uint16ToFp16 = uint16Val;
  fp16_t int32ToFp16 = int32Val;
  fp16_t uint32ToFp16 = uint32Val;
  fp16_t largeUint32ToFp16 = largeUint32Val;
  fp16_t resVal = val;
  resVal = largeVal;

  EXPECT_EQ(uint32Val, 8);
}
} // namespace fe
