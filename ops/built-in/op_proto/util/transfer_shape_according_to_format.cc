/**
 * @file transfer_shape_according_to_format.cpp
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief set shape according to original format and current format
 *
 * @version 1.0
 *
 */
#include "transfer_shape_according_to_format.h"
#include "framework/omg/omg_inner_types.h"
namespace ge {
ShapeTransferAccordingToFormat::ShapeTransferAccordingToFormat(void) {
  getNewShapeFuncMap = {
      {ge::FORMAT_NCHW, std::make_shared<GetNewShapeByAxisValueAndFormat>(
          GetNCHWShapeByAxisValue)},
      {ge::FORMAT_NHWC, std::make_shared<GetNewShapeByAxisValueAndFormat>(
          GetNHWCShapeByAxisValue)},
      {ge::FORMAT_NC1HWC0, std::make_shared<GetNewShapeByAxisValueAndFormat>(
          GetNC1HWC0ShapeByAxisValue)},
      {ge::FORMAT_FRACTAL_Z,
       std::make_shared<GetNewShapeByAxisValueAndFormat>(
           GetFzShapeByAxisValue)},
      {ge::FORMAT_HWCN,
       std::make_shared<GetNewShapeByAxisValueAndFormat>(
           GetHWCNShapeByAxisValue)},
      {ge::FORMAT_C1HWNCoC0,
       std::make_shared<GetNewShapeByAxisValueAndFormat>(
           GetC1HWNCoC0ShapeByAxisValue)},
      {ge::FORMAT_FRACTAL_NZ,
       std::make_shared<GetNewShapeByAxisValueAndFormat>(
          GetNzShapeByAxisValue)}};

  mapOfDtypeAndC0 = {
      {ge::DT_FLOAT16, SHAPE_NUMBER_16},
      {ge::DT_FLOAT, SHAPE_NUMBER_16},
      {ge::DT_INT8, SHAPE_NUMBER_32},
      {ge::DT_INT16, SHAPE_NUMBER_16},
      {ge::DT_INT32, SHAPE_NUMBER_16},
      {ge::DT_INT64, SHAPE_NUMBER_16},
      {ge::DT_UINT8, SHAPE_NUMBER_16},
      {ge::DT_UINT16, SHAPE_NUMBER_32},
      {ge::DT_UINT32, SHAPE_NUMBER_16},
      {ge::DT_UINT64, SHAPE_NUMBER_16},
      {ge::DT_BOOL, SHAPE_NUMBER_16}
  };
}

bool ShapeTransferAccordingToFormat::GetNCHWShapeByAxisValue(
    ge::GeShape& newShape,
    const int64_t& implType,
    const vector<int64_t>& axisValue,
    const vector<int64_t>& ndValue) {
  CHECK(axisValue.empty(), LOG_INFO("AxisValue is empty!"), return true);
  /* axisValue is initialized as a size 6 vector. */
  std::vector<int64_t> newDimVec;
  newDimVec.push_back(axisValue[AXIS_N]);
  newDimVec.push_back(axisValue[AXIS_C]);
  newDimVec.push_back(axisValue[AXIS_H]);
  newDimVec.push_back(axisValue[AXIS_W]);
  newShape = ge::GeShape(newDimVec);
  return true;
}

bool ShapeTransferAccordingToFormat::GetNHWCShapeByAxisValue(
    ge::GeShape &newShape,
    const int64_t &implType,
    const vector<int64_t>& axisValue,
    const vector<int64_t>& ndValue) {
  CHECK(axisValue.empty(), LOG_INFO("AxisValue is empty!"), return true);
  /* axisValue is initialized as a size 6 vector. */
  std::vector<int64_t> newDimVec;
  newDimVec.push_back(axisValue[AXIS_N]);
  newDimVec.push_back(axisValue[AXIS_H]);
  newDimVec.push_back(axisValue[AXIS_W]);
  newDimVec.push_back(axisValue[AXIS_C]);
  newShape = ge::GeShape(newDimVec);
  return true;
}

bool ShapeTransferAccordingToFormat::GetNC1HWC0ShapeByAxisValue(
    ge::GeShape &newShape,
    const int64_t &implType,
    const vector<int64_t>& axisValue,
    const vector<int64_t>& ndValue) {
  CHECK(axisValue.empty(), LOG_INFO("AxisValue is empty!"), return true);
  /* axisValue is initialized as a size 6 vector. */
  std::vector<int64_t> newDimVec;
  if (implType == EN_IMPL_HW_TBE || implType == EN_IMPL_CUSTOM_TBE ||
      implType == EN_IMPL_NON_PERSISTENT_CUSTOM_TBE) {
    newDimVec.push_back(axisValue[AXIS_N]);
    newDimVec.push_back(axisValue[AXIS_C1]);
    newDimVec.push_back(axisValue[AXIS_H]);
    newDimVec.push_back(axisValue[AXIS_W]);
    newDimVec.push_back(axisValue[AXIS_C0]);
    newShape = ge::GeShape(newDimVec);
  } else {
    newDimVec.push_back(axisValue[AXIS_N]);
    newDimVec.push_back(axisValue[AXIS_C]);
    newDimVec.push_back(axisValue[AXIS_H]);
    newDimVec.push_back(axisValue[AXIS_W]);
    newShape = ge::GeShape(newDimVec);
  }
  return true;
}

bool ShapeTransferAccordingToFormat::GetFzShapeByAxisValue(
    ge::GeShape &newShape,
    const int64_t &implType,
    const vector<int64_t>& axisValue,
    const vector<int64_t>& ndValue) {
  CHECK(axisValue.empty(), LOG_INFO("AxisValue is empty!"), return true);
  /* axisValue is initialized as a size 6 vector. */
  std::vector<int64_t> newDimVec;

  if (ndValue.size() == SIZE_OF_CN) {
    auto sizeOfOriginalVec = ndValue.size();
    std::vector<int64_t> newDimVec = ndValue;
    /* sizeOfOriginalVec - 1 mean the last value of original vec
     * sizeOfOriginalVec - 2 mean the second last value of original vec */
    newDimVec[sizeOfOriginalVec - MINUS_VALUE_ONE] = DivisionCeiling(
        ndValue[sizeOfOriginalVec - MINUS_VALUE_ONE], SHAPE_NUMBER_16);
    newDimVec[sizeOfOriginalVec - MINUS_VALUE_TWO] = DivisionCeiling(
        ndValue[sizeOfOriginalVec - MINUS_VALUE_TWO], axisValue[AXIS_C0]);
    newDimVec.push_back(SHAPE_NUMBER_16);
    newDimVec.push_back(axisValue[AXIS_C0]);
    newShape = ge::GeShape(newDimVec);
  } else {
    if (implType == EN_IMPL_HW_TBE || implType == EN_IMPL_CUSTOM_TBE ||
        implType == EN_IMPL_NON_PERSISTENT_CUSTOM_TBE) {
      int64_t hwc1 = axisValue[AXIS_C1] * axisValue[AXIS_H] * axisValue[AXIS_W];
      newDimVec.push_back(hwc1);
      newDimVec.push_back(DivisionCeiling(axisValue[AXIS_N], NI));
      newDimVec.push_back(NI);
      newDimVec.push_back(axisValue[AXIS_C0]);
      newShape = ge::GeShape(newDimVec);
    } else {
      newDimVec.push_back(axisValue[AXIS_N]);
      newDimVec.push_back(axisValue[AXIS_C]);
      newDimVec.push_back(axisValue[AXIS_H]);
      newDimVec.push_back(axisValue[AXIS_W]);
      newShape = ge::GeShape(newDimVec);
    }
  }

  return true;
}

bool ShapeTransferAccordingToFormat::GetHWCNShapeByAxisValue(
    ge::GeShape &newShape,
    const int64_t &implType,
    const vector<int64_t>& axisValue,
    const vector<int64_t>& ndValue) {
  CHECK(axisValue.empty(), LOG_INFO("AxisValue is empty!"), return true);
  /* axisValue is initialized as a size 6 vector. */
  std::vector<int64_t> newDimVec;
  newDimVec.push_back(axisValue[AXIS_H]);
  newDimVec.push_back(axisValue[AXIS_W]);
  newDimVec.push_back(axisValue[AXIS_C]);
  newDimVec.push_back(axisValue[AXIS_N]);
  newShape = ge::GeShape(newDimVec);
  return true;
}

bool ShapeTransferAccordingToFormat::GetC1HWNCoC0ShapeByAxisValue(
    ge::GeShape &newShape,
    const int64_t &implType,
    const vector<int64_t>& axisValue,
    const vector<int64_t>& ndValue) {
  CHECK(axisValue.empty(), LOG_INFO("AxisValue is empty!"), return true);
  /* axisValue is initialized as a size 6 vector. */
  std::vector<int64_t> newDimVec;
  newDimVec.push_back(axisValue[AXIS_C1]);
  newDimVec.push_back(axisValue[AXIS_H]);
  newDimVec.push_back(axisValue[AXIS_W]);
  newDimVec.push_back(axisValue[AXIS_N]);
  newDimVec.push_back(axisValue[AXIS_Co]);
  newDimVec.push_back(axisValue[AXIS_C0]);
  newShape = ge::GeShape(newDimVec);
  return true;
}

bool ShapeTransferAccordingToFormat::GetNzShapeByAxisValue(
    ge::GeShape &newShape,
    const int64_t &implType,
    const vector<int64_t>& axisValue,
    const vector<int64_t>& ndValue) {
  CHECK(ndValue.empty(), LOG_INFO("ndValue is empty!"), return true);
  CHECK(axisValue.empty() || axisValue.size() <= AXIS_C0,
        LOG_INFO("AxisValue is empty or its size %zu <= AXIS_C0[%u]",
                 axisValue.size(), AXIS_C0),
        return true);
  uint32_t sizeOfOriginalVec = ndValue.size();\
  if (sizeOfOriginalVec < MINIMUM_NZ_SHAPE_DIM_NUM) {
    LOG_INFO("ndValue's dim num is less than 2!");
    return true;
  }
  /* axisValue is initialized as a size 6 vector. */
  std::vector<int64_t> newDimVec = ndValue;

  /* sizeOfOriginalVec - 1 mean the last value of original vec
   * sizeOfOriginalVec - 2 mean the second last value of original vec */
  newDimVec[sizeOfOriginalVec - MINUS_VALUE_ONE] = DivisionCeiling(
      ndValue[sizeOfOriginalVec - MINUS_VALUE_TWO], (int64_t)SHAPE_NUMBER_16);

  newDimVec[sizeOfOriginalVec - MINUS_VALUE_TWO] = DivisionCeiling(
      ndValue[sizeOfOriginalVec - MINUS_VALUE_ONE], axisValue[AXIS_C0]);
  newDimVec.push_back(SHAPE_NUMBER_16);
  newDimVec.push_back(axisValue[AXIS_C0]);
  newShape = ge::GeShape(newDimVec);
  return true;
}

bool ShapeTransferAccordingToFormat::GetShapeAccordingToFormat(
    ShapeAndFormat& shapeAndFormatInfo, int64_t* c) {
  /* The default new shape is old shape */
  shapeAndFormatInfo.newShape = shapeAndFormatInfo.oldShape;
  if (shapeAndFormatInfo.oldFormat >= ge::FORMAT_RESERVED ||
      shapeAndFormatInfo.newFormat >= ge::FORMAT_RESERVED) {
    LOG_ERROR("Old format %u or new format %u is invalid!",
            shapeAndFormatInfo.oldFormat, shapeAndFormatInfo.newFormat);
    return false;
  }

  if (shapeAndFormatInfo.currentDataType >= ge::DT_UNDEFINED) {
    LOG_ERROR("currentDataType %u is invalid!",
            shapeAndFormatInfo.currentDataType);
    return false;
  }
  AxisUtil *axisutil_object = new AxisUtil();
  if (!axisutil_object->HasAxisValueFunc(shapeAndFormatInfo.oldFormat)) {
    delete axisutil_object;
    return true;
  }

  auto iterGetNewShapeFunc =
      getNewShapeFuncMap.find(shapeAndFormatInfo.newFormat);
  if (iterGetNewShapeFunc == getNewShapeFuncMap.end()) {
    LOG_INFO("Can not get new shape of new format %u!",
            shapeAndFormatInfo.newFormat);
    delete axisutil_object;
    return true;
  }
  LOG_INFO("Original format %u, new format %u",
          shapeAndFormatInfo.oldFormat,
          shapeAndFormatInfo.newFormat);
  GetNewShapeByAxisValueAndFormatPtr getNewShapeFunc =
      iterGetNewShapeFunc->second;
  CHECK_NOTNULL(getNewShapeFunc);
  std::vector<int64_t> axisValue;
  for (uint32_t i = 0; i < AXIS_BOTTOM; i++) {
    axisValue.push_back(1);
  }
  std::vector<int64_t> ndValue;
  uint32_t c0;
  if (mapOfDtypeAndC0.empty()) {
    c0 = SHAPE_NUMBER_16;
  } else {
    auto iterGetC0 = mapOfDtypeAndC0.find(shapeAndFormatInfo.currentDataType);
    if (iterGetC0 == mapOfDtypeAndC0.end()) {
      LOG_ERROR("Dtype is not support.");
      delete axisutil_object;
      return true;
    }
    c0 = iterGetC0->second;
  }

  // The value of C0 should be 4 while format is 5HD-4 or FRAZ-4
  if (shapeAndFormatInfo.newFormat == ge::FORMAT_NC1HWC0_C04) {
    c0 = SHAPE_DIM_VALUE_C04;
  }

  bool status = axisutil_object->GetAxisValueByOriginFormat(
      shapeAndFormatInfo.oldFormat,
      shapeAndFormatInfo.oldShape.GetDims(), c0, axisValue,
      ndValue);
  if (status != true &&
      shapeAndFormatInfo.newFormat != ge::FORMAT_FRACTAL_NZ) {
    delete axisutil_object;
    return true;
  }
  delete axisutil_object;

  (*getNewShapeFunc)(shapeAndFormatInfo.newShape,
                     shapeAndFormatInfo.opImplType,
                     axisValue, ndValue);
  if (c != nullptr) {
    *c = axisValue[AXIS_C];
  }
  return true;
}
};  // namespace ge