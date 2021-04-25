/**
* @file op_runner.cpp
*
* Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#include "op_runner.h"
#include <limits>
#include "common.h"

using namespace std;

extern bool g_isDevice;
const size_t DEFAULT_WHITE_COUNT = 10;
const size_t DEFAULT_PRECISION_COUNT = 4;

OpRunner::OpRunner(OpTestDesc *opDesc) : opDesc_(opDesc)
{
    numInputs_ = opDesc->inputDesc.size();
    numOutputs_ = opDesc->outputDesc.size();
}

OpRunner::~OpRunner()
{
    for (size_t i = 0; i < inputBuffers_.size(); ++i) {
        (void)aclDestroyDataBuffer(inputBuffers_[i]);
    }
    for (size_t i = 0; i < devInputs_.size(); ++i) {
        if (devInputs_[i] != nullptr) {
            (void)aclrtFree(devInputs_[i]);
        }
        if (g_isDevice) {
            if (hostInputs_[i] != nullptr) {
                (void)aclrtFree(hostInputs_[i]);
            }
        } else {
            if (hostInputs_[i] != nullptr) {
                (void)aclrtFreeHost(hostInputs_[i]);
            }
        }
    }

    for (size_t i = 0; i < outputBuffers_.size(); ++i) {
        (void)aclDestroyDataBuffer(outputBuffers_[i]);
        (void)aclrtFree(devOutputs_[i]);
        if (g_isDevice) {
            (void)aclrtFree(hostOutputs_[i]);
        } else {
            (void)aclrtFreeHost(hostOutputs_[i]);
        }
    }
}

bool OpRunner::Init()
{
    bool InputInitResult = InputsInit();
    bool OutputInitResult = OutputsInit();
    if (InputInitResult && OutputInitResult) {
        return true;
    } else {
        return false;
    }
}

bool OpRunner::InputsInit()
{
    for (size_t i = 0; i < numInputs_; ++i) {
        auto size = GetInputSize(i);
        // if size is zero, input is an optional
        void *devMem = nullptr;
        if (size != 0) {
            if (aclrtMalloc(&devMem, size, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_ERROR_NONE) {
                ERROR_LOG("Malloc device memory for input[%zu] failed", i);
                return false;
            }
            devInputs_.emplace_back(devMem);
            void *hostMem = nullptr;
            if (g_isDevice) {
                if (aclrtMalloc(&hostMem, size, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_ERROR_NONE) {
                    ERROR_LOG("Malloc device memory for input[%zu] failed", i);
                    return false;
                }
            } else {
                if (aclrtMallocHost(&hostMem, size) != ACL_ERROR_NONE) {
                    ERROR_LOG("Malloc device memory for input[%zu] failed", i);
                    return false;
                }
            }
            if (hostMem == nullptr) {
                ERROR_LOG("Malloc memory for input[%zu] failed", i);
                return false;
            }
            hostInputs_.emplace_back(hostMem);
        }
        inputBuffers_.emplace_back(aclCreateDataBuffer(devMem, size));
    }
    return true;
}

bool OpRunner::OutputsInit()
{
    for (size_t i = 0; i < numOutputs_; ++i) {
        auto size = GetOutputSize(i);
        void *devMem = nullptr;
        if (aclrtMalloc(&devMem, size, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_ERROR_NONE) {
            ERROR_LOG("Malloc device memory for output[%zu] failed", i);
            return false;
        }
        devOutputs_.emplace_back(devMem);
        outputBuffers_.emplace_back(aclCreateDataBuffer(devMem, size));

        void *hostOutput = nullptr;
        if (g_isDevice) {
            if (aclrtMalloc(&hostOutput, size, ACL_MEM_MALLOC_NORMAL_ONLY)!= ACL_ERROR_NONE) {
                ERROR_LOG("Malloc device memory for output[%zu] failed", i);
                return false;
            }
        } else {
            if (aclrtMallocHost(&hostOutput, size) != ACL_ERROR_NONE) {
                ERROR_LOG("Malloc device memory for output[%zu] failed", i);
                return false;
            }
        }
        if (hostOutput == nullptr) {
            ERROR_LOG("Malloc host memory for output[%zu] failed", i);
            return false;
        }
        hostOutputs_.emplace_back(hostOutput);
    }
    return true;
}

const size_t OpRunner::NumInputs()
{
    return numInputs_;
}

const size_t OpRunner::NumOutputs()
{
    return numOutputs_;
}

const size_t OpRunner::GetInputSize(size_t index) const
{
    if (index >= opDesc_->inputDesc.size()) {
        ERROR_LOG("index out of range. index = %zu, numInputs = %zu", index, numInputs_);
        return 0;
    }

    return aclGetTensorDescSize(opDesc_->inputDesc[index]);
}

std::vector<int64_t> OpRunner::GetInputShape(size_t index) const
{
    std::vector<int64_t> ret;
    if (index >= opDesc_->inputDesc.size()) {
        ERROR_LOG("index out of range. index = %zu, numInputs = %zu", index, numInputs_);
        return ret;
    }

    auto desc = opDesc_->inputDesc[index];
    for (size_t i = 0; i < aclGetTensorDescNumDims(desc); ++i) {
        int64_t dimSize;
        if (aclGetTensorDescDimV2(desc, i, &dimSize) != ACL_SUCCESS) {
            ERROR_LOG("get dimension size from tensor desc failed, dimension index = %zu", i);
            ret.clear();
            return ret;
        }
        ret.emplace_back(dimSize);
    }

    return ret;
}

std::vector<int64_t> OpRunner::GetOutputShape(size_t index) const
{
    std::vector<int64_t> ret;
    if (index >= opDesc_->outputDesc.size()) {
        ERROR_LOG("index out of range. index = %zu, numOutputs = %zu", index, numOutputs_);
        return ret;
    }

    auto desc = opDesc_->outputDesc[index];
    for (size_t i = 0; i < aclGetTensorDescNumDims(desc); ++i) {
        int64_t dimSize;
        if (aclGetTensorDescDimV2(desc, i, &dimSize) != ACL_SUCCESS) {
            ERROR_LOG("get dimension size from tensor desc failed, dimension index = %zu", i);
            ret.clear();
            return ret;
        }
        ret.emplace_back(dimSize);
    }
    return ret;
}

size_t OpRunner::GetInputElementCount(size_t index) const
{
    if (index >= opDesc_->inputDesc.size()) {
        ERROR_LOG("index out of range. index = %zu, numInputs = %zu", index, numInputs_);
        return 0;
    }

    return aclGetTensorDescElementCount(opDesc_->inputDesc[index]);
}

size_t OpRunner::GetOutputElementCount(size_t index) const
{
    if (index >= opDesc_->outputDesc.size()) {
        ERROR_LOG("index out of range. index = %zu, numOutputs = %zu", index, numOutputs_);
        return 0;
    }

    return aclGetTensorDescElementCount(opDesc_->outputDesc[index]);
}

size_t OpRunner::GetOutputSize(size_t index) const
{
    if (index >= opDesc_->outputDesc.size()) {
        ERROR_LOG("index out of range. index = %zu, numOutputs = %zu", index, numOutputs_);
        return 0;
    }

    return aclGetTensorDescSize(opDesc_->outputDesc[index]);
}

bool OpRunner::RunOp()
{
    for (size_t i = 0; i < devInputs_.size(); ++i) {
        auto size = GetInputSize(i);
        aclrtMemcpyKind kind = ACL_MEMCPY_HOST_TO_DEVICE;
        if (g_isDevice) {
            kind = ACL_MEMCPY_DEVICE_TO_DEVICE;
        }
        if (aclrtMemcpy(devInputs_[i], size, hostInputs_[i], size, kind) != ACL_ERROR_NONE) {
            ERROR_LOG("Copy input[%zu] failed", i);
            return false;
        }
        INFO_LOG("Copy input[%zu] success", i);
    }

    aclrtStream stream = nullptr;
    if (aclrtCreateStream(&stream) != ACL_ERROR_NONE) {
        ERROR_LOG("Create stream failed");
        return false;
    }
    INFO_LOG("Create stream success");

    auto ret = aclopExecuteV2(opDesc_->opType.c_str(),
                              numInputs_,
                              opDesc_->inputDesc.data(),
                              inputBuffers_.data(),
                              numOutputs_,
                              opDesc_->outputDesc.data(),
                              outputBuffers_.data(),
                              opDesc_->opAttr,
                              stream);
    if (ret == ACL_ERROR_OP_TYPE_NOT_MATCH || ret == ACL_ERROR_OP_INPUT_NOT_MATCH ||
        ret == ACL_ERROR_OP_OUTPUT_NOT_MATCH || ret == ACL_ERROR_OP_ATTR_NOT_MATCH) {
        ERROR_LOG("[%s] op with the given description is not compiled. Please run atc first", opDesc_->opType.c_str());
        return false;
    } else if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("Execute %s failed. ret = %d", opDesc_->opType.c_str(), ret);
        return false;
    }
    INFO_LOG("Execute %s success", opDesc_->opType.c_str());

    if (aclrtSynchronizeStream(stream) != ACL_ERROR_NONE) {
        ERROR_LOG("Synchronize stream failed");
        return false;
    }
    INFO_LOG("Synchronize stream success");

    for (size_t i = 0; i < numOutputs_; ++i) {
        auto size = GetOutputSize(i);
        aclrtMemcpyKind kind = ACL_MEMCPY_DEVICE_TO_HOST;
        if (g_isDevice) {
            kind = ACL_MEMCPY_DEVICE_TO_DEVICE;
        }
        if (aclrtMemcpy(hostOutputs_[i], size, devOutputs_[i], size, kind) != ACL_ERROR_NONE) {
            INFO_LOG("Copy output[%zu] success", i);
            return false;
        }
        INFO_LOG("Copy output[%zu] success", i);
    }

    (void) aclrtDestroyStream(stream);
    return true;
}


template<typename T>
void DoPrintData(const T *data, size_t count, size_t elementsPerRow)
{
    ios::fmtflags f(cout.flags());
    for (size_t i = 0; i < count; ++i) {
        std::cout << std::setw(DEFAULT_WHITE_COUNT) << data[i];
        if (i % elementsPerRow == elementsPerRow - 1) {
            std::cout << std::endl;
        }
    }
    cout.flags(f);
}

void DoPrintFp16Data(const aclFloat16 *data, size_t count, size_t elementsPerRow)
{
    ios::fmtflags f(cout.flags()); // save old format
    for (size_t i = 0; i < count; ++i) {
        std::cout << std::setw(DEFAULT_WHITE_COUNT) << \
        std::setprecision(DEFAULT_PRECISION_COUNT) << \
        aclFloat16ToFloat(data[i]);
        if (i % elementsPerRow == elementsPerRow - 1) {
            std::cout << std::endl;
        }
    }
    cout.flags(f);
}

void PrintData(const void *data, size_t count, aclDataType dataType, size_t elementsPerRow)
{
    if (data == nullptr) {
        ERROR_LOG("Print data failed. data is nullptr");
        return;
    }

    switch (dataType) {
        case ACL_BOOL:
            DoPrintData(reinterpret_cast<const bool *>(data), count, elementsPerRow);
            break;
        case ACL_INT8:
            DoPrintData(reinterpret_cast<const int8_t *>(data), count, elementsPerRow);
            break;
        case ACL_UINT8:
            DoPrintData(reinterpret_cast<const uint8_t *>(data), count, elementsPerRow);
            break;
        case ACL_INT16:
            DoPrintData(reinterpret_cast<const int16_t *>(data), count, elementsPerRow);
            break;
        case ACL_UINT16:
            DoPrintData(reinterpret_cast<const uint16_t *>(data), count, elementsPerRow);
            break;
        case ACL_INT32:
            DoPrintData(reinterpret_cast<const int32_t *>(data), count, elementsPerRow);
            break;
        case ACL_UINT32:
            DoPrintData(reinterpret_cast<const uint32_t *>(data), count, elementsPerRow);
            break;
        case ACL_INT64:
            DoPrintData(reinterpret_cast<const int64_t *>(data), count, elementsPerRow);
            break;
        case ACL_UINT64:
            DoPrintData(reinterpret_cast<const uint64_t *>(data), count, elementsPerRow);
            break;
        case ACL_FLOAT16:
            DoPrintFp16Data(reinterpret_cast<const aclFloat16 *>(data), count, elementsPerRow);
            break;
        case ACL_FLOAT:
            DoPrintData(reinterpret_cast<const float *>(data), count, elementsPerRow);
            break;
        case ACL_DOUBLE:
            DoPrintData(reinterpret_cast<const double *>(data), count, elementsPerRow);
            break;
        default:
            ERROR_LOG("Unsupported type: %d", dataType);
    }
}

void OpRunner::PrintInput(size_t index, size_t numElementsPerRow)
{
    if (index >= opDesc_->inputDesc.size()) {
        ERROR_LOG("index out of range. index = %zu, numOutputs = %zu", index, numInputs_);
        return;
    }

    auto desc = opDesc_->inputDesc[index];
    PrintData(hostInputs_[index], GetInputElementCount(index), aclGetTensorDescType(desc), numElementsPerRow);
}

void OpRunner::PrintOutput(size_t index, size_t numElementsPerRow)
{
    if (index >= opDesc_->outputDesc.size()) {
        ERROR_LOG("index out of range. index = %zu, numOutputs = %zu", index, numOutputs_);
        return;
    }

    auto desc = opDesc_->outputDesc[index];
    PrintData(hostOutputs_[index], GetOutputElementCount(index), aclGetTensorDescType(desc), numElementsPerRow);
}
