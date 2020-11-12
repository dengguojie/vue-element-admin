/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
 * Description:
 */
#include "aicpu_schedule/aicpu_sharder/aicpu_sharder.h"

namespace aicpu {
SharderNonBlock::SharderNonBlock() : schedule_(NULL), doTask_(NULL), cpuCoreNum_(0) {}

SharderNonBlock &SharderNonBlock::GetInstance()
{
    static SharderNonBlock sharderNonBlock;
    return sharderNonBlock;
}

void SharderNonBlock::ParallelFor(int64_t total, int64_t perUnitSize, const SharderWork &work)
{
    work(0, total);
}

uint32_t SharderNonBlock::GetCPUNum()
{
    return 1;
}
}
