#include "toolchain/slog.h"

#include <stdio.h>
#include <string.h>
#include <stdarg.h>

#define op_ut_log_print(fmt) \
	va_list args;va_start(args,fmt);vprintf(fmt,args);va_end(args);printf("\n");

//#define op_ut_log_print(fmt) \
//    return;

void DlogErrorInner(int module_id, const char *fmt, ...)
{
    op_ut_log_print(fmt);
}

void DlogWarnInner(int module_id, const char *fmt, ...)
{
    op_ut_log_print(fmt);
}

void DlogInfoInner(int module_id, const char *fmt, ...)
{
    op_ut_log_print(fmt);
}

void DlogDebugInner(int module_id, const char *fmt, ...)
{
    op_ut_log_print(fmt);
}

void DlogEventInner(int module_id, const char *fmt, ...)
{
    op_ut_log_print(fmt);
}

void DlogInner(int moduleId, int level, const char *fmt, ...)
{
    op_ut_log_print(fmt);
}

void DlogWithKVInner(int moduleId, int level, KeyValue* pstKVArray, int kvNum, const char *fmt, ...)
{
    op_ut_log_print(fmt);
}

int dlog_getlevel(int module_id, int *enable_event)
{
    return DLOG_DEBUG;
}

int CheckLogLevel(int moduleId, int logLevel)
{
    return 1;
}
