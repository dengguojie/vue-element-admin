aux_source_directory(src SLOG_OPS_STUB_SRC)

include_directories(${TOP_DIR}/inc)

add_library(slog_ops_stub SHARED ${SLOG_OPS_STUB_SRC})