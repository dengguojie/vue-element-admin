if(NOT ENABLE_TEST)
  set(ALL_JSON_FILES "")
  set(AICPU_JSON_OUT_PATH "${CMAKE_CURRENT_BINARY_DIR}/config/aicpu_kernel")

  set(INI_PARSER "${CMAKE_CURRENT_SOURCE_DIR}/op_info_cfg/parser/parser_ini.py")
  if(NOT EXISTS "${INI_PARSER}")
    message(FATAL_ERROR "Cannot find the the ini parser ${INI_PARSER}")
  endif()

  set(CUSTOM_FLAG "")
  if(WITH_CUSTOM_OP)
    set(CUSTOM_FLAG "-c")
  endif()

  file(MAKE_DIRECTORY "${AICPU_JSON_OUT_PATH}")
  add_custom_command(OUTPUT "${AICPU_JSON_OUT_PATH}/aicpu_kernel.json"
    COMMAND "${HI_PYTHON}" "${INI_PARSER}"
    "${CMAKE_CURRENT_SOURCE_DIR}/op_info_cfg/aicpu_kernel/aicpu_kernel.ini"
    "${CMAKE_CURRENT_SOURCE_DIR}/op_info_cfg/aicpu_cust_kernel/aicpu_cust_kernel.ini"
    "${AICPU_JSON_OUT_PATH}/aicpu_kernel.json"
    "${CUSTOM_FLAG}" 1>/dev/null
    COMMENT "Parsing aicpu_kernel.ini and aicpu_cust_kernel.ini file"
  )

  add_custom_command(OUTPUT "${AICPU_JSON_OUT_PATH}/host_cpu_kernel.json"
    COMMAND "${HI_PYTHON}" "${INI_PARSER}"
    "${CMAKE_CURRENT_SOURCE_DIR}/op_info_cfg/host_cpu_kernel/host_cpu_kernel.ini"
    "${AICPU_JSON_OUT_PATH}/host_cpu_kernel.json"
    "${CUSTOM_FLAG}" 1>/dev/null
    COMMENT "Parsing host_cpu_kernel.ini file"
  )

  list(APPEND ALL_JSON_FILES "${AICPU_JSON_OUT_PATH}/aicpu_kernel.json")
  list(APPEND ALL_JSON_FILES "${AICPU_JSON_OUT_PATH}/host_cpu_kernel.json")
  set(TBE_OP_PATH "${INSTALL_PATH}/opp/op_impl/built-in")

  add_custom_target(aicpu_ops_json_info ALL
    DEPENDS ${ALL_JSON_FILES}
  )
  if(BUILD_OPEN_PROJECT)
    cann_install(
      TARGET      aicpu_ops_json_info
      FILES "${CMAKE_CURRENT_BINARY_DIR}/config/aicpu_kernel/aicpu_kernel.json"
      FILES "${CMAKE_CURRENT_BINARY_DIR}/config/aicpu_kernel/host_cpu_kernel.json"
      DESTINATION "${TBE_OP_PATH}/aicpu/aicpu_kernel/config"
    )
  else()
    install(
      FILES       ${ALL_JSON_FILES}
      DESTINATION ${INSTALL_LIBRARY_DIR}
      OPTIONAL
    )
  endif()

  add_subdirectory(impl)
  add_subdirectory(context)
  add_subdirectory(folding)

endif()
