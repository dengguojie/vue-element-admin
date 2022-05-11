if(NOT ENABLE_TEST)
  set(ALL_JSON_FILES "")
  set(DVPP_JSON_OUT_PATH "${CMAKE_CURRENT_BINARY_DIR}/config/dvpp_kernel")

  set(INI_PARSER "${CMAKE_CURRENT_SOURCE_DIR}/op_info_cfg/parser/parser_ini.py")
  if(NOT EXISTS "${INI_PARSER}")
    message(FATAL_ERROR "Cannot find the the dvpp ini parser ${INI_PARSER}")
  endif()

  set(CUSTOM_FLAG "")
  if(WITH_CUSTOM_OP)
    set(CUSTOM_FLAG "-c")
  endif()

  file(MAKE_DIRECTORY "${DVPP_JSON_OUT_PATH}")

  add_custom_command(OUTPUT "${DVPP_JSON_OUT_PATH}/dvpp_kernel.json"
    COMMAND "${HI_PYTHON}" "${INI_PARSER}"
    "${CMAKE_CURRENT_SOURCE_DIR}/op_info_cfg/dvpp_kernel/dvpp_kernel.ini"
    "${DVPP_JSON_OUT_PATH}/dvpp_kernel.json"
    "${CUSTOM_FLAG}" 1>/dev/null
    COMMENT "Parsing dvpp_kernel.ini file"
  )

  list(APPEND ALL_JSON_FILES "${DVPP_JSON_OUT_PATH}/dvpp_kernel.json")
  set(TBE_OP_PATH "${INSTALL_PATH}/opp/op_impl/built-in")

  add_custom_target(dvpp_ops_json_info ALL
    DEPENDS ${ALL_JSON_FILES}
  )
  if(BUILD_OPEN_PROJECT)
    cann_install(
      TARGET      dvpp_ops_json_info
      FILES "${CMAKE_CURRENT_BINARY_DIR}/config/dvpp_kernel/dvpp_kernel.json"
      DESTINATION "${TBE_OP_PATH}/dvpp/dvpp_kernel/config"
    )
  else()
    install(
      FILES       ${ALL_JSON_FILES}
      DESTINATION ${INSTALL_LIBRARY_DIR}
      OPTIONAL
    )
  endif()
endif()
