set(ALL_JSON_FILES "")
file(GLOB_RECURSE OP_FILES "op_info_cfg/*.ini")
foreach(_file ${OP_FILES})
  set(INI_PARSER "${CMAKE_CURRENT_SOURCE_DIR}/op_info_cfg/parser/parser_ini.py")
  if(NOT EXISTS "${INI_PARSER}")
    message(FATAL_ERROR "Cannot find the the ini parser ${INI_PARSER}")
  endif()

  string(REPLACE ${CMAKE_CURRENT_SOURCE_DIR}/op_info_cfg/ "" _rel_path "${_file}")
  get_filename_component(_ini_name "${_rel_path}" NAME)
  get_filename_component(_parent_dir "${_rel_path}" DIRECTORY)
  string(REPLACE ".ini" ".json" _json_name "${_ini_name}")
  set(_out_path "${CMAKE_CURRENT_BINARY_DIR}/config/${_parent_dir}")
  file(MAKE_DIRECTORY "${_out_path}")

  set(_out_json "${_out_path}/${_json_name}")
  add_custom_command(OUTPUT "${_out_json}"
    COMMAND "${HI_PYTHON}" "${INI_PARSER}" "${_file}" "${_out_json}" 1>/dev/null
    COMMENT "Parsing ini file ${_ini_name}"
  )
  list(APPEND ALL_JSON_FILES "${_out_json}")
endforeach()

set(TBE_OP_PATH "${INSTALL_PATH}/opp/op_impl/built-in")

add_custom_target(aicpu_ops_json_info ALL
  DEPENDS ${ALL_JSON_FILES}
)
if(BUILD_OPEN_PROJECT)
  cann_install(
    TARGET      aicpu_ops_json_info
    FILES "${CMAKE_CURRENT_BINARY_DIR}/config/aicpu_kernel/aicpu_kernel.json"
    DESTINATION "${TBE_OP_PATH}/aicpu/aicpu_kernel/config"
  )
  cann_install(
    TARGET      aicpu_ops_json_info
    FILES       "${CMAKE_CURRENT_BINARY_DIR}/config/tf_kernel/tf_kernel.json"
    DESTINATION "${TBE_OP_PATH}/aicpu/tf_kernel/config"
  )
else()
install(
  FILES       ${ALL_JSON_FILES}
  DESTINATION ${INSTALL_LIBRARY_DIR}
  OPTIONAL
)
add_subdirectory(impl)
add_subdirectory(context)
endif()

