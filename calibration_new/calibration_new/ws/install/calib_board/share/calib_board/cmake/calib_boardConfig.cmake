# generated from ament/cmake/core/templates/nameConfig.cmake.in

# prevent multiple inclusion
if(_calib_board_CONFIG_INCLUDED)
  # ensure to keep the found flag the same
  if(NOT DEFINED calib_board_FOUND)
    # explicitly set it to FALSE, otherwise CMake will set it to TRUE
    set(calib_board_FOUND FALSE)
  elseif(NOT calib_board_FOUND)
    # use separate condition to avoid uninitialized variable warning
    set(calib_board_FOUND FALSE)
  endif()
  return()
endif()
set(_calib_board_CONFIG_INCLUDED TRUE)

# output package information
if(NOT calib_board_FIND_QUIETLY)
  message(STATUS "Found calib_board: 0.0.1 (${calib_board_DIR})")
endif()

# warn when using a deprecated package
if(NOT "" STREQUAL "")
  set(_msg "Package 'calib_board' is deprecated")
  # append custom deprecation text if available
  if(NOT "" STREQUAL "TRUE")
    set(_msg "${_msg} ()")
  endif()
  # optionally quiet the deprecation message
  if(NOT ${calib_board_DEPRECATED_QUIET})
    message(DEPRECATION "${_msg}")
  endif()
endif()

# flag package as ament-based to distinguish it after being find_package()-ed
set(calib_board_FOUND_AMENT_PACKAGE TRUE)

# include all config extra files
set(_extras "")
foreach(_extra ${_extras})
  include("${calib_board_DIR}/${_extra}")
endforeach()
