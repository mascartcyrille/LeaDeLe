## Version: $Id$
##
######################################################################
##
### Commentary:
##
######################################################################
##
### Change Log:
##
######################################################################
##
### Code:

project(leadele VERSION ${nm-learning_VERSION})

## #############################################################################
##
## #############################################################################

set(layer_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

## #################################################################
## Sources
## #################################################################

set(${PROJECT_NAME}_DATA_HEADERS
  data/mnist.hpp
  data/image.hpp)

set(${PROJECT_NAME}_LEARNING_HEADERS
  learning/linear-model.hpp)

## #################################################################
## Build rules
## #################################################################

add_library(${PROJECT_NAME} INTERFACE)

target_link_libraries(${PROJECT_NAME} INTERFACE nm-spikes)
target_link_libraries(${PROJECT_NAME} INTERFACE tbb)
# target_link_libraries(${PROJECT_NAME} INTERFACE ${TBB_IMPORTED_TARGETS})

target_include_directories(${PROJECT_NAME} INTERFACE
  $<BUILD_INTERFACE:${layer_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
  $<BUILD_INTERFACE:${layer_BINARY_DIR}>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
  $<INSTALL_INTERFACE:include>)

## ###################################################################
## Install rules - files
## ###################################################################

install(FILES ${${PROJECT_NAME}_DATA_HEADERS}
  DESTINATION include/${PROJECT_NAME}/data)
install(FILES ${${PROJECT_NAME}_LEARNING_HEADERS}
  DESTINATION include/${PROJECT_NAME}/learning)

## ###################################################################
## Install rules - targets
## ###################################################################

install(TARGETS ${PROJECT_NAME} EXPORT layer-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

## #############################################################################
##
## #############################################################################

export(EXPORT layer-targets FILE "${CMAKE_BINARY_DIR}/${PROJECT_NAME}Targets.cmake")

######################################################################
### CMakeLists.txt ends here
