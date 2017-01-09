cmake_minimum_required (VERSION 2.8)
SET(CMAKE_BUILD_TYPE Release)
project (itkslave)
 
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})
SET(ELASTIX_PATH $ENV{HOME}/Multimodal/external/elastix/src)
SET(ELASTIX_BIN_PATH $ENV{HOME}/Multimodal/external/elastix/build/bin)

#find_package(Lua REQUIRED)
FIND_PACKAGE(Torch REQUIRED)
find_package(ITK REQUIRED)

#FIND_PACKAGE(Boost COMPONENTS thread system filesystem regex REQUIRED)
#IF (Boost_FOUND)
#    INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
#    ADD_DEFINITIONS( "-DHAS_BOOST" )
#ENDIF()

SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -g -fPIC")

include_directories (
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LUA_INCLUDE_DIR}
)
include(${ITK_USE_FILE})

# Find OpenCL.
find_package( OpenCL REQUIRED )
if( NOT ${OPENCL_FOUND} )
  message(FATAL_ERROR "Could not find OpenCL")
else()
  # Force OPENCL_OPTIMIZATION_MAD_ENABLE to on
  if( DEFINED OPENCL_OPTIMIZATION_MAD_ENABLE )
    set( OPENCL_OPTIMIZATION_MAD_ENABLE ON CACHE BOOL
      "Allow a * b + c to be replaced by a mad. The mad computes a * b + c with reduced accuracy." FORCE )
  endif()
endif()

include_directories(${OPENCL_INCLUDE_DIRS})

set( CMAKE_INCLUDE_DIRECTORIES_BEFORE ON )

set(ELASTIX_INCLUDE_DIRS
  ${ELASTIX_PATH}/Common
  ${ELASTIX_PATH}/Common/OpenCL
  ${ELASTIX_PATH}/Common/OpenCL/Factories  
  ${ELASTIX_PATH}/Common/OpenCL/Filters
  ${ELASTIX_PATH}/Common/OpenCL/Kernels  
  ${ELASTIX_PATH}/Common/OpenCL/ITKimprovements
)
include_directories(${ELASTIX_INCLUDE_DIRS})

set(ELASTIX_LIBRARIES
  ${ELASTIX_BIN_PATH}/libelastix.so
)
 
set (SOURCES
    itkslave.cpp
)

add_library(${PROJECT_NAME} MODULE ${SOURCES})
target_link_libraries (${PROJECT_NAME} ${ITK_LIBRARIES} ${LUA_LIBRARIES} ${OPENCL_LIBRARIES} ${ELASTIX_LIBRARIES} TH)# luaT

SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES
  PREFIX "lib"
  IMPORT_PREFIX "lib")
    
INSTALL(TARGETS ${PROJECT_NAME}
  RUNTIME DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}"
  LIBRARY DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}")





