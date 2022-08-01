# ======================================================================== #
# Copyright 2020-2020 Ingo Wald                                            #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
# ======================================================================== #

cmake_minimum_required(VERSION 3.12)

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.18)
  cmake_policy(SET CMP0104 NEW)
endif()

project(Cutee-OWL VERSION 0.0.1)

# option(CUTEEOWL_USE_CUDA "Build with CUDA frame buffer implementation" ON)
#if(CUTEEOWL_USE_CUDA)
#  find_package(CUDAToolkit)
#endif()

if (NOT (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR}))
  set(CUTEE_OWL_IS_SUBPROJECT ON)
else()
  set(CUTEE_OWL_IS_SUBPROJECT OFF)
endif()

# ------------------------------------------------------------------
# add owl, for vec3f stuff used in viewer
# ------------------------------------------------------------------
if (NOT TARGET owl::owl)
  set(owl_dir ${CMAKE_SOURCE_DIR}/../owl)
  add_subdirectory(${owl_dir} external_owl EXCLUDE_FROM_ALL)
endif()


set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# iw - remove QT's hardcoded "-fPIC" from the target compile options, because
# this will cause issues when later 'linking' this target (via linking to qtOWL)
# to targets that get compiled with other compilers such as nvcc (which do not
# understand this target). Proper way is to remove this hardcoded flag here,
# and in the 'real' app set the POSITION_INDEPENDENT_CODE flag to on, as well
# as set CMAKE_CXX_COMPILE_OPTIONS_PIE to "-fPIC" for those targets that
# actually use a C++ compiler that understands this.
find_package(Qt5Widgets REQUIRED)
get_property(core_options TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS)
if(${core_options})
string(REPLACE "-fPIC" "" new_core_options ${core_options})
set_property(TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS ${new_core_options})
endif()
set_property(TARGET Qt5::Core PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE ON)
  

# ------------------------------------------------------------------
# the owlQT library with viewer widget etc
# ------------------------------------------------------------------
add_subdirectory(qtOWL)

# ------------------------------------------------------------------
# sample app(s)
# ------------------------------------------------------------------

IF (NOT CUTEE_OWL_IS_SUBPROJECT)
  set( CMAKE_CXX_COMPILE_OPTIONS_PIE "-fPIC" )
  enable_language(CUDA)
  
  add_executable(sampleViewer apps/sample.cpp)
  set_property(TARGET sampleViewer PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE ON)
  target_link_libraries(sampleViewer PRIVATE qtOWL  Qt5::Widgets)
endif()
#endif()








