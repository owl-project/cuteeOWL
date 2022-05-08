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

option(CUTEEOWL_USE_CUDA "Build with CUDA frame buffer implementation" ON)
if(CUTEEOWL_USE_CUDA)
  find_package(CUDAToolkit)
endif()

# ------------------------------------------------------------------
# add owl, for vec3f stuff used in viewer
# ------------------------------------------------------------------
if (NOT TARGET owl::owl)
  set(owl_dir ${CMAKE_SOURCE_DIR}/../owl)
  add_subdirectory(${owl_dir} external_owl EXCLUDE_FROM_ALL)
endif()

# ------------------------------------------------------------------
# the owlQT library with viewer widget etc
# ------------------------------------------------------------------
add_subdirectory(qtOWL)

# ------------------------------------------------------------------
# sample app(s)
# ------------------------------------------------------------------



if(CUDAToolkit_FOUND)
  add_executable(sampleViewer apps/sample.cpp)
  target_link_libraries(sampleViewer PRIVATE qtOWL)
endif()








