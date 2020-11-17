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

if (POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
endif()
project(Cutee-OWL VERSION 0.0.1)
cmake_minimum_required(VERSION 3.10)

find_package(Qt5Widgets REQUIRED)

# As moc files are generated in the binary dir, tell CMake
# to always look for includes there:
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# ------------------------------------------------------------------
# add owl, for vec3f stuff used in viewer
# ------------------------------------------------------------------
if (NOT (TARGET owl_static))
  message("owl hasn't been included yet...")
  set(owl_dir ${CMAKE_CURRENT_SOURCE_DIR}/../owl)
  add_subdirectory(${owl_dir} external_owl EXCLUDE_FROM_ALL)
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${owl_dir}/owl/cmake/")
  include(configure_build_type)
endif()


# ------------------------------------------------------------------
# the owlQT library with viewer widget etc
# ------------------------------------------------------------------
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory(qtOWL)


# ------------------------------------------------------------------
# sample app(s)
# ------------------------------------------------------------------

# Tell CMake to run moc when necessary:
set(CMAKE_AUTOMOC ON)

include_directories(${QT_OWL_INCLUDES})

add_executable(sampleViewer
  apps/sample.cpp
  )

QT_OWL_LINK(sampleViewer)

set(QT_OWL_INCLUDES  ${QT_OWL_INCLUDES}  PARENT_SCOPE)
set(QT_OWL_LIBRARIES ${QT_OWL_LIBRARIES} PARENT_SCOPE)
