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

# Tell CMake to run moc when necessary:
set(CMAKE_AUTOMOC ON)
# As moc files are generated in the binary dir, tell CMake
# to always look for includes there:
set(CMAKE_INCLUDE_CURRENT_DIR ON)

include_directories(${OWL_INCLUDES})
find_package(OpenGL REQUIRED)
find_package(CUDA REQUIRED)
add_library(qtOWL STATIC
  OWLViewer.cpp
  InspectMode.cpp
  FlyMode.cpp
  Camera.cpp

  AlphaEditor.h
  AlphaEditor.cpp
  XFEditor.h
  XFEditor.cpp

  ColorMaps.h
  ColorMaps.cpp
  )
target_link_libraries(qtOWL
  ${OPENGL_opengl_LIBRARY}
  ${CUDA_CUDA_LIBRARY}
  ${CUDA_CUDART_LIBRARY}
  )
qt5_use_modules(qtOWL
  Widgets
  )

set(QT_OWL_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/.. ${OWL_INCLUDES} PARENT_SCOPE)
set(QT_OWL_LIBRARIES qtOWL PARENT_SCOPE)
  
macro(OWL_QT_LINK target)
  target_link_libraries(${target} ${QT_OWL_LIBRARIES})
  qt5_use_modules(${target} Widgets)
endmacro()
