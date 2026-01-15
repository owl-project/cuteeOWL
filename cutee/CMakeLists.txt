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

set(CMAKE_AUTOMOC ON)

get_property(core_options TARGET Qt::Core PROPERTY INTERFACE_COMPILE_OPTIONS)
if(${core_options})
string(REPLACE "-fPIC" "" new_core_options ${core_options})
endif(${core_options})
#set_property(TARGET Qt::Core PROPERTY INTERFACE_COMPILE_OPTIONS ${new_core_options})
#set_property(TARGET Qt::Core PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE ON)


add_library(qtOWL STATIC)
target_sources(qtOWL PRIVATE
  OWLViewer.cpp
  InspectMode.cpp
  FlyMode.cpp
  Camera.cpp
  CameraManip.cpp

  AlphaEditor.h
  AlphaEditor.cpp
  XFEditor.h
  XFEditor.cpp

  ColorMaps.h
  ColorMaps.cpp
  )
if (APPLE)
  target_link_libraries(qtOWL PUBLIC Qt::Widgets Qt::OpenGLWidgets
    #OpenGL::GL
    cuteeOWL-config
    )
else()
  target_link_libraries(qtOWL PUBLIC
    Qt::Core
    Qt::Widgets
    Qt::OpenGLWidgets
    OpenGL::GL
    cuteeOWL-config
)
endif()
target_include_directories(qtOWL PUBLIC ${CMAKE_CURRENT_LIST_DIR}/..)
if (APPLE)
  target_compile_definitions(qtOWL PUBLIC -DGL_SILENCE_DEPRECATION=1)
endif()

# target_include_directories(qtOWL
#   PUBLIC
#   ${PROJECT_SOURCE_DIR}
#   )
# target_link_libraries(qtOWL #PUBLIC
#   OpenGL::OpenGL
#   cuteeOWL-config
#   #  OpenGL::GL
#   #  ${OPENGL_opengl_LIBRARY}
#   #  ${CUDA_CUDA_LIBRARY}
#   #  ${CUDA_CUDART_LIBRARY}
# )
  
# qt5_use_modules(qtOWL
#   Widgets
#   )
# macro(QT_OWL_LINK target)
#   target_link_libraries(${target} ${QT_OWL_LIBRARIES})
#   qt5_use_modules(${target} Widgets)
# endmacro()
