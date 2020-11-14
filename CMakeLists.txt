cmake_minimum_required(VERSION 3.16)

project(OWLQTViewer)

#find_package(Qt4 COMPONENTS QTCORE QTGUI)
#find_package(
find_package(Qt5Widgets REQUIRED)

# Tell CMake to run moc when necessary:
set(CMAKE_AUTOMOC ON)
# As moc files are generated in the binary dir, tell CMake
# to always look for includes there:
set(CMAKE_INCLUDE_CURRENT_DIR ON)
  
add_subdirectory(owlViewer)

#include_directories(${Qt5Widgets_INCLUDES})
#add_definitions(${Qt5Widgets_DEFINITIONS})
#set(CMAKE_CXX_FLAGS "${Qt5Widgets_EXECUTABLE_COMPILE_FLAGS}")
 
#add_library(owlQTViewer STATIC
#  owlViewer/OWLViewerWidget.cpp
#  )
#qt5_use_modules(owlQTViewer Widgets)




add_executable(sampleViewer
  apps/sample.cpp
  )
#target_link_libraries(
#  firstTest
#  ${Qt5Widgets_LIBRARIES}
#  )
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/owlViewer)

qt5_use_modules(sampleViewer Widgets)
target_link_libraries(sampleViewer
  owlQTViewer
  )
