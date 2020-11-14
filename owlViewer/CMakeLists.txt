# Tell CMake to run moc when necessary:
set(CMAKE_AUTOMOC ON)
# As moc files are generated in the binary dir, tell CMake
# to always look for includes there:
set(CMAKE_INCLUDE_CURRENT_DIR ON)
  
#include_directories(${Qt5Widgets_INCLUDES})
#add_definitions(${Qt5Widgets_DEFINITIONS})
#set(CMAKE_CXX_FLAGS "${Qt5Widgets_EXECUTABLE_COMPILE_FLAGS}")
 
add_library(owlQTViewer STATIC
  OWLViewer.cpp
  )
qt5_use_modules(owlQTViewer
  Widgets
  )


