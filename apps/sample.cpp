// ======================================================================== //
// Copyright 2019-2020 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include <QApplication>
#include <QPushButton>
#include <QMainWindow>

#include "owlQT/OWLViewer.h"

int main(int argc, char **argv)
{
  QApplication app(argc,argv);
  
  owlQT::OWLViewer viewer;
  viewer.show();
  // QPushButton button("Hello World!");
  // button.show();

  QMainWindow secondWindow;
  QPushButton button2("Hello World!",&secondWindow);
  button2.show();
  // OWLViewerWidget viewer(secondWindow);
  // secondWindow.setWindowFlags(Qt::Window | Qt::FramelessWindowHint); 
  secondWindow.show();
  
  return app.exec();
}
// https://code.qt.io/cgit/qt/qtbase.git/tree/examples/opengl/textures/glwidget.cpp?h=5.15
