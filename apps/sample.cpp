#include <QApplication>
#include <QPushButton>
#include <QMainWindow>

#include "OWLViewer.h"

int main(int argc, char **argv)
{
  QApplication app(argc,argv);
  
  OWLViewerWidget viewer;
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
