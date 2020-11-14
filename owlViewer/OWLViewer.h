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

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QOpenGLBuffer>
#include <QTimer>

QT_FORWARD_DECLARE_CLASS(QOpenGLShaderProgram);
QT_FORWARD_DECLARE_CLASS(QOpenGLTexture)

class OWLViewerWidget : public QOpenGLWidget, protected QOpenGLFunctions
{
  Q_OBJECT

public:
  using QOpenGLWidget::QOpenGLWidget;

  OWLViewerWidget() {
    printf("creating timer...\n");
    connect(&timer, SIGNAL(timeout()), this, SLOT(update()));
    timer.start(1);
  }
  
  ~OWLViewerWidget();

  QSize minimumSizeHint() const override;
  QSize sizeHint() const override;
  // void rotateBy(int xAngle, int yAngle, int zAngle);
  // void setClearColor(const QColor &color);

signals:
  void clicked();

protected:
  void initializeGL() override;
  void paintGL() override;
  void resizeGL(int width, int height) override;
  void mousePressEvent(QMouseEvent *event) override;
  void mouseMoveEvent(QMouseEvent *event) override;
  void mouseReleaseEvent(QMouseEvent *event) override;

private:
  // void makeObject();

  QColor clearColor = Qt::black;
  QPoint lastPos;
  // int xRot = 0;
  // int yRot = 0;
  // int zRot = 0;
  // QOpenGLTexture *textures[6] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
  // QOpenGLShaderProgram *program = nullptr;
  QOpenGLBuffer vbo;
  QTimer timer;
};
