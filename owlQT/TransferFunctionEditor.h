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

namespace owlQT {

  /*! base abstraction for a camera that can generate rays. For this
    viewer, we assume we're dealine with a camera that has a
    rectangular viewing plane that's in focus, and a circular (and
    possible single-point) lens for depth of field. At some later
    point this should also capture time.{t0,t1} for motion blur, but
    let's leave this out for now. */
  struct SimpleCamera
  {
    inline SimpleCamera() {}
    SimpleCamera(const Camera &camera);

    struct {
      vec3f lower_left;
      vec3f horizontal;
      vec3f vertical;
    } screen;
    struct {
      vec3f center;
      vec3f du;
      vec3f dv;
      float radius { 0.f };
    } lens;
  };
    
  /*! snaps a given vector to one of the three coordinate axis;
    useful for pbrt models in which the upvector sometimes isn't
    axis-aligend */
  inline vec3f getUpVector(const vec3f &v)
  {
    int dim = arg_max(abs(v));
    vec3f up(0);
    up[dim] = v[dim] < 0.f ? -1.f : 1.f;
    return up;
  }

  class OWLViewer : public QOpenGLWidget, protected QOpenGLFunctions
  {
    Q_OBJECT

  public:
    using QOpenGLWidget::QOpenGLWidget;

    OWLViewer() {
      printf("creating timer...\n");
      connect(&timer, SIGNAL(timeout()), this, SLOT(update()));
      timer.start(1);
    }
  
    ~OWLViewer();

    // ------------------------------------------------------------------
    // VIEWER LOGIC
    // ------------------------------------------------------------------
    
    /*! snaps a given vector to one of the three coordinate axis;
      useful for pbrt models in which the upvector sometimes isn't
      axis-aligend */
    static vec3f getUpVector(const vec3f &v);


    // ------------------------------------------------------------------
    // QT STUFF
    // ------------------------------------------------------------------
    
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
}
