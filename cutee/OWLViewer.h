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

#pragma once

#include "cutee/Camera.h"
#include "cutee/CameraManip.h"

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QTimer>
#include <QApplication>
#include <QMainWindow>

typedef struct cudaGraphicsResource* cudaGraphicsResource_t;

QT_FORWARD_DECLARE_CLASS(QOpenGLShaderProgram);
QT_FORWARD_DECLARE_CLASS(QOpenGLTexture)

namespace cutee {
  
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
    
  class OWLViewer : public QOpenGLWidget, protected QOpenGLFunctions
  {
    Q_OBJECT

  public:
    using QOpenGLWidget::QOpenGLWidget;

    // ------------------------------------------------------------------
    // VIEWER LOGIC
    // ------------------------------------------------------------------
    
    /*! snaps a given vector to one of the three coordinate axis;
      useful for pbrt models in which the upvector sometimes isn't
      axis-aligend */
    static vec3f getUpVector(const vec3f &v);


    OWLViewer(const std::string &title = "OWL Sample Viewer",
              const vec2i &initWindowSize=vec2i(1200,800),
              bool visible=true
              // ,
              // const vec3f &cameraInitFrom = vec3f(0,0,-1),
              // const vec3f &cameraInitAt   = vec3f(0,0,0),
              // const vec3f &cameraInitUp   = vec3f(0,1,0),
              // const float worldScale      = 1.f
              );
      

    ~OWLViewer();


    /*! window notifies us that we got resized */     
    virtual void resize(const vec2i &newSize);

    void setTitle(const std::string &s);

    /*! gets called whenever the viewer needs us to re-render out widget */
    virtual void render() {}
      
    /*! draw framebuffer using OpenGL */
    virtual void draw();

    struct ButtonState {
      bool  isPressed        { false };
      vec2i posFirstPressed  { -1 };
      vec2i posLastSeen      { -1 };
      bool  shiftWhenPressed { false };
      bool  ctrlWhenPressed  { false };
      bool  altWhenPressed   { false };
    };

    ButtonState leftButton;
    ButtonState rightButton;
    ButtonState centerButton;
    vec2i       lastMousePosition { -1,-1 };

    /*! this gets called when the window determines that the mouse
      got _moved_ to the given position */
    virtual void mouseMotion(const vec2i &newMousePosition);

    /*! mouse got dragged with left button pressedn, by 'delta'
      pixels, at last position where */
    virtual void mouseDragLeft  (const vec2i &where, const vec2i &delta);
      
    /*! mouse got dragged with left button pressedn, by 'delta'
      pixels, at last position where */
    virtual void mouseDragCenter(const vec2i &where, const vec2i &delta);
      
    /*! mouse got dragged with left button pressedn, by 'delta'
      pixels, at last position where */
    virtual void mouseDragRight (const vec2i &where, const vec2i &delta);
      
    /*! mouse button got either pressed or released at given location */
    virtual void mouseButtonLeft  (const vec2i &where, bool pressed);
      
    /*! mouse button got either pressed or released at given location */
    virtual void mouseButtonCenter(const vec2i &where, bool pressed);
      
    /*! mouse button got either pressed or released at given location */
    virtual void mouseButtonRight (const vec2i &where, bool pressed);

    /*! this gets called when the user presses a key on the keyboard ... */
    virtual void key(char key, const vec2i &/*where*/);
    
    /*! this gets called when the user presses a 'special' key on
      the keyboard (cursor keys) ... */
    virtual void special(QKeyEvent *event, const vec2i &/*where*/);

    /*! set a new window aspect ratio for the camera, update the
      camera, and notify the app */
    void setAspect(const float aspect)
    {
      camera.setAspect(aspect);
      updateCamera();
    }

    /*! set a new orientation for the camera, update the camera, and
      notify the app */
    void setCameraOrientation(/* camera origin    : */const vec3f &origin,
                              /* point of interest: */const vec3f &interest,
                              /* up-vector        : */const vec3f &up,
                              /* fovy, in degrees : */float fovyInDegrees);
    void getCameraOrientation(/* camera origin    : */vec3f &origin,
                              /* point of interest: */vec3f &interest,
                              /* up-vector        : */vec3f &up,
                              /* fovy, in degrees : */float & fovyInDegrees);

    void setCameraOptions(float fovy,
                          float focalDistance);

    /*! this function gets called whenever any camera manipulator
      updates the camera. gets called AFTER all values have been updated */
    virtual void cameraChanged() {}

    /*! return currently active window size */
    vec2i getWindowSize() const { return fbSize; }
    static vec2i getScreenSize();
      
    Camera &getCamera() { return camera; }

    /*! helper function that dumps the current frame buffer in a png
      file of given name */
    void screenShot(const std::string &fileName);
      
      
    const SimpleCamera getSimplifiedCamera() const
    {
      return SimpleCamera(camera);
    }

    std::shared_ptr<CameraManipulator> cameraManipulator;
    std::shared_ptr<CameraManipulator> inspectModeManipulator;
    std::shared_ptr<CameraManipulator> flyModeManipulator;

    void enableFlyMode();
    enum RotateMode { POI, Arcball };
    void enableInspectMode(RotateMode rm,
                           const box3f &validPoiRange=box3f(),
                           float minPoiDist=1e-3f,
                           float maxPoiDist=std::numeric_limits<float>::infinity());
    void enableInspectMode(const box3f &validPoiRange=box3f(),
                           float minPoiDist=1e-3f,
                           float maxPoiDist=std::numeric_limits<float>::infinity());
    void setWorldScale(const float worldScale)
    {
      camera.motionSpeed = worldScale / sqrtf(3.f);
    }

    /*! re-computes the 'camera' from the 'cameracontrol', and notify
      app that the camera got changed */
    void updateCamera();

    // void showAndRun();
      
    void mouseButton(int button, int action, int mods);
      
    /*! the full camera state we are manipulating */
    Camera camera;

    vec2i getMousePos() const;
    // inline vec2i getMousePos() const
    // {
    //   double x,y;
    //   glfwGetCursorPos(handle,&x,&y);
    //   return vec2i((int)x, (int)y);
    // }
  private:
    friend struct CameraManipulator;
    friend struct CameraInspectMode;
    friend struct CameraFlyMode;

  protected:

    
      
    vec2i    fbSize { 0 };
    /*! what we'll return as sizeHint when qt asks us for it */
    vec2i    userSizeHint { 0,0 };
    GLuint   fbTexture  {0};
    cudaGraphicsResource_t cuDisplayTexture { 0 };
    uint32_t *fbPointer { nullptr };
      
    /*! the glfw window handle */
    // GLFWwindow *handle { nullptr };
    vec2i lastMousePos = { -1,-1 };

    /*! tracks whether we could successfully do cuda resource
      binding to the GL display texture; if not, we'll have to
      fall back to a slower path with glTexImage */
    bool resourceSharingSuccessful;


    
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
    void keyPressEvent(QKeyEvent *event) override;
    // void resizeEvent(QResizeEvent* event) override;


  private:
    // void makeObject();

    // QColor clearColor = Qt::black;
    // QPoint lastPos;
    // int xRot = 0;
    // int yRot = 0;
    // int zRot = 0;
    // QOpenGLTexture *textures[6] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    // QOpenGLShaderProgram *program = nullptr;
    // QOpenGLBuffer vbo;
    QTimer timer;
  };
}
