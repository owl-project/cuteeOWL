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

#include "OWLViewer.h"
#include <QOpenGLShaderProgram>
#include <QOpenGLTexture>
#include <QMouseEvent>
#include <math.h>
#include <QApplication>
#include <QDesktopWidget>

#include "InspectMode.h"
#include "FlyMode.h"
#include <sstream>

#if CUTEEOWL_USE_CUDA
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#endif

// eventually to go into 'apps/'
#define STB_IMAGE_WRITE_IMPLEMENTATION 1
#include "stb/stb_image_write.h"

namespace qtOWL {

  inline const char* getGLErrorString( GLenum error )
  {
    switch( error )
      {
      case GL_NO_ERROR:            return "No error";
      case GL_INVALID_ENUM:        return "Invalid enum";
      case GL_INVALID_VALUE:       return "Invalid value";
      case GL_INVALID_OPERATION:   return "Invalid operation";
        //case GL_STACK_OVERFLOW:      return "Stack overflow";
        //case GL_STACK_UNDERFLOW:     return "Stack underflow";
      case GL_OUT_OF_MEMORY:       return "Out of memory";
        //case GL_TABLE_TOO_LARGE:     return "Table too large";
      default:                     return "Unknown GL error";
      }
  }



#define DO_GL_CHECK
#ifdef DO_GL_CHECK
#    define GL_CHECK( call )                                            \
  do                                                                    \
    {                                                                   \
      call;                                                             \
      GLenum err = glGetError();                                        \
      if( err != GL_NO_ERROR )                                          \
        {                                                               \
          std::stringstream ss;                                         \
          ss << "GL error " <<  getGLErrorString( err ) << " at "       \
             << __FILE__  << "(" <<  __LINE__  << "): " << #call        \
             << std::endl;                                              \
          std::cerr << ss.str() << std::endl;                           \
          throw std::runtime_error( ss.str().c_str() );                 \
        }                                                               \
    }                                                                   \
  while (0)


#    define GL_CHECK_ERRORS( )                                          \
  do                                                                    \
    {                                                                   \
      GLenum err = glGetError();                                        \
      if( err != GL_NO_ERROR )                                          \
        {                                                               \
          std::stringstream ss;                                         \
          ss << "GL error " <<  getGLErrorString( err ) << " at "       \
             << __FILE__  << "(" <<  __LINE__  << ")";                  \
          std::cerr << ss.str() << std::endl;                           \
          throw std::runtime_error( ss.str().c_str() );                 \
        }                                                               \
    }                                                                   \
  while (0)

#else
#    define GL_CHECK( call )   do { call; } while(0)
#    define GL_CHECK_ERRORS( ) do { ;     } while(0)
#endif



  OWLViewer::OWLViewer(const std::string &title,
                       const vec2i &initWindowSize,
                       bool visible)
    : userSizeHint(initWindowSize)
  {
    connect(&timer, SIGNAL(timeout()), this, SLOT(update()));
    timer.start(1);

    if (initWindowSize != vec2i(0))
      ((QWidget *)this)->resize(QSize(initWindowSize.x,initWindowSize.y));
  }



  OWLViewer::~OWLViewer()
  {
    // makeCurrent();
    // doneCurrent();
  }

  QSize OWLViewer::minimumSizeHint() const
  {
    return QSize(50, 50);
  }

  QSize OWLViewer::sizeHint() const
  {
    if (userSizeHint != vec2i(0))
      return QSize(userSizeHint.x,userSizeHint.y);
    else {
      QRect rec = QApplication::desktop()->screenGeometry();
      int height = rec.height();
      int width = rec.width();
      return QSize(width/2, height/2);
    }
  }

  void OWLViewer::setTitle(const std::string &s)
  {
    setWindowTitle(s.c_str());
  }

  void OWLViewer::initializeGL()
  {
    initializeOpenGLFunctions();
    glDisable(GL_DEPTH_TEST);
  }

  /* this is the _QT_ widget event handler for redrawing .... */
  void OWLViewer::paintGL()
  {
    static double lastCameraUpdate = -1.f;
    if (camera.lastModified != lastCameraUpdate) {
      cameraChanged();
      lastCameraUpdate = camera.lastModified;
    }

    // call virtual render method that asks child class to render into
    // the fbPointer/frame buffer texture....
    render();
    // and display on the screen via opengl texture draw call
    draw();
  }

  vec2i OWLViewer::getMousePos() const
  {
    QPoint p = mapFromGlobal(QCursor::pos());
    vec2i where = {p.x(),p.y()};
    return where;
  }

  void OWLViewer::keyPressEvent(QKeyEvent *event)
  {
    QPoint p = mapFromGlobal(QCursor::pos());
    vec2i where = {p.x(),p.y()};
    const QString keyText = event->text();
    if (keyText.size() == 1) {
      // ugh - this is not very qt like - but makes this OWLViewer
      // implementation behave omre like its GLUT and GLFW
      // counterparts....
      key(keyText.data()[0].toLatin1(),where);
    }
    else
      special(event,where);

    QWidget::keyPressEvent(event);
  }

  void OWLViewer::resizeGL(int width, int height)
  {
    resize({width,height});
  }


  /*! helper function that dumps the current frame buffer in a png
    file of given name */
  void OWLViewer::screenShot(const std::string &fileName)
  {
    const uint32_t *fb
      = (const uint32_t*)fbPointer;

    std::vector<uint32_t> pixels;
    for (int y=0;y<fbSize.y;y++) {
      const uint32_t *line = fb + (fbSize.y-1-y)*fbSize.x;
      for (int x=0;x<fbSize.x;x++) {
        pixels.push_back(line[x] | (0xff << 24));
      }
    }
    stbi_write_png(fileName.c_str(),fbSize.x,fbSize.y,4,
                   pixels.data(),fbSize.x*sizeof(uint32_t));
    std::cout << "#owl.viewer: frame buffer written to " << fileName << std::endl;
  }

  vec2i OWLViewer::getScreenSize()
  {
    QRect rec = QApplication::desktop()->screenGeometry();
    int height = rec.height();
    int width = rec.width();
    return {width, height};
  }

  float computeStableEpsilon(float f)
  {
    return abs(f) * float(1./(1<<21));
  }

  float computeStableEpsilon(const vec3f v)
  {
    return max(max(computeStableEpsilon(v.x),
                   computeStableEpsilon(v.y)),
               computeStableEpsilon(v.z));
  }

  SimpleCamera::SimpleCamera(const Camera &camera)
  {
    auto &easy = *this;
    easy.lens.center = camera.position;
    easy.lens.radius = 0.f;
    easy.lens.du     = camera.frame.vx;
    easy.lens.dv     = camera.frame.vy;

    const float minFocalDistance
      = max(computeStableEpsilon(camera.position),
            computeStableEpsilon(camera.frame.vx));

    /*
      tan(fov/2) = (height/2) / dist
      -> height = 2*tan(fov/2)*dist
    */
    float screen_height
      = 2.f*tanf(camera.fovyInDegrees/2.f * (float)M_PI/180.f)
      * max(minFocalDistance,camera.focalDistance);
    easy.screen.vertical   = screen_height * camera.frame.vy;
    easy.screen.horizontal = screen_height * camera.aspect * camera.frame.vx;
    easy.screen.lower_left
      = //easy.lens.center
        /* NEGATIVE z axis! */
      - max(minFocalDistance,camera.focalDistance) * camera.frame.vz
      - 0.5f * easy.screen.vertical
      - 0.5f * easy.screen.horizontal;
    // easy.lastModified = getCurrentTime();
  }

  // ==================================================================
  // actual viewerwidget class
  // ==================================================================

  void OWLViewer::resize(const vec2i &newSize)
  {
    // glfwMakeContextCurrent(handle);
#if CUTEEOWL_USE_CUDA
    if (fbPointer)
      cudaFree(fbPointer);
    cudaMallocManaged(&fbPointer,newSize.x*newSize.y*sizeof(uint32_t));
#else
# pragma message("not building with CUDA toolkit, frame buffer allocated in host memory!")
    if (fbPointer)
      delete[] fbPointer;
    fbPointer = new uint32_t[newSize.x*newSize.y];
#endif

    fbSize = newSize;
    // bool firstResize = false;
    if (fbTexture == 0) {
      GL_CHECK(glGenTextures(1, &fbTexture));
      // firstResize = true;
    }
    else {
#if CUTEEOWL_USE_CUDA
      if (cuDisplayTexture) {
        cudaGraphicsUnregisterResource(cuDisplayTexture);
        cuDisplayTexture = 0;
      }
#endif
    }

    GL_CHECK(glBindTexture(GL_TEXTURE_2D, fbTexture));
    // GL_CHECK(glActiveTexture(GL_TEXTURE0));
    GL_CHECK(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, newSize.x, newSize.y, 0, GL_RGBA,
                          GL_UNSIGNED_BYTE, nullptr));

#if CUTEEOWL_USE_CUDA
    // We need to re-register when resizing the texture
    cudaError_t rc = cudaGraphicsGLRegisterImage
      (&cuDisplayTexture, fbTexture, GL_TEXTURE_2D, 0);
#endif

    // if (firstResize || !firstResize && resourceSharingSuccessful) {
    //   const char *forceSlowDisplay = getenv("OWL_NO_CUDA_RESOURCE_SHARING");
#if CUTEEOWL_FORCE_SLOW_DISPLAY
# pragma message("forcing slow display in qt-owl viewer!")
    bool forceSlowDisplay = true;
#else
    bool forceSlowDisplay = false;
#endif

#if CUTEEOWL_USE_CUDA
    if (rc != cudaSuccess || forceSlowDisplay) {
      std::cout << OWL_TERMINAL_RED
                << "Warning: Could not do CUDA graphics resource sharing "
                << "for the display buffer texture ("
                << cudaGetErrorString(cudaGetLastError())
                << ")... falling back to slower path"
                << OWL_TERMINAL_DEFAULT
                << std::endl;
      resourceSharingSuccessful = false;
      if (cuDisplayTexture) {
        cudaGraphicsUnregisterResource(cuDisplayTexture);
        cuDisplayTexture = 0;
      }
    } else {
      resourceSharingSuccessful = true;
    }
#else
    resourceSharingSuccessful = false;
#endif
    setAspect(fbSize.x/float(fbSize.y));
  }


  /*! re-draw the current frame. This function itself isn't
    virtual, but it calls the framebuffer's render(), which
    is */
  void OWLViewer::draw()
  {
    if (resourceSharingSuccessful) {
#if CUTEEOWL_USE_CUDA
      GL_CHECK(cudaGraphicsMapResources(1, &cuDisplayTexture));

      cudaArray_t array;
      GL_CHECK(cudaGraphicsSubResourceGetMappedArray(&array, cuDisplayTexture, 0, 0));
      {
        cudaMemcpy2DToArray(array,
                            0,
                            0,
                            reinterpret_cast<const void *>(fbPointer),
                            fbSize.x * sizeof(uint32_t),
                            fbSize.x * sizeof(uint32_t),
                            fbSize.y,
                            cudaMemcpyDeviceToDevice);
      }
#endif
    } else {
      GL_CHECK(glBindTexture(GL_TEXTURE_2D, fbTexture));
      glEnable(GL_TEXTURE_2D);
      GL_CHECK(glTexSubImage2D(GL_TEXTURE_2D,0,
                               0,0,
                               fbSize.x, fbSize.y,
                               GL_RGBA, GL_UNSIGNED_BYTE, fbPointer));
    }

    glDisable(GL_LIGHTING);
    glColor3f(1, 1, 1);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, fbTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glDisable(GL_DEPTH_TEST);

    glViewport(0, 0, fbSize.x, fbSize.y);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.f, (float)fbSize.x, 0.f, (float)fbSize.y, -1.f, 1.f);

    glBegin(GL_QUADS);
    {
      glTexCoord2f(0.f, 0.f);
      glVertex3f(0.f, 0.f, 0.f);

      glTexCoord2f(0.f, 1.f);
      glVertex3f(0.f, (float)fbSize.y, 0.f);

      glTexCoord2f(1.f, 1.f);
      glVertex3f((float)fbSize.x, (float)fbSize.y, 0.f);

      glTexCoord2f(1.f, 0.f);
      glVertex3f((float)fbSize.x, 0.f, 0.f);
    }
    glEnd();
#if CUTEEOWL_USE_CUDA
    if (resourceSharingSuccessful) {
      GL_CHECK(cudaGraphicsUnmapResources(1, &cuDisplayTexture));
    }
#endif
  }

  /*! re-computes the 'camera' from the 'cameracontrol', and notify
    app that the camera got changed */
  void OWLViewer::updateCamera()
  {
    // camera.digestInto(simpleCamera);
    // if (isActive)
    camera.lastModified = getCurrentTime();
  }

  void OWLViewer::enableInspectMode(RotateMode rm,
                                    const box3f &validPoiRange,
                                    float minPoiDist,
                                    float maxPoiDist)
  {
    inspectModeManipulator
      = std::make_shared<CameraInspectMode>
      (this,validPoiRange,minPoiDist,maxPoiDist,
       rm==POI? CameraInspectMode::POI: CameraInspectMode::Arcball);
    cameraManipulator = inspectModeManipulator;
  }

  void OWLViewer::enableInspectMode(const box3f &validPoiRange,
                                    float minPoiDist,
                                    float maxPoiDist)
  {
    enableInspectMode(POI,validPoiRange,minPoiDist,maxPoiDist);
  }

  void OWLViewer::enableFlyMode()
  {
    flyModeManipulator
      = std::make_shared<CameraFlyMode>(this);
    cameraManipulator = flyModeManipulator;
  }

  /*! this gets called when the window determines that the mouse got
    _moved_ to the given position */
  void OWLViewer::mouseMotion(const vec2i &newMousePosition)
  {
    if (lastMousePosition == newMousePosition) return;
    
    if (lastMousePosition != vec2i(-1)) {
      if (leftButton.isPressed && leftButton.ctrlWhenPressed)
        // let left-plus-ctrl emulate center
        mouseDragCenter(newMousePosition,newMousePosition-lastMousePosition);
      else if (leftButton.isPressed)
        mouseDragLeft  (newMousePosition,newMousePosition-lastMousePosition);
      else if (centerButton.isPressed)
        mouseDragCenter(newMousePosition,newMousePosition-lastMousePosition);
      else if (rightButton.isPressed)
        mouseDragRight (newMousePosition,newMousePosition-lastMousePosition);
    }
    lastMousePosition = newMousePosition;
  }

  void OWLViewer::mouseDragLeft  (const vec2i &where, const vec2i &delta)
  {
    if (cameraManipulator) cameraManipulator->mouseDragLeft(where,delta);
  }

  /*! mouse got dragged with left button pressedn, by 'delta' pixels,
      at last position where */
  void OWLViewer::mouseDragCenter(const vec2i &where, const vec2i &delta)
  {
    if (cameraManipulator) cameraManipulator->mouseDragCenter(where,delta);
  }

  /*! mouse got dragged with left button pressedn, by 'delta' pixels,
      at last position where */
  void OWLViewer::mouseDragRight (const vec2i &where, const vec2i &delta)
  {
    if (cameraManipulator) cameraManipulator->mouseDragRight(where,delta);
  }

  /*! mouse button got either pressed or released at given location */
  void OWLViewer::mouseButtonLeft  (const vec2i &where, bool pressed)
  {
    if (cameraManipulator) cameraManipulator->mouseButtonLeft(where,pressed);

    lastMousePosition = where;
  }

  /*! mouse button got either pressed or released at given location */
  void OWLViewer::mouseButtonCenter(const vec2i &where, bool pressed)
  {
    if (cameraManipulator) cameraManipulator->mouseButtonCenter(where,pressed);

    lastMousePosition = where;
  }

  /*! mouse button got either pressed or released at given location */
  void OWLViewer::mouseButtonRight (const vec2i &where, bool pressed)
  {
    if (cameraManipulator) cameraManipulator->mouseButtonRight(where,pressed);

    lastMousePosition = where;
  }

  /*! this gets called when the user presses a key on the keyboard ... */
  void OWLViewer::key(char key, const vec2i &where)
  {
    if (cameraManipulator) cameraManipulator->key(key,where);
  }



  /*! this gets called when the user presses a key on the keyboard ... */
  void OWLViewer::special(QKeyEvent *key, const vec2i &where)
  {
    if (cameraManipulator) cameraManipulator->special(key,where);
  }


  void OWLViewer::mousePressEvent(QMouseEvent *event)
  {
    const bool pressed = true;//(action == GLFW_PRESS);
    lastMousePos = getMousePos();
        //Do stuff

    switch(event->button()) {
    case Qt::LeftButton://GLFW_MOUSE_BUTTON_LEFT:
      leftButton.isPressed        = pressed;
      leftButton.shiftWhenPressed
        = QGuiApplication::keyboardModifiers().testFlag(Qt::ShiftModifier);
      leftButton.ctrlWhenPressed
        = QGuiApplication::keyboardModifiers().testFlag(Qt::ControlModifier);
      // leftButton.shiftWhenPressed = (mods & GLFW_MOD_SHIFT  );
      // leftButton.ctrlWhenPressed  = (mods & GLFW_MOD_CONTROL);
      // leftButton.altWhenPressed   = (mods & GLFW_MOD_ALT    );
      mouseButtonLeft(lastMousePos, pressed);
      break;
    case Qt::MidButton://GLFW_MOUSE_BUTTON_MIDDLE:
      centerButton.isPressed = pressed;
      // centerButton.shiftWhenPressed = (mods & GLFW_MOD_SHIFT  );
      // centerButton.ctrlWhenPressed  = (mods & GLFW_MOD_CONTROL);
      // centerButton.altWhenPressed   = (mods & GLFW_MOD_ALT    );
      mouseButtonCenter(lastMousePos, pressed);
      break;
    case Qt::RightButton://GLFW_MOUSE_BUTTON_RIGHT:
      rightButton.isPressed = pressed;
      // rightButton.shiftWhenPressed = (mods & GLFW_MOD_SHIFT  );
      // rightButton.ctrlWhenPressed  = (mods & GLFW_MOD_CONTROL);
      // rightButton.altWhenPressed   = (mods & GLFW_MOD_ALT    );
      mouseButtonRight(lastMousePos, pressed);
      break;
    }
    // // lastPos = event->pos();
  }

  void OWLViewer::mouseMoveEvent(QMouseEvent *event)
  {
    mouseMotion({event->x(),event->y()});
  }

  void OWLViewer::mouseReleaseEvent(QMouseEvent *event)
  {
    const bool pressed = false;//(action == GLFW_PRESS);
    lastMousePos = getMousePos();
    switch(event->button()) {
    case Qt::LeftButton://GLFW_MOUSE_BUTTON_LEFT:
      leftButton.isPressed        = pressed;
      // leftButton.shiftWhenPressed = (mods & GLFW_MOD_SHIFT  );
      // leftButton.ctrlWhenPressed  = (mods & GLFW_MOD_CONTROL);
      // leftButton.altWhenPressed   = (mods & GLFW_MOD_ALT    );
      mouseButtonLeft(lastMousePos, pressed);
      break;
    case Qt::MidButton://GLFW_MOUSE_BUTTON_MIDDLE:
      centerButton.isPressed = pressed;
      // centerButton.shiftWhenPressed = (mods & GLFW_MOD_SHIFT  );
      // centerButton.ctrlWhenPressed  = (mods & GLFW_MOD_CONTROL);
      // centerButton.altWhenPressed   = (mods & GLFW_MOD_ALT    );
      mouseButtonCenter(lastMousePos, pressed);
      break;
    case Qt::RightButton://GLFW_MOUSE_BUTTON_RIGHT:
      rightButton.isPressed = pressed;
      // rightButton.shiftWhenPressed = (mods & GLFW_MOD_SHIFT  );
      // rightButton.ctrlWhenPressed  = (mods & GLFW_MOD_CONTROL);
      // rightButton.altWhenPressed   = (mods & GLFW_MOD_ALT    );
      mouseButtonRight(lastMousePos, pressed);
      break;
    }
    // // lastPos = event->pos();
    // const bool pressed = false;//true;//(action == GLFW_PRESS);
    //   lastMousePos = getMousePos();
    //   switch(button) {
    //   case GLFW_MOUSE_BUTTON_LEFT:
    //     leftButton.isPressed        = pressed;
    //     leftButton.shiftWhenPressed = (mods & GLFW_MOD_SHIFT  );
    //     leftButton.ctrlWhenPressed  = (mods & GLFW_MOD_CONTROL);
    //     leftButton.altWhenPressed   = (mods & GLFW_MOD_ALT    );
    //     mouseButtonLeft(lastMousePos, pressed);
    //     break;
    //   case GLFW_MOUSE_BUTTON_MIDDLE:
    //     centerButton.isPressed = pressed;
    //     centerButton.shiftWhenPressed = (mods & GLFW_MOD_SHIFT  );
    //     centerButton.ctrlWhenPressed  = (mods & GLFW_MOD_CONTROL);
    //     centerButton.altWhenPressed   = (mods & GLFW_MOD_ALT    );
    //     mouseButtonCenter(lastMousePos, pressed);
    //     break;
    //   case GLFW_MOUSE_BUTTON_RIGHT:
    //     rightButton.isPressed = pressed;
    //     rightButton.shiftWhenPressed = (mods & GLFW_MOD_SHIFT  );
    //     rightButton.ctrlWhenPressed  = (mods & GLFW_MOD_CONTROL);
    //     rightButton.altWhenPressed   = (mods & GLFW_MOD_ALT    );
    //     mouseButtonRight(lastMousePos, pressed);
    //     break;
    //   }
    emit clicked();
  }

  /*! set a new orientation for the camera, update the camera, and
    notify the app */
  void OWLViewer::setCameraOrientation(/* camera origin    : */const vec3f &origin,
                                       /* point of interest: */const vec3f &interest,
                                       /* up-vector        : */const vec3f &up,
                                       /* fovy, in degrees : */float fovyInDegrees)
  {
    camera.setOrientation(origin,interest,up,fovyInDegrees,false);
    updateCamera();
  }

  void OWLViewer::getCameraOrientation(/* camera origin    : */vec3f &origin,
                                       /* point of interest: */vec3f &interest,
                                       /* up-vector        : */vec3f &up,
                                       /* fovy, in degrees : */float & fovyInDegrees)
  {
    origin = camera.position;
    interest = -camera.poiDistance * camera.frame.vz + camera.position;
    up = camera.upVector;
    fovyInDegrees = camera.fovyInDegrees;
  }

}
