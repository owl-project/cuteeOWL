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

#include "qtOWL/ColorMaps.h"

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
// #include <QOpenGLBuffer>
// #include <QTimer>

// #include <QLabel>
// #include <QComboBox>
// #include <QVBoxLayout>
// #include <QFormLayout>
// #include <QDoubleSpinBox>

QT_FORWARD_DECLARE_CLASS(QOpenGLShaderProgram);
QT_FORWARD_DECLARE_CLASS(QOpenGLTexture)

namespace qtOWL {

  using namespace owl;
  using namespace owl::common;

  typedef interval<float> range1f;
  
  /*! a widget that displays - and allows to draw into the alpha
      channel of - a "color+alpha" transfer function. The widget
      consists of two halves - the upper 80% displays the current
      transfer function using a graph (with color in x from the color
      channels, and height in y from the alpha channel); the lower 20%
      can optionally display a histogram of corresponding values. 
    
      Editing the alpha channel works by drawing with the mouse into
      it, using either left or right mouse button: left button sets
      alpha value at given cursor pos to where teh curor is in the
      window; right button sets alpha at that x positoin to either 0.f
      (if in lower two-thirds of widget), or 1.f (if in upper third).
  */
  class AlphaEditor : public QOpenGLWidget, protected QOpenGLFunctions
  {
    Q_OBJECT

    friend class XFEditor;
    
  public:
    using QOpenGLWidget::QOpenGLWidget;

    AlphaEditor(const ColorMap &cm);
    ~AlphaEditor();

    typedef enum { KEEP_ALPHA, OVERWRITE_ALPHA } SetColorMapModeMode;
    
    void setColorMap(const ColorMap &cm, SetColorMapModeMode mode);
    
    /*! return the current color map */
    const ColorMap &getColorMap() const { return colorMap; }

    // ------------------------------------------------------------------
    // qt stuff
    // ------------------------------------------------------------------
    QSize minimumSizeHint() const override;
    QSize sizeHint() const override;
    
  signals:
    void colorMapChanged(qtOWL::AlphaEditor *);

  protected:
    void initializeGL() override;
    void paintGL() override;
    void resizeGL(int width, int height) override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;

  private:
    void drawLastToCurrent(vec2i to);
    void drawIntoAlpha(vec2f from,
                       vec2f to,
                       bool set);
    void drawIntoAlphaAbsolute(const vec2i &from,
                               const vec2i &to,
                               bool set);

    /*! the color map we're currently editing */
    ColorMap colorMap;
    
    int width, height;
    bool leftButtonPressed = 0;
    bool rightButtonPressed = 0;
    vec2i lastPos;

    /*! height of histogram area, relative to total widget height */
    const float histogramHeight = .2f;
    
    /*! histogram of scalar values corresponding to the scalar range
        for which we draw the transfer function. histogram values are
        absolute (we will normalize in drawing); histogram may be
        empty, in which case it gets ignored. histogram does not have
        to have same x resolution as transfer functoin */
    std::vector<float> histogram;
  };

}
