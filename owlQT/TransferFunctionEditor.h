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

#include "ColorMaps.h"

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QOpenGLBuffer>
#include <QTimer>

#include <QLabel>
#include <QComboBox>
#include <QVBoxLayout>
#include <QFormLayout>
#include <QDoubleSpinBox>

QT_FORWARD_DECLARE_CLASS(QOpenGLShaderProgram);
QT_FORWARD_DECLARE_CLASS(QOpenGLTexture)

namespace owlQT {

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

  public:
    using QOpenGLWidget::QOpenGLWidget;

    AlphaEditor(const ColorMap &cm);
    ~AlphaEditor();

    typedef enum { KEEP_ALPHA, OVERWRITE_ALPHA } SetColorMapModeMode;
    
    void setColorMap(const ColorMap &cm, SetColorMapModeMode mode);


    // ------------------------------------------------------------------
    // qt stuff
    // ------------------------------------------------------------------
    QSize minimumSizeHint() const override;
    QSize sizeHint() const override;
    
  signals:
    void colorMapChanged();
    // void clicked();

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

      
    std::vector<vec4f> xf;
    // QColor clearColor = Qt::black;
    // QPoint lastPos;
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



  struct XFEditor : public QWidget
  {
    Q_OBJECT

  public:
    XFEditor()
    {
      QGridLayout *rangeLayout = new QGridLayout;//(2,2);
      
      domain_lower = new QDoubleSpinBox;
      domain_upper = new QDoubleSpinBox;

      range1f domain(0.f,1.f);
      domain_lower->setValue(domain.lower);
      domain_lower->setRange(domain.lower,domain.upper);
      domain_lower->setSingleStep((domain.upper-domain.lower)/40.f);

      domain_upper->setValue(domain.upper);
      domain_upper->setRange(domain.lower,domain.upper);
      domain_upper->setSingleStep((domain.upper-domain.lower)/40.f);

      rangeLayout->addWidget(new QLabel("upper"),0,0);
      rangeLayout->addWidget(domain_lower,1,0);
      rangeLayout->addWidget(new QLabel("upper"),0,1);
      rangeLayout->addWidget(domain_upper,1,1);
      QWidget *rangeWidget = new QWidget;
      rangeWidget->setLayout(rangeLayout);
      
      alphaEditor = new AlphaEditor(colorMaps.getMap(0));
      
      cmSelector = new QComboBox;
      for (auto cmName : colorMaps.getNames())
        cmSelector->addItem(QString(cmName.c_str()));

      layout = new QFormLayout;
      layout->addWidget(alphaEditor);
      layout->addWidget(cmSelector);

      
      layout->addWidget(rangeWidget);
      setLayout(layout);
      
      // connect(&timer, SIGNAL(timeout()), this, SLOT(update()));
      connect(cmSelector, SIGNAL(currentIndexChanged(int)),
              this, SLOT(cmSelectionChanged(int)));
    }

  public slots:
    /*! we'll have the qcombobox that selsects the desired color map
      call this, and then update the alpha editor */
    void cmSelectionChanged(int idx)
    {
      alphaEditor->setColorMap(colorMaps.getMap(idx),AlphaEditor::KEEP_ALPHA);
    }
    
  signals:
    /*! this gets emitted every time the app 'might' want to check
        that the color map got changed (either by drawing into, or by
        selecting a new color map */
    void colorMapChanged();

  private:
    AlphaEditor *alphaEditor;
    QFormLayout *layout;
    ColorMapLibrary colorMaps;
    QComboBox *cmSelector;
    QDoubleSpinBox *domain_lower, *domain_upper;
  };

}
