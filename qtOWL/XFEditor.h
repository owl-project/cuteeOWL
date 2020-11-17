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

#include "qtOWL/AlphaEditor.h"

#include <QFormLayout>
#include <QComboBox>
#include <QDoubleSpinBox>

// #include <QOpenGLWidget>
// #include <QOpenGLFunctions>
// #include <QOpenGLBuffer>
// #include <QTimer>

// #include <QLabel>
// #include <QVBoxLayout>

QT_FORWARD_DECLARE_CLASS(QOpenGLShaderProgram);
QT_FORWARD_DECLARE_CLASS(QOpenGLTexture)

namespace qtOWL {

  using namespace owl;
  using namespace owl::common;

  struct XFEditor;
  
  typedef interval<float> range1f;

  /*! main transfer function editor widget: a widget with an alpha
      editor, and smaller widgets to set lower and upper bound, and
      opacity scale */
  struct XFEditor : public QWidget
  {
    Q_OBJECT

  public:
    XFEditor();

    const ColorMap &getColorMap() const;
    
  signals:
    /*! the color map (in the alpha editor) got changed; either
        because the user drew in it, or selected anothe ron from the
        drobox, etc */
    void colorMapChanged(qtOWL::XFEditor *);
    void opacityScaleChanged(double);
                                     
  public slots:
    /*! we'll have the qcombobox that selsects the desired color map
      call this, and then update the alpha editor */
    void cmSelectionChanged(int idx);

    /*! the alpha editor child widget changed something to the color
        map*/
    void alphaEditorChanged(qtOWL::AlphaEditor *ae);
    
  private:
    ColorMapLibrary colorMaps;
    AlphaEditor    *alphaEditor;
    QFormLayout    *layout;
    QComboBox      *cmSelector;
    QDoubleSpinBox *domain_lower;
    QDoubleSpinBox *domain_upper;
    QDoubleSpinBox *opacityScaleSpinBox;
  };

}
