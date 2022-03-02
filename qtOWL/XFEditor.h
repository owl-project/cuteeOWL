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
#include <QLineEdit>

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
    /*! construct a new xf editor widget around the given 'domain'
        (ie, range of valid inputs) of a scalar field. */
    XFEditor(const range1f &domain);

    /*! load either a transfer function written with saveTo(), or
        whole RGBA maps from a png file */
    void loadFrom(const std::string &fileName);

    /*! dump entire transfer function to file */
    void saveTo(const std::string &fileName);

  signals:
    /*! the color map (in the alpha editor) got changed; either
        because the user drew in it, or selected anothe ron from the
        drobox, etc */
    void colorMapChanged(qtOWL::XFEditor *);
    void opacityScaleChanged(double);
    void rangeChanged(range1f);

  public slots:
    /*! we'll have the qcombobox that selsects the desired color map
      call this, and then update the alpha editor */
    void cmSelectionChanged(int idx);

    /*! the alpha editor child widget changed something to the color
        map*/
    void alphaEditorChanged(qtOWL::AlphaEditor *ae);

  private slots:
    /*! gets called by the absolute domain qlineedits */
    void emitAbsRangeChanged(const QString &);
    /*! gets called by the relative domain double spinners */
    void emitRelRangeChanged(double);

  public:
    /* both different rangeChanged() signals route to this one
       implementaion, since type doesn't matter */
    void signal_rangeChanged();

    // for saving/restoring all editor fields:
    range1f getAbsDomain() const;
    range1f getRelDomain() const;
    float   getOpacityScale() const;
    const ColorMap &getColorMap() const;

    void setColorMap(const std::vector<vec4f> &cm);
    void setAbsDomain(const range1f &range);
    void setRelDomain(const range1f &range);
    void setOpacityScale(float value);
  private:
    ColorMapLibrary colorMaps;
    AlphaEditor    *alphaEditor;
    QFormLayout    *layout;
    QComboBox      *cmSelector;
    QLineEdit      *abs_domain_lower;
    QLineEdit      *abs_domain_upper;
    QDoubleSpinBox *rel_domain_lower;
    QDoubleSpinBox *rel_domain_upper;
    range1f         dataValueRange{0.f, 1.f};
    QDoubleSpinBox *opacityScaleSpinBox;
  };

}
