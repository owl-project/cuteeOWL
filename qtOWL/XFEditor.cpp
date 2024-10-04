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

#include <cassert>
#include "qtOWL/XFEditor.h"
#include <QLabel>
#include <QtGlobal>
#include <fstream>

namespace qtOWL {

  XFEditor::XFEditor() : XFEditor(range1f(0.f, 1.f)) {}

  XFEditor::XFEditor(const range1f &domain)
    : dataValueRange(domain)
  {
    QGridLayout *gridLayout = new QGridLayout;//(2,2);

    // -------------------------------------------------------
    // general info:
    // -------------------------------------------------------
    std::string infoText
      = "value range of data (absolute) ["
      +std::to_string(domain.lower)
      +".."
      +std::to_string(domain.upper)
      +"]";
    gridLayout->addWidget(new QLabel(QString(infoText.c_str())),0,0,1,3);
    gridLayout->addWidget(new QLabel("domain (abs)"),1,0);

    // -------------------------------------------------------
    // absolute xf domain (editable, no spinners, absolute values)
    // -------------------------------------------------------
    abs_domain_lower = new QLineEdit;
    abs_domain_lower->setValidator(new QDoubleValidator(domain.lower,domain.upper,3));
    abs_domain_lower->setText(QString::number(domain.lower));
    gridLayout->addWidget(abs_domain_lower,1,1);

    abs_domain_upper = new QLineEdit;
    abs_domain_upper->setValidator(new QDoubleValidator(domain.lower,domain.upper,3));
    abs_domain_upper->setText(QString::number(domain.upper));
    gridLayout->addWidget(abs_domain_upper,1,2);


    // -------------------------------------------------------
    // relative range (within the absolute one)
    // -------------------------------------------------------
    gridLayout->addWidget(new QLabel("domain (rel)"),2,0);

    rel_domain_lower = new QDoubleSpinBox;
    rel_domain_lower->setValue(0.f);
    rel_domain_lower->setDecimals(3);
    rel_domain_lower->setRange(-1000.f,1000.f);
    rel_domain_lower->setSingleStep(0.01f);

    rel_domain_upper = new QDoubleSpinBox;
    rel_domain_upper->setValue(100.f);
    rel_domain_upper->setDecimals(3);
    rel_domain_upper->setRange(-1000.f,1000.f);
    rel_domain_upper->setSingleStep(0.01f);

    gridLayout->addWidget(rel_domain_lower,2,1);
    gridLayout->addWidget(rel_domain_upper,2,2);
    // gridLayout->addWidget(new QLabel("lower"),0,0);
    // gridLayout->addWidget(domain_lower,0,1);
    // gridLayout->addWidget(new QLabel("upper"),1,0);
    // gridLayout->addWidget(domain_upper,1,1);

    // -------------------------------------------------------
    // opacity scale
    // -------------------------------------------------------
    opacityScaleSpinBox = new QDoubleSpinBox;
    // opacityScaleSpinBox->setDecimals(3);
    // opacityScaleSpinBox->setValue(1.f);
    // opacityScaleSpinBox->setRange(0.f,1.f);
    // opacityScaleSpinBox->setSingleStep(.03f);
    opacityScaleSpinBox->setDecimals(3);
    opacityScaleSpinBox->setSingleStep(1.f);
    opacityScaleSpinBox->setRange(0.f,200.f);
    opacityScaleSpinBox->setValue(100.f);
    gridLayout->addWidget(new QLabel("opacity scale"),3,0);
    gridLayout->addWidget(opacityScaleSpinBox,3,1);

    
    // opacityScaleLayout->addWidget(new QLabel("Opacity scale"),0,0);
    // opacityScaleLayout->addWidget(opacityScaleSpinBox);
    // QWidget *opacityScaleWidget = new QWidget;
    // opacityScaleWidget->setLayout(opacityScaleLayout);

    QWidget *valuesWidget = new QWidget;
    valuesWidget->setLayout(gridLayout);

    alphaEditor = new AlphaEditor(colorMaps.getMap(0));

    cmSelector = new QComboBox;
    for (auto cmName : colorMaps.getNames())
      cmSelector->addItem(QString(cmName.c_str()));

    QVBoxLayout *opacityScaleLayout = new QVBoxLayout;

    layout = new QFormLayout;
    layout->addWidget(alphaEditor);
    layout->addWidget(cmSelector);
    layout->addWidget(valuesWidget);
    setLayout(layout);

    // connect(&timer, SIGNAL(timeout()), this, SLOT(update()));
    connect(cmSelector, SIGNAL(currentIndexChanged(int)),
            this, SLOT(cmSelectionChanged(int)));
    connect(alphaEditor, SIGNAL(colorMapChanged(qtOWL::AlphaEditor*)),
            this, SLOT(alphaEditorChanged(qtOWL::AlphaEditor*)));
    connect(abs_domain_lower, QOverload<const QString &>::of(&QLineEdit::textChanged),
            this, &qtOWL::XFEditor::emitAbsRangeChanged);
    connect(abs_domain_upper, QOverload<const QString &>::of(&QLineEdit::textChanged),
            this, &qtOWL::XFEditor::emitAbsRangeChanged);
    connect(rel_domain_lower, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &qtOWL::XFEditor::emitRelRangeChanged);
    connect(rel_domain_upper, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &qtOWL::XFEditor::emitRelRangeChanged);
    connect(opacityScaleSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &qtOWL::XFEditor::opacityScaleChanged);
  }

  /*! we'll have the qcombobox that selsects the desired color map
    call this, and then update the alpha editor */
  void XFEditor::cmSelectionChanged(int idx)
  {
    alphaEditor->setColorMap(colorMaps.getMap(idx),
                             AlphaEditor::KEEP_ALPHA);
  }

  /*! the alpha editor child widget changed something to the color
    map*/
  void XFEditor::alphaEditorChanged(AlphaEditor *ae)
  {
    //colorMapChangedCB(&ae->xf);
    emit colorMapChanged(this);
    emit opacityScaleChanged(getOpacityScale());
    emit signal_rangeChanged();
  }

  inline range1f order(const range1f v)
  {
    return { std::min(v.lower,v.upper),std::max(v.lower,v.upper) };
  }

  inline float lerp(const range1f v, const float f)
  {
    return (1.f-f)*v.lower + f*v.upper;
  }

  /*! one of the range spin boxes' value changed */
  void XFEditor::signal_rangeChanged()
  {
    range1f absRange(abs_domain_lower->text().toDouble(),
                     abs_domain_upper->text().toDouble());
    absRange = order(absRange);

    range1f relRange(.01f*rel_domain_lower->value(),
                     .01f*rel_domain_upper->value());
    relRange = order(relRange);

    range1f finalRange(lerp(absRange,relRange.lower),
                       lerp(absRange,relRange.upper));
    emit rangeChanged(finalRange);
  }

  /*! one of the range spin boxes' value changed */
  void XFEditor::emitRelRangeChanged(double)
  {
    signal_rangeChanged();
  }
  /*! one of the range spin boxes' value changed */
  void XFEditor::emitAbsRangeChanged(const QString &)
  {
    signal_rangeChanged();
  }

  const ColorMap &XFEditor::getColorMap() const
  {
    return alphaEditor->getColorMap();
  }

  range1f XFEditor::getAbsDomain() const
  {
    range1f absRange(abs_domain_lower->text().toDouble(),
                     abs_domain_upper->text().toDouble());
    absRange = order(absRange);
    return absRange;
  }
  
  range1f XFEditor::getRelDomain() const
  {
    range1f relRange(rel_domain_lower->text().toDouble(),
                     rel_domain_upper->text().toDouble());
    relRange = order(relRange);
    return relRange;
  }
  
  float   XFEditor::getOpacityScale() const
  {
    return opacityScaleSpinBox->value();
  }

  void XFEditor::setOpacityScale(float v) 
  {
    return opacityScaleSpinBox->setValue(v);
  }

  void XFEditor::setColorMap(const std::vector<vec4f> &cm) 
  {
    ColorMap colorMap;
    for (auto v : cm) colorMap.push_back(v);
    alphaEditor->setColorMap(colorMap,AlphaEditor::OVERWRITE_ALPHA);
  }

  void XFEditor::setAbsDomain(const range1f &r) 
  {
    // range1f absRange(abs_domain_lower->text().toDouble(),
    //                  abs_domain_upper->text().toDouble());
    // absRange = order(absRange);
    abs_domain_lower->setText(QString::number(r.lower));
    abs_domain_upper->setText(QString::number(r.upper));
    signal_rangeChanged();
  }
  
  void XFEditor::setRelDomain(const range1f &r) 
  {
    // range1f relRange(rel_domain_lower->text().toDouble(),
    //                  rel_domain_upper->text().toDouble());
    // relRange = order(relRange);
    rel_domain_lower->setValue(r.lower);
    rel_domain_upper->setValue(r.upper);
    signal_rangeChanged();
  }

  


  static const size_t xfFileFormatMagic = 0x1235abc000;
  /*! load either a transfer function written with saveTo(), or
      whole RGBA maps from a png file */
  void XFEditor::loadFrom(const std::string &fileName)
  {
    std::ifstream in(fileName,std::ios::binary);
    size_t magic;
    in.read((char*)&magic,sizeof(xfFileFormatMagic));
    if (magic != xfFileFormatMagic) {
      // Try loading as png
      try {
        in.close();
        // Determine file size
        in.open(fileName,std::ios::binary|std::ios::ate);
        size_t numBytes = in.tellg();
        // Load whole file content
        in.close();
        std::vector<uint8_t> asPNG(numBytes);
        in.open(fileName,std::ios::binary);
        in.read((char *)asPNG.data(),numBytes);
        ColorMap colorMap = ColorMap::fromPNG(asPNG.data(),numBytes);
        // Add as color map
        colorMaps.addColorMap(colorMap,fileName);
        cmSelector->clear();
        for (auto cmName : colorMaps.getNames())
          cmSelector->addItem(QString(cmName.c_str()));
        // Select the color map, but make sure the
        // combo box's signal won't trigger
        cmSelector->blockSignals(true);
        cmSelector->setCurrentIndex(0);
        cmSelector->blockSignals(false);
        // Add as alpha map (make sure to use the resampled map here!)
        alphaEditor->setColorMap(colorMaps.getMap(0),AlphaEditor::OVERWRITE_ALPHA);
      } catch (...) {
        throw std::runtime_error(fileName+": not a valid '.xf' "
                                 "transfer function file!?");
      }
    } else {
      float floatVal;
      in.read((char*)&floatVal,sizeof(floatVal));
      if (floatVal < 20) {
        std::cout << "=======================================================" << std::endl;
        std::cout <<
          "#cutee: warning - you seem to have loaded a very old\n"
          "#cutee: transfer function .xf file with - according to\n"
          "#cutee: how we currently store xf files - a *very* low\n"
          "#cutee: density scale ... fixing this by trying to move\n"
          "#cutee: this value into a more useful range ...\n"
          "#cutee: ...but might want to consider updating that xf file!\n"
          ;
        std::cout << "=======================================================" << std::endl;
        floatVal = std::max(80.f,std::min(120.f,floatVal*10));
      }

      
      opacityScaleSpinBox->setValue(floatVal);

      in.read((char*)&floatVal,sizeof(floatVal));
      abs_domain_lower->setText(QString::number(floatVal));
      in.read((char*)&floatVal,sizeof(floatVal));
      abs_domain_upper->setText(QString::number(floatVal));

      in.read((char*)&floatVal,sizeof(floatVal));
      rel_domain_lower->setValue(floatVal);
      in.read((char*)&floatVal,sizeof(floatVal));
      rel_domain_upper->setValue(floatVal);

      ColorMap colorMap;
      int numColorMapValues;
      in.read((char*)&numColorMapValues,sizeof(numColorMapValues));
      colorMap.resize(numColorMapValues);
      in.read((char*)colorMap.data(),colorMap.size()*sizeof(colorMap[0]));
      alphaEditor->setColorMap(colorMap,AlphaEditor::OVERWRITE_ALPHA);
    }

    std::cout << "loaded xf from " << fileName << std::endl;
    emit colorMapChanged(this);
  }


  /*! dump entire transfer function to file */
  void XFEditor::saveTo(const std::string &fileName)
  {
    std::ofstream out(fileName,std::ios::binary);
    out.write((char*)&xfFileFormatMagic,sizeof(xfFileFormatMagic));

    float floatVal;
    floatVal = opacityScaleSpinBox->value();
    out.write((char*)&floatVal,sizeof(floatVal));

    floatVal = abs_domain_lower->text().toDouble();
    out.write((char*)&floatVal,sizeof(floatVal));
    floatVal = abs_domain_upper->text().toDouble();
    out.write((char*)&floatVal,sizeof(floatVal));

    floatVal = rel_domain_lower->value();
    out.write((char*)&floatVal,sizeof(floatVal));
    floatVal = rel_domain_upper->value();
    out.write((char*)&floatVal,sizeof(floatVal));

    const auto &colorMap = getColorMap();
    int numColorMapValues = colorMap.size();
    out.write((char*)&numColorMapValues,sizeof(numColorMapValues));
    out.write((char*)colorMap.data(),colorMap.size()*sizeof(colorMap[0]));
    std::cout << "#qtOWL.XFEditor: saved transfer function to "
              <<  fileName << std::endl;
  }

  
}
