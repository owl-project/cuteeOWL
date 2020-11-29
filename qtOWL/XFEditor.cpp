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
    abs_domain_lower->setValidator(new QDoubleValidator(domain.lower,domain.upper,0));
    abs_domain_lower->setText(QString::number(domain.lower));
    gridLayout->addWidget(abs_domain_lower,1,1);

    abs_domain_upper = new QLineEdit;
    abs_domain_upper->setValidator(new QDoubleValidator(domain.lower,domain.upper,0));
    abs_domain_upper->setText(QString::number(domain.upper));
    gridLayout->addWidget(abs_domain_upper,1,2);


    // -------------------------------------------------------
    // relative range (within the absolute one)
    // -------------------------------------------------------
    gridLayout->addWidget(new QLabel("domain (rel)"),2,0);

    rel_domain_lower = new QDoubleSpinBox;
    rel_domain_lower->setValue(0.f);
    rel_domain_lower->setDecimals(0);
    rel_domain_lower->setRange(0.f,100.f);
    rel_domain_lower->setSingleStep(1.f);

    rel_domain_upper = new QDoubleSpinBox;
    rel_domain_upper->setValue(100.f);
    rel_domain_upper->setDecimals(0);
    rel_domain_upper->setRange(0.f,100.f);
    rel_domain_upper->setSingleStep(1.f);

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
    opacityScaleSpinBox->setDecimals(3);
    opacityScaleSpinBox->setValue(1.f);
    opacityScaleSpinBox->setRange(0.f,1.f);
    opacityScaleSpinBox->setSingleStep(.03f);
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

  static const size_t xfFileFormatMagic = 0x1235abc000;
  /*! load transfer function written with saveTo() */
  void XFEditor::loadFrom(const std::string &fileName)
  {
    std::ifstream in(fileName,std::ios::binary);
    size_t magic;
    in.read((char*)&magic,sizeof(xfFileFormatMagic));
    if (magic != xfFileFormatMagic)
      throw std::runtime_error(fileName+": not a valid '.xf' "
                               "transfer function file!?");
    float floatVal;
    in.read((char*)&floatVal,sizeof(floatVal));
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
