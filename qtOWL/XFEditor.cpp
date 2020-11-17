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

  XFEditor::XFEditor()
  {
    QGridLayout *gridLayout = new QGridLayout;//(2,2);
      
    domain_lower = new QDoubleSpinBox;
    domain_upper = new QDoubleSpinBox;

    range1f domain(0.f,1.f);
    domain_lower->setValue(domain.lower);
    domain_lower->setRange(domain.lower,domain.upper);
    domain_lower->setSingleStep((domain.upper-domain.lower)/40.f);
    gridLayout->addWidget(new QLabel("lower"),0,0);
    gridLayout->addWidget(domain_lower,0,1);

    domain_upper->setValue(domain.upper);
    domain_upper->setRange(domain.lower,domain.upper);
    domain_upper->setSingleStep((domain.upper-domain.lower)/40.f);
    gridLayout->addWidget(new QLabel("upper"),1,0);
    gridLayout->addWidget(domain_upper,1,1);

    opacityScaleSpinBox = new QDoubleSpinBox;
    opacityScaleSpinBox->setDecimals(3);
    opacityScaleSpinBox->setValue(1.f);
    opacityScaleSpinBox->setRange(0.f,1.f);
    opacityScaleSpinBox->setSingleStep(.03f);
    gridLayout->addWidget(new QLabel("opacity scale"),2,0);
    gridLayout->addWidget(opacityScaleSpinBox,2,1);
    
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
    connect(domain_lower, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &qtOWL::XFEditor::emitRangeChanged);
    connect(domain_upper, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &qtOWL::XFEditor::emitRangeChanged);
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

  /*! one of the range spin boxes' value changed */
  void XFEditor::emitRangeChanged(double value)
  {
    if (sender() == domain_lower)
      emit rangeChanged({value,domain_upper->value()});
    else if (sender() == domain_upper)
      emit rangeChanged({domain_lower->value(),value});
    else
      assert(0);
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
    domain_lower->setValue(floatVal);
    in.read((char*)&floatVal,sizeof(floatVal));
    domain_upper->setValue(floatVal);
    
    ColorMap colorMap;
    int numColorMapValues;
    in.read((char*)&numColorMapValues,sizeof(numColorMapValues));
    colorMap.resize(numColorMapValues);
    in.read((char*)colorMap.data(),colorMap.size()*sizeof(colorMap[0]));
    alphaEditor->setColorMap(colorMap,AlphaEditor::OVERWRITE_ALPHA);
    std::cout << "loaded xf from " << fileName << std::endl;
  }
  
  
  /*! dump entire transfer function to file */
  void XFEditor::saveTo(const std::string &fileName)
  {
    std::ofstream out(fileName,std::ios::binary);
    out.write((char*)&xfFileFormatMagic,sizeof(xfFileFormatMagic));

    float floatVal;
    floatVal = opacityScaleSpinBox->value();
    out.write((char*)&floatVal,sizeof(floatVal));
    floatVal = domain_lower->value();
    out.write((char*)&floatVal,sizeof(floatVal));
    floatVal = domain_upper->value();
    out.write((char*)&floatVal,sizeof(floatVal));
    
    const auto &colorMap = getColorMap();
    int numColorMapValues = colorMap.size();
    out.write((char*)&numColorMapValues,sizeof(numColorMapValues));
    out.write((char*)colorMap.data(),colorMap.size()*sizeof(colorMap[0]));
    std::cout << "#qtOWL.XFEditor: saved transfer function to "
              <<  fileName << std::endl;
  }
  
}
