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

#include "qtOWL/XFEditor.h"
#include <QLabel>
#include <QtGlobal>

namespace qtOWL {

  XFEditor::XFEditor()
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

    QVBoxLayout *opacityScaleLayout = new QVBoxLayout;

    opacityScaleSpinBox = new QDoubleSpinBox;
    opacityScaleSpinBox->setDecimals(3);
    opacityScaleSpinBox->setValue(1.);
    opacityScaleSpinBox->setRange(0.,1.);
    opacityScaleSpinBox->setSingleStep(.03);
    opacityScaleLayout->addWidget(new QLabel("Opacity scale"),0,0);
    opacityScaleLayout->addWidget(opacityScaleSpinBox);
    QWidget *opacityScaleWidget = new QWidget;
    opacityScaleWidget->setLayout(opacityScaleLayout);

    layout = new QFormLayout;
    layout->addWidget(alphaEditor);
    layout->addWidget(cmSelector);

      
    layout->addWidget(opacityScaleWidget);
    layout->addWidget(rangeWidget);
    setLayout(layout);
      
    // connect(&timer, SIGNAL(timeout()), this, SLOT(update()));
    connect(cmSelector, SIGNAL(currentIndexChanged(int)),
            this, SLOT(cmSelectionChanged(int)));
    connect(alphaEditor, SIGNAL(colorMapChanged(qtOWL::AlphaEditor*)),
            this, SLOT(alphaEditorChanged(qtOWL::AlphaEditor*)));
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
  
  const ColorMap &XFEditor::getColorMap() const
  {
    return alphaEditor->getColorMap();
  }
    
}
