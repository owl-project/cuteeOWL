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

#include <QApplication>
#include <QPushButton>
#include <QMainWindow>
#include <QLabel>
#include <QVBoxLayout>
#include <QFormLayout>
#include "QComboBox"
// BorderLayout Sample

#include "qtOWL/OWLViewer.h"
#include "qtOWL/XFEditor.h"

#include <cuda_runtime.h>
//#include <cuda_gl_interop.h>


namespace qtOWL {
  struct SampleViewer : public OWLViewer{

    void render() override
    {
      /* compute a simple test pattern ..... of course, we _should_ be
         doin that in cuda on the device, but for now let's keep the
         camkefile simple and just do it on the host, then copy to
         device FB */
      static int g_frameID = 0;
      int frameID = g_frameID++;

      std::vector<int> hostFB(fbSize.x*fbSize.y);
      for (int iy=0;iy<fbSize.y;iy++)
        for (int ix=0;ix<fbSize.x;ix++) {
          int r = (ix+frameID)%256;
          int g = (iy+frameID)%256;
          int b = (ix+iy+frameID)%256;
          int rgba
            = (r<<0)
            | (g<<8)
            | (b<<16)
            | (255<<24);
          hostFB[fbSize.x*iy+ix] = rgba;
        }
      cudaMemcpy(fbPointer,hostFB.data(),hostFB.size()*sizeof(int),cudaMemcpyDefault);
    }

    void cameraChanged() override
    {
      const SimpleCamera camera = getSimplifiedCamera();
    }

  };


  extern "C" int main(int argc, char **argv)
  {
    QApplication app(argc,argv);

    SampleViewer viewer;
    viewer.show();
    const box3f worldBounds(vec3f(-1),vec3f(+1));
    viewer.enableFlyMode();
    viewer.enableInspectMode(worldBounds);


    XFEditor    *xfEditor = new XFEditor;
    QFormLayout *layout   = new QFormLayout;
    layout->addWidget(xfEditor);

    // Set QWidget as the central layout of the main window
    QMainWindow secondWindow;
    secondWindow.setCentralWidget(xfEditor);

    secondWindow.show();

    return app.exec();
  }

}

