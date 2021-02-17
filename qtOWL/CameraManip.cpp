// ======================================================================== //
// Copyright 2018-2020 Ingo Wald                                            //
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

#include "CameraManip.h"
#include "OWLViewer.h"

namespace qtOWL {

  /*! this gets called when the user presses a key on the keyboard ... */
  void CameraManipulator::key(char key, const vec2i &where)
  {
    // int key = event->key();
    Camera &fc = viewer->camera;

    switch(key) {
    case 'f':
    case 'F':
      if (viewer->flyModeManipulator)
        viewer->cameraManipulator = viewer->flyModeManipulator;
      break;
    case 'i':
    case 'I':
      if (viewer->inspectModeManipulator)
        viewer->cameraManipulator = viewer->inspectModeManipulator;
      break;
    case '+':
    case '=':
      fc.motionSpeed *= 2.f;
      std::cout << "# viewer: new motion speed is " << fc.motionSpeed << std::endl;
      break;
    case '-':
    case '_':
      fc.motionSpeed /= 2.f;
      std::cout << "# viewer: new motion speed is " << fc.motionSpeed << std::endl;
      break;
    case 'C': {
      std::cout << "(C)urrent camera:" << std::endl;
      std::cout << "- from :" << fc.position << std::endl;
      std::cout << "- poi  :" << fc.getPOI() << std::endl;
      std::cout << "- upVec:" << fc.upVector << std::endl;
      std::cout << "- frame:" << fc.frame << std::endl;

      const vec3f vp = fc.position;
      const vec3f vi = fc.getPOI();
      const vec3f vu = fc.upVector;
      const float fovy = fc.getFovyInDegrees();
      std::cout << "(suggested cmdline format, for apps that support this:) "
                << std::endl
                << " --camera"
                << " " << vp.x << " " << vp.y << " " << vp.z
                << " " << vi.x << " " << vi.y << " " << vi.z
                << " " << vu.x << " " << vu.y << " " << vu.z
                << " -fovy " << fovy
                << std::endl;
    } break;
    case 'x':
    case 'X':
      fc.setUpVector(fc.upVector==vec3f(1,0,0)?vec3f(-1,0,0):vec3f(1,0,0));
      viewer->updateCamera();
      break;
    case 'y':
    case 'Y':
      fc.setUpVector(fc.upVector==vec3f(0,1,0)?vec3f(0,-1,0):vec3f(0,1,0));
      viewer->updateCamera();
      break;
    case 'z':
    case 'Z':
      fc.setUpVector(fc.upVector==vec3f(0,0,1)?vec3f(0,0,-1):vec3f(0,0,1));
      viewer->updateCamera();
      break;
    default:
      break;
    }
  }

} // ::owlQT

