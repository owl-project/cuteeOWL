// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
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

#include "Camera.h"

#include <QKeyEvent>

#include <vector>
#include <memory>
#ifdef _GNUC_
#include <unistd.h>
#endif

namespace qtOWL {

  using namespace owl;
  using namespace owl::common;
  
  struct OWLViewer;

  // ------------------------------------------------------------------
  /*! abstract base class that allows to manipulate a renderable
    camera */
  struct CameraManipulator {
    CameraManipulator(OWLViewer *viewer) : viewer(viewer) {}

   /*! this gets called when the user presses a key on the keyboard ... */
    virtual void key(char key, const vec2i &/*where*/);
    

    //  /*! this gets called when the user presses a key on the keyboard ... */
    // virtual void key(QKeyEvent *event);

    /*! this gets called when the user presses a key on the keyboard ... */
     virtual void special(QKeyEvent *event, const vec2i &where) { };

    /*! mouse got dragged with left button pressedn, by 'delta'
      pixels, at last position where */
    virtual void mouseDragLeft  (const vec2i &where, const vec2i &delta) {}

    /*! mouse got dragged with left button pressedn, by 'delta'
      pixels, at last position where */
    virtual void mouseDragCenter(const vec2i &where, const vec2i &delta) {}

    /*! mouse got dragged with left button pressedn, by 'delta'
      pixels, at last position where */
    virtual void mouseDragRight (const vec2i &where, const vec2i &delta) {}

    /*! mouse button got either pressed or released at given location */
    virtual void mouseButtonLeft  (const vec2i &where, bool pressed) {}

    /*! mouse button got either pressed or released at given location */
    virtual void mouseButtonCenter(const vec2i &where, bool pressed) {}

    /*! mouse button got either pressed or released at given location */
    virtual void mouseButtonRight (const vec2i &where, bool pressed) {}

  protected:
    OWLViewer *const viewer;
  };

} // ::owlQT

