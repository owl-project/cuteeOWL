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

#include "owl/common/math/box.h"
#include "owl/common/math/LinearSpace.h"
#include <map>
#include <vector>
#include <string>

namespace owlQT {

  using namespace owl;
  using namespace owl::common;

  /*! convert PNG rgb image into a color map; if PNG doesn't have an
      alpha channel, alpha channel will be set to constant 1.f */
  struct ColorMap : public std::vector<vec4f>
  {
    static ColorMap fromPNG(const uint8_t asPNG[],
                            size_t numBytes);
  };

  ColorMap resample(const ColorMap &cm, int numSamples);


  
  /*! class that abstracts a library of different, predefined color
      maps that are embedded in this library */
  struct ColorMapLibrary
  {
    /*! return a color map library of hardcoded, embedded color
      maps */
    static ColorMapLibrary loadDefaults();
    
    /*! returns a std::vector with all the names of color maps known
      in this library */
    std::vector<std::string> names() const;

    /*! find colro map with given name, and reutrn pointer to it; or
        null if this does not exist */
    const ColorMap *find(const std::string &name) const;

    /*! list of color maps stored in this library */
    std::vector<std::pair<std::string,ColorMap>> knownMaps;
  };

}

