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

#include "Camera.h"

namespace qtOWL {

  vec3f Camera::getFrom() const
  {
    return position;
  }
    
  vec3f Camera::getAt() const
  {
    return position - frame.vz;
  }

  vec3f Camera::getUp() const
  {
    return frame.vy;
  }
      
  void Camera::setFovy(const float fovy)
  {
    this->fovyInDegrees = fovy;
  }

  void Camera::setAspect(const float aspect)
  {
    this->aspect = aspect;
  }

  void Camera::setFocalDistance(float focalDistance)
  {
    this->focalDistance = focalDistance;
  }

  /*! tilt the frame around the z axis such that the y axis is "facing upwards" */
  void Camera::forceUpFrame()
  {
    // frame.vz remains unchanged
    if (fabsf(dot(frame.vz,upVector)) < 1e-6f)
      // looking along upvector; not much we can do here ...
      return;
    frame.vx = normalize(cross(upVector,frame.vz));
    frame.vy = normalize(cross(frame.vz,frame.vx));
  }

  void Camera::setOrientation(/* camera origin    : */const vec3f &origin,
                              /* point of interest: */const vec3f &interest,
                              /* up-vector        : */const vec3f &up,
                              /* fovy, in degrees : */float fovyInDegrees,
                              /* set focal dist?  : */bool  setFocalDistance)
  {
    this->fovyInDegrees = fovyInDegrees;
    position = origin;
    upVector = up;
    frame.vz
      = (interest==origin)
      ? vec3f(0,0,1)
      : /* negative because we use NEGATIZE z axis */ - normalize(interest - origin);
    frame.vx = cross(up,frame.vz);
    if (dot(frame.vx,frame.vx) < 1e-8f)
      frame.vx = vec3f(0,1,0);
    else
      frame.vx = normalize(frame.vx);
    // frame.vx
    //   = (fabs(dot(up,frame.vz)) < 1e-6f)
    //   ? vec3f(0,1,0)
    //   : normalize(cross(up,frame.vz));
    frame.vy = normalize(cross(frame.vz,frame.vx));
    poiDistance = length(interest-origin);
    if (setFocalDistance) focalDistance = poiDistance;
    forceUpFrame();
  }

} // ::owlQT

