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

#include "TransferFunctionEditor.h"
#include <QOpenGLShaderProgram>
#include <QOpenGLTexture>
#include <QMouseEvent>
#include <math.h>
#include <QApplication>
#include <QDesktopWidget>

namespace owlQT {

  AlphaEditor::AlphaEditor()
  {
    const int N = 128;
    
    const vec4f xf0(0.f,0.f,1.f,0.f);
    const vec4f xf1(1.f,0.f,0.f,1.f);;
    for (int i=0;i<N;i++) {
      float f = i / (N-1.f);
      xf.push_back((1.f-f)*xf0+f*xf1);
    }
  }
  
  AlphaEditor::~AlphaEditor()
  {}
  
  QSize AlphaEditor::minimumSizeHint() const
  {
    return QSize(250, 150);
  }
  
  QSize AlphaEditor::sizeHint() const
  {
    QRect rec = QApplication::desktop()->screenGeometry();
    int height = rec.height();
    int width = rec.width();
    return QSize(width/6, height/6);
  }

  void AlphaEditor::initializeGL()
  {
    initializeOpenGLFunctions();
    glDisable(GL_DEPTH_TEST);
  }

  void AlphaEditor::paintGL()
  {
    // makeCurrent();
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.f, width, 0.f, height, -1000.f, 1000.f);
    glViewport(0.f, 0.f, width, height);
      
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glScalef(width,height,1.f);

    // ------------------------------------------------------------------
    // paint background using window palette
    // ------------------------------------------------------------------
    QPalette palette
      = QApplication::palette();
    QColor bgColor
      = palette.color(QPalette::Active,QPalette::Window);
    
    glClearColor(bgColor.redF(),
                 bgColor.greenF(),
                 bgColor.blueF(),
                 bgColor.alphaF());
    glClear(GL_COLOR_BUFFER_BIT);

    const int alphaCount = xf.size();

    // ------------------------------------------------------------------
    // draw the alpha map
    // ------------------------------------------------------------------
    glPushMatrix();
    {
      glTranslatef(0.f,histogramHeight,1.f);
      glScalef(1.f,1.f-histogramHeight,1.f);
      for (int i=0;i<alphaCount-1;i++) {
        int i0 = i;
        int i1 = i+1;

        const vec4f c0 = xf[i0];
        const vec4f c1 = xf[i1];
        float x0 = i0 / float(alphaCount-1.f);
        float x1 = i1 / float(alphaCount-1.f);
        glBegin( GL_QUADS );
        {
          glColor4f(c0.x,c0.y,c0.z,1.f);
          glVertex2f( x0,0.f );
          glVertex2f( x0,c0.w );
        
          glColor4f(c1.x,c1.y,c1.z,1.f);
          glVertex2f( x1,c1.w );
          glVertex2f( x1,0.f );
        }
        glEnd();
      }
    }
    glPopMatrix();


    // ------------------------------------------------------------------
    // draw the histogram
    // ------------------------------------------------------------------
    if (!histogram.empty()) {
      glColor4f(1.f-bgColor.redF(),
                1.f-bgColor.greenF(),
                1.f-bgColor.blueF(),
                1.f);
      float maxHistogramValue = 0.f;
      for (auto v : histogram)
        maxHistogramValue = std::max(maxHistogramValue,v);
      glPushMatrix();
      {
        glTranslatef(0.f,histogramHeight,1.f);
        glScalef(1.f,1.f-histogramHeight,1.f);
        for (int i=0;i<histogram.size();i++) {
          float x0 = (i+0) / float(histogram.size());
          float x1 = (i+1) / float(histogram.size());
          float height = histogram[i]/maxHistogramValue;
          
          glVertex2f( x0,0.f );
          glVertex2f( x0,height );
          
          glVertex2f( x1,height );
          glVertex2f( x1,0.f );
        }
      }
      glPopMatrix();
    }
    // doneCurrent();
  }




  
  /*! draw with the mouse into the alpha window, using relative
    coordinates (0,0 lower lef, 1,1 upper right) */
  void AlphaEditor::drawIntoAlpha(vec2f from,
                                  vec2f to,
                                  bool set)
  {
    if (from.x > to.x) std::swap(from,to);

    int x0 = int(floorf(from.x * (xf.size()-1) + .5f));
    int x1 = int(floorf(to.x * (xf.size()-1) + .5f));
    for (int x=x0;x<=x1;x++) {
      float f = (x0==x1)?1.f:((x-x0)/float(x1-x0));
      float y = (1.f-f)*from.y+f*to.y;
      if (x>=0 && x<xf.size())
        if (set)
          xf[x].w = min(1.f,max(0.f,y));
        else
          xf[x].w = y > .65f ? 1.f : 0.f;
    }
  }

  /*! draw with the mouse into the alpha window, using absolute
    parent window coordinates */
  void AlphaEditor::drawIntoAlphaAbsolute(const vec2i &from,
                                          const vec2i &to,
                                          bool set)
  {
    vec2i parentSize = { width, height };//getWindowSize();
    vec2f parent_rel_from = (vec2f)from * rcp(vec2f(parentSize));
    vec2f parent_rel_to   = (vec2f)to * rcp(vec2f(parentSize));

    // invert mouse pos to start at lower left, not top left
    parent_rel_from.y = 1.f-parent_rel_from.y;
    parent_rel_to.y = 1.f-parent_rel_to.y;

    vec2f scale(1.f, 1.f-histogramHeight);
    vec2f offset(0.f,histogramHeight);

    drawIntoAlpha((parent_rel_from-offset)*rcp(scale),
                  (parent_rel_to-offset)*rcp(scale),
                  set);

    emit colorMapChanged();
    update();
  }

  /*! draw from last pixel where we ended drawing to new mouse
    position */
  void AlphaEditor::drawLastToCurrent(vec2i newPos)
  {
    // lastPos = newPos;
    if (leftButtonPressed)
      drawIntoAlphaAbsolute(lastPos,newPos,true);
    else if (rightButtonPressed)
      drawIntoAlphaAbsolute(lastPos,newPos,false);
    lastPos = newPos;
  }
  


  
  void AlphaEditor::resizeGL(int width, int height)
  {
    this->width = width;
    this->height = height;
  }

  void AlphaEditor::mousePressEvent(QMouseEvent *event)
  {
    if (event->button() == Qt::LeftButton)
      leftButtonPressed = true;
    if (event->button() == Qt::RightButton)
      rightButtonPressed = true;

    lastPos = { event->pos().x(), event->pos().y() };
    drawLastToCurrent(lastPos); // yes, draw "last to last" (single click) ...
  }

  void AlphaEditor::mouseReleaseEvent(QMouseEvent *event)
  {
    if (event->button() == Qt::LeftButton)
      leftButtonPressed = false;
    if (event->button() == Qt::RightButton)
      rightButtonPressed = false;
    
    lastPos = { event->pos().x(), event->pos().y() };
    drawLastToCurrent(lastPos); // yes, draw "last to last" (single click) ...
    
    // emit clicked();
  }

  void AlphaEditor::mouseMoveEvent(QMouseEvent *event)
  {
    vec2i newPos{event->pos().x(),event->pos().y()};
    
    drawLastToCurrent(newPos); 
  }

}
