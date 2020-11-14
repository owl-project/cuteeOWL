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

#include "OWLViewer.h"
#include <QOpenGLShaderProgram>
#include <QOpenGLTexture>
#include <QMouseEvent>
#include <math.h>
#include <QApplication>
#include <QDesktopWidget>

namespace owlQT {
  OWLViewer::~OWLViewer()
  {
  
    makeCurrent();
    // vbo.destroy();
    // for (int i = 0; i < 6; ++i)
    //     delete textures[i];
    // delete program;
    doneCurrent();
  }

  QSize OWLViewer::minimumSizeHint() const
  {
    return QSize(50, 50);
  }

  QSize OWLViewer::sizeHint() const
  {
    QRect rec = QApplication::desktop()->screenGeometry();
    int height = rec.height();
    int width = rec.width();
    return QSize(width/2, height/2);
    // return QSize(200, 200);
  }

  // void OWLViewer::rotateBy(int xAngle, int yAngle, int zAngle)
  // {
  //     xRot += xAngle;
  //     yRot += yAngle;
  //     zRot += zAngle;
  //     update();
  // }

  // void OWLViewer::setClearColor(const QColor &color)
  // {
  //     clearColor = color;
  //     update();
  // }

  void OWLViewer::initializeGL()
  {
    initializeOpenGLFunctions();
    glDisable(GL_DEPTH_TEST);

    // makeObject();

    //     glEnable(GL_DEPTH_TEST);
    //     glEnable(GL_CULL_FACE);

    // #define PROGRAM_VERTEX_ATTRIBUTE 0
    // #define PROGRAM_TEXCOORD_ATTRIBUTE 1

    //     QOpenGLShader *vshader = new QOpenGLShader(QOpenGLShader::Vertex, this);
    //     const char *vsrc =
    //         "attribute highp vec4 vertex;\n"
    //         "attribute mediump vec4 texCoord;\n"
    //         "varying mediump vec4 texc;\n"
    //         "uniform mediump mat4 matrix;\n"
    //         "void main(void)\n"
    //         "{\n"
    //         "    gl_Position = matrix * vertex;\n"
    //         "    texc = texCoord;\n"
    //         "}\n";
    //     vshader->compileSourceCode(vsrc);

    //     QOpenGLShader *fshader = new QOpenGLShader(QOpenGLShader::Fragment, this);
    //     const char *fsrc =
    //         "uniform sampler2D texture;\n"
    //         "varying mediump vec4 texc;\n"
    //         "void main(void)\n"
    //         "{\n"
    //         "    gl_FragColor = texture2D(texture, texc.st);\n"
    //         "}\n";
    //     fshader->compileSourceCode(fsrc);

    //     program = new QOpenGLShaderProgram;
    //     program->addShader(vshader);
    //     program->addShader(fshader);
    //     program->bindAttributeLocation("vertex", PROGRAM_VERTEX_ATTRIBUTE);
    //     program->bindAttributeLocation("texCoord", PROGRAM_TEXCOORD_ATTRIBUTE);
    //     program->link();

    //     program->bind();
    //     program->setUniformValue("texture", 0);
  }

  void OWLViewer::paintGL()
  {
    static float clearColor = 0.f;
    // glClearColor(clearColor.redF(), clearColor.greenF(), clearColor.blueF(), clearColor.alphaF());
    clearColor = fmodf(clearColor+.01f,1.f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(clearColor,clearColor,clearColor,1.f);
    printf("bla\n");
    // QMatrix4x4 m;
    // m.ortho(-0.5f, +0.5f, +0.5f, -0.5f, 4.0f, 15.0f);
    // m.translate(0.0f, 0.0f, -10.0f);
    // m.rotate(xRot / 16.0f, 1.0f, 0.0f, 0.0f);
    // m.rotate(yRot / 16.0f, 0.0f, 1.0f, 0.0f);
    // m.rotate(zRot / 16.0f, 0.0f, 0.0f, 1.0f);

    // program->setUniformValue("matrix", m);
    // program->enableAttributeArray(PROGRAM_VERTEX_ATTRIBUTE);
    // program->enableAttributeArray(PROGRAM_TEXCOORD_ATTRIBUTE);
    // program->setAttributeBuffer(PROGRAM_VERTEX_ATTRIBUTE, GL_FLOAT, 0, 3, 5 * sizeof(GLfloat));
    // program->setAttributeBuffer(PROGRAM_TEXCOORD_ATTRIBUTE, GL_FLOAT, 3 * sizeof(GLfloat), 2, 5 * sizeof(GLfloat));

    // for (int i = 0; i < 6; ++i) {
    //     textures[i]->bind();
    //     glDrawArrays(GL_TRIANGLE_FAN, i * 4, 4);
    // }
  }

  void OWLViewer::resizeGL(int width, int height)
  {
    int side = qMin(width, height);
    glViewport((width - side) / 2, (height - side) / 2, side, side);
  }

  void OWLViewer::mousePressEvent(QMouseEvent *event)
  {
    lastPos = event->pos();
  }

  void OWLViewer::mouseMoveEvent(QMouseEvent *event)
  {
    int dx = event->x() - lastPos.x();
    int dy = event->y() - lastPos.y();

    printf("mouse %i %i\n",dx,dy);
    // if (event->buttons() & Qt::LeftButton) {
    //     rotateBy(8 * dy, 8 * dx, 0);
    // } else if (event->buttons() & Qt::RightButton) {
    //     rotateBy(8 * dy, 0, 8 * dx);
    // }
    lastPos = event->pos();
  }

  void OWLViewer::mouseReleaseEvent(QMouseEvent * /* event */)
  {
    emit clicked();
  }

  // void OWLViewer::makeObject()
  // {
  //   printf("makeObject\n");
  // static const int coords[6][4][3] = {
  //     { { +1, -1, -1 }, { -1, -1, -1 }, { -1, +1, -1 }, { +1, +1, -1 } },
  //     { { +1, +1, -1 }, { -1, +1, -1 }, { -1, +1, +1 }, { +1, +1, +1 } },
  //     { { +1, -1, +1 }, { +1, -1, -1 }, { +1, +1, -1 }, { +1, +1, +1 } },
  //     { { -1, -1, -1 }, { -1, -1, +1 }, { -1, +1, +1 }, { -1, +1, -1 } },
  //     { { +1, -1, +1 }, { -1, -1, +1 }, { -1, -1, -1 }, { +1, -1, -1 } },
  //     { { -1, -1, +1 }, { +1, -1, +1 }, { +1, +1, +1 }, { -1, +1, +1 } }
  // };

  // for (int j = 0; j < 6; ++j)
  //     textures[j] = new QOpenGLTexture(QImage(QString(":/images/side%1.png").arg(j + 1)).mirrored());

  // QVector<GLfloat> vertData;
  // for (int i = 0; i < 6; ++i) {
  //     for (int j = 0; j < 4; ++j) {
  //         // vertex position
  //         vertData.append(0.2 * coords[i][j][0]);
  //         vertData.append(0.2 * coords[i][j][1]);
  //         vertData.append(0.2 * coords[i][j][2]);
  //         // texture coordinate
  //         vertData.append(j == 0 || j == 3);
  //         vertData.append(j == 0 || j == 1);
  //     }
  // }

  // vbo.create();
  // vbo.bind();
  // vbo.allocate(vertData.constData(), vertData.count() * sizeof(GLfloat));
  // }
}
