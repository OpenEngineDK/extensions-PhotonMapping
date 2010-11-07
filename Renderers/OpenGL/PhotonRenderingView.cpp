// Photon Mapping rendering view.
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <Renderers/OpenGL/PhotonRenderingView.h>
#include <Meta/CUDA.h>
#include <Meta/OpenGL.h>
#include <Geometry/GeometrySet.h>
#include <Resources/DataBlock.h>
#include <Scene/PhotonNode.h>
#include <Utils/CUDA/TriangleMap.h>
#include <Utils/CUDA/PhotonMap.h>
#include <Utils/CUDA/Raytracer.h>
#include <Utils/CUDA/Utils.h>

#include <Logging/Logger.h>

namespace OpenEngine {
    namespace Renderers {
        namespace OpenGL {
            
            PhotonRenderingView::PhotonRenderingView()
                : RenderingView(), triangleMap(NULL), photonMap(NULL), 
                  renderPhotons(false), renderTree(false){
            }
            
            void PhotonRenderingView::Handle(RenderingEventArg arg){
                // Send arg to parent
                RenderingView::Handle(arg);
                
                if (arg.renderer.GetCurrentStage() == IRenderer::RENDERER_INITIALIZE){
                    Initialize(arg);
                    UpdateGeometry();
                }else if (arg.renderer.GetCurrentStage() == IRenderer::RENDERER_PREPROCESS){
                    if (renderPhotons)
                        RenderPhotons();
                    if (renderTree)
                        RenderTree(arg);
                }else if (arg.renderer.GetCurrentStage() == IRenderer::RENDERER_PROCESS){
                    raytracer->Trace(&arg.canvas);
                }
            }

            void PhotonRenderingView::Initialize(RenderingEventArg arg) {
                INITIALIZE_CUDA();

                triangleMap = new TriangleMap(arg.canvas.GetScene());
                raytracer = new RayTracer(triangleMap);
                
                unsigned int size = (1<<17)-7;
                //photonMap = new PhotonMap(size);
                IDataBlockPtr vertices = IDataBlockPtr(new DataBlock<4, float>(size));
                map<string, IDataBlockPtr> attr;
                attr["vertex"] = vertices;
                photons = GeometrySetPtr(new GeometrySet(attr));
                arg.renderer.BindDataBlock(vertices.get());
            }

            void PhotonRenderingView::UpdateGeometry(){
                logger.info << "Pew pew, triangles everywhere" << logger.end;
                triangleMap->Create();
                CHECK_FOR_CUDA_ERROR();
                //ShootPhotons();
            }

            void PhotonRenderingView::ShootPhotons(){
                logger.info << "Pew pew, photons everywhere" << logger.end;
                photonMap->Create();
                CHECK_FOR_CUDA_ERROR();
            }

            void PhotonRenderingView::RenderPhotons(){
                // Copy Photons to OpenGL buffer
                photonMap->photons.MapToDataBlocks(photons->GetVertices().get());
                
                glColor3f(0.0f, 1.0f, 0.0f);
                glEnableClientState(GL_VERTEX_ARRAY);
                IDataBlockPtr verts = photons->GetDataBlock("vertex");
                glBindBuffer(GL_ARRAY_BUFFER, verts->GetID());
                glVertexPointer(verts->GetDimension(), GL_FLOAT, 0, 0);
                glDrawArrays(GL_POINTS, 0, verts->GetSize());
                glDisableClientState(GL_VERTEX_ARRAY);
            }

            void PhotonRenderingView::RenderTree(RenderingEventArg arg){
                unsigned int size = photonMap->upperNodes->size * 12;

                if (upperNodes == NULL || upperNodes->GetSize() < size){

                    // @TODO destroy old datablocks on gpu

                    // Allocate new datablocks to store debug geometry
                    IDataBlockPtr positions = IDataBlockPtr(new DataBlock<3, float>(size));
                    IDataBlockPtr colors = IDataBlockPtr(new DataBlock<3, float>(size));
                    map<string, IDataBlockPtr> attr;
                    attr["position"] = positions;
                    attr["color"] = colors;
                    upperNodes = GeometrySetPtr(new GeometrySet(attr));
                    arg.renderer.BindDataBlock(positions.get());
                    arg.renderer.BindDataBlock(colors.get());
                }
                
                IDataBlockPtr p = upperNodes->GetDataBlock("position");
                IDataBlockPtr c = upperNodes->GetDataBlock("color");

                photonMap->upperNodes->MapToDataBlocks(p.get(), c.get());

                glEnableClientState(GL_VERTEX_ARRAY);
                glBindBuffer(GL_ARRAY_BUFFER, p->GetID());
                glVertexPointer(p->GetDimension(), GL_FLOAT, 0, 0);
                
                glEnableClientState(GL_COLOR_ARRAY);
                glBindBuffer(GL_ARRAY_BUFFER, c->GetID());
                glColorPointer(c->GetDimension(), GL_FLOAT, 0, 0);
                
                glDrawArrays(GL_LINES, 0, size);

                glDisableClientState(GL_VERTEX_ARRAY);
                glDisableClientState(GL_COLOR_ARRAY);
            }

        }
    }
}
