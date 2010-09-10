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

#include <Logging/Logger.h>

namespace OpenEngine {
    namespace Renderers {
        namespace OpenGL {
            
            PhotonRenderingView::PhotonRenderingView()
                : RenderingView(), photonTree(NULL), 
                  renderPhotons(true), renderTree(true){
            }
            
            void PhotonRenderingView::Handle(RenderingEventArg arg){
                // Send arg to parent
                RenderingView::Handle(arg);
                
                if (arg.renderer.GetCurrentStage() == IRenderer::RENDERER_INITIALIZE){
                    Initialize(arg);
                    ShootPhotons();
                }else if (arg.renderer.GetCurrentStage() == IRenderer::RENDERER_PREPROCESS){
                    if (renderPhotons)
                        RenderPhotons();
                    if (renderTree)
                        RenderTree(arg);
                }
            }

            void PhotonRenderingView::Initialize(RenderingEventArg arg) {
                INITIALIZE_CUDA();
                unsigned int size = 1<<16;
                photonTree = new PhotonKDTree(size);
                IDataBlockPtr vertices = IDataBlockPtr(new DataBlock<3, float>(size));
                map<string, IDataBlockPtr> attr;
                attr["vertex"] = vertices;
                photons = GeometrySetPtr(new GeometrySet(attr));
                arg.renderer.BindDataBlock(vertices.get());                
            }

            void PhotonRenderingView::ShootPhotons(){
                logger.info << "Pew pew, photons everywhere" << logger.end;
                photonTree->Create();
                CHECK_FOR_CUDA_ERROR();
            }

            void PhotonRenderingView::RenderPhotons(){
                // Copy Photons to OpenGL buffer
                photonTree->photons.MapToDataBlocks(photons->GetVertices().get());
                
                glColor3f(0.0f, 1.0f, 0.0f);
                glEnableClientState(GL_VERTEX_ARRAY);
                IDataBlockPtr verts = photons->GetDataBlock("vertex");
                glBindBuffer(GL_ARRAY_BUFFER, verts->GetID());
                glVertexPointer(verts->GetDimension(), GL_FLOAT, 0, 0);
                glDrawArrays(GL_POINTS, 0, verts->GetSize());
                glDisableClientState(GL_VERTEX_ARRAY);
            }

            void PhotonRenderingView::RenderTree(RenderingEventArg arg){
                unsigned int size = photonTree->upperNodes.size * 12;

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

                photonTree->upperNodes.MapToDataBlocks(p.get(), c.get());

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
