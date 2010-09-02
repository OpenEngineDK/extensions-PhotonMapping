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
                : RenderingView(), photonTree(NULL), renderPhotons(true){
            }
            
            void PhotonRenderingView::Handle(RenderingEventArg arg){
                // Send arg to parent
                RenderingView::Handle(arg);
                
                if (arg.renderer.GetCurrentStage() == IRenderer::RENDERER_INITIALIZE){
                    Initialize(arg);
                    ShootPhotons();
                }else if (arg.renderer.GetCurrentStage() == IRenderer::RENDERER_PREPROCESS){
                    //ShootPhotons();
                    if (renderPhotons){
                        RenderPhotons();
                    }
                }
            }

            void PhotonRenderingView::Initialize(RenderingEventArg arg) {
                INITIALIZE_CUDA();
                unsigned int size = 1<<3;
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
                glVertexPointer(verts->GetDimension(), GL_FLOAT, 0, 0);
                glDrawArrays(GL_POINTS, 0, verts->GetSize());
                glDisableClientState(GL_VERTEX_ARRAY);
            }

            void PhotonRenderingView::RenderTree(){
                
            }
            
        }
    }
}
