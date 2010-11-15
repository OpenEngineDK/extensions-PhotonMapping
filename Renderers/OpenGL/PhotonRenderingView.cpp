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
#include <Utils/CUDA/Utils.h>

#include <Utils/CUDA/BruteTracer.h>
#include <Utils/CUDA/Raytracer.h>

#include <Logging/Logger.h>

namespace OpenEngine {
    namespace Renderers {
        namespace OpenGL {
            
            PhotonRenderingView::PhotonRenderingView()
                : RenderingView(), triangleMap(NULL), 
                  renderTree(false), raytrace(true){
            }
            
            void PhotonRenderingView::Handle(RenderingEventArg arg){
                // Send arg to parent
                RenderingView::Handle(arg);
                
                if (arg.renderer.GetCurrentStage() == IRenderer::RENDERER_INITIALIZE){
                    Initialize(arg);
                    UpdateGeometry();
                }else if (arg.renderer.GetCurrentStage() == IRenderer::RENDERER_PREPROCESS){
                    //if (renderTree)
                    //RenderTree(arg);
                }else if (arg.renderer.GetCurrentStage() == IRenderer::RENDERER_PROCESS){
                    if (raytrace){
                        cudaGraphicsResource *pboResource;
                        cudaGraphicsGLRegisterBuffer(&pboResource, pbo->GetID(), cudaGraphicsMapFlagsWriteDiscard);
                        cudaGraphicsMapResources(1, &pboResource, 0);
                        CHECK_FOR_CUDA_ERROR();
                        
                        size_t bytes;
                        uchar4* pixels;
                        cudaGraphicsResourceGetMappedPointer((void**)&pixels, &bytes,
                                                             pboResource);
                        CHECK_FOR_CUDA_ERROR();
                        
                        raytracer->Trace(&arg.canvas, pixels);
                        
                        cudaGraphicsUnmapResources(1, &pboResource, 0);
                        cudaGraphicsUnregisterResource(pboResource);
                        CHECK_FOR_CUDA_ERROR();
                        
                        glMatrixMode(GL_MODELVIEW);
                        glLoadIdentity();
                        
                        glMatrixMode(GL_PROJECTION);
                        glLoadIdentity();
                        glOrtho(0,1,0,1,0,1);
                        
                        glDisable(GL_DEPTH_TEST);
                        glRasterPos2i(0, 0);
                        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo->GetID());
                        CHECK_FOR_GL_ERROR();
                        glDrawPixels(arg.canvas.GetWidth(), arg.canvas.GetHeight(), 
                                     GL_RGBA, GL_UNSIGNED_BYTE, 0);
                        CHECK_FOR_GL_ERROR();
                        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
                        CHECK_FOR_GL_ERROR();
                    }
                }
            }

            void PhotonRenderingView::Initialize(RenderingEventArg arg) {
                INITIALIZE_CUDA();

                triangleMap = new TriangleMap(arg.canvas.GetScene());
                raytracer = new RayTracer(triangleMap);
                //raytracer = new BruteTracer(triangleMap->geom);
                //raytracer->SetVisualizeRays(true);

                int size = arg.canvas.GetWidth() * arg.canvas.GetHeight();
                pbo = IDataBlockPtr(new DataBlock<4, unsigned char>(size, NULL, PIXEL_UNPACK));
                arg.renderer.BindDataBlock(pbo.get());                
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
            }

            void PhotonRenderingView::UpdateGeometry(){
                logger.info << "Pew pew, triangles everywhere" << logger.end;
                triangleMap->Create();
                //triangleMap->Create();
                CHECK_FOR_CUDA_ERROR();
            }

            /*
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
            */
        }
    }
}
