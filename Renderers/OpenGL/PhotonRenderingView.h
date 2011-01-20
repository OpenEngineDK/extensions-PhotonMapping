// Photon Mapping rendering view.
// -------------------------------------------------------------------
// Copyright (C) 2010 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _PHOTON_MAPPING_RENDERING_VIEW_H_
#define _PHOTON_MAPPING_RENDERING_VIEW_H_

#include <Renderers/OpenGL/RenderingView.h>

#include <string>
#include <boost/shared_ptr.hpp>

namespace OpenEngine {
    namespace Geoemtry {
        class GeometrySet;
        typedef boost::shared_ptr<GeometrySet> GeometrySetPtr;
    }
    namespace Resources {
        class IDataBlock;
        typedef boost::shared_ptr<IDataBlock> IDataBlockPtr;
    }
    namespace Utils {
        namespace CUDA {
            class TriangleMap;
            class IRayTracer;
        }
    }
    namespace Renderers {
        namespace OpenGL {

            using namespace Utils::CUDA;

            class PhotonRenderingView : public RenderingView {
            protected:
                TriangleMap* triangleMap;
                IRayTracer* raytracer;

                bool updateTree;
                string rayTracerName;

                IDataBlockPtr pbo;
                
                bool renderTree, raytrace;
                GeometrySetPtr upperNodes;
                
            public:
                PhotonRenderingView();
                virtual ~PhotonRenderingView() {}

                virtual void Handle(RenderingEventArg arg);

                void Initialize(RenderingEventArg arg);
                void UpdateGeometry();

                TriangleMap* GetTriangleMap() const { return triangleMap; }
                IRayTracer* GetRayTracer() const { return raytracer; }

                void SetTreeUpdate(bool u) { updateTree = u; }
                bool GetTreeUpdate() { return updateTree; }

                void SetRayTracerName(string name) { rayTracerName = name; }
                string GetRayTracerName() { return rayTracerName; }
                //void RenderTree(RenderingEventArg arg);
            };

        }
    }
}

#endif // _PHOTON_MAPPING_RENDERING_VIEW_H_
