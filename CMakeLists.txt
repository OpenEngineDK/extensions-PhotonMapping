SET ( EXTENSION_NAME "Extensions_PhotonMapping")

# Create the extension library
CUDA_ADD_LIBRARY( "CUDA_KERNELS"
  Math/CUDA/Matrix.h
  Scene/KDNode.cu
#  Scene/KDPhotonUpperNode.cu
#  Scene/PhotonUpperNode.cu
#  Scene/PhotonLowerNode.cu
#  Scene/PhotonNode.cu
  Scene/TriangleNode.cu
  Scene/CUDAMeshNode.cu
  Resources/CUDA/CUDADataBlock.h
#  Utils/CUDA/NodeChildren.cu
#  Utils/CUDA/UpperNodeLeafList.cu
  Utils/CUDA/Segments.cu
  Utils/CUDA/SplittingPlanes.cu
  Utils/CUDA/GeometryList.cu
  Utils/CUDA/IRayTracer.cu
#  Utils/CUDA/BruteTracer.cu
  Utils/CUDA/Raytracer.cu
  Utils/CUDA/ShortStack.cu
  Utils/CUDA/ITriangleMapCreator.h
  Utils/CUDA/TriangleMapUpperCreator.cu
  Utils/CUDA/TriangleMapSAHCreator.cu
  Utils/CUDA/TriangleMapBalancedCreator.cu
#  Utils/CUDA/TriangleMapBalancedFast.cu
  Utils/CUDA/TriangleMap.cu
#  Utils/CUDA/PhotonMap.cu
#  Utils/CUDA/PhotonMapCheck.cu
#  Utils/CUDA/PhotonMapLowerNodes.cu
)

TARGET_LINK_LIBRARIES("CUDA_KERNELS"
  ${CUDA_CUT_LIBRARIES} #cutil
  ${CUDA_CUDPP_LIBRARY}
)  

# Create the extension library
ADD_LIBRARY( ${EXTENSION_NAME}
  Renderers/OpenGL/PhotonRenderingView.h
  Renderers/OpenGL/PhotonRenderingView.cpp
)

TARGET_LINK_LIBRARIES( ${EXTENSION_NAME}
  OpenEngine_Scene
  OpenEngine_Renderers
  OpenEngine_Resources
  Extensions_OpenGLRenderer
  ${OPENGL_LIBRARY}
  ${GLEW_LIBRARIES}
  ${SDL_LIBRARY}
  Extensions_CUDA
  "CUDA_KERNELS"
)
