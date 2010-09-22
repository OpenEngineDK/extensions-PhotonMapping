SET ( EXTENSION_NAME "Extensions_PhotonMapping")

# Create the extension library
CUDA_ADD_LIBRARY( "CUDA_KERNELS"
#  Utils/CUDA/Photon/KDTreeConstruction.cu
  Utils/CUDA/PhotonKDTree.cu
  Utils/CUDA/PhotonMap.cu
  Utils/CUDA/PhotonMapCheck.cu
  Scene/KDNode.cu
  Scene/KDPhotonUpperNode.cu
  Scene/PhotonUpperNode.cu
  Scene/PhotonLowerNode.cu
  Scene/PhotonNode.cu
  Utils/CUDA/AABBVar.cu
  Utils/CUDA/NodeChildren.cu
  Utils/CUDA/UpperNodeLeafList.cu
  Utils/CUDA/splitVar.cu
  Utils/CUDA/Segments.cu
  Utils/CUDA/SplittingPlanes.cu
)

TARGET_LINK_LIBRARIES("CUDA_KERNELS"
  ${CUDA_CUT_LIBRARIES} #cutil
)  

MESSAGE( " cut: ${CUDA_CUT_LIBRARIES} ")#cutil

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
