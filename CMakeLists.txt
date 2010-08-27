SET ( EXTENSION_NAME "Extensions_PhotonMapping")

# Create the extension library
CUDA_ADD_LIBRARY( "CUDA_KERNELS"
#  Utils/CUDA/Photon/KDTreeConstruction.cu
  Utils/CUDA/PhotonKDTree.cu
  Scene/KDPhotonUpperNode.cu
  Utils/CUDA/AABBVar.cu
  Utils/CUDA/splitVar.cu
  Scene/PhotonNode.cu
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
