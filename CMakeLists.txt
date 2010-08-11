SET ( EXTENSION_NAME "Extensions_PhotonMapping")

# Create the extension library
CUDA_ADD_LIBRARY( "CUDA_KERNELS"
  Utils/CUDA/KDTreeConstruction.cu
)

TARGET_LINK_LIBRARIES("CUDA_KERNELS"
  ${CUDA_CUT_LIBRARIES} #cutil
)  

MESSAGE( " cut: ${CUDA_CUT_LIBRARIES} ")#cutil

# Create the extension library
ADD_LIBRARY( ${EXTENSION_NAME}
  Utils/CUDA/KDTree.h
  Renderers/OpenGL/PhotonRenderingView.h
  Renderers/OpenGL/PhotonRenderingView.cpp
)

TARGET_LINK_LIBRARIES( ${EXTENSION_NAME}
  OpenEngine_Scene
  OpenEngine_Renderers
  OpenEngine_Resources
  Extensions_OpenGLRenderer
  Extensions_OpenGLSelection
  ${OPENGL_LIBRARY}
  ${GLEW_LIBRARIES}
  ${SDL_LIBRARY}
  "CUDA_KERNELS"
)
