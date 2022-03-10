# Define collections of links that have to be made for morphologica
set(MORPH_LIBS_CORE ${ARMADILLO_LIBRARY} ${ARMADILLO_LIBRARIES} ${HDF5_C_LIBRARIES} ${LAPACK_LIBRARIES})
set(MORPH_LIBS_GL ${OpenCV_LIBS} OpenGL::GL Freetype::Freetype glfw)

# An exe
add_executable(exe1 exe1.cpp)
target_compile_definitions(exe1 PUBLIC FLT=float VISUALISE=1)
target_link_libraries(exe1 ${MORPH_LIBS_CORE} ${MORPH_LIBS_GL})
