cmake_minimum_required( VERSION 2.8 )
project(directBA)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 14)

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_PREFIX_PATH "/usr/local/opencv/opencv-3.4.1/share/OpenCV")


FIND_PACKAGE(Eigen3 REQUIRED)
FIND_PACKAGE(CSparse REQUIRED)
FIND_PACKAGE(Pangolin REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(Sophus REQUIRED)
FIND_PACKAGE(Boost REQUIRED)
FIND_PACKAGE(G2O REQUIRED)

SET(G2O_LIBS g2o_cli g2o_ext_freeglut_minimal g2o_simulator g2o_solver_slam2d_linear 
    g2o_types_icp g2o_types_slam2d g2o_types_sba g2o_types_slam3d g2o_core g2o_interface 
    g2o_solver_csparse g2o_solver_structure_only g2o_csparse_extension g2o_opengl_helper
    g2o_solver_dense g2o_stuff g2o_types_sclam2d g2o_parser g2o_solver_pcg g2o_types_data
    g2o_types_sim3 cxsparse)

include_directories(${Eigen3_INCLUDE_DIR}
                    ${CSPARSE_INCLUDE_DIR}
                    ${Sophus_INCLUDE_DIRS}
                    ${OpenCV_INCLUDE_DIRS}
                    ${Boost_INCLUDE_DIRS}
                    ${G2O_INCLUDE_DIRS}
                    ${Pangolin_INCLUDE_DIRS})

add_executable(directg2oBA directg2oBA.cpp)


target_link_libraries(directg2oBA
                        ${G2O_LIBS} fmt
                        ${CSPARSE_LIBRARY} fmt
                        ${Sophus_LIBRARIES} fmt
                        ${OpenCV_LIBRARIES} fmt
                        ${Boost_LIBRARIES} fmt
                        ${G2O_LIBRARIES} fmt
                        ${Pangolin_LIBRARIES} fmt)
