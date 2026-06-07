cmake_minimum_required(VERSION 3.20)
project(Infer CXX)

option(BUILD_UNITTEST "BUILD UNITTEST" OFF)
option(FEATHER_DEBUG "Enable Feather debug logging and operator timers" OFF)
set(CXX_STANDARD 17)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

List(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(util)

Normalize(BUILD_UNITTEST)

if (CONAN_EXPORTED)
    include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
    conan_basic_setup(TARGETS)
else()
    if (NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
        file(DOWNLOAD
            "https://raw.githubusercontent.com/conan-io/cmake-conan/master/conan.cmake"
            "${CMAKE_BINARY_DIR}/conan.cmake" SHOW_PROGRESS)
    endif()
    include(${CMAKE_BINARY_DIR}/conan.cmake)
    conan_cmake_run(CONANFILE conanfile.py
        BASIC_SETUP CMAKE_TARGETS UPDATE 
        BUILD missing
        PROFILE ${CONAN_PROFILE_PATH}
        SETTINGS build_type=${CMAKE_BUILD_TYPE}
        OPTIONS 
        BUILD_UNITTEST=${BUILD_UNITTEST})
endif()


list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(SRC "src/core/memory.cc"
    "src/core/dead_node_elimination_pass.cc"
    "src/core/dim.cc"
    "src/core/graph_pass.cc"
    "src/core/tensor.cc"
    "src/core/graph.cc"
    "src/core/graph_lowering.cc"
    "src/core/operator_registry.cc"
    "src/core/static_graph.cc"
    "src/model/model_io.cc"
    "src/model/weight_store.cc"
    "src/operator/register_all.cc"
    "src/operator/conv2d_op.cc"
    "src/operator/add_op.cc"
    "src/operator/concat_op.cc"
    "src/operator/fc_op.cc"
    "src/operator/flatten_op.cc"
    "src/operator/gemm_op.cc"
    "src/operator/identity_op.cc"
    "src/operator/matmul_op.cc"
    "src/operator/mul_op.cc"
    "src/operator/pool_op.cc"
    "src/operator/pow_op.cc"
    "src/operator/relu_op.cc"
    "src/operator/reshape_op.cc"
    "src/operator/resize_op.cc"
    "src/operator/slice_op.cc"
    "src/operator/softmax_op.cc"
    "src/operator/sigmoid_op.cc"
    "src/operator/split_op.cc"
    "src/operator/transpose_op.cc"
    "src/kernel/common/add.cc"
    "src/kernel/common/concat.cc"
    "src/kernel/common/conv2d.cc"
    "src/kernel/common/fc.cc"
    "src/kernel/common/flatten.cc"
    "src/kernel/common/gemm.cc"
    "src/kernel/common/identity.cc"
    "src/kernel/common/matmul.cc"
    "src/kernel/common/mul.cc"
    "src/kernel/common/pool.cc"
    "src/kernel/common/pow.cc"
    "src/kernel/register_all.cc"
    "src/kernel/common/reshape.cc"
    "src/kernel/common/relu.cc"
    "src/kernel/common/slice.cc"
    "src/kernel/common/softmax.cc"
    "src/kernel/common/sigmoid.cc"
    "src/kernel/common/split.cc"
    "src/kernel/common/transpose.cc"
    "src/kernel/common/resize.cc"
    "src/kernel/x86/add.cc"
    "src/kernel/x86/concat.cc"
    "src/kernel/x86/conv2d.cc"
    "src/kernel/x86/gemm.cc"
    "src/kernel/x86/matmul.cc"
    "src/kernel/x86/mul.cc"
    "src/kernel/x86/pool.cc"
    "src/kernel/x86/pow.cc"
    "src/kernel/x86/relu.cc"
    "src/kernel/x86/resize.cc"
    "src/kernel/x86/reshape.cc"
    "src/kernel/x86/sigmoid.cc"
    "src/kernel/x86/split.cc"
    "src/kernel/x86/transpose.cc"
    "src/demo/image_io.cc"
    "src/demo/yolov5_postprocess.cc"
    "src/demo/yolov5_runner.cc")

add_library(infer ${SRC})
add_executable(yolov5_demo demo/yolov5_infer_demo.cc)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(infer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
if (FEATHER_DEBUG)
    target_compile_definitions(infer PUBLIC FEATHER_DEBUG)
endif()
set_source_files_properties(
    src/kernel/x86/add.cc
    src/kernel/x86/concat.cc
    src/kernel/x86/conv2d.cc
    src/kernel/x86/gemm.cc
    src/kernel/x86/matmul.cc
    src/kernel/x86/mul.cc
    src/kernel/x86/pool.cc
    src/kernel/x86/pow.cc
    src/kernel/x86/relu.cc
    src/kernel/x86/resize.cc
    src/kernel/x86/reshape.cc
    src/kernel/x86/sigmoid.cc
    src/kernel/x86/split.cc
    src/kernel/x86/transpose.cc
    PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
target_include_directories(yolov5_demo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(yolov5_demo infer)
message("${CMAKE_CURRENT_SOURCE_DIR}")

if (BUILD_UNITTEST)
    # MACRO(UNITTEST TESTNAME EXTRA_LIBS)
    #     message("name ${TESTNAME}")
    #     add_executable(${TESTNAME} ${TESTNAME}.cc ${CMAKE_CURRENT_SOURCE_DIR}/unittest_main.cc)
    #     message("test >>>> " ${CMAKE_CURRENT_SOURCE_DIR})
    #     target_include_directories(${TESTNAME}
    #         PRIVATE
    #         ${CMAKE_CURRENT_SOURCE_DIR}/../include
    #         ${CMAKE_CURRENT_SOURCE_DIR/include})
    #     target_link_libraries(${TESTNAME} ${EXTRA_LIBS} CONAN_PKG::gtest)
    # ENDMACRO(UNITTEST)
    file(GLOB TESTS ${CMAKE_CURRENT_SOURCE_DIR}/test/*.cc)
    add_executable(unit_tests ${TESTS})
    target_include_directories(unit_tests PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(unit_tests infer CONAN_PKG::gtest)
    enable_testing()
    # add_subdirectory(test)
endif()
message("\n")
message("unittest: " ${BUILD_UNITTEST})
message("debug: " ${FEATHER_DEBUG})
message("\n")
