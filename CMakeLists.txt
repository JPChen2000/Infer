cmake_minimum_required(VERSION 3.20)
project(Infer CXX)

option(BUILD_UNITTEST "BUILD UNITTEST" OFF)
option(FEATHER_DEBUG "Enable Feather debug logging and operator timers" OFF)
option(FEATHER_ENABLE_CUDA "Enable CUDA kernels when a CUDA compiler is available" ON)
option(FEATHER_ENABLE_CUDNN "Enable cuDNN acceleration when cuDNN is available" ON)
option(FEATHER_ENABLE_OPENMP "Enable OpenMP acceleration for CPU kernels" ON)
if (DEFINED FEATHER_WITH_CUDA)
    # Preserve the historical command-line spelling while keeping
    # FEATHER_WITH_CUDA as the detected backend state below.
    set(FEATHER_ENABLE_CUDA "${FEATHER_WITH_CUDA}" CACHE BOOL
        "Enable CUDA kernels when a CUDA compiler is available" FORCE)
    unset(FEATHER_WITH_CUDA CACHE)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

List(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(util)

find_package(Threads REQUIRED)
if (FEATHER_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
endif()

set(FEATHER_WITH_CUDA OFF)
set(FEATHER_WITH_CUDNN OFF)
if (FEATHER_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if (CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)
        set(FEATHER_WITH_CUDA ON)
        if (FEATHER_ENABLE_CUDNN)
            find_path(CUDNN_INCLUDE_DIR cudnn.h
                HINTS
                ${CUDAToolkit_INCLUDE_DIRS}
                /usr/local/cuda/include
                /usr/include)
            find_library(CUDNN_LIBRARY cudnn
                HINTS
                /usr/local/cuda/targets/x86_64-linux/lib
                /usr/local/cuda/lib64
                /usr/lib/x86_64-linux-gnu)
            if (CUDNN_INCLUDE_DIR AND CUDNN_LIBRARY)
                set(FEATHER_WITH_CUDNN ON)
            else()
                message(WARNING "cuDNN not found, CUDA Conv2D will use fallback kernels only")
            endif()
        endif()
    else()
        message(WARNING "CUDA compiler not found, CUDA kernels will not be built")
    endif()
endif()

Normalize(BUILD_UNITTEST)

if (CONAN_EXPORTED)
    include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
    conan_basic_setup(TARGETS)
else()
    if (NOT DEFINED ENV{CONAN_USER_HOME})
        set(ENV{CONAN_USER_HOME} "${CMAKE_SOURCE_DIR}/.conan_local")
    endif()
    set(CONAN_CMAKE_PATH "${CMAKE_BINARY_DIR}/conan.cmake")
    set(CONAN_CMAKE_URL
        "https://raw.githubusercontent.com/conan-io/cmake-conan/0.18.1/conan.cmake")
    set(NEEDS_CONAN_CMAKE_DOWNLOAD TRUE)
    if (EXISTS "${CONAN_CMAKE_PATH}")
        file(READ "${CONAN_CMAKE_PATH}" CONAN_CMAKE_CONTENTS)
        if (CONAN_CMAKE_CONTENTS MATCHES "conan_cmake_run")
            set(NEEDS_CONAN_CMAKE_DOWNLOAD FALSE)
        endif()
    endif()
    if (NEEDS_CONAN_CMAKE_DOWNLOAD)
        file(DOWNLOAD
            "${CONAN_CMAKE_URL}" "${CONAN_CMAKE_PATH}" SHOW_PROGRESS
            STATUS CONAN_CMAKE_DOWNLOAD_STATUS TLS_VERIFY ON)
        list(GET CONAN_CMAKE_DOWNLOAD_STATUS 0 CONAN_CMAKE_DOWNLOAD_RESULT)
        list(GET CONAN_CMAKE_DOWNLOAD_STATUS 1 CONAN_CMAKE_DOWNLOAD_MESSAGE)
        if (NOT CONAN_CMAKE_DOWNLOAD_RESULT EQUAL 0)
            message(FATAL_ERROR
                "Failed to download the Conan 1 CMake helper: ${CONAN_CMAKE_DOWNLOAD_MESSAGE}")
        endif()
    endif()
    include("${CONAN_CMAKE_PATH}")
    if (NOT COMMAND conan_cmake_run)
        message(FATAL_ERROR "The Conan CMake helper does not define conan_cmake_run.")
    endif()
    set(CONAN_PROFILE_ARGUMENT)
    if (DEFINED CONAN_PROFILE_PATH AND NOT "${CONAN_PROFILE_PATH}" STREQUAL "")
        set(CONAN_PROFILE_ARGUMENT PROFILE "${CONAN_PROFILE_PATH}")
    endif()
    conan_cmake_run(CONANFILE conanfile.py
        BASIC_SETUP CMAKE_TARGETS UPDATE
        BUILD missing
        ${CONAN_PROFILE_ARGUMENT}
        SETTINGS build_type=${CMAKE_BUILD_TYPE}
        OPTIONS
        BUILD_UNITTEST=${BUILD_UNITTEST})
endif()


list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(SRC "src/core/memory.cc"
    "src/core/dim.cc"
    "src/core/tensor.cc"
    "src/core/graph.cc"
    "src/core/graph_lowering.cc"
    "src/core/operator_registry.cc"
    "src/core/static_graph.cc"
    "src/pass/dead_node_elimination_pass.cc"
    "src/pass/graph_pass.cc"
    "src/pass/identity_elimination_pass.cc"
    "src/pass/matmul_add_fusion_pass.cc"
    "src/pass/qwen_gated_delta_fusion_pass.cc"
    "src/pass/qwen_matmul_add_fusion_pass.cc"
    "src/pass/qwen_rms_norm_fusion_pass.cc"
    "src/pass/no_op_elimination_pass.cc"
    "src/pass/resize_concat_fusion_pass.cc"
    "src/pass/reshape_chain_elimination_pass.cc"
    "src/pass/sigmoid_mul_fusion_pass.cc"
    "src/pass/yolo_decode_fusion_pass.cc"
    "src/model/model_io.cc"
    "src/model/layout_transform.cc"
    "src/model/weight_store.cc"
    "src/operator/register_all.cc"
    "src/operator/conv2d_op.cc"
    "src/operator/add_op.cc"
    "src/operator/batch_normalization_op.cc"
    "src/operator/div_op.cc"
    "src/operator/erf_op.cc"
    "src/operator/exp_op.cc"
    "src/operator/sin_op.cc"
    "src/operator/cos_op.cc"
    "src/operator/neg_op.cc"
    "src/operator/softplus_op.cc"
    "src/operator/concat_op.cc"
    "src/operator/fc_op.cc"
    "src/operator/flatten_op.cc"
    "src/operator/global_average_pool_op.cc"
    "src/operator/gemm_op.cc"
    "src/operator/identity_op.cc"
    "src/operator/matmul_op.cc"
    "src/operator/mul_op.cc"
    "src/operator/pool_op.cc"
    "src/operator/pow_op.cc"
    "src/operator/relu_op.cc"
    "src/operator/reshape_op.cc"
    "src/operator/resize_op.cc"
    "src/operator/resize_concat_op.cc"
    "src/operator/silu_op.cc"
    "src/operator/slice_op.cc"
    "src/operator/softmax_op.cc"
    "src/operator/sigmoid_op.cc"
    "src/operator/split_op.cc"
    "src/operator/transpose_op.cc"
    "src/operator/unsqueeze_op.cc"
    "src/operator/squeeze_op.cc"
    "src/operator/cast_op.cc"
    "src/operator/reduce_mean_op.cc"
    "src/operator/reduce_sum_op.cc"
    "src/operator/gather_op.cc"
    "src/operator/equal_op.cc"
    "src/operator/sqrt_op.cc"
    "src/operator/sub_op.cc"
    "src/operator/tanh_op.cc"
    "src/operator/shape_op.cc"
    "src/operator/constant_of_shape_op.cc"
    "src/operator/expand_op.cc"
    "src/operator/where_op.cc"
    "src/operator/qwen_gated_delta_op.cc"
    "src/operator/qwen_rms_norm_op.cc"
    "src/operator/yolo_decode_op.cc"
    "src/kernel/common/add.cc"
    "src/kernel/common/batch_normalization.cc"
    "src/kernel/common/div.cc"
    "src/kernel/common/erf.cc"
    "src/kernel/common/exp.cc"
    "src/kernel/common/sin.cc"
    "src/kernel/common/cos.cc"
    "src/kernel/common/neg.cc"
    "src/kernel/common/softplus.cc"
    "src/kernel/common/concat.cc"
    "src/kernel/common/conv2d.cc"
    "src/kernel/common/fc.cc"
    "src/kernel/common/global_average_pool.cc"
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
    "src/kernel/common/silu.cc"
    "src/kernel/common/slice.cc"
    "src/kernel/common/softmax.cc"
    "src/kernel/common/sigmoid.cc"
    "src/kernel/common/split.cc"
    "src/kernel/common/transpose.cc"
    "src/kernel/common/unsqueeze.cc"
    "src/kernel/common/squeeze.cc"
    "src/kernel/common/cast.cc"
    "src/kernel/common/reduce_mean.cc"
    "src/kernel/common/reduce_sum.cc"
    "src/kernel/common/gather.cc"
    "src/kernel/common/equal.cc"
    "src/kernel/common/shape.cc"
    "src/kernel/common/constant_of_shape.cc"
    "src/kernel/common/expand.cc"
    "src/kernel/common/where.cc"
    "src/kernel/common/qwen_gated_delta.cc"
    "src/kernel/common/qwen_rms_norm.cc"
    "src/kernel/common/sqrt.cc"
    "src/kernel/common/sub.cc"
    "src/kernel/common/tanh.cc"
    "src/kernel/common/resize.cc"
    "src/kernel/common/resize_concat.cc"
    "src/kernel/common/yolo_decode.cc"
    "src/kernel/x86/add.cc"
    "src/kernel/x86/cast.cc"
    "src/kernel/x86/concat.cc"
    "src/kernel/x86/conv2d.cc"
    "src/kernel/x86/div.cc"
    "src/kernel/x86/direct_conv_fp32.cc"
    "src/kernel/x86/expand.cc"
    "src/kernel/x86/fc.cc"
    "src/kernel/x86/flatten.cc"
    "src/kernel/x86/gemm.cc"
    "src/kernel/x86/identity.cc"
    "src/kernel/x86/linear.cc"
    "src/kernel/x86/linear_fp16.cc"
    "src/kernel/x86/linear_fp32.cc"
    "src/kernel/x86/matmul.cc"
    "src/kernel/x86/mul.cc"
    "src/kernel/x86/pointwise_conv_fp32.cc"
    "src/kernel/x86/pool.cc"
    "src/kernel/x86/pow.cc"
    "src/kernel/x86/reduce_sum.cc"
    "src/kernel/x86/reduce_mean.cc"
    "src/kernel/x86/relu.cc"
    "src/kernel/x86/resize.cc"
    "src/kernel/x86/reshape.cc"
    "src/kernel/x86/silu.cc"
    "src/kernel/x86/sigmoid.cc"
    "src/kernel/x86/slice.cc"
    "src/kernel/x86/split.cc"
    "src/kernel/x86/softmax.cc"
    "src/kernel/x86/transpose.cc"
    "src/kernel/x86/qwen_gated_delta.cc"
    "src/kernel/x86/qwen_rms_norm.cc"
    "src/kernel/x86/yolo_decode.cc"
    "src/demo/classification_demo.cc"
    "src/demo/classification_runner.cc"
    "src/demo/image_io.cc"
    "src/demo/qwen_chat_template.cc"
    "src/demo/qwen_demo.cc"
    "src/demo/qwen_runner.cc"
    "src/demo/qwen_tokenizer.cc"
    "src/demo/yolov5_postprocess.cc"
    "src/demo/yolov5_runner.cc")

set(CUDA_SRC
    "src/kernel/cuda/add.cu"
    "src/kernel/cuda/batch_normalization.cu"
    "src/kernel/cuda/div.cu"
    "src/kernel/cuda/erf.cu"
    "src/kernel/cuda/exp.cu"
    "src/kernel/cuda/sin.cu"
    "src/kernel/cuda/cos.cu"
    "src/kernel/cuda/neg.cu"
    "src/kernel/cuda/softplus.cu"
    "src/kernel/cuda/concat.cu"
    "src/kernel/cuda/conv2d.cu"
    "src/kernel/cuda/fc.cu"
    "src/kernel/cuda/flatten.cu"
    "src/kernel/cuda/gemm.cu"
    "src/kernel/cuda/global_average_pool.cu"
    "src/kernel/cuda/identity.cu"
    "src/kernel/cuda/matmul.cu"
    "src/kernel/cuda/mul.cu"
    "src/kernel/cuda/pool.cu"
    "src/kernel/cuda/pow.cu"
    "src/kernel/cuda/register_all.cu"
    "src/kernel/cuda/relu.cu"
    "src/kernel/cuda/reshape.cu"
    "src/kernel/cuda/resize.cu"
    "src/kernel/cuda/resize_concat.cu"
    "src/kernel/cuda/runtime.cu"
    "src/kernel/cuda/silu.cu"
    "src/kernel/cuda/sigmoid.cu"
    "src/kernel/cuda/slice.cu"
    "src/kernel/cuda/softmax.cu"
    "src/kernel/cuda/split.cu"
    "src/kernel/cuda/unsqueeze.cu"
    "src/kernel/cuda/squeeze.cu"
    "src/kernel/cuda/cast.cu"
    "src/kernel/cuda/reduce_mean.cu"
    "src/kernel/cuda/reduce_sum.cu"
    "src/kernel/cuda/gather.cu"
    "src/kernel/cuda/equal.cu"
    "src/kernel/cuda/expand.cu"
    "src/kernel/cuda/where.cu"
    "src/kernel/cuda/transpose.cu"
    "src/kernel/cuda/sqrt.cu"
    "src/kernel/cuda/sub.cu"
    "src/kernel/cuda/tanh.cu"
    "src/kernel/cuda/yolo_decode.cu")

if (FEATHER_WITH_CUDA)
    list(APPEND SRC ${CUDA_SRC})
endif()

add_library(infer ${SRC})
target_link_libraries(infer
    PUBLIC
    Threads::Threads
    )
if (FEATHER_ENABLE_OPENMP)
    target_compile_definitions(infer PUBLIC FEATHER_WITH_OPENMP)
    target_link_libraries(infer PUBLIC OpenMP::OpenMP_CXX)
endif()
if (FEATHER_WITH_CUDA)
    target_compile_definitions(infer PUBLIC FEATHER_WITH_CUDA)
    target_link_libraries(infer PUBLIC CUDA::cudart CUDA::cublas)
    if (FEATHER_WITH_CUDNN)
        target_compile_definitions(infer PUBLIC FEATHER_WITH_CUDNN)
        target_include_directories(infer PUBLIC ${CUDNN_INCLUDE_DIR})
        target_link_libraries(infer PUBLIC ${CUDNN_LIBRARY})
    endif()
    set_target_properties(infer PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_EXTENSIONS OFF)
endif()

add_executable(yolov5_demo demo/yolov5_infer_demo.cc)
add_executable(yolov5_benchmark_demo demo/yolov5_benchmark_demo.cc)
add_executable(resnet_demo demo/resnet_infer_demo.cc)
add_executable(resent_demo demo/resent_infer_demo.cc)
add_executable(repvgg_demo demo/repvgg_infer_demo.cc)
add_executable(vit_demo demo/vit_infer_demo.cc)
add_executable(qwen_demo demo/qwen_infer_demo.cc)
add_executable(fth_layout_convert demo/fth_layout_convert_demo.cc)
add_executable(x86_conv_benchmark_demo demo/x86_conv_benchmark_demo.cc)
add_executable(x86_linear_benchmark_demo demo/x86_linear_benchmark_demo.cc)
add_executable(x86_softmax_benchmark_demo demo/x86_softmax_benchmark_demo.cc)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(infer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
if (FEATHER_DEBUG)
    target_compile_definitions(infer PUBLIC FEATHER_DEBUG)
endif()
set_source_files_properties(
    src/kernel/x86/add.cc
    src/kernel/x86/cast.cc
    src/kernel/x86/concat.cc
    src/kernel/x86/conv2d.cc
    src/kernel/x86/div.cc
    src/kernel/x86/direct_conv_fp32.cc
    src/kernel/x86/expand.cc
    src/kernel/x86/fc.cc
    src/kernel/x86/flatten.cc
    src/kernel/x86/gemm.cc
    src/kernel/x86/identity.cc
    src/kernel/x86/linear.cc
    src/kernel/x86/linear_fp16.cc
    src/kernel/x86/linear_fp32.cc
    src/kernel/x86/matmul.cc
    src/kernel/x86/mul.cc
    src/kernel/x86/pointwise_conv_fp32.cc
    src/kernel/x86/pool.cc
    src/kernel/x86/pow.cc
    src/kernel/x86/reduce_sum.cc
    src/kernel/x86/reduce_mean.cc
    src/kernel/x86/relu.cc
    src/kernel/x86/resize.cc
    src/kernel/x86/reshape.cc
    src/kernel/x86/silu.cc
    src/kernel/x86/sigmoid.cc
    src/kernel/x86/slice.cc
    src/kernel/x86/split.cc
    src/kernel/x86/softmax.cc
    src/kernel/x86/transpose.cc
    src/kernel/x86/qwen_gated_delta.cc
    src/kernel/x86/qwen_rms_norm.cc
    src/kernel/x86/yolo_decode.cc
    PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
target_include_directories(yolov5_demo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(yolov5_demo infer)
target_include_directories(resnet_demo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(resnet_demo infer)
target_include_directories(resent_demo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(resent_demo infer)
target_include_directories(repvgg_demo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(repvgg_demo infer)
target_include_directories(vit_demo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(vit_demo infer)
target_include_directories(qwen_demo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(qwen_demo infer)
target_include_directories(yolov5_benchmark_demo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(yolov5_benchmark_demo infer CONAN_PKG::benchmark)
target_include_directories(fth_layout_convert PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(fth_layout_convert infer)
target_include_directories(x86_conv_benchmark_demo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(x86_conv_benchmark_demo infer CONAN_PKG::benchmark)
target_include_directories(x86_linear_benchmark_demo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(x86_linear_benchmark_demo infer CONAN_PKG::benchmark)
target_include_directories(x86_softmax_benchmark_demo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(x86_softmax_benchmark_demo infer CONAN_PKG::benchmark)
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
message("cuda: " ${FEATHER_WITH_CUDA})
message("cudnn: " ${FEATHER_WITH_CUDNN})
message("\n")
