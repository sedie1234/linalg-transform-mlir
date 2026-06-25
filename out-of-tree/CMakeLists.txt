#===------------------------------------------------------------------------===
# linalg-transform-mlir out-of-tree pass infrastructure
#
# affine-mlir/out-of-tree 를 시작점으로 복제 후 linalg+transform 학습용으로 적응.
# 이 워크스페이스는 linalg dialect 의 high-level semantic 보존 변환과
# transform dialect 의 명시적 schedule IR 을 학습한다.
#
# 후속 work(M1~M7)은 lib/Passes/ 에 새 .cpp 를 추가하고
# lib/Passes/PassRegistration.cpp 에 한 줄만 등록하면 자동 빌드된다.
#
# 빌드 방법 (out-of-tree/ 에서, 또는 -S out-of-tree 로):
#   cmake -S . -B build -G Ninja \
#         -DMLIR_DIR=$HOME/llvm-project/build/lib/cmake/mlir \
#         -DLLVM_DIR=$HOME/llvm-project/build/lib/cmake/llvm \
#         -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j 8
#
# install step 불필요. LLVM build directory 의 cmake config 를 그대로 사용.
#===------------------------------------------------------------------------===

cmake_minimum_required(VERSION 3.20.0)
project(linalg-transform-passes LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# LLVM/MLIR 의 cmake config 를 찾아서 그대로 사용 (install 불필요)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# 본인 pass 라이브러리 + standalone driver
add_subdirectory(lib/Passes)
add_subdirectory(tools/my-mlir-opt)
