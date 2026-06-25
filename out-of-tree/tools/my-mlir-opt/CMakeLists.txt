# my-mlir-opt: linalg-transform-mlir 학습용 standalone driver

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
  MLIROptLib
  MyLinalgTransformPasses
)

add_llvm_executable(my-mlir-opt main.cpp)
llvm_update_compile_flags(my-mlir-opt)
target_link_libraries(my-mlir-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(my-mlir-opt)
