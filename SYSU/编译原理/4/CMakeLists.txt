file(GLOB _classic_src *.cpp *.hpp *.c *.h)

add_executable(task4-classic ${_classic_src})

target_include_directories(task4-classic PRIVATE . ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(task4-classic PRIVATE ${LLVM_INCLUDE_DIRS})

target_link_libraries(task4-classic LLVM)

file(GLOB _llm_src llm/*.cpp llm/*.hpp llm/*.c llm/*.h)

add_executable(task4-llm ${_classic_src} ${_llm_src})

target_include_directories(task4-llm PRIVATE . ${CMAKE_CURRENT_BINARY_DIR} llm)
target_include_directories(task4-llm PRIVATE ${LLVM_INCLUDE_DIRS})

target_link_libraries(task4-llm LLVM pybind11::embed)

target_compile_definitions(task4-llm
                           PRIVATE TASK4_LLM TASK4_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

add_custom_target(task4 DEPENDS task4-classic task4-llm)
