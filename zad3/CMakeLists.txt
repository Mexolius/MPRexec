project(lab3_openMP)

find_package(OpenMP) 

add_executable(lab3_openMP main_lab3.cpp)
target_include_directories(lab3_openMP PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(lab3_openMP PRIVATE OpenMP::OpenMP_CXX)