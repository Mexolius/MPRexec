project(lab4_openMP)

find_package(OpenMP) 

add_executable(lab4_openMP main_lab4.cpp)
target_include_directories(lab4_openMP PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(lab4_openMP PRIVATE OpenMP::OpenMP_CXX)