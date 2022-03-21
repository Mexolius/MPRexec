project(lab2_PI C)

find_package(MPI) 

add_executable(lab2_PI main_lab2.c)
target_include_directories(lab2_PI PRIVATE ${MPI_INCLUDE_PATH})
target_link_libraries(lab2_PI PRIVATE ${MPI_C_LIBRARIES})