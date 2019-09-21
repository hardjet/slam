message(STATUS "In: ${CMAKE_CURRENT_SOURCE_DIR}")

add_library(${PROJECT_NAME} SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/config.cc
        )

add_subdirectory(camera)
add_subdirectory(util)
add_subdirectory(feature)
add_subdirectory(data)
add_subdirectory(match)
add_subdirectory(solve)

target_sources(${PROJECT_NAME} PRIVATE)

set_target_properties(${PROJECT_NAME} PROPERTIES
        OUTPUT_NAME openvslam
        LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

target_link_libraries(${PROJECT_NAME}
        Threads::Threads
        ${OpenCV_LIBS}
        ${G2O_LIBS}
        ${LAPACK_LIBRARIES}
        ${CXSPARSE_LIBRARIES}
        ${SUITESPARSE_LIBRARIES}
        ${YAMLCPP_LIBRARIES}
        ${BOW_LIBS})

message(STATUS "Out: ${CMAKE_CURRENT_SOURCE_DIR}")
