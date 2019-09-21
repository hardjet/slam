cmake_minimum_required(VERSION 3.0)

message(STATUS "In: ${CMAKE_CURRENT_SOURCE_DIR}")
set(TEST_NAME "test_match")


add_subdirectory(openvslam)



add_executable(${TEST_NAME}
        test_match.cpp
        util/image_util.cc)

# set output directory for executables
set_target_properties(${TEST_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

# setup stack trace logger
if(USE_STACK_TRACE_LOGGER)
    target_compile_definitions(${TEST_NAME}
            PRIVATE -DUSE_STACK_TRACE_LOGGER)
    target_link_libraries(${TEST_NAME} PRIVATE
            ${GLOG_LIBRARIES})
endif()

# setup google-perftools
if(USE_GOOGLE_PERFTOOLS)
    target_compile_definitions(${TEST_NAME}
            PRIVATE -DUSE_GOOGLE_PERFTOOLS)
    target_link_libraries(${TEST_NAME} PRIVATE
            ${GPERFTOOLS_LIBRARIES})
endif()

target_link_libraries(${TEST_NAME} PRIVATE
        Threads::Threads
        ${OpenCV_LIBS}
        ${YAMLCPP_LIBRARIES}
        ${BOW_LIBS}
        ${PROJECT_NAME}
)



add_executable(orb_compare
        orb_compare.cpp)

# set output directory for executables
set_target_properties(orb_compare PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

# setup stack trace logger
if(USE_STACK_TRACE_LOGGER)
    target_compile_definitions(orb_compare
            PRIVATE -DUSE_STACK_TRACE_LOGGER)
    target_link_libraries(orb_compare PRIVATE
            ${GLOG_LIBRARIES})
endif()

# setup google-perftools
if(USE_GOOGLE_PERFTOOLS)
    target_compile_definitions(orb_compare
            PRIVATE -DUSE_GOOGLE_PERFTOOLS)
    target_link_libraries(orb_comparePRIVATE
            ${GPERFTOOLS_LIBRARIES})
endif()

target_link_libraries(orb_compare PRIVATE
        Threads::Threads
        ${OpenCV_LIBS}
        ${YAMLCPP_LIBRARIES}
        ${BOW_LIBS}
        ${PROJECT_NAME}
        )


message(STATUS "Out: ${CMAKE_CURRENT_SOURCE_DIR}")



