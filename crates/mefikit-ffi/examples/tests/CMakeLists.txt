add_executable(mefikit_cpp_tests test_mefikit.cpp)
target_link_libraries(mefikit_cpp_tests PRIVATE mefikit::mefikit)

# The suite reads the repository's real .med test meshes. ctest runs the binary
# from the build tree, so it cannot find them relative to the working directory.
target_compile_definitions(mefikit_cpp_tests PRIVATE
    MEFIKIT_TEST_DATA_DIR="${MEFIKIT_REPO_ROOT}/tests/data")

add_test(NAME mefikit_cpp_tests COMMAND mefikit_cpp_tests)
