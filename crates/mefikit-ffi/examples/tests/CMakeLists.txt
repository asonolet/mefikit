add_executable(mefikit_cpp_tests test_mefikit.cpp)
target_link_libraries(mefikit_cpp_tests PRIVATE mefikit::mefikit)

add_test(NAME mefikit_cpp_tests COMMAND mefikit_cpp_tests)
