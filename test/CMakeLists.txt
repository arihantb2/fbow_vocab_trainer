find_package(GTest REQUIRED)

if(TARGET yaml-cpp::yaml-cpp)
    set(YAML_CPP_TARGET yaml-cpp::yaml-cpp)
else()
    set(YAML_CPP_TARGET yaml-cpp)
endif()

add_executable(fbow_vocab_trainer_unit_tests
    test_fbow_trainer_config.cpp
    test_fbow_trainer_features.cpp
    ../src/config/fbow_trainer_config.cpp
    ../src/features/fbow_trainer_features.cpp
)

target_compile_features(fbow_vocab_trainer_unit_tests PRIVATE cxx_std_11)

target_include_directories(fbow_vocab_trainer_unit_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

target_link_libraries(fbow_vocab_trainer_unit_tests
    PRIVATE
    GTest::gtest
    GTest::gtest_main
    ${YAML_CPP_TARGET}
    Boost::filesystem
    ${OpenCV_LIBS}
)

add_test(NAME fbow_vocab_trainer_unit_tests COMMAND fbow_vocab_trainer_unit_tests)
