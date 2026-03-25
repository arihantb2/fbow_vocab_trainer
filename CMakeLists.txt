cmake_minimum_required(VERSION 3.10)

project(fbow_vocab_trainer VERSION 0.0.1 LANGUAGES CXX)

include(CTest)

find_package(OpenCV REQUIRED)
find_package(fbow REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem)

add_executable(fbow_vocab_trainer
    src/config/fbow_trainer_config.cpp
    src/dataset/fbow_trainer_dataset.cpp
    src/features/fbow_trainer_features.cpp
    src/fbow_trainer_main.cpp
    src/train/fbow_trainer_train.cpp
    src/vocabulary_test/fbow_trainer_test.cpp
)

target_compile_features(fbow_vocab_trainer PRIVATE cxx_std_11)

target_include_directories(fbow_vocab_trainer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    "${fbow_INCLUDE_DIRS}/fbow"
)

if(TARGET yaml-cpp::yaml-cpp)
    set(YAML_CPP_TARGET yaml-cpp::yaml-cpp)
else()
    set(YAML_CPP_TARGET yaml-cpp)
endif()

target_link_libraries(fbow_vocab_trainer
    PRIVATE
    ${YAML_CPP_TARGET}
    Boost::filesystem
    ${fbow_LIBS}
    ${OpenCV_LIBS}
)

install(TARGETS fbow_vocab_trainer
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

if(BUILD_TESTING)
    add_subdirectory(test)
endif()
