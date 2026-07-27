add_library(forge_image_io STATIC
    image/image_io.cpp
    image/preprocess.cpp
)
add_library(forge_audio_io STATIC
    audio/audio_io.cpp
)
add_library(forge_audio_dsp STATIC
    audio/resample.cpp
)

foreach(target IN ITEMS forge_image_io forge_audio_io forge_audio_dsp)
    set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(${target} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_features(${target} PUBLIC cxx_std_17)
    if (MSVC)
        target_compile_options(${target} PRIVATE /EHsc /utf-8 /W3)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra)
    endif()
endforeach()

# Forge already manages stb_image through the stable-diffusion.cpp revision.
# Only this private utility target sees that upstream header; category APIs and
# consumers remain dependency-free and exchange decoded pixels instead.
set(FORGE_STB_IMAGE_DIR
    "${CMAKE_SOURCE_DIR}/src/categories/visual_generation/providers/stable_diffusion_cpp/stable-diffusion.cpp/thirdparty"
    CACHE PATH "Directory containing stb_image.h for Forge media I/O")
if (NOT EXISTS "${FORGE_STB_IMAGE_DIR}/stb_image.h")
    message(FATAL_ERROR
        "forge_image_io requires stb_image.h; initialize stable-diffusion.cpp or set FORGE_STB_IMAGE_DIR")
endif()
target_include_directories(forge_image_io PRIVATE ${FORGE_STB_IMAGE_DIR})
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|IntelLLVM")
    target_compile_options(forge_image_io PRIVATE -Wno-unused-function)
elseif (MSVC)
    target_compile_options(forge_image_io PRIVATE /wd4505)
else()
    target_compile_options(forge_image_io PRIVATE -Wno-unused-function)
endif()
