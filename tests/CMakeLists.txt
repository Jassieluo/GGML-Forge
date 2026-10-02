add_subdirectory(media_test)
add_subdirectory(ops_test)
add_subdirectory(nn_test)

if (INFERENCE_BUILD_TTS)
    add_subdirectory(model_config_test)
endif()
if (INFERENCE_BUILD_ASR)
    add_subdirectory(asr_test)
endif()
if (INFERENCE_BUILD_LLM)
    add_subdirectory(llm_test)
endif()
if (INFERENCE_BUILD_VISUAL_GENERATION)
    add_subdirectory(visual_test)
endif()
if (INFERENCE_BUILD_VISUAL_PERCEPTION)
    add_subdirectory(visual_perception_test)
endif()
if (INFERENCE_BUILD_SERVER)
    add_subdirectory(server_test)
endif()

find_package(Python3 QUIET COMPONENTS Interpreter)
if (Python3_Interpreter_FOUND)
    foreach(script_test IN ITEMS artifact quantization model_schema maintenance_updates)
        add_test(
            NAME scripts.${script_test}
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/scripts/test_${script_test}.py
        )
        set_tests_properties(scripts.${script_test} PROPERTIES
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            LABELS "scripts;unit"
        )
    endforeach()
    if (INFERENCE_BUILD_TTS)
        add_test(
            NAME scripts.gpt_sovits_provider
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/scripts/test_gpt_sovits_provider.py
        )
        set_tests_properties(scripts.gpt_sovits_provider PROPERTIES
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            LABELS "scripts;tts;provider;unit"
        )
    endif()
    if (INFERENCE_BUILD_VISUAL_PERCEPTION AND
        (INFERENCE_BUILD_INSTANCE_PERCEPTION OR INFERENCE_BUILD_IMAGE_CLASSIFICATION))
        add_test(
            NAME scripts.yolo_conversion
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/scripts/test_yolo_conversion.py
        )
        set_tests_properties(scripts.yolo_conversion PROPERTIES
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            LABELS "scripts;visual_perception;instance;yolo;provider;unit"
        )
    endif()
    if (INFERENCE_BUILD_VISUAL_PERCEPTION AND INFERENCE_BUILD_SEMANTIC_SEGMENTATION)
        add_test(
            NAME scripts.lraspp_conversion
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/scripts/test_lraspp_conversion.py
        )
        set_tests_properties(scripts.lraspp_conversion PROPERTIES
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            LABELS "scripts;visual_perception;segmentation;lraspp;provider;unit"
        )
    endif()
    if (INFERENCE_BUILD_VISUAL_PERCEPTION AND INFERENCE_BUILD_DEPTH_ESTIMATION)
        add_test(
            NAME scripts.stereonet_conversion
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/scripts/test_stereonet_conversion.py
        )
        set_tests_properties(scripts.stereonet_conversion PROPERTIES
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            LABELS "scripts;visual_perception;depth;stereonet;provider;unit"
        )
    endif()
endif()
