#pragma once

#ifdef _WIN32
#  ifdef GGML_OPS_EXT_CUDA_SHARED
#    define GGML_OPS_EXT_CUDA_API __declspec(dllexport)
#  else
#    define GGML_OPS_EXT_CUDA_API __declspec(dllimport)
#  endif
#else
#  define GGML_OPS_EXT_CUDA_API
#endif

namespace ggml_ops_ext {
namespace cuda {

GGML_OPS_EXT_CUDA_API void register_backend();

} // namespace cuda
} // namespace ggml_ops_ext
