#pragma once

#include "ggml-backend.h"

#ifndef GGML_OPS_EXT_CPU_API
#ifdef _WIN32
#define GGML_OPS_EXT_CPU_API extern "C" __declspec(dllimport)
#else
#define GGML_OPS_EXT_CPU_API extern "C"
#endif
#endif

GGML_OPS_EXT_CPU_API void ggml_ops_ext_cpu_set_n_threads(ggml_backend_t backend, int n_threads);

namespace ggml_ops_ext {
namespace cpu {

int backend_thread_count(ggml_backend_t backend);

} // namespace cpu
} // namespace ggml_ops_ext
