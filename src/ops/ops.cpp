#include "ops/ops.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include <algorithm>
#include <deque>
#include <vector>
#include <string>
#include <mutex>
#include <memory>
#include <unordered_map>
#include <atomic>
#include <cstring>
#include <iostream>

namespace ggml_ops_ext {

struct registered_backend {
    std::string name_prefix;
    std::vector<ops_kernel_entry> kernels;
};

static std::deque<registered_backend> g_registered_backends;
static std::mutex g_registry_mutex;
static std::atomic<bool> g_registry_frozen { false };
static std::mutex g_hook_mutex;
static size_t g_hook_users = 0;
static std::mutex g_backend_lanes_mutex;
static std::unordered_map<ggml_backend_t, std::weak_ptr<std::recursive_mutex>> g_backend_lanes;

#include "ggml-ops-ext-bridge.h"

static std::shared_ptr<std::recursive_mutex> backend_lane(ggml_backend_t backend) {
    std::lock_guard<std::mutex> lock(g_backend_lanes_mutex);
    if (g_backend_lanes.size() > 64) {
        for (auto it = g_backend_lanes.begin(); it != g_backend_lanes.end();) {
            if (it->second.expired()) it = g_backend_lanes.erase(it);
            else ++it;
        }
    }
    auto& weak = g_backend_lanes[backend];
    auto lane = weak.lock();
    if (!lane) {
        lane = std::make_shared<std::recursive_mutex>();
        weak = lane;
    }
    return lane;
}

static const registered_backend* find_ops_backend_by_name(const char* backend_name) {
    if (!backend_name) return nullptr;
    std::unique_lock<std::mutex> lock(g_registry_mutex, std::defer_lock);
    if (!g_registry_frozen.load(std::memory_order_acquire)) lock.lock();
    for (const auto& iface : g_registered_backends) {
        const size_t prefix_len = iface.name_prefix.size();
        if (std::strncmp(backend_name, iface.name_prefix.c_str(), prefix_len) == 0) {
            return &iface;
        }
    }
    return nullptr;
}

static const ops_kernel_entry* select_kernel(
    const registered_backend* ops_backend,
    const ops_request& request,
    ops_probe_result* selected_probe = nullptr
) {
    if (!ops_backend) return nullptr;
    const ops_kernel_entry* selected = nullptr;
    ops_probe_result best_probe;
    for (const auto& kernel : ops_backend->kernels) {
        if (kernel.op_id != request.op_id || !kernel.execute) continue;
        const ops_probe_result probe = kernel.probe
            ? kernel.probe(request)
            : ops_probe_result(true);
        if (!probe.supported) continue;
        if (!selected || kernel.priority > selected->priority) {
            selected = &kernel;
            best_probe = probe;
        }
    }
    if (selected_probe) *selected_probe = best_probe;
    return selected;
}

static int ops_ext_hook_impl(ggml_backend_t backend, struct ggml_tensor* node) {
    if (!node || (int)node->op < GGML_OP_OPS_VIRT_BASE) {
        return GGML_OPS_EXT_NOT_HANDLED;
    }

    const ops_status status = execute_ops_kernel(backend, node);
    if (status.code == ops_status_code::not_handled) return GGML_OPS_EXT_NOT_HANDLED;
    return status ? GGML_OPS_EXT_SUCCESS : GGML_OPS_EXT_FAILED;
}

static bool ops_ext_supports_impl(
    ggml_backend_dev_t device,
    const struct ggml_tensor* node
) {
    if (!device || !node || (int)node->op < GGML_OP_OPS_VIRT_BASE) return false;
    return probe_ops_kernel({
        device, (int)node->op, const_cast<ggml_tensor* const*>(node->src), GGML_MAX_SRC,
        node->op_params, sizeof(node->op_params), const_cast<ggml_tensor*>(node)
    }).supported;
}

bool register_ops_backend(const ops_backend_registration& registration) {
    if (!registration.backend_name_prefix || !registration.kernels || registration.n_kernels <= 0) return false;
    std::lock_guard<std::mutex> lock(g_registry_mutex);
    if (g_registry_frozen.load(std::memory_order_relaxed)) return false;
    for (const auto& registered : g_registered_backends) {
        if (registered.name_prefix == registration.backend_name_prefix) {
            return false;
        }
    }
    registered_backend backend;
    backend.name_prefix = registration.backend_name_prefix;
    backend.kernels.assign(registration.kernels, registration.kernels + registration.n_kernels);
    g_registered_backends.push_back(std::move(backend));
    return true;
}

ops_probe_result probe_ops_kernel(
    ggml_backend_dev_t device,
    int op_id,
    struct ggml_tensor* const* srcs,
    int n_srcs,
    const void* params,
    size_t params_size
) {
    return probe_ops_kernel({ device, op_id, srcs, n_srcs, params, params_size, nullptr });
}

ops_probe_result probe_ops_kernel(const ops_request& request) {
    if (!request.device) return { false, 0, "device is null" };
    const registered_backend* backend = find_ops_backend_by_name(ggml_backend_dev_name(request.device));
    ops_probe_result probe;
    if (!select_kernel(backend, request, &probe)) {
        return { false, 0, "no compatible kernel" };
    }
    return probe;
}

static ops_status execute_ops_kernel_impl(ggml_backend_t backend, struct ggml_tensor* node,
                                          bool take_lane) {
    if (!backend || !node) {
        return ops_status::error(ops_status_code::invalid_request, "backend or node is null");
    }
    ggml_backend_dev_t device = ggml_backend_get_device(backend);
    const registered_backend* registered = find_ops_backend_by_name(ggml_backend_name(backend));
    const ops_request request = {
        device, (int)node->op, node->src, GGML_MAX_SRC,
        node->op_params, sizeof(node->op_params), node
    };
    const ops_kernel_entry* kernel = select_kernel(registered, request);
    if (!kernel) {
        return ops_status::error(ops_status_code::not_handled, "no compatible kernel");
    }
    const ops_execution_context context = { backend, device, nullptr, nullptr, 0 };
    // Backend kernels may throw (e.g. sycl::exception at submit); letting that
    // unwind through the extern "C" graph-compute boundary aborts the process.
    try {
        if (take_lane) {
            ops_backend_lane_guard lane(backend);
            return kernel->execute(context, node);
        }
        return kernel->execute(context, node);
    } catch (const std::exception& e) {
        std::cerr << "[ggml-ops-ext] kernel '" << kernel->name << "' threw: " << e.what() << std::endl;
        return ops_status::error(ops_status_code::execution_failed, "kernel threw an exception");
    } catch (...) {
        std::cerr << "[ggml-ops-ext] kernel '" << kernel->name << "' threw an unknown exception" << std::endl;
        return ops_status::error(ops_status_code::execution_failed, "kernel threw an exception");
    }
}

ops_status execute_ops_kernel(ggml_backend_t backend, struct ggml_tensor* node) {
    return execute_ops_kernel_impl(backend, node, true);
}

ops_backend_lane_guard::ops_backend_lane_guard(ggml_backend_t backend) {
    if (!backend) return;
    auto lane = backend_lane(backend);
    lane_ = lane.get();
    owner_ = std::move(lane);
    static_cast<std::recursive_mutex*>(lane_)->lock();
}

ops_backend_lane_guard::~ops_backend_lane_guard() {
    if (lane_) static_cast<std::recursive_mutex*>(lane_)->unlock();
}

enum ggml_status ops_backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph* graph) {
    if (!backend || !graph) return GGML_STATUS_FAILED;
    ops_backend_lane_guard lane(backend);
    return ggml_backend_graph_compute(backend, graph);
}

enum ggml_status ops_backend_sched_graph_compute(ggml_backend_sched_t sched,
                                                 struct ggml_cgraph* graph) {
    if (!sched || !graph) return GGML_STATUS_FAILED;
    std::vector<ggml_backend_t> backends;
    const int count = ggml_backend_sched_get_n_backends(sched);
    for (int i = 0; i < count; ++i) {
        ggml_backend_t backend = ggml_backend_sched_get_backend(sched, i);
        if (backend) backends.push_back(backend);
    }
    // Lock in a stable global order so two schedulers sharing backends cannot
    // deadlock against each other.
    std::sort(backends.begin(), backends.end());
    backends.erase(std::unique(backends.begin(), backends.end()), backends.end());
    std::vector<std::shared_ptr<std::recursive_mutex>> lanes;
    lanes.reserve(backends.size());
    for (ggml_backend_t backend : backends) {
        lanes.push_back(backend_lane(backend));
        lanes.back()->lock();
    }
    const enum ggml_status status = ggml_backend_sched_graph_compute(sched, graph);
    for (auto it = lanes.rbegin(); it != lanes.rend(); ++it) (*it)->unlock();
    return status;
}

// The vtable signature cannot propagate a status, but a failed kernel leaves
// dst uninitialized while the graph keeps running — never let that pass
// silently.
static void ops_vtable_report_failure(struct ggml_tensor* node, const ops_status& status) {
    std::cerr << "[ggml-ops-ext] kernel failed for op " << (int)node->op
              << " (node '" << node->name << "'): "
              << (status.message ? status.message : "unknown error")
              << " — output left zero/uninitialized" << std::endl;
}

// CPU inline dispatch runs on a compute-pool thread inside a graph pass that
// the caller already serialized via ops_backend_graph_compute; taking the
// recursive lane mutex here would deadlock whenever that pool thread is not
// the caller thread (non-OpenMP threadpool builds).
static void ops_vtable_cpu_adapter(ggml_backend_t backend, struct ggml_tensor* node) {
    const ops_status status = execute_ops_kernel_impl(backend, node, /*take_lane=*/false);
    if (!status) ops_vtable_report_failure(node, status);
}

static void ops_vtable_backend_adapter(ggml_backend_t backend, struct ggml_tensor* node) {
    const ops_status status = execute_ops_kernel(backend, node);
    if (!status) ops_vtable_report_failure(node, status);
}

void acquire_ops_hook() {
    std::lock_guard<std::mutex> lock(g_hook_mutex);
    if (g_hook_users++ == 0) {
        {
            std::lock_guard<std::mutex> registry_lock(g_registry_mutex);
            g_registry_frozen.store(true, std::memory_order_release);
        }
        for (int op = GGML_OP_OPS_VIRT_BASE; op < GGML_OP_OPS_VIRT_COUNT; ++op) {
            g_ggml_cpu_op_vtable[op] = ops_vtable_cpu_adapter;
            g_ggml_cuda_op_vtable[op] = ops_vtable_backend_adapter;
            g_ggml_sycl_op_vtable[op] = ops_vtable_backend_adapter;
        }
        ggml_ops_ext_bridge_set_hooks(ops_ext_hook_impl, ops_ext_supports_impl);
    }
}

void release_ops_hook() {
    std::lock_guard<std::mutex> lock(g_hook_mutex);
    if (g_hook_users == 0) return;
    if (--g_hook_users == 0) {
        for (int op = GGML_OP_OPS_VIRT_BASE; op < GGML_OP_OPS_VIRT_COUNT; ++op) {
            g_ggml_cpu_op_vtable[op] = nullptr;
            g_ggml_cuda_op_vtable[op] = nullptr;
            g_ggml_sycl_op_vtable[op] = nullptr;
        }
        ggml_ops_ext_bridge_set_hooks(nullptr, nullptr);
    }
}

struct ggml_tensor* ops_new_virtual_node(
    struct ggml_context* ctx,
    ops_virt_op_type op,
    ggml_type type,
    int n_dims,
    const int64_t* ne,
    int n_srcs,
    struct ggml_tensor** srcs
) {
    if (type < 0 || type >= GGML_TYPE_COUNT) {
        std::cerr << "[ops_new_virtual_node ERROR] Invalid type=" << type 
                  << " for op=" << op << ", n_dims=" << n_dims 
                  << ", n_srcs=" << n_srcs << std::endl;
        if (n_srcs > 0 && srcs && srcs[0]) {
            std::cerr << "  srcs[0] name=" << srcs[0]->name 
                      << ", type=" << srcs[0]->type << std::endl;
        }
    }
    struct ggml_tensor* result = ggml_new_tensor(ctx, type, n_dims, ne);
    result->op = (enum ggml_op)op;
    GGML_ASSERT(n_srcs <= GGML_MAX_SRC);
    for (int i = 0; i < n_srcs; ++i) {
        result->src[i] = srcs[i];
    }
    return result;
}

} // namespace ggml_ops_ext

// Global namespace custom operator wrapper functions
bool ggml_ops_backend_supports_op(
    ggml_backend_t backend,
    int op_id,
    struct ggml_tensor* const* srcs,
    int n_srcs,
    const void* params,
    size_t params_size
) {
    if (!backend) return false;
    return ggml_ops_ext::probe_ops_kernel(
        ggml_backend_get_device(backend), op_id, srcs, n_srcs, params, params_size).supported;
}
