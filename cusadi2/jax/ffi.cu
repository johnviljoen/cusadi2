// XLA FFI handler for cusadi2 plans, plus the nanobind module that installs
// plans from Python (cusadi2.jax._ffi).
//
// Liveness (the spineax pattern): plans live in a process-wide registry keyed
// by plan id, held by shared_ptr. Every XLA custom call instantiates a State
// that points at the registry entry, so a plan stays alive for as long as any
// compiled executable references it, even after Python frees it.
//
// Each call is one kernel launch on XLA's stream with the argument buffer
// pointers passed by value (Program::evaluate_ptrs): no per-call staging.
#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include "xla/ffi/api/ffi.h"

// Let CUDA failures inside the header surface as Python exceptions.
#define CUSADI2_CHECK(call)                                                        \
    do {                                                                           \
        cudaError_t _e = (call);                                                   \
        if (_e != cudaSuccess)                                                     \
            throw std::runtime_error(std::string("cusadi2: CUDA error: ") +        \
                                     cudaGetErrorString(_e));                      \
    } while (0)
#include "cusadi2.h"

namespace ffi = xla::ffi;
namespace nb = nanobind;
using namespace nb::literals;

// One build per scalar type (-DCUSADI2_SCALAR=double|float, module name
// -DCUSADI2_FFI_MODULE=_ffi_f64|_ffi_f32): the kernel and Program are fixed to
// a scalar at compile time, so each dtype is its own extension module.
using Scalar = cusadi2::Scalar;
constexpr ffi::DataType kDType = std::is_same_v<Scalar, float> ? ffi::F32 : ffi::F64;
#ifndef CUSADI2_FFI_MODULE
#define CUSADI2_FFI_MODULE _ffi_f64
#endif

namespace {

struct Entry {
    cusadi2::Plan plan;
    cusadi2::Program prog;
};

struct Registry {
    static std::mutex mtx;
    static std::unordered_map<int64_t, std::shared_ptr<Entry>> map;
    static std::shared_ptr<Entry> get(int64_t id) {
        std::lock_guard<std::mutex> lock(mtx);
        auto it = map.find(id);
        return it == map.end() ? nullptr : it->second;
    }
};
std::mutex Registry::mtx;
std::unordered_map<int64_t, std::shared_ptr<Entry>> Registry::map;

// XLA-managed per-custom-call state: keeps the registry entry alive.
struct PlanState {
    static ffi::TypeId id;
    std::shared_ptr<Entry> entry;
};
ffi::TypeId PlanState::id = {};
XLA_FFI_TypeInfo kPlanStateInfo = ffi::MakeTypeInfo<PlanState>();   // deleter for XLA-owned states

ffi::ErrorOr<std::unique_ptr<PlanState>> Instantiate(int64_t plan_id, int64_t batch) {
    (void)batch;
    auto e = Registry::get(plan_id);
    if (!e)
        return ffi::Error::InvalidArgument("cusadi2: plan " + std::to_string(plan_id) +
                                           " is not installed (freed before compilation?)");
    auto st = std::make_unique<PlanState>();
    st->entry = std::move(e);
    return st;
}

ffi::Error Execute(cudaStream_t stream, PlanState* st, ffi::RemainingArgs args,
                   ffi::RemainingRets rets, int64_t plan_id, int64_t batch) {
    const Entry& e = *st->entry;
    const cusadi2::Plan& p = e.plan;
    if ((int)args.size() != p.n_in() || (int)rets.size() != p.n_out())
        return ffi::Error::InvalidArgument(
            "cusadi2: plan " + std::to_string(plan_id) + " expects " + std::to_string(p.n_in()) +
            " inputs / " + std::to_string(p.n_out()) + " outputs, got " + std::to_string(args.size()) +
            " / " + std::to_string(rets.size()));
    if (p.n_in() > CUSADI2_MAX_ARGS || p.n_out() > CUSADI2_MAX_ARGS)
        return ffi::Error::InvalidArgument("cusadi2: too many arguments for the by-value launch");

    const Scalar* in[CUSADI2_MAX_ARGS] = {};
    Scalar* out[CUSADI2_MAX_ARGS] = {};
    for (int a = 0; a < p.n_in(); ++a) {
        auto buf = args.get<ffi::Buffer<kDType>>(a);
        if (!buf.has_value()) return buf.error();
        size_t want = p.in[a].len((int)batch);
        if (buf.value().element_count() != want)
            return ffi::Error::InvalidArgument(
                "cusadi2: input " + std::to_string(a) + " has " + std::to_string(buf.value().element_count()) +
                " elements, plan wants " + std::to_string(want) + " for batch " + std::to_string(batch));
        in[a] = buf.value().typed_data();
    }
    for (int k = 0; k < p.n_out(); ++k) {
        auto r = rets.get<ffi::Buffer<kDType>>(k);
        if (!r.has_value()) return r.error();
        size_t want = p.out[k].len((int)batch);
        if ((*r.value()).element_count() != want)
            return ffi::Error::InvalidArgument(
                "cusadi2: output " + std::to_string(k) + " has " + std::to_string((*r.value()).element_count()) +
                " elements, plan wants " + std::to_string(want) + " for batch " + std::to_string(batch));
        out[k] = (*r.value()).typed_data();
    }
    int err = e.prog.evaluate_ptrs(in, out, (int)batch, stream);
    if (err)
        return ffi::Error::Internal(std::string("cusadi2: launch failed: ") +
                                    cudaGetErrorString((cudaError_t)err));
    return ffi::Error::Success();
}

}  // namespace

XLA_FFI_DEFINE_HANDLER(kCusadi2Instantiate, Instantiate,
                       ffi::Ffi::BindInstantiate().Attr<int64_t>("plan_id").Attr<int64_t>("batch"));
XLA_FFI_DEFINE_HANDLER(kCusadi2Execute, Execute,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::PlatformStream<cudaStream_t>>()
                           .Ctx<ffi::State<PlanState>>()
                           .RemainingArgs()
                           .RemainingRets()
                           .Attr<int64_t>("plan_id")
                           .Attr<int64_t>("batch"));

NB_MODULE(CUSADI2_FFI_MODULE, m) {
    m.doc() = "cusadi2 XLA FFI handler and plan registry";
    m.attr("DTYPE") = std::is_same_v<Scalar, float> ? "float32" : "float64";

    m.def(
        "install_plan",
        [](int64_t plan_id, nb::bytes data, int block_size, bool serial, size_t scratch_bytes,
           bool thread_major) {
            auto e = std::make_shared<Entry>();
            e->plan = cusadi2::load_plan_from_memory(data.c_str(), data.size(),
                                                     "plan " + std::to_string(plan_id));
            if (block_size) e->plan.block_size = block_size;
            e->prog.upload(e->plan);
            if (serial) e->prog.use_serial(scratch_bytes, 256, 1 << 20, thread_major);
            std::lock_guard<std::mutex> lock(Registry::mtx);
            Registry::map[plan_id] = std::move(e);
        },
        "plan_id"_a, "data"_a, "block_size"_a = 0, "serial"_a = false,
        "scratch_bytes"_a = size_t(256) << 20, "thread_major"_a = false,
        "Upload a plan (the bytes of a plan file) to the current device under plan_id.");

    m.def(
        "free_plan",
        [](int64_t plan_id) {
            std::lock_guard<std::mutex> lock(Registry::mtx);
            Registry::map.erase(plan_id);   // executables that hold a State keep it alive
        },
        "plan_id"_a);

    m.def("has_plan", [](int64_t plan_id) { return Registry::get(plan_id) != nullptr; }, "plan_id"_a);

    m.def("plan_nnz", [](int64_t plan_id) {
        auto e = Registry::get(plan_id);
        if (!e) throw std::runtime_error("cusadi2: plan not installed");
        std::vector<int> nnz_in, nnz_out;
        for (auto& a : e->plan.in) nnz_in.push_back(a.nnz());
        for (auto& a : e->plan.out) nnz_out.push_back(a.nnz());
        return std::make_pair(nnz_in, nnz_out);
    }, "plan_id"_a);

    m.def("type_id", []() { return nb::capsule(reinterpret_cast<void*>(&PlanState::id)); });
    m.def("type_info", []() { return nb::capsule(reinterpret_cast<void*>(&kPlanStateInfo)); });
    m.def("handler", []() {
        nb::dict d;
        d["instantiate"] = nb::capsule(reinterpret_cast<void*>(kCusadi2Instantiate));
        d["execute"] = nb::capsule(reinterpret_cast<void*>(kCusadi2Execute));
        return d;
    });
    m.attr("MAX_ARGS") = CUSADI2_MAX_ARGS;
}
