// include/strata/core/glm_gpu_experts.hpp - glm5-next's routed experts in VRAM.
//
// The PR that brought glm5-next computes every routed expert on the CPU (`GlmExpertPool`).  On a 24 GB card
// with 64 GB of RAM and a 92 GiB expert set, that is ~2.5 tok/s: each token reads 42 layers x 8 experts from
// RAM or disk on the CPU.  This is the first family's split, applied to GLM: a per-layer set of experts lives
// in VRAM, the GPU computes the ones it holds with `native_expert_grouped`, and the CPU pool only gets the
// misses.  Slots are filled as experts are routed (per-layer quota), then a slot goes to an expert routed
// clearly more often than the least routed one it holds (LFU: one swap a layer a token at most, counts
// halved every 4096 tokens so the set follows the conversation).  A run warms up over its first tokens.
//
// Off unless `--glm-gpu-experts` is given.  STRATA_GLM_GPU_CHECK=1 also computes every hit on the CPU and
// prints the largest difference, which is how this was checked against the CPU pool.
#pragma once

#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

class GlmGpuExperts {
public:
    GlmGpuExperts() = default;
    ~GlmGpuExperts();
    GlmGpuExperts(const GlmGpuExperts&) = delete;
    GlmGpuExperts& operator=(const GlmGpuExperts&) = delete;

    /// `gu_type` / `d_type` / `blob_bytes`: per layer, from the pack's native layout (`blob_bytes[l] == 0` for a
    /// dense layer).  `budget_bytes`: the VRAM the slots may take, split evenly over the MoE layers.
    bool init(ExpertSource* src, const std::vector<int>& gu_type, const std::vector<int>& d_type,
              const std::vector<uint64_t>& blob_bytes, int64_t n_expert, int64_t k, int64_t n_embd, int64_t n_ff,
              int64_t budget_bytes, std::string& err);
    bool valid() const { return cache_.valid(); }

    /// One decode token, one MoE layer.  `ids` are the routed experts (HOST, k of them).  The hits are launched on
    /// `stream` and write their UNWEIGHTED rows `i` of `parts_dev` (k x n_embd).  `miss` receives the positions
    /// `i` the CPU still has to compute.  `cur_dev`: the layer's normed FFN input on the device.
    bool run_hits(int64_t layer, const float* cur_dev, const int32_t* ids, float* parts_dev, void* stream,
                  std::vector<int32_t>& miss, std::string& err);

    /// After the CPU computed the misses: copy them into this layer's free slots (async on `stream`).
    bool admit(int64_t layer, const int32_t* ids, const std::vector<int32_t>& miss, void* stream, std::string& err);

    /// STRATA_GLM_GPU_CHECK: the last `run_hits` launched hits for these positions.
    const std::vector<int32_t>& last_hits() const { return hit_pos_; }

    int64_t hits() const { return hits_; }
    int64_t misses() const { return misses_; }
    int64_t resident() const { return admitted_; }
    int64_t slots() const { return cache_.full_slots(); }
    double gib() const { return cache_.gib(); }

private:
    ExpertCache cache_;
    ExpertSource* src_ = nullptr;
    int64_t n_layers_ = 0, n_expert_ = 0, k_ = 0, n_embd_ = 0, n_ff_ = 0;
    std::vector<strata::kernels::NativeExpertLayout> lay_;
    std::vector<int64_t> lo_, hi_, next_;      ///< per layer: its slot range and the next free slot in it
    std::vector<int32_t> slot_;                ///< [layer * n_expert + expert] -> slot or kNotResident
    std::vector<int32_t> owner_;               ///< [slot] -> the expert it holds, or -1
    std::vector<uint32_t> count_;              ///< [layer * n_expert + expert]: how often it was routed
    int64_t moe_layers_ = 0, calls_ = 0, swaps_ = 0, rep_hits_ = 0, rep_miss_ = 0;
    void* xq_ = nullptr;                       ///< the token's q8_1 activation
    void* scratch_ = nullptr;
    unsigned long long* d_ptr_ = nullptr;      ///< device: one group per hit, its blob's address
    int32_t* d_idx_ = nullptr;                 ///< device: start[k+1] | n_groups | dst[k] | tok[k]
    unsigned long long* h_ptr_ = nullptr;      ///< pinned host mirrors
    int32_t* h_idx_ = nullptr;
    std::vector<uint8_t*> stage_;              ///< pinned staging, one blob per admission in flight
    std::vector<int32_t> hit_pos_;
    int64_t hits_ = 0, misses_ = 0, admitted_ = 0;
};

}  // namespace strata::core
