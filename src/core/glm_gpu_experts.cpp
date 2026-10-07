// src/core/glm_gpu_experts.cpp - see include/strata/core/glm_gpu_experts.hpp.
#include "strata/core/glm_gpu_experts.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace strata::core {

namespace {
constexpr size_t kQ8_1Block = 36;   // 32 int8 + half d + half s
}

GlmGpuExperts::~GlmGpuExperts() {
    if (xq_) cudaFree(xq_);
    if (scratch_) cudaFree(scratch_);
    if (d_ptr_) cudaFree(d_ptr_);
    if (d_idx_) cudaFree(d_idx_);
    if (h_ptr_) cudaFreeHost(h_ptr_);
    if (h_idx_) cudaFreeHost(h_idx_);
    for (uint8_t* p : stage_) cudaFreeHost(p);
}

bool GlmGpuExperts::init(ExpertSource* src, const std::vector<int>& gu_type, const std::vector<int>& d_type,
                         const std::vector<uint64_t>& blob_bytes, int64_t n_expert, int64_t k, int64_t n_embd,
                         int64_t n_ff, int64_t budget_bytes, std::string& err) {
    src_ = src;
    n_layers_ = (int64_t) blob_bytes.size();
    n_expert_ = n_expert;
    k_ = k;
    n_embd_ = n_embd;
    n_ff_ = n_ff;
    lay_.assign((size_t) n_layers_, {});
    int64_t moe_layers = 0, moe_bytes = 0;
    for (int64_t l = 0; l < n_layers_; ++l) {
        if (blob_bytes[(size_t) l] == 0) continue;
        if (!strata::kernels::native_expert_supported(gu_type[(size_t) l], d_type[(size_t) l], n_embd, n_ff)) {
            err = "glm gpu experts: layer " + std::to_string(l) + " has no GPU kernels for its formats";
            return false;
        }
        lay_[(size_t) l] = strata::kernels::native_expert_layout(gu_type[(size_t) l], d_type[(size_t) l], n_embd, n_ff);
        if (lay_[(size_t) l].bytes != blob_bytes[(size_t) l]) {
            err = "glm gpu experts: layer " + std::to_string(l) + ": the pack's blob is " +
                  std::to_string(blob_bytes[(size_t) l]) + " B, the kernels' layout " +
                  std::to_string(lay_[(size_t) l].bytes) + " B";
            return false;
        }
        ++moe_layers;
        moe_bytes += (int64_t) blob_bytes[(size_t) l];
    }
    if (moe_layers == 0) { err = "glm gpu experts: no MoE layer"; return false; }
    // An even number of slots per MoE layer: a layer routes k of n_expert every token, so a quota that favours
    // the first layers seen (the first family's R4.2g lesson) would leave the others with nothing.
    const int64_t per_layer = std::min<int64_t>(n_expert, budget_bytes / moe_bytes);
    if (per_layer < 1) { err = "glm gpu experts: the VRAM budget does not hold one expert per layer"; return false; }
    std::vector<int64_t> sizes;
    lo_.assign((size_t) n_layers_, 0);
    hi_.assign((size_t) n_layers_, 0);
    for (int64_t l = 0; l < n_layers_; ++l) {
        lo_[(size_t) l] = (int64_t) sizes.size();
        if (blob_bytes[(size_t) l] != 0) sizes.insert(sizes.end(), (size_t) per_layer, (int64_t) blob_bytes[(size_t) l]);
        hi_[(size_t) l] = (int64_t) sizes.size();
    }
    next_ = lo_;
    if (!cache_.open_sized(sizes, n_layers_, n_expert, err)) return false;
    slot_.assign((size_t) (n_layers_ * n_expert), kNotResident);
    owner_.assign((size_t) cache_.full_slots(), -1);
    count_.assign((size_t) (n_layers_ * n_expert), 0);
    moe_layers_ = moe_layers;

    const size_t xq_bytes = (size_t) (n_embd / 32) * kQ8_1Block;
    const size_t scratch = strata::kernels::native_expert_scratch_bytes(k, n_ff);
    if (cudaMalloc(&xq_, xq_bytes) != cudaSuccess || cudaMalloc(&scratch_, scratch) != cudaSuccess ||
        cudaMalloc((void**) &d_ptr_, (size_t) k * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc((void**) &d_idx_, (size_t) (3 * k + 2) * sizeof(int32_t)) != cudaSuccess ||
        cudaMallocHost((void**) &h_ptr_, (size_t) k * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMallocHost((void**) &h_idx_, (size_t) (3 * k + 2) * sizeof(int32_t)) != cudaSuccess) {
        err = std::string("glm gpu experts: buffers: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    uint64_t max_blob = 0;
    for (uint64_t b : blob_bytes) max_blob = std::max(max_blob, b);
    stage_.assign((size_t) k, nullptr);
    for (auto& p : stage_) {
        if (cudaMallocHost((void**) &p, (size_t) max_blob) != cudaSuccess) {
            err = std::string("glm gpu experts: staging: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
    }
    std::fprintf(stderr, "strata generate: glm5-next experts in VRAM: %lld slots a MoE layer, %lld in all, %.2f GiB "
                         "(the rest on the CPU; slots fill as experts are routed)\n",
                 (long long) per_layer, (long long) cache_.full_slots(), cache_.gib());
    return true;
}

bool GlmGpuExperts::run_hits(int64_t layer, const float* cur_dev, const int32_t* ids, float* parts_dev, void* stream,
                             std::vector<int32_t>& miss, std::string& err) {
    miss.clear();
    hit_pos_.clear();
    // Every 4096 tokens the counts are halved, so an expert hot an hour ago does not keep its slot for ever.
    if (++calls_ % (4096 * moe_layers_) == 0)
        for (uint32_t& c : count_) c >>= 1;
    for (int64_t i = 0; i < k_; ++i) {
        const int32_t e = ids[i];
        if (e >= 0 && e < n_expert_) ++count_[(size_t) (layer * n_expert_ + e)];
        const int32_t s = (e >= 0 && e < n_expert_) ? slot_[(size_t) (layer * n_expert_ + e)] : kNotResident;
        if (s == kNotResident) {
            miss.push_back((int32_t) i);
            continue;
        }
        h_ptr_[hit_pos_.size()] = (unsigned long long) (uintptr_t) cache_.device_slot(s);
        hit_pos_.push_back((int32_t) i);
    }
    hits_ += (int64_t) hit_pos_.size();
    misses_ += (int64_t) miss.size();
    rep_hits_ += (int64_t) hit_pos_.size();
    rep_miss_ += (int64_t) miss.size();
    if (calls_ % (256 * moe_layers_) == 0) {
        std::fprintf(stderr, "strata glm gpu experts: last 256 tokens %.1f%% hits, %lld resident, %lld swaps so far\n",
                     100.0 * (double) rep_hits_ / (double) std::max<int64_t>(1, rep_hits_ + rep_miss_),
                     (long long) admitted_, (long long) swaps_);
        rep_hits_ = rep_miss_ = 0;
    }
    const int32_t ng = (int32_t) hit_pos_.size();
    if (ng == 0) return true;
    // One group per hit, one entry per group: start[g] = g, dst = the hit's position in `parts`, tok = 0.
    int32_t* start = h_idx_;
    int32_t* n_groups = h_idx_ + k_ + 1;
    int32_t* dst = h_idx_ + k_ + 2;
    int32_t* tok = h_idx_ + 2 * k_ + 2;
    for (int32_t g = 0; g <= ng; ++g) start[g] = g;
    *n_groups = ng;
    for (int32_t g = 0; g < ng; ++g) { dst[g] = hit_pos_[(size_t) g]; tok[g] = 0; }
    cudaStream_t cs = (cudaStream_t) stream;
    if (cudaMemcpyAsync(d_ptr_, h_ptr_, (size_t) ng * sizeof(unsigned long long), cudaMemcpyHostToDevice, cs) !=
            cudaSuccess ||
        cudaMemcpyAsync(d_idx_, h_idx_, (size_t) (3 * k_ + 2) * sizeof(int32_t), cudaMemcpyHostToDevice, cs) !=
            cudaSuccess) {
        err = std::string("glm gpu experts: index upload: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    strata::kernels::quantize_q8_1_rows(cur_dev, 1, n_embd_, xq_, stream);
    strata::kernels::native_expert_grouped(lay_[(size_t) layer], d_ptr_, d_idx_, d_idx_ + k_ + 1, d_idx_ + k_ + 2,
                                           d_idx_ + 2 * k_ + 2, k_, k_, xq_, scratch_, parts_dev, stream, ng);
    return true;
}

bool GlmGpuExperts::admit(int64_t layer, const int32_t* ids, const std::vector<int32_t>& miss, void* stream,
                          std::string& err) {
    const int64_t bytes = (int64_t) lay_[(size_t) layer].bytes;
    size_t used = 0;
    const uint32_t* cnt = count_.data() + (size_t) (layer * n_expert_);
    bool swapped = false;
    for (int32_t i : miss) {
        const int32_t e = ids[i];
        if (e < 0 || e >= n_expert_ || slot_[(size_t) (layer * n_expert_ + e)] != kNotResident) continue;
        int32_t s = kNotResident;
        if (next_[(size_t) layer] < hi_[(size_t) layer]) {
            s = (int32_t) next_[(size_t) layer]++;
            ++admitted_;
        } else if (!swapped) {
            // LFU: the least routed expert this layer holds gives its slot up, if the newcomer is clearly hotter
            // (+2, so two experts of equal heat do not trade places every token).  One swap a layer a token.
            int32_t victim = kNotResident;
            for (int64_t v = lo_[(size_t) layer]; v < hi_[(size_t) layer]; ++v)
                if (victim == kNotResident || cnt[owner_[(size_t) v]] < cnt[owner_[(size_t) victim]]) victim = (int32_t) v;
            if (victim == kNotResident || cnt[e] < cnt[owner_[(size_t) victim]] + 2) continue;
            slot_[(size_t) (layer * n_expert_ + owner_[(size_t) victim])] = kNotResident;
            s = victim;
            swapped = true;
            ++swaps_;
        } else {
            continue;
        }
        uint8_t* host = stage_[used++];
        if (!src_->copy_blob(layer, e, host)) {
            err = "glm gpu experts: reading layer " + std::to_string(layer) + " expert " + std::to_string(e);
            return false;
        }
        // The copy is ordered on `stream` behind this layer's hit kernel, which may still be reading a victim
        // slot; the next read of the slot is a later token's kernel on the same stream, behind the copy.  The
        // staging buffer is reused only after the caller's next stream sync.
        if (!cache_.fill_slot(s, host, stream, err, bytes)) return false;
        slot_[(size_t) (layer * n_expert_ + e)] = s;
        owner_[(size_t) s] = e;
    }
    return true;
}

}  // namespace strata::core
