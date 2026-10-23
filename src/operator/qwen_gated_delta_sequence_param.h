// Copyright 2026 JPChen2000 and the Feather contributors.
// SPDX-License-Identifier: Apache-2.0

#ifndef FEATHER_OPERATOR_QWEN_GATED_DELTA_SEQUENCE_PARAM_H
#define FEATHER_OPERATOR_QWEN_GATED_DELTA_SEQUENCE_PARAM_H

#include "src/operator/params.h"

namespace feather {
namespace operators {

// Vectors use [1,T,H,D], gates use [1,T,H], and state uses [1,H,K,V].
// Each token updates the FP32 state before reading it with that token's q.
struct QwenGatedDeltaSequenceParam : ParamBase {
    std::shared_ptr<Tensor> state, k, v, beta, decay, q, next_state, out;
    template <typename Visitor> void VisitTensors(Visitor&& visit) {
        visit(state); visit(k); visit(v); visit(beta); visit(decay); visit(q);
        visit(next_state); visit(out);
    }
};

namespace qwen_sequence_detail {
inline bool HasCapacity(const Tensor& tensor, size_t bytes) {
    return tensor.IsInitialized() && tensor.memory_size() >= bytes && tensor.storage() &&
           tensor.offset() <= tensor.storage()->size() && bytes <= tensor.storage()->size() - tensor.offset();
}

inline bool Overlaps(const Tensor& a, const Tensor& b) {
    if (&a == &b) return true;
    if (!a.storage() || !b.storage()) return false;
    if (a.storage() == b.storage()) {
        return a.offset() <= b.offset() ? b.offset() - a.offset() < a.memory_size()
                                         : a.offset() - b.offset() < b.memory_size();
    }
    // Distinct Storage wrappers may still refer to the same host Buffer.
    if (!a.IsInitialized() || !b.IsInitialized() || !a.storage()->materialized() ||
        !b.storage()->materialized()) return false;
    const auto x = reinterpret_cast<uintptr_t>(a.raw_data());
    const auto y = reinterpret_cast<uintptr_t>(b.raw_data());
    return x <= y ? y - x < a.memory_size() : x - y < b.memory_size();
}

inline bool Validate(const QwenGatedDeltaSequenceParam* p, bool require_buffers) {
    if (!p || !p->state || !p->k || !p->v || !p->beta || !p->decay || !p->q ||
        !p->next_state || !p->out) return false;
    const auto& s = p->state->dims();
    const auto& k = p->k->dims();
    if (s.size() != 4 || s[0] != 1 || s[1] <= 0 || s[2] <= 0 || s[3] <= 0 ||
        k.size() != 4 || k[0] != 1 || k[1] <= 0 || k[2] != s[1] || k[3] != s[2]) return false;
    const std::vector<int64_t> key_shape = {1, k[1], s[1], s[2]};
    const std::vector<int64_t> value_shape = {1, k[1], s[1], s[3]};
    const std::vector<int64_t> gate_shape = {1, k[1], s[1]};
    if (p->q->dims().data() != key_shape || p->v->dims().data() != value_shape ||
        p->beta->dims().data() != gate_shape || p->decay->dims().data() != gate_shape) return false;
    for (const auto& input : {p->state, p->k, p->v, p->beta, p->decay, p->q}) {
        size_t bytes = 0;
        if (input->data_type() != DataType::FP32 ||
            !CheckedByteCount(input->dims().data(), sizeof(float), &bytes) ||
            (require_buffers && !HasCapacity(*input, bytes)) ||
            Overlaps(*input, *p->next_state) || Overlaps(*input, *p->out)) return false;
    }
    if (Overlaps(*p->next_state, *p->out)) return false;
    size_t state_bytes = 0, output_bytes = 0;
    if (!CheckedByteCount(s.data(), sizeof(float), &state_bytes) ||
        !CheckedByteCount(value_shape, sizeof(float), &output_bytes)) return false;
    for (const auto& output : {p->next_state, p->out}) {
        if (output->data_type() != DataType::UNKNOWN && output->data_type() != DataType::FP32) return false;
    }
    if (require_buffers &&
        (p->next_state->dims().data() != s.data() || p->out->dims().data() != value_shape ||
         p->next_state->data_type() != DataType::FP32 || p->out->data_type() != DataType::FP32 ||
         !HasCapacity(*p->next_state, state_bytes) || !HasCapacity(*p->out, output_bytes))) return false;
    return true;
}
} // namespace qwen_sequence_detail
} // namespace operators
} // namespace feather
#endif
