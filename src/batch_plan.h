#pragma once
#include <any>
#include <cfenv>
#include <cstddef>
#include <cstdint>
#include <optional>
#if defined(__SSE__)
#include <xmmintrin.h>
#endif

namespace tesseract_decoder::batch_detail {
// Private frozen owner only. Public mutable decoders use short BatchPlanScope.
enum class PreparationPolicy { LazyRows, EagerRows };
constexpr size_t eager_row_allocation_limit=1024*1024;
template<class Rows>
bool eager_rows_fit_budget(const Rows& rows, size_t vector_overhead,
                          size_t ready_bytes, size_t entry_bytes) {
  if (!entry_bytes || vector_overhead > eager_row_allocation_limit ||
      ready_bytes > eager_row_allocation_limit-vector_overhead) return false;
  const size_t per_row=vector_overhead+ready_bytes;
  if (per_row && rows.size()>eager_row_allocation_limit/per_row) return false;
  size_t remaining=eager_row_allocation_limit-rows.size()*per_row;
  for (const auto& row : rows) {
    if (row.size()>remaining/entry_bytes) return false;
    remaining-=row.size()*entry_bytes;
  }
  return true;
}
struct BatchPlanState {
  explicit BatchPlanState(const void* decoder, PreparationPolicy policy = PreparationPolicy::EagerRows)
      : owner(decoder), policy(policy) {}
  const void* const owner;
  const PreparationPolicy policy;
  std::any prepared;
  std::optional<bool> compact_graph_eligible;
  uint64_t builds = 0, hits = 0;
  uint64_t eager_builds = 0, eager_hits = 0, compact_builds = 0, compact_hits = 0;
  uint64_t environment = UINT64_MAX;
  void invalidate() { prepared.reset(); compact_graph_eligible.reset(); }
  BatchPlanState(const BatchPlanState&) = delete;
  BatchPlanState& operator=(const BatchPlanState&) = delete;
};
inline uint64_t batch_environment() {
  uint64_t key = static_cast<uint32_t>(std::fegetround());
#if defined(__SSE__)
  // Control bits, including rounding, DAZ, FTZ and exception masks; exclude flags.
  key |= uint64_t(_mm_getcsr() & 0xffc0U) << 32;
#endif
  return key;
}
inline bool batch_environment_supported() {
#if defined(__SSE__)
  // Stim's normal environment may enable FTZ. Stable DAZ/FTZ is valid:
  // preparation and use execute under identical controls recorded in the key.
  return std::fegetround() == FE_TONEAREST;
#else
  // Other architectures can have flush/rounding controls not represented above.
  return false;
#endif
}
struct BatchPlanFrame {
  BatchPlanState* state;
  BatchPlanFrame* ancestor;
};
extern thread_local BatchPlanState* active_batch_plan;
extern thread_local BatchPlanFrame* active_batch_frame;
// Defined in the decoder translation unit: a gate must not silently exercise
// only bypass paths if TLS discovery or floating environment differs.
void verify_batch_plan_visible(const BatchPlanState* expected);
class BatchPlanActivation {
 public:
  explicit BatchPlanActivation(BatchPlanState& state)
      : frame_{&state, active_batch_frame}, previous_(active_batch_plan) {
    const auto environment = batch_environment();
    if (state.environment != environment) {
      state.invalidate();
      state.environment = environment;
    }
    active_batch_frame = &frame_;
    active_batch_plan = &state;
  }
  ~BatchPlanActivation() {
    active_batch_frame = frame_.ancestor;
    active_batch_plan = previous_;
  }
  BatchPlanActivation(const BatchPlanActivation&) = delete;
  BatchPlanActivation& operator=(const BatchPlanActivation&) = delete;
 private:
  BatchPlanFrame frame_;
  BatchPlanState* previous_;
};
class BatchPlanScope {
 public:
  explicit BatchPlanScope(const void* owner)
      : state_(owner, PreparationPolicy::LazyRows), activation_(state_) {}
  BatchPlanScope(const BatchPlanScope&) = delete;
  BatchPlanScope& operator=(const BatchPlanScope&) = delete;
  const BatchPlanState& stats() const { return state_; }
 private:
  BatchPlanState state_;
  BatchPlanActivation activation_;
};
class SuspendBatchPlan {
 public:
  SuspendBatchPlan() : previous_(active_batch_plan), previous_frame_(active_batch_frame) {
    active_batch_plan = nullptr;
    active_batch_frame = nullptr;
  }
  ~SuspendBatchPlan() {
    for (auto* frame = previous_frame_; frame; frame = frame->ancestor)
      frame->state->invalidate();
    active_batch_plan = previous_;
    active_batch_frame = previous_frame_;
  }
  SuspendBatchPlan(const SuspendBatchPlan&) = delete;
  SuspendBatchPlan& operator=(const SuspendBatchPlan&) = delete;
 private:
  BatchPlanState* previous_;
  BatchPlanFrame* previous_frame_;
};
}
