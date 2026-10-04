// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Validated large-code net-parent default candidate; not a release declaration.
// Feature selection is implementation-local and changes no public header.
// The opt-out takes precedence over externally supplied algorithm feature flags.
// Existing ensemble certificate disable switches remain independent.
#ifdef TESSERACT_DISABLE_ALGORITHM_OPTIMIZATIONS
#undef TESSERACT_LAZY_CHILD_COUNTS
#undef TESSERACT_EARLY_CHILD_BEAM
#undef TESSERACT_RANKED_COSTS
#undef TESSERACT_RANK_VALUE_TABLE
#undef TESSERACT_COMPACT_RANK_METADATA
#undef TESSERACT_COMPACT_TUPLES
#undef TESSERACT_FACTORED_NEIGHBOR_INIT
#undef TESSERACT_EXACT_NEIGHBOR_RESERVE
#undef TESSERACT_NET_SIBLING_COUNTS
#undef TESSERACT_LOCAL_RANK_ROWS
#undef TESSERACT_SPLIT_COMPACT_STORAGE
#undef TESSERACT_BYTE_RANK_METADATA
#undef TESSERACT_HYBRID_STATE
#undef TESSERACT_NET_PARENT_COUNTS
#undef TESSERACT_INCREMENTAL_STATE
#undef TESSERACT_PACKED_COSTS
#undef TESSERACT_PREFIX_CACHE
#undef TESSERACT_SKIP_TERMINAL_STATE
#else
#ifndef TESSERACT_LAZY_CHILD_COUNTS
#define TESSERACT_LAZY_CHILD_COUNTS
#endif
#ifndef TESSERACT_EARLY_CHILD_BEAM
#define TESSERACT_EARLY_CHILD_BEAM
#endif
#ifndef TESSERACT_RANKED_COSTS
#define TESSERACT_RANKED_COSTS
#endif
#ifndef TESSERACT_RANK_VALUE_TABLE
#define TESSERACT_RANK_VALUE_TABLE
#endif
#ifndef TESSERACT_COMPACT_RANK_METADATA
#define TESSERACT_COMPACT_RANK_METADATA
#endif
#ifndef TESSERACT_COMPACT_TUPLES
#define TESSERACT_COMPACT_TUPLES
#endif
#ifndef TESSERACT_FACTORED_NEIGHBOR_INIT
#define TESSERACT_FACTORED_NEIGHBOR_INIT
#endif
#ifndef TESSERACT_EXACT_NEIGHBOR_RESERVE
#define TESSERACT_EXACT_NEIGHBOR_RESERVE
#endif
#ifndef TESSERACT_NET_SIBLING_COUNTS
#define TESSERACT_NET_SIBLING_COUNTS
#endif
#ifndef TESSERACT_LOCAL_RANK_ROWS
#define TESSERACT_LOCAL_RANK_ROWS
#endif
#ifndef TESSERACT_SPLIT_COMPACT_STORAGE
#define TESSERACT_SPLIT_COMPACT_STORAGE
#endif
#ifndef TESSERACT_BYTE_RANK_METADATA
#define TESSERACT_BYTE_RANK_METADATA
#endif
#ifndef TESSERACT_HYBRID_STATE
#define TESSERACT_HYBRID_STATE
#endif
#ifndef TESSERACT_NET_PARENT_COUNTS
#define TESSERACT_NET_PARENT_COUNTS
#endif
#endif

#include "tesseract.h"
#include "batch_plan.h"
#include <memory>
#if defined(TESSERACT_NET_PARENT_COUNTS) && !defined(TESSERACT_HYBRID_STATE)
#error "TESSERACT_NET_PARENT_COUNTS requires TESSERACT_HYBRID_STATE"
#endif
#if defined(TESSERACT_VERIFY_NET_PARENT_COUNTS) && !defined(TESSERACT_VERIFY_HYBRID_STATE)
#define TESSERACT_VERIFY_HYBRID_STATE
#endif
#if defined(TESSERACT_NET_SIBLING_COUNTS) && !defined(TESSERACT_LAZY_CHILD_COUNTS)
#error "TESSERACT_NET_SIBLING_COUNTS requires TESSERACT_LAZY_CHILD_COUNTS"
#endif
#if defined(TESSERACT_VERIFY_NET_SIBLING_COUNTS) && !defined(TESSERACT_VERIFY_COMPACT_STATE)
// Reuse the existing independent uint32 full-state reconstruction oracle.
#define TESSERACT_VERIFY_COMPACT_STATE
#endif

#include <algorithm>
#include <bit>
#include <boost/functional/hash.hpp>  // For boost::hash_range
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <utility>
#include <type_traits>

namespace {

#ifdef TESSERACT_SPLIT_COMPACT_STORAGE
// Same vector-like operations as the trial uses. Counts are byte-addressable;
// blocker writes touch only their packed bitset, not the count byte.
template <typename Value>
class SplitCompactStorage {
 public:
  struct Reference {
    boost::dynamic_bitset<>::reference error_blocked;
    uint8_t& detectors_count;
  };
  SplitCompactStorage() = default;
  explicit SplitCompactStorage(size_t size) : counts(size, 0), blockers(size, false) {}
  Reference operator[](size_t index) { return {blockers[index], counts[index]}; }
  Value operator[](size_t index) const {
    Value result{};
    result.error_blocked = blockers[index];
    result.detectors_count = counts[index];
    return result;
  }
  size_t size() const { return counts.size(); }
  void clear() { counts.clear(); blockers.clear(); }
  void push_back(const Value& value) {
    counts.push_back(value.detectors_count);
    blockers.push_back(bool(value.error_blocked));
  }

 private:
  std::vector<uint8_t> counts;
  boost::dynamic_bitset<> blockers;
};
#endif

template <typename Callback>
class ScopeExit {
 public:
  explicit ScopeExit(Callback callback) : callback(std::move(callback)) {}
  ScopeExit(const ScopeExit&) = delete;
  ScopeExit& operator=(const ScopeExit&) = delete;
  ~ScopeExit() noexcept {
    callback();
  }

 private:
  Callback callback;
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
  os << "[";
  bool is_first = true;
  for (const auto& x : vec) {
    if (!is_first) {
      os << ", ";
    }
    is_first = false;
    os << x;
  }
  return os << "]";
}

int suggest_sparsify_reactivate_limit_capped(size_t num_detectors, int sparsify_base_degree,
                                             int max_limit) {
  if (sparsify_base_degree < 0) {
    throw std::invalid_argument("sparsify_base_degree must be >= 0.");
  }
  if (num_detectors == 0 || max_limit <= 0) {
    return 0;
  }
  double exponent = static_cast<double>(sparsify_base_degree) - 2.0;
  double max_result = static_cast<double>(max_limit);
  double log_result =
      exponent * std::log(4.5) - std::log(3.0) + std::log(static_cast<double>(num_detectors));
  if (!std::isfinite(log_result) || log_result >= std::log(max_result)) {
    return max_limit;
  }
  double result = std::exp(log_result);
  if (!std::isfinite(result)) {
    return max_limit;
  }
  double rounded = std::round(result);
  if (rounded >= max_result) {
    return max_limit;
  }
  return static_cast<int>(rounded);
}

};  // namespace

namespace std {
template <>
struct hash<boost::dynamic_bitset<>> {
  size_t operator()(const boost::dynamic_bitset<>& bs) const {
    // Delegate to Boost's internal hash_value for dynamic_bitset
    // This is the correct and most efficient way.
    return boost::hash_value(bs);
  }
};
}  // namespace std

namespace tesseract_decoder {

std::string TesseractConfig::str() {
  auto& config = *this;
  std::stringstream ss;
  ss << "TesseractConfig(";
  ss << "dem=DetectorErrorModel_Object" << ", ";
  ss << "det_beam=" << config.det_beam << ", ";
  ss << "no_revisit_dets=" << config.no_revisit_dets << ", ";

  ss << "verbose=" << config.verbose << ", ";
  ss << "merge_errors=" << config.merge_errors << ", ";
  ss << "pqlimit=" << config.pqlimit << ", ";
  ss << "num_detector_orders=" << config.detector_orders.size() << ", ";
  ss << "det_penalty=" << config.det_penalty << ", ";
  ss << "create_visualization=" << config.create_visualization;
  ss << ")";
  return ss.str();
}

int suggest_sparsify_reactivate_limit(size_t num_detectors, int sparsify_base_degree) {
  return suggest_sparsify_reactivate_limit_capped(num_detectors, sparsify_base_degree,
                                                  std::numeric_limits<int>::max());
}

std::string Node::str() {
  std::stringstream ss;
  auto& self = *this;
  ss << "Node(";
  ss << "error_chain_idx=" << self.error_chain_idx << ", ";
  ss << "cost=" << self.cost << ", ";
  ss << "num_dets=" << self.num_dets << ", ";
  return ss.str();
}

bool Node::operator>(const Node& other) const {
  return cost > other.cost || (cost == other.cost && num_dets < other.num_dets);
}

double TesseractDecoder::get_detcost(
    size_t d, const std::vector<DetectorCostTuple>& detector_cost_tuples) const {
  return get_detcost(d, detector_cost_tuples, d2e);
}

double TesseractDecoder::get_detcost(size_t d,
                                     const std::vector<DetectorCostTuple>& detector_cost_tuples,
                                     const std::vector<std::vector<int>>& active_d2e) const {
  double min_cost = INF;
  uint32_t min_det_cost_det_count = std::numeric_limits<uint32_t>::max();
  double error_cost;
  ErrorCost ec;
  DetectorCostTuple dct;

  for (int ei : active_d2e[d]) {
    ec = error_costs[ei];
    if (ec.likelihood_cost * min_det_cost_det_count >=
        min_cost * errors[ei].symptom.detectors.size())
      break;

    dct = detector_cost_tuples[ei];
    if (!dct.error_blocked) {
      error_cost = ec.likelihood_cost;
      if (error_cost * min_det_cost_det_count < min_cost * dct.detectors_count) {
        min_cost = error_cost;
        min_det_cost_det_count = dct.detectors_count;
      }
    }
  }

  return (min_cost / min_det_cost_det_count) + config.det_penalty;
}

TesseractDecoder::TesseractDecoder(TesseractConfig config_) : config(std::move(config_)) {
  if (config.detector_orders.empty()) {
    throw std::invalid_argument("At least one detector order is required.");
  }
  config.dem = common::flatten(config.dem);

  std::vector<size_t> dem_error_map(config.dem.count_errors());
  std::iota(dem_error_map.begin(), dem_error_map.end(), 0);

  if (config.merge_errors) {
    std::vector<size_t> merge_map;
    config.dem = common::merge_indistinguishable_errors(config.dem, merge_map);
    common::chain_error_maps(dem_error_map, merge_map);
  }

  std::vector<size_t> nonzero_map;
  config.dem = common::remove_zero_probability_errors(config.dem, nonzero_map);
  common::chain_error_maps(dem_error_map, nonzero_map);

  dem_error_to_error = std::move(dem_error_map);
  error_to_dem_error = common::invert_error_map(dem_error_to_error, config.dem.count_errors());

  resolve_detector_orders(config.detector_orders, config.dem);
  errors = get_errors_from_dem(config.dem);
  if (config.verbose) {
    for (auto& error : errors) {
      std::cout << error.str() << "\n";
    }
    std::cout << std::flush;
  }
  num_detectors = config.dem.count_detectors();
  num_errors = config.dem.count_errors();
  num_observables = config.dem.count_observables();
  initialize_structures(config.dem.count_detectors());
  if (config.create_visualization) {
    auto detectors = get_detector_coords(config.dem);
    visualizer.add_detector_coords(detectors);
    visualizer.add_errors(errors);
  }
}

TesseractDecoder::~TesseractDecoder() = default;

size_t TesseractDecoder::error_count() const {
  return errors.size();
}

const common::Symptom& TesseractDecoder::error_symptom(size_t error_index) const {
  return errors.at(error_index).symptom;
}

double TesseractDecoder::error_probability(size_t error_index) const {
  return errors.at(error_index).get_probability();
}

size_t TesseractDecoder::dem_error_index(size_t error_index) const {
  return error_to_dem_error.at(error_index);
}

size_t TesseractDecoder::retained_error_index(size_t dem_error_index) const {
  if (dem_error_index >= dem_error_to_error.size()) {
    throw std::out_of_range("DEM error index " + std::to_string(dem_error_index) +
                            " is out of range for " + std::to_string(dem_error_to_error.size()) +
                            " DEM errors.");
  }
  size_t error_index = dem_error_to_error[dem_error_index];
  if (error_index == std::numeric_limits<size_t>::max()) {
    throw std::invalid_argument("DEM error index " + std::to_string(dem_error_index) +
                                " was removed from the decoder.");
  }
  return error_index;
}

std::vector<size_t> TesseractDecoder::collect_affected_detectors(
    const std::vector<size_t>& modified_error_indices) const {
  for (size_t error_index : modified_error_indices) {
    if (error_index >= errors.size()) {
      throw std::out_of_range("Modified error index " + std::to_string(error_index) +
                              " is out of range for " + std::to_string(errors.size()) +
                              " decoder errors.");
    }
  }

  std::vector<uint8_t> affected(d2e.size());
  std::vector<size_t> affected_detectors;
  for (size_t error_index : modified_error_indices) {
    for (int detector : edets[error_index]) {
      if (!affected[detector]) {
        affected[detector] = true;
        affected_detectors.push_back(detector);
      }
    }
  }
  return affected_detectors;
}

void TesseractDecoder::update_internal_costs(const std::vector<size_t>& modified_error_indices,
                                             const std::vector<size_t>& affected_detectors) {
  for (size_t ei : modified_error_indices) {
    double min_cost = errors[ei].symptom.detectors.empty()
                          ? errors[ei].likelihood_cost
                          : errors[ei].likelihood_cost / errors[ei].symptom.detectors.size();
    error_costs[ei] = {errors[ei].likelihood_cost, min_cost};
  }

  for (size_t detector : affected_detectors) {
    std::sort(d2e[detector].begin(), d2e[detector].end(), [this](size_t idx_a, size_t idx_b) {
      return error_costs[idx_a].min_cost < error_costs[idx_b].min_cost ||
             (error_costs[idx_a].min_cost == error_costs[idx_b].min_cost && idx_a < idx_b);
    });
  }
}

TesseractDecoder::ErrorProbabilityRollback TesseractDecoder::apply_error_probability_updates(
    const std::vector<ErrorProbabilityUpdate>& probability_updates) {
  std::vector<size_t> error_indices;
  error_indices.reserve(probability_updates.size());
  for (const auto& update : probability_updates) {
    if (!std::isfinite(update.probability) || update.probability <= 0 || update.probability >= 1) {
      throw std::invalid_argument("Error probability must be finite and between 0 and 1.");
    }
    error_indices.push_back(retained_error_index(update.dem_error_index));
  }

  ErrorProbabilityRollback rollback;
  rollback.previous_likelihood_costs.reserve(probability_updates.size());
  rollback.modified_error_indices.reserve(probability_updates.size());
  std::vector<uint8_t> modified(errors.size());
  for (size_t k = 0; k < probability_updates.size(); ++k) {
    size_t error_index = error_indices[k];
    if (!modified[error_index]) {
      rollback.previous_likelihood_costs.emplace_back(error_index,
                                                      errors[error_index].likelihood_cost);
      rollback.modified_error_indices.push_back(error_index);
      modified[error_index] = true;
    }
  }
  rollback.affected_detectors = collect_affected_detectors(rollback.modified_error_indices);

  for (size_t k = 0; k < probability_updates.size(); ++k) {
    errors[error_indices[k]].set_with_probability(probability_updates[k].probability);
  }
  update_internal_costs(rollback.modified_error_indices, rollback.affected_detectors);
  return rollback;
}

void TesseractDecoder::restore_error_probabilities(const ErrorProbabilityRollback& rollback) {
  for (const auto& [error_index, likelihood_cost] : rollback.previous_likelihood_costs) {
    errors[error_index].likelihood_cost = likelihood_cost;
  }
  update_internal_costs(rollback.modified_error_indices, rollback.affected_detectors);
}

void TesseractDecoder::initialize_structures(size_t num_detectors) {
  d2e.resize(num_detectors);
  edets.resize(num_errors);

  for (size_t ei = 0; ei < num_errors; ++ei) {
    edets[ei] = errors[ei].symptom.detectors;
    for (int d : edets[ei]) {
      d2e[d].push_back(ei);
    }
  }

  // Initial fill of error_costs and sorting of d2e for all errors
  error_costs.reserve(errors.size());
  for (size_t i = 0; i < errors.size(); ++i) {
    double min_cost = errors[i].symptom.detectors.empty()
                          ? errors[i].likelihood_cost
                          : errors[i].likelihood_cost / errors[i].symptom.detectors.size();
    error_costs.push_back({errors[i].likelihood_cost, min_cost});
  }

  for (size_t d = 0; d < num_detectors; ++d) {
    std::sort(d2e[d].begin(), d2e[d].end(), [this](size_t idx_a, size_t idx_b) {
      return error_costs[idx_a].min_cost < error_costs[idx_b].min_cost ||
             (error_costs[idx_a].min_cost == error_costs[idx_b].min_cost && idx_a < idx_b);
    });
  }

  eneighbors.resize(num_errors);

#ifdef TESSERACT_FACTORED_NEIGHBOR_INIT
  // Factor the repeated union through detectors:
  // B[d] = union of edets[e] for e incident on d.
  // Then neighbors[e] = union(B[d], d in edets[e]) minus edets[e].
  // Bound the optional D-by-D table including row-object overhead. The
  // overflow-safe quotient check falls back to one scratch union if too big.
  constexpr size_t neighbor_table_budget = size_t(256) * 1024 * 1024;
  using NeighborBits = boost::dynamic_bitset<>;
  using NeighborBlock = NeighborBits::block_type;
  constexpr size_t neighbor_block_bits = std::numeric_limits<NeighborBlock>::digits;
  const size_t neighbor_blocks = num_detectors / neighbor_block_bits +
                                  (num_detectors % neighbor_block_bits != 0);
  const size_t neighbor_row_bytes = neighbor_blocks * sizeof(NeighborBlock) +
                                    sizeof(NeighborBits) + 2 * sizeof(void*);
  const bool use_factored_table = num_detectors == 0 ||
                                 neighbor_row_bytes <= neighbor_table_budget / num_detectors;
  std::vector<NeighborBits> detector_neighbors;
  if (use_factored_table) {
    detector_neighbors.reserve(num_detectors);
    for (size_t d = 0; d < num_detectors; ++d) {
      detector_neighbors.emplace_back(num_detectors, false);
      for (int oei : d2e[d]) {
        for (int od : edets[oei]) detector_neighbors.back()[od] = true;
      }
    }
  }

  NeighborBits neighbor_scratch(num_detectors, false);
  for (size_t ei = 0; ei < num_errors; ++ei) {
    neighbor_scratch.reset();
    if (use_factored_table) {
      for (int d : edets[ei]) neighbor_scratch |= detector_neighbors[d];
    } else {
      // Sparse fallback retains O(D) bits of temporary storage.
      for (int d : edets[ei]) {
        for (int oei : d2e[d]) {
          for (int od : edets[oei]) neighbor_scratch[od] = true;
        }
      }
    }
    for (int d : edets[ei]) neighbor_scratch[d] = false;
    auto& row = eneighbors[ei];
#ifdef TESSERACT_EXACT_NEIGHBOR_RESERVE
    const size_t appended_neighbors = neighbor_scratch.count();
    if (appended_neighbors > row.max_size() - row.size())
      throw std::length_error("Neighbor vector exceeds maximum size");
    row.reserve(row.size() + appended_neighbors);
#endif
#ifdef TESSERACT_VERIFY_FACTORED_NEIGHBOR_INIT
    const size_t old_size = row.size();
#endif
    for (size_t d = neighbor_scratch.find_first(); d != boost::dynamic_bitset<>::npos;
         d = neighbor_scratch.find_next(d)) {
      row.push_back(d);
    }
#ifdef TESSERACT_VERIFY_FACTORED_NEIGHBOR_INIT
    // All rows are verified, one at a time, using bounded O(D) bytes rather
    // than recreating the original E-by-D error bitset table.
    std::vector<uint8_t> reference(num_detectors, 0);
    for (int d : edets[ei]) {
      for (int oei : d2e[d]) {
        for (int od : edets[oei]) reference[od] = 1;
      }
    }
    for (int d : edets[ei]) reference[d] = 0;
    size_t output_index = old_size;
    for (size_t d = 0; d < num_detectors; ++d) {
      if (!reference[d]) continue;
      if (output_index >= row.size() || size_t(row[output_index++]) != d)
        throw std::logic_error("Factored neighbor initialization differs");
    }
    if (output_index != row.size())
      throw std::logic_error("Factored neighbor initialization has extra entries");
#endif
  }
#else
  std::vector<boost::dynamic_bitset<>> edets_bitsets(num_errors,
                                                     boost::dynamic_bitset<>(num_detectors));
  for (size_t ei = 0; ei < num_errors; ++ei) {
    for (int d : edets[ei]) {
      edets_bitsets[ei][d] = 1;
    }
  }

  for (size_t ei = 0; ei < num_errors; ++ei) {
    boost::dynamic_bitset<> neighbor_set(num_detectors, false);
    for (int d : edets[ei]) {
      for (int oei : d2e[d]) {
        // Unify detectors from neighboring errors
        neighbor_set |= edets_bitsets[oei];
      }
    }
    // Remove detectors from error's own set
    neighbor_set &= ~edets_bitsets[ei];

#ifdef TESSERACT_EXACT_NEIGHBOR_RESERVE
    const size_t appended_neighbors = neighbor_set.count();
    auto& row = eneighbors[ei];
    if (appended_neighbors > row.max_size() - row.size())
      throw std::length_error("Neighbor vector exceeds maximum size");
    row.reserve(row.size() + appended_neighbors);
#endif
    for (size_t d = neighbor_set.find_first(); d != boost::dynamic_bitset<>::npos;
         d = neighbor_set.find_next(d)) {
      eneighbors[ei].push_back(d);
    }
  }
#endif

  if (config.sparsify_errors) {
    if (config.sparsify_base_degree <= 0) {
      throw std::invalid_argument(
          "sparsify_base_degree must be > 0 when sparsify_errors is enabled.");
    }
    if (config.sparsify_max_degree < -1) {
      throw std::invalid_argument("sparsify_max_degree must be >= -1.");
    }
    if (config.sparsify_reactivate_limit < -1) {
      throw std::invalid_argument("sparsify_reactivate_limit must be >= -1.");
    }
    if (config.sparsify_max_degree >= 0 &&
        config.sparsify_max_degree < config.sparsify_base_degree) {
      throw std::invalid_argument("sparsify_max_degree must be >= sparsify_base_degree.");
    }

    if (config.sparsify_reactivate_limit == -1) {
      int error_count_limit = static_cast<int>(
          std::min(num_errors, static_cast<size_t>(std::numeric_limits<int>::max())));
      config.sparsify_reactivate_limit = suggest_sparsify_reactivate_limit_capped(
          config.dem.count_detectors(), config.sparsify_base_degree, error_count_limit);
    }

    sparsify_mandatory_errors.clear();
    sparsify_optional_errors.clear();
    for (size_t ei = 0; ei < num_errors; ++ei) {
      int degree = errors[ei].symptom.detectors.size();
      if (degree <= config.sparsify_base_degree) {
        sparsify_mandatory_errors.push_back(ei);
      } else if (degree > config.sparsify_base_degree &&
                 (config.sparsify_max_degree == -1 || degree <= config.sparsify_max_degree)) {
        sparsify_optional_errors.push_back(ei);
      }
    }
    sparse_error_active.assign(num_errors, 0);
    sparse_d2e.resize(num_detectors);
  }
}

void TesseractDecoder::decode_to_errors(const std::vector<uint64_t>& detections) {
  predicted_errors_buffer.clear();
  low_confidence_flag = false;
  if (detections.empty()) {
    return;
  }
  if (config.sparsify_errors) {
    build_sparse_d2e(detections);
  }
  const auto& active_d2e = config.sparsify_errors ? sparse_d2e : d2e;

  std::vector<size_t> best_errors;
  double best_cost = std::numeric_limits<double>::max();
  if (config.detector_orders.empty()) {
    throw std::runtime_error("Detector orders list must not be empty before decoding.");
  }

#ifndef TESSERACT_DISABLE_CERTIFIED_ENSEMBLE_STOP
  // Every solution must include an incident error for each fired detector.
  // With nonnegative weights, its floating-point cost is at least each such
  // weight. Min/max require no rounded fractional sums. Later trials replace
  // the incumbent only on strictly lower cost, including when costs tie.
  bool bound_computed = false;
  double ensemble_lower_bound = -INF;
#ifndef TESSERACT_DISABLE_PAIR_CERTIFICATE
  double pair_minimum_weight = -INF;
  double pair_singleton_bound = -INF;
  bool pair_bound_computed = false;
#endif
  auto incumbent_is_certified = [&](size_t remaining_trials) {
    (void)remaining_trials;
    if (config.verbose || config.create_visualization ||
        best_cost == std::numeric_limits<double>::max()) return false;
    if (!bound_computed) {
      bound_computed = true;
      auto canonical_detections = detections;
      std::sort(canonical_detections.begin(), canonical_detections.end());
      canonical_detections.erase(
          std::unique(canonical_detections.begin(), canonical_detections.end()),
          canonical_detections.end());
      double minimum_weight = INF;
      double minimum_singleton = INF;
      for (const auto& error : errors) {
        if (!std::isfinite(error.likelihood_cost) || error.likelihood_cost < 0) return false;
        minimum_weight = std::min(minimum_weight, error.likelihood_cost);
        if (std::equal(error.symptom.detectors.begin(), error.symptom.detectors.end(),
                       canonical_detections.begin(), canonical_detections.end())) {
          minimum_singleton = std::min(minimum_singleton, error.likelihood_cost);
        }
      }
      // Nonempty syndromes require one or more errors. A singleton must match
      // the whole syndrome; any longer nonnegative sum is at least min + min,
      // including under the decoder's ordinary floating-point accumulation.
      const double cardinality_bound = std::min(minimum_singleton, minimum_weight + minimum_weight);
      ensemble_lower_bound = 0;
      for (size_t d : detections) {
        double cheapest = INF;
        for (int ei : d2e[d]) cheapest = std::min(cheapest, errors[ei].likelihood_cost);
        ensemble_lower_bound = std::max(ensemble_lower_bound, cheapest);
      }
      ensemble_lower_bound = std::max(ensemble_lower_bound, cardinality_bound);
#ifndef TESSERACT_DISABLE_PAIR_CERTIFICATE
      pair_minimum_weight = minimum_weight;
      pair_singleton_bound = minimum_singleton;
#endif
    }
#ifndef TESSERACT_DISABLE_PAIR_CERTIFICATE
    const double triple_bound = (pair_minimum_weight + pair_minimum_weight) + pair_minimum_weight;
    // If even the optimistic triple bound cannot certify the incumbent,
    // defer the index until a later trial finds a cheaper incumbent.
    // The 12-trial gate amortizes index construction on measured workloads;
    // skipping this optional certificate never changes the winning result.
    if (remaining_trials >= 12 && best_cost > ensemble_lower_bound &&
        best_cost <= triple_bound && !pair_bound_computed) {
      pair_bound_computed = true;
      std::unordered_map<std::vector<int>, double, boost::hash<std::vector<int>>> cheapest;
      cheapest.reserve(errors.size());
      for (const auto& error : errors) {
        std::vector<int> symptom(error.symptom.detectors.begin(), error.symptom.detectors.end());
        auto [it, inserted] = cheapest.emplace(std::move(symptom), error.likelihood_cost);
        if (!inserted) it->second = std::min(it->second, error.likelihood_cost);
      }
      std::vector<int> syndrome(detections.begin(), detections.end());
      std::sort(syndrome.begin(), syndrome.end());
      syndrome.erase(std::unique(syndrome.begin(), syndrome.end()), syndrome.end());
      std::vector<int> complement;
      double pair_bound = INF;
      for (const auto& [symptom, weight] : cheapest) {
        complement.clear();
        std::set_symmetric_difference(symptom.begin(), symptom.end(), syndrome.begin(), syndrome.end(),
                                      std::back_inserter(complement));
        auto it = cheapest.find(complement);
        if (it != cheapest.end()) pair_bound = std::min(pair_bound, weight + it->second);
      }
      ensemble_lower_bound = std::max(ensemble_lower_bound,
          std::min({pair_singleton_bound, pair_bound, triple_bound}));
    }
#endif
    return best_cost <= ensemble_lower_bound;
  };
#endif

  if (config.beam_climbing) {
    int beam = 0;
    int order_index = 0;
    for (int trial = 0; trial < std::max(config.det_beam + 1, int(config.detector_orders.size()));
         ++trial) {
      decode_to_errors_with_graph(detections, order_index, beam, active_d2e);
      double local_cost = cost_from_errors(predicted_errors_buffer);
      if (!low_confidence_flag && local_cost < best_cost) {
        best_errors = predicted_errors_buffer;
        best_cost = local_cost;
      }
      if (config.verbose) {
        std::cout << "for detector_order " << order_index << " beam " << beam
                  << " got low confidence " << low_confidence_flag << " and cost " << local_cost
                  << " and obs_mask " << get_flipped_observables(predicted_errors_buffer)
                  << ". Best cost so far: " << best_cost << std::endl;
      }
#ifndef TESSERACT_DISABLE_CERTIFIED_ENSEMBLE_STOP
      if (trial + 1 < std::max(config.det_beam + 1, int(config.detector_orders.size())) &&
          incumbent_is_certified(
              std::max(config.det_beam + 1, int(config.detector_orders.size())) - trial - 1)) break;
#endif
      beam += 1;
      order_index += 1;
      beam %= (config.det_beam + 1);
      order_index %= config.detector_orders.size();
    }
  } else {
    for (size_t order_index = 0; order_index < config.detector_orders.size(); ++order_index) {
      decode_to_errors_with_graph(detections, order_index, config.det_beam, active_d2e);
      double local_cost = cost_from_errors(predicted_errors_buffer);
      if (!low_confidence_flag && local_cost < best_cost) {
        best_errors = predicted_errors_buffer;
        best_cost = local_cost;
      }
      if (config.verbose) {
        std::cout << "for detector_order " << order_index << " beam " << config.det_beam
                  << " got low confidence " << low_confidence_flag << " and cost " << local_cost
                  << " and obs_mask " << get_flipped_observables(predicted_errors_buffer)
                  << ". Best cost so far: " << best_cost << std::endl;
      }
#ifndef TESSERACT_DISABLE_CERTIFIED_ENSEMBLE_STOP
      if (order_index + 1 < config.detector_orders.size() &&
          incumbent_is_certified(config.detector_orders.size() - order_index - 1)) break;
#endif
    }
  }
  predicted_errors_buffer = best_errors;
  low_confidence_flag = best_cost == std::numeric_limits<double>::max();
}

void TesseractDecoder::flip_detectors_and_block_errors(
    size_t detector_order_index, int64_t error_chain_idx, boost::dynamic_bitset<>& detectors,
    std::vector<DetectorCostTuple>& detector_cost_tuples,
    const std::vector<std::vector<int>>& active_d2e) const {
  int64_t walker_idx = error_chain_idx;
  while (walker_idx != -1) {
    const auto& node = error_chain_arena[walker_idx];
    size_t ei = node.error_index;
    size_t min_detector = node.min_detector;

    for (int oei : active_d2e[min_detector]) {
      detector_cost_tuples[oei].error_blocked = 1;
      if (oei == ei) break;
    }

    for (int d : edets[ei]) {
      detectors[d] = !detectors[d];
    }
    walker_idx = node.parent_idx;
  }
}

void TesseractDecoder::decode_to_errors(const std::vector<uint64_t>& detections,
                                        size_t detector_order_index, size_t detector_beam) {
  if (config.sparsify_errors) {
    build_sparse_d2e(detections);
  }
  const auto& active_d2e = config.sparsify_errors ? sparse_d2e : d2e;
  decode_to_errors_with_graph(detections, detector_order_index, detector_beam, active_d2e);
}

void TesseractDecoder::decode_to_errors_with_graph(
    const std::vector<uint64_t>& detections, size_t detector_order_index, size_t detector_beam,
    const std::vector<std::vector<int>>& active_d2e) {
  auto decode_trial = [&]<typename TrialTuple, typename TupleStorage = std::vector<TrialTuple>>() {
#ifdef TESSERACT_COMPACT_STATS
    std::cerr << "compact_stats tuple_bytes=" << sizeof(TrialTuple)
              << " errors=" << num_errors << " detectors=" << num_detectors
              << " input_count=" << detections.size() << '\n';
#endif
    // Generic copy of the original arithmetic: uint32 min-count sentinel and
    // the same products, comparisons, final division and penalty addition.
    auto trial_plain_detcost = [&](size_t d, const TupleStorage& tuples) {
      double min_cost = INF;
      uint32_t min_count = std::numeric_limits<uint32_t>::max();
      for (int ei : active_d2e[d]) {
        const auto ec = error_costs[ei];
        if (ec.likelihood_cost * min_count >= min_cost * errors[ei].symptom.detectors.size()) break;
        const auto dct = tuples[ei];
        if (!dct.error_blocked && ec.likelihood_cost * min_count < min_cost * dct.detectors_count) {
          min_cost = ec.likelihood_cost;
          min_count = dct.detectors_count;
        }
      }
      return min_cost / min_count + config.det_penalty;
    };
    auto trial_reconstruct = [&](int64_t index, boost::dynamic_bitset<>& bits,
                                 TupleStorage& tuples) {
      while (index != -1) {
        const auto& edge = error_chain_arena[index];
        for (int ei : active_d2e[edge.min_detector]) {
          tuples[ei].error_blocked = 1;
          if (size_t(ei) == edge.error_index) break;
        }
        for (int d : edets[edge.error_index]) bits[d] = !bits[d];
        index = edge.parent_idx;
      }
    };

#if defined(TESSERACT_NET_SIBLING_COUNTS) && defined(TESSERACT_NET_SIBLING_STATS)
  uint64_t net_baseline_apply = 0, net_baseline_undo = 0, net_actual_updates = 0;
  uint64_t net_children = 0, net_common_detectors = 0;
  auto incidence_count = [&](size_t child) {
    uint64_t result = 0;
    for (int d : edets[child]) result += active_d2e[d].size();
    return result;
  };
  ScopeExit report_net_sibling_stats([&] {
    std::cerr << "net_sibling_stats children=" << net_children
              << " baseline_apply=" << net_baseline_apply
              << " baseline_undo=" << net_baseline_undo
              << " actual_updates=" << net_actual_updates
              << " saved_updates=" << net_baseline_apply + net_baseline_undo - net_actual_updates
              << " shared_detectors=" << net_common_detectors << std::endl;
  });
#endif
  const std::vector<size_t>& detector_at_position =
      config.detector_orders.at(detector_order_index).get_order();

#ifdef TESSERACT_RANKED_COSTS
  // Cross-product comparisons are converted to integer ranks only after an
  // exhaustive check of all weight/count pairs used during this trial.
  struct RankType { double weight; uint32_t count; uint32_t group; };
#ifdef TESSERACT_BYTE_RANK_METADATA
  struct RankError { uint8_t group; uint8_t lower_rank; };
  static_assert(sizeof(RankError) == 2);
  constexpr uint32_t metadata_no_rank = std::numeric_limits<uint8_t>::max();
#elif defined(TESSERACT_COMPACT_RANK_METADATA)
  struct RankError { uint16_t descriptor; uint16_t lower_rank; };
  constexpr uint32_t metadata_no_rank = std::numeric_limits<uint16_t>::max();
#else
  struct RankError { uint32_t descriptor; uint32_t lower_rank; };
  constexpr uint32_t metadata_no_rank = std::numeric_limits<uint32_t>::max();
#endif
  constexpr uint32_t rank_stride = 17;
  constexpr uint32_t no_rank = std::numeric_limits<uint32_t>::max();
  struct PreparedRanks {
    std::vector<RankError> ranked_errors;
    std::vector<uint32_t> count_ranks;
#ifdef TESSERACT_RANK_VALUE_TABLE
    std::vector<double> count_values;
#endif
#ifdef TESSERACT_LOCAL_RANK_ROWS
    struct LocalRankEntry { int error; RankError rank; };
    std::vector<std::vector<LocalRankEntry>> rows;
    std::vector<uint8_t> ready;
#endif
    bool safe = false;
    bool eager_rows = false;
    uint64_t weights = 0, types = 0;
  };
  auto* batch = batch_detail::active_batch_plan;
  if (batch && (batch->owner != this || config.sparsify_errors || config.verbose ||
                config.create_visualization || &active_d2e != &d2e ||
                !batch_detail::batch_environment_supported())) batch = nullptr;
  if (batch && batch->environment != batch_detail::batch_environment()) {
    batch->invalidate();
    batch->environment = batch_detail::batch_environment();
  }
  std::shared_ptr<PreparedRanks> prepared;
  if (batch) {
    auto* found = std::any_cast<std::shared_ptr<PreparedRanks>>(&batch->prepared);
    if (found) prepared = *found;
  }
  const bool prepare_now = !prepared;
  if (prepare_now) {
    prepared = std::make_shared<PreparedRanks>();
    prepared->safe = config.pqlimit >= 1024;
#ifdef TESSERACT_LOCAL_RANK_ROWS
    prepared->rows.resize(num_detectors);
    prepared->ready.assign(num_detectors, 0);
#endif
    if (batch) ++batch->builds;
  } else if (batch) ++batch->hits;
  auto& ranked_errors = prepared->ranked_errors;
  auto& count_ranks = prepared->count_ranks;
#ifdef TESSERACT_RANK_VALUE_TABLE
  auto& count_values = prepared->count_values;
#endif
  auto& rank_safe = prepared->safe;
#ifdef TESSERACT_RANK_STATS
  uint64_t rank_stats_weights = prepared->weights, rank_stats_types = prepared->types;
  uint64_t rank_stats_calls = 0, rank_stats_entries = 0;
  uint64_t rank_stats_hits = 0, rank_stats_fallbacks = 0;
  ScopeExit emit_rank_stats([&] {
    std::cerr << "rank_stats enabled=" << rank_safe
              << " metadata_bytes=" << sizeof(RankError)
              << " errors=" << num_errors
              << " weights=" << rank_stats_weights
              << " types=" << rank_stats_types
              << " calls=" << rank_stats_calls
              << " entries=" << rank_stats_entries
              << " witness_hits=" << rank_stats_hits
              << " fallback_calls=" << rank_stats_fallbacks << '\n';
  });
#endif
  if (prepare_now) {
  if (rank_safe) {
    std::unordered_map<uint64_t, uint32_t> group_map;
    std::vector<double> weights;
    std::vector<uint32_t> degrees;
    ranked_errors.reserve(num_errors);
    for (size_t ei = 0; ei < num_errors; ++ei) {
      const double weight = error_costs[ei].likelihood_cost;
      const size_t degree = errors[ei].symptom.detectors.size();
      if (!(weight > 0) || !std::isfinite(weight) ||
          !std::isfinite(weight * double(no_rank)) || degree >= rank_stride) {
        rank_safe = false; break;
      }
      auto [it, inserted] = group_map.emplace(std::bit_cast<uint64_t>(weight), weights.size());
      if (inserted) { weights.push_back(weight); degrees.push_back(0); }
      const uint32_t group = it->second;
      degrees[group] = std::max(degrees[group], uint32_t(degree));
#ifdef TESSERACT_BYTE_RANK_METADATA
      // Check before narrowing: group255 is valid; group256 requires fallback.
      if (group > std::numeric_limits<uint8_t>::max()) { rank_safe = false; break; }
      ranked_errors.push_back({static_cast<uint8_t>(group), uint8_t(metadata_no_rank)});
#else
      ranked_errors.push_back({
          static_cast<decltype(RankError::descriptor)>((group << 5) | uint32_t(degree)),
          static_cast<decltype(RankError::lower_rank)>(metadata_no_rank)});
      if (weights.size() > 512) { rank_safe = false; break; }
#endif
    }
    prepared->weights = weights.size();
#ifdef TESSERACT_RANK_STATS
    rank_stats_weights = weights.size();
#endif
    if (rank_safe) {
      std::vector<RankType> types;
      for (uint32_t g = 0; g < weights.size(); ++g) {
        for (uint32_t count = 1; count <= degrees[g]; ++count)
          types.push_back({weights[g], count, g});
      }
      prepared->types = types.size();
#ifdef TESSERACT_RANK_STATS
      rank_stats_types = types.size();
#endif
      // Division supplies a provisional total ordering. Pairwise validation,
      // rather than assuming FP ratios and cross products agree, establishes
      // that it is valid for this exact collection of binary64 weights.
      std::sort(types.begin(), types.end(), [](const auto& a, const auto& b) {
        return a.weight / a.count < b.weight / b.count;
      });
      count_ranks.assign(weights.size() * rank_stride, no_rank);
      uint32_t rank = 0;
      for (size_t i = 0; i < types.size(); ++i) {
        const auto& a = types[i];
        if (i && types[i-1].weight * a.count != a.weight * types[i-1].count) ++rank;
        count_ranks[a.group * rank_stride + a.count] = rank;
      }
#ifdef TESSERACT_BYTE_RANK_METADATA
      // At most255 ranks: valid ranks0..254, reserving255 for no candidate.
      // No lower_rank byte is stored before this global eligibility check.
      if (rank >= metadata_no_rank) rank_safe = false;
#endif
      for (const auto& a : types) {
        for (const auto& b : types) {
          if ((a.weight * b.count < b.weight * a.count) !=
              (count_ranks[a.group * rank_stride + a.count] <
               count_ranks[b.group * rank_stride + b.count])) { rank_safe = false; break; }
        }
        if (!rank_safe) break;
      }
      if (rank_safe) {
        for (size_t ei = 0; ei < ranked_errors.size(); ++ei) {
          auto& entry = ranked_errors[ei];
#ifdef TESSERACT_BYTE_RANK_METADATA
          const uint32_t lower = count_ranks[uint32_t(entry.group) * rank_stride +
                                             errors[ei].symptom.detectors.size()];
#else
          const uint32_t lower = count_ranks[(entry.descriptor >> 5) * rank_stride + (entry.descriptor & 31)];
#endif
          entry.lower_rank = static_cast<decltype(RankError::lower_rank)>(
              lower == no_rank ? metadata_no_rank : lower);
        }
#ifdef TESSERACT_RANK_VALUE_TABLE
        // Index by the actual weight/count type, never by comparison rank:
        // rounded cross-product ties can have different rounded quotients.
        count_values.resize(count_ranks.size());
        for (const auto& type : types) {
          count_values[type.group * rank_stride + type.count] =
              (type.weight / type.count) + config.det_penalty;
        }
#endif
      }
    }
  }
#ifdef TESSERACT_LOCAL_RANK_ROWS
    bool eager_rows_valid = false;
    if (batch && rank_safe && batch->policy == batch_detail::PreparationPolicy::EagerRows) {
      eager_rows_valid = active_d2e.size() == num_detectors &&
          batch_detail::eager_rows_fit_budget(active_d2e,sizeof(prepared->rows[0]),
                                              sizeof(prepared->ready[0]),sizeof(prepared->rows[0][0]));
      if (eager_rows_valid) {
        for (const auto& row : active_d2e) {
          for (int ei : row) {
            if (ei < 0 || size_t(ei) >= ranked_errors.size()) { eager_rows_valid=false; break; }
          }
          if (!eager_rows_valid) break;
        }
      }
    } else eager_rows_valid=false;
    if (eager_rows_valid) {
      for (size_t d=0; d<num_detectors; ++d) {
        auto& row=prepared->rows[d];
        row.reserve(active_d2e[d].size());
        for (int ei : active_d2e[d]) row.push_back({ei,ranked_errors[ei]});
        prepared->ready[d]=1;
      }
      prepared->eager_rows=true;
      ++batch->eager_builds;
    }
#endif
    // Publish only after preparation completes successfully.
    if (batch) batch->prepared = prepared;
  }
#ifdef TESSERACT_LOCAL_RANK_ROWS
  auto& local_rank_rows = prepared->rows;
  auto& local_rank_ready = prepared->ready;
#endif
  if (batch && !prepare_now && prepared->eager_rows) ++batch->eager_hits;
  auto trial_detcost = [&](size_t d, const TupleStorage& tuples) {
#ifdef TESSERACT_RANK_STATS
    ++rank_stats_calls;
    if (!rank_safe) ++rank_stats_fallbacks;
#endif
    if (!rank_safe) return trial_plain_detcost(d, tuples);
    uint32_t best_rank = no_rank;
#ifdef TESSERACT_RANK_VALUE_TABLE
    uint32_t best_type = no_rank;
#else
    uint32_t best_count = no_rank;
    int best_error = -1;
#endif
#ifdef TESSERACT_LOCAL_RANK_ROWS
    if (!local_rank_ready[d]) {
      auto& row = local_rank_rows[d];
      row.reserve(active_d2e[d].size());
      for (int ei : active_d2e[d]) row.push_back({ei, ranked_errors[ei]});
      local_rank_ready[d] = 1;
    }
    for (const auto& local_entry : local_rank_rows[d]) {
      const int ei = local_entry.error;
#else
    for (int ei : active_d2e[d]) {
#endif
#ifdef TESSERACT_RANK_STATS
      ++rank_stats_entries;
#endif
#ifdef TESSERACT_LOCAL_RANK_ROWS
      const auto entry = local_entry.rank;
#else
      const auto entry = ranked_errors[ei];
#endif
      const uint32_t lower_rank = entry.lower_rank == metadata_no_rank ? no_rank : entry.lower_rank;
      if (lower_rank >= best_rank) break;
      const auto dct = tuples[ei];
      if (!dct.error_blocked) {
#ifdef TESSERACT_BYTE_RANK_METADATA
        if constexpr (std::is_same_v<TrialTuple, DetectorCostTuple>) {
          // Wide dispatch can contain duplicate inputs; keep exact guard.
          if (dct.detectors_count > errors[ei].symptom.detectors.size()) {
#ifdef TESSERACT_RANK_STATS
            ++rank_stats_fallbacks;
#endif
            return trial_plain_detcost(d, tuples);
          }
        }
        const uint32_t type_index = uint32_t(entry.group) * rank_stride + dct.detectors_count;
#else
        if (dct.detectors_count > (entry.descriptor & 31)) {
#ifdef TESSERACT_RANK_STATS
          ++rank_stats_fallbacks;
#endif
          return trial_plain_detcost(d, tuples);
        }
        const uint32_t type_index = (entry.descriptor >> 5) * rank_stride + dct.detectors_count;
#endif
        const uint32_t rank = count_ranks[type_index];
        if (rank < best_rank) {
          best_rank = rank;
#ifdef TESSERACT_RANK_VALUE_TABLE
          best_type = type_index;
#else
          best_count = dct.detectors_count; best_error = ei;
#endif
        }
      }
    }
#ifdef TESSERACT_RANK_VALUE_TABLE
    const double value = best_type == no_rank
        ? (INF / no_rank) + config.det_penalty : count_values[best_type];
#else
    const double value = ((best_error == -1 ? INF : error_costs[best_error].likelihood_cost) / best_count) + config.det_penalty;
#endif
#if defined(TESSERACT_VERIFY_RANKED_COSTS) || defined(TESSERACT_VERIFY_BYTE_RANK_METADATA)
    if (std::bit_cast<uint64_t>(value) != std::bit_cast<uint64_t>(trial_plain_detcost(d, tuples)))
      throw std::logic_error("Ranked cost differs from original computation");
#endif
    return value;
  };
#elif defined(TESSERACT_PREFIX_CACHE)
  // Trial-local: graph, weights and penalty stay fixed. Only the tuples read
  // before the original early exit can affect its return value. The error at
  // the early-exit boundary reads no tuple and needs no snapshot.
  struct PrefixEntry {
    bool valid = false;
    double value = 0;
    TupleStorage tuples;
  };
  std::vector<PrefixEntry> prefix_cache(num_detectors);
  auto trial_detcost = [&](size_t d, const TupleStorage& tuples) {
    auto& entry = prefix_cache[d];
    const auto& row = active_d2e[d];
    bool hit = entry.valid;
    for (size_t i = 0; hit && i < entry.tuples.size(); ++i) {
      const auto& now = tuples[row[i]];
      const auto& old = entry.tuples[i];
      hit = now.error_blocked == old.error_blocked &&
            now.detectors_count == old.detectors_count;
    }
    if (!hit) {
      entry.tuples.clear();
      double min_cost = INF;
      uint32_t min_count = std::numeric_limits<uint32_t>::max();
      for (int ei : row) {
        const auto ec = error_costs[ei];
        if (ec.likelihood_cost * min_count >=
            min_cost * errors[ei].symptom.detectors.size()) break;
        const auto dct = tuples[ei];
        entry.tuples.push_back(dct);
        if (!dct.error_blocked &&
            ec.likelihood_cost * min_count < min_cost * dct.detectors_count) {
          min_cost = ec.likelihood_cost;
          min_count = dct.detectors_count;
        }
      }
      entry.value = (min_cost / min_count) + config.det_penalty;
      entry.valid = true;
    }
#ifdef TESSERACT_VERIFY_PREFIX_CACHE
    const double reference = trial_plain_detcost(d, tuples);
    if (std::bit_cast<uint64_t>(entry.value) != std::bit_cast<uint64_t>(reference)) {
      throw std::logic_error("Detector-cost cache differs from original computation");
    }
#endif
    return entry.value;
  };
#elif defined(TESSERACT_PACKED_COSTS)
  struct PackedCost {
    double weight;
    size_t degree;
  };
  // Rebuilt per trial: temporary probability updates are reflected directly.
  std::vector<PackedCost> packed_costs;
  // Avoid the measured setup regression on tiny budgets and sparse graphs.
  if (config.pqlimit >= 1024 && !config.sparsify_errors) {
    packed_costs.reserve(num_errors);
    for (size_t ei = 0; ei < num_errors; ++ei) {
      packed_costs.push_back({error_costs[ei].likelihood_cost, errors[ei].symptom.detectors.size()});
    }
  }
  auto trial_detcost = [&](size_t d, const TupleStorage& tuples) {
    if (packed_costs.empty()) return trial_plain_detcost(d, tuples);
    double min_cost = INF;
    uint32_t min_count = std::numeric_limits<uint32_t>::max();
    for (int ei : active_d2e[d]) {
      const auto ec = packed_costs[ei];
      if (ec.weight * min_count >= min_cost * ec.degree) break;
      const auto dct = tuples[ei];
      if (!dct.error_blocked && ec.weight * min_count < min_cost * dct.detectors_count) {
        min_cost = ec.weight;
        min_count = dct.detectors_count;
      }
    }
    const double value = (min_cost / min_count) + config.det_penalty;
#ifdef TESSERACT_VERIFY_PACKED_COSTS
    if (std::bit_cast<uint64_t>(value) !=
        std::bit_cast<uint64_t>(trial_plain_detcost(d, tuples))) {
      throw std::logic_error("Packed detector cost differs from original computation");
    }
#endif
    return value;
  };
#else
  auto trial_detcost = [&](size_t d, const TupleStorage& tuples) {
    return trial_plain_detcost(d, tuples);
  };
#endif

  predicted_errors_buffer.clear();
  low_confidence_flag = false;
  error_chain_arena.clear();
  // Can technically be larger than pqlimit, but we need an initial guess on how many nodes we
  // will process from the queue. Only reserve if pqlimit is a reasonable (finite) value;
  // reserving SIZE_MAX bytes would throw std::length_error.
  if (config.pqlimit != std::numeric_limits<size_t>::max()) {
    error_chain_arena.reserve(config.pqlimit);
  }

  std::priority_queue<Node, std::vector<Node>, std::greater<Node>> pq;
  std::unordered_map<size_t, std::unordered_set<boost::dynamic_bitset<>>> visited_detectors;

  boost::dynamic_bitset<> initial_detectors(num_detectors, false);
  TupleStorage initial_detector_cost_tuples(num_errors);

  for (size_t d : detections) {
    if (d >= num_detectors) {
      throw std::runtime_error(
          "Symptom " + std::to_string(d) +
          " references a detector >= num_detectors (= " + std::to_string(num_detectors) + ").");
    }
    initial_detectors[d] = true;
    for (int ei : active_d2e[d]) {
      ++initial_detector_cost_tuples[ei].detectors_count;
    }
  }

  double initial_cost = 0;
  for (size_t d : detections) {
    initial_cost += trial_detcost(d, initial_detector_cost_tuples);
  }

  if (initial_cost == INF) {
    low_confidence_flag = true;
    return;
  }

  size_t min_num_dets = detections.size();
  size_t max_num_dets = min_num_dets + detector_beam;

  boost::dynamic_bitset<> next_detectors;
  TupleStorage next_detector_cost_tuples;

  pq.push({initial_cost, min_num_dets, 0, -1});
  size_t num_pq_pushed = 1;

#if defined(TESSERACT_INCREMENTAL_STATE) || defined(TESSERACT_HYBRID_STATE)
  // Experimental exact state recovery: undo to the common ancestor, then
  // apply only the new suffix. Reference counts preserve overlapping blockers.
  boost::dynamic_bitset<> detectors = initial_detectors;
  TupleStorage detector_cost_tuples(num_errors);
  std::vector<size_t> blocking_counts(num_errors, 0);
  std::vector<int64_t> current_path, target_path;
#ifdef TESSERACT_HYBRID_STATE
  // Only counts written for an expanded parent need clearing. Blocker counts
  // are retained independently through the ancestry transition.
  std::vector<int> count_touched;
#ifdef TESSERACT_NET_PARENT_COUNTS
  // Counts start at zero, so the last counted residual is the empty set, not
  // initial_detectors. It advances only when a parent reaches expansion.
  boost::dynamic_bitset<> counted_parent_detectors(num_detectors, false);
  boost::dynamic_bitset<> parent_count_delta(num_detectors, false);
#ifdef TESSERACT_NET_PARENT_STATS
  uint64_t net_parent_expanded = 0, net_parent_flips = 0, net_parent_updates = 0;
  uint64_t net_parent_recount_incidences = 0;
  ScopeExit report_net_parent_stats([&] {
    std::cerr << "net_parent_stats expanded=" << net_parent_expanded
              << " delta_detectors=" << net_parent_flips
              << " net_incidences=" << net_parent_updates
              << " original_recount_incidences=" << net_parent_recount_incidences
              << std::endl;
  });
#endif
#endif
#else
  for (size_t d = detectors.find_first(); d != boost::dynamic_bitset<>::npos;
       d = detectors.find_next(d)) {
    for (int ei : active_d2e[d]) ++detector_cost_tuples[ei].detectors_count;
  }
#endif
  auto apply_chain_edge = [&](int64_t index, bool undo) {
    const auto& edge = error_chain_arena[index];
    for (int ei : active_d2e[edge.min_detector]) {
      if (undo) {
        --blocking_counts[ei];
      } else {
        ++blocking_counts[ei];
      }
      detector_cost_tuples[ei].error_blocked = blocking_counts[ei] != 0;
      if (size_t(ei) == edge.error_index) break;
    }
    for (int d : edets[edge.error_index]) {
      detectors[d] = !detectors[d];
#ifndef TESSERACT_HYBRID_STATE
      const int fired = detectors[d] ? 1 : -1;
      for (int ei : active_d2e[d]) detector_cost_tuples[ei].detectors_count += fired;
#endif
    }
  };
#endif

#if defined(TESSERACT_VERIFY_COMPACT_STATE) || defined(TESSERACT_VERIFY_SPLIT_STORAGE)
  auto verify_trial_state = [&](int64_t chain, size_t child_error, size_t branch_detector,
                                const boost::dynamic_bitset<>& bits,
                                const TupleStorage& tuples) {
    // Independently reconstruct using the unchanged public uint32 helper and
    // recount from the resulting syndrome; never convert per heuristic query.
    auto expected_bits = initial_detectors;
    std::vector<DetectorCostTuple> expected(num_errors);
    flip_detectors_and_block_errors(detector_order_index, chain, expected_bits, expected, active_d2e);
    if (child_error != std::numeric_limits<size_t>::max()) {
      for (int ei : active_d2e[branch_detector]) {
        expected[ei].error_blocked = 1;
        if (size_t(ei) == child_error) break;
      }
      for (int d : edets[child_error]) expected_bits[d] = !expected_bits[d];
    }
    for (size_t d = expected_bits.find_first(); d != boost::dynamic_bitset<>::npos;
         d = expected_bits.find_next(d)) {
      for (int ei : active_d2e[d]) ++expected[ei].detectors_count;
    }
    if (bits != expected_bits) throw std::logic_error("Compact syndrome differs from reference");
    for (size_t ei = 0; ei < num_errors; ++ei) {
      if (uint32_t(tuples[ei].detectors_count) != expected[ei].detectors_count ||
          uint32_t(tuples[ei].error_blocked) != expected[ei].error_blocked)
        throw std::logic_error("Compact tuple differs from independently reconstructed uint32 state");
    }
  };
#endif
  while (!pq.empty()) {
    const Node node = pq.top();
    pq.pop();

    if (node.num_dets > max_num_dets) continue;

#ifdef TESSERACT_SKIP_TERMINAL_STATE
    // A solved node needs only its error chain. Diagnostic modes below still
    // reconstruct the state they display. No subsequent node consumes it.
    if (node.num_dets == 0 && !config.verbose && !config.create_visualization) {
      predicted_errors_buffer.resize(node.depth);
      int64_t walker_idx = node.error_chain_idx;
      for (size_t i = 0; i < node.depth; ++i) {
        predicted_errors_buffer[node.depth - 1 - i] =
            error_to_dem_error[error_chain_arena[walker_idx].error_index];
        walker_idx = error_chain_arena[walker_idx].parent_idx;
      }
      return;
    }
#endif

#if defined(TESSERACT_INCREMENTAL_STATE) || defined(TESSERACT_HYBRID_STATE)
    target_path.clear();
    for (int64_t index = node.error_chain_idx; index != -1;
         index = error_chain_arena[index].parent_idx) {
      target_path.push_back(index);
    }
    std::reverse(target_path.begin(), target_path.end());
    size_t common = 0;
    while (common < current_path.size() && common < target_path.size() &&
           current_path[common] == target_path[common]) ++common;
    for (size_t i = current_path.size(); i > common; --i) {
      apply_chain_edge(current_path[i - 1], true);
    }
    for (size_t i = common; i < target_path.size(); ++i) {
      apply_chain_edge(target_path[i], false);
    }
    current_path.swap(target_path);
#if defined(TESSERACT_VERIFY_INCREMENTAL_STATE) && !defined(TESSERACT_HYBRID_STATE)
    auto expected_detectors = initial_detectors;
    TupleStorage expected_tuples(num_errors);
    trial_reconstruct(node.error_chain_idx, expected_detectors, expected_tuples);
    for (size_t d = 0; d < num_detectors; ++d) {
      if (expected_detectors[d]) {
        for (int ei : active_d2e[d]) ++expected_tuples[ei].detectors_count;
      }
    }
    if (detectors != expected_detectors) throw std::logic_error("Residual state differs");
    for (size_t ei = 0; ei < num_errors; ++ei) {
      if (detector_cost_tuples[ei].error_blocked != expected_tuples[ei].error_blocked ||
          detector_cost_tuples[ei].detectors_count != expected_tuples[ei].detectors_count) {
        throw std::logic_error("Incremental error state differs");
      }
    }
#endif
#else
    boost::dynamic_bitset<> detectors = initial_detectors;
    TupleStorage detector_cost_tuples(num_errors);
    trial_reconstruct(node.error_chain_idx, detectors, detector_cost_tuples);
#endif

    if (node.num_dets == 0) {
      if (config.create_visualization) {
        visualizer.add_activated_errors(node.error_chain_idx, error_chain_arena);
        visualizer.add_activated_detectors(detectors, num_detectors);
      }
      if (config.verbose) {
        std::cout << "activated_errors = ";
        int64_t walker_idx = node.error_chain_idx;
        while (walker_idx != -1) {
          std::cout << error_chain_arena[walker_idx].error_index << ", ";
          walker_idx = error_chain_arena[walker_idx].parent_idx;
        }
        std::cout << std::endl;
        std::cout << "activated_detectors = ";
        for (size_t d = 0; d < num_detectors; ++d) {
          if (detectors[d]) {
            std::cout << d << ", ";
          }
        }
        std::cout << std::endl;
        std::cout.precision(13);
        std::cout << "Decoding complete. Cost: " << node.cost
                  << " num_pq_pushed = " << num_pq_pushed << std::endl;
      }
      predicted_errors_buffer.resize(node.depth);
      int64_t walker_idx = node.error_chain_idx;
      for (size_t i = 0; i < node.depth; ++i) {
        predicted_errors_buffer[node.depth - 1 - i] =
            error_to_dem_error[error_chain_arena[walker_idx].error_index];
        walker_idx = error_chain_arena[walker_idx].parent_idx;
      }
      return;
    }

    if (config.no_revisit_dets && !visited_detectors[node.num_dets].insert(detectors).second)
      continue;

    if (config.create_visualization) {
      visualizer.add_activated_errors(node.error_chain_idx, error_chain_arena);
      visualizer.add_activated_detectors(detectors, num_detectors);
    }
    if (config.verbose) {
      std::cout.precision(13);
      std::cout << "len(pq) = " << pq.size() << " num_pq_pushed = " << num_pq_pushed << std::endl;
      std::cout << "num_dets = " << node.num_dets << " max_num_dets = " << max_num_dets
                << " cost = " << node.cost << std::endl;
      std::cout << "activated_errors = ";
      int64_t walker_idx = node.error_chain_idx;
      while (walker_idx != -1) {
        std::cout << error_chain_arena[walker_idx].error_index << ", ";
        walker_idx = error_chain_arena[walker_idx].parent_idx;
      }
      std::cout << std::endl;
      std::cout << "activated_detectors = ";
      for (size_t d = 0; d < num_detectors; ++d) {
        if (detectors[d]) {
          std::cout << d << ", ";
        }
      }
      std::cout << std::endl;
    }

    if (node.num_dets < min_num_dets) {
      min_num_dets = node.num_dets;
      if (config.no_revisit_dets) {
        for (size_t i = min_num_dets + detector_beam + 1; i <= max_num_dets; ++i) {
          visited_detectors[i].clear();
        }
      }
      max_num_dets = std::min(max_num_dets, min_num_dets + detector_beam);
    }

#ifdef TESSERACT_HYBRID_STATE
#ifdef TESSERACT_NET_PARENT_COUNTS
    // Ancestry recovery may visit rejected parents; count state deliberately
    // stays tied to the last expanded parent until this exact boundary.
    parent_count_delta = counted_parent_detectors;
    parent_count_delta ^= detectors;
    for (size_t d = parent_count_delta.find_first(); d != boost::dynamic_bitset<>::npos;
         d = parent_count_delta.find_next(d)) {
      const int fired = detectors[d] ? 1 : -1;
      for (int ei : active_d2e[d]) detector_cost_tuples[ei].detectors_count += fired;
#ifdef TESSERACT_NET_PARENT_STATS
      ++net_parent_flips;
      net_parent_updates += active_d2e[d].size();
#endif
    }
    counted_parent_detectors = detectors;
#ifdef TESSERACT_NET_PARENT_STATS
    ++net_parent_expanded;
    for (size_t d = detectors.find_first(); d != boost::dynamic_bitset<>::npos;
         d = detectors.find_next(d)) net_parent_recount_incidences += active_d2e[d].size();
#endif
#else
    for (int ei : count_touched) detector_cost_tuples[ei].detectors_count = 0;
    count_touched.clear();
    for (size_t d = detectors.find_first(); d != boost::dynamic_bitset<>::npos;
         d = detectors.find_next(d)) {
      for (int ei : active_d2e[d]) {
        auto& count = detector_cost_tuples[ei].detectors_count;
        if (count == 0) count_touched.push_back(ei);
        ++count;
      }
    }
#endif
#ifdef TESSERACT_VERIFY_HYBRID_STATE
    auto expected_detectors = initial_detectors;
    std::vector<DetectorCostTuple> expected_tuples(num_errors);
    flip_detectors_and_block_errors(detector_order_index, node.error_chain_idx,
                                    expected_detectors, expected_tuples, active_d2e);
    for (size_t d = 0; d < num_detectors; ++d) {
      if (expected_detectors[d]) {
        for (int ei : active_d2e[d]) ++expected_tuples[ei].detectors_count;
      }
    }
    if (detectors != expected_detectors) throw std::logic_error("Hybrid residual differs");
    for (size_t ei = 0; ei < num_errors; ++ei) {
      if (detector_cost_tuples[ei].error_blocked != expected_tuples[ei].error_blocked ||
          detector_cost_tuples[ei].detectors_count != expected_tuples[ei].detectors_count) {
        throw std::logic_error("Hybrid tuple differs");
      }
    }
#endif
#elif !defined(TESSERACT_INCREMENTAL_STATE)
    for (size_t d = 0; d < num_detectors; ++d) {
      if (!detectors[d]) continue;
      for (int ei : active_d2e[d]) {
        ++detector_cost_tuples[ei].detectors_count;
      }
    }
#endif

#if defined(TESSERACT_VERIFY_COMPACT_STATE) || defined(TESSERACT_VERIFY_SPLIT_STORAGE)
    verify_trial_state(node.error_chain_idx, std::numeric_limits<size_t>::max(), 0,
                       detectors, detector_cost_tuples);
#endif
    next_detector_cost_tuples = detector_cost_tuples;

    size_t min_detector = std::numeric_limits<size_t>::max();
    for (size_t position = 0; position < num_detectors; ++position) {
      const size_t detector = detector_at_position[position];
      if (detectors[detector]) {
        min_detector = detector;
        break;
      }
    }

    size_t prev_ei = std::numeric_limits<size_t>::max();
#if defined(TESSERACT_NET_SIBLING_COUNTS) && defined(TESSERACT_NET_SIBLING_STATS)
    size_t net_baseline_previous = std::numeric_limits<size_t>::max();
#endif
    std::vector<double> detector_cost_cache(num_detectors, -1);

    for (int ei : active_d2e[min_detector]) {
      if (detector_cost_tuples[ei].error_blocked) continue;

#ifdef TESSERACT_EARLY_CHILD_BEAM
      // |S xor E| = |S| + |E| - 2|S intersect E|. Parent counts already
      // provide the intersection, so reject without copying the bitset.
      if (node.num_dets + edets[ei].size() -
              2 * size_t(detector_cost_tuples[ei].detectors_count) > max_num_dets) {
        next_detector_cost_tuples[ei].error_blocked = 1;
        continue;
      }
#endif

#if defined(TESSERACT_NET_SIBLING_COUNTS) && defined(TESSERACT_NET_SIBLING_STATS)
      // Shadow exactly when the existing lazy implementation would undo its
      // last accepted child, including before a subsequently rejected sibling.
      if (net_baseline_previous != std::numeric_limits<size_t>::max())
        net_baseline_undo += incidence_count(net_baseline_previous);
      net_baseline_previous = std::numeric_limits<size_t>::max();
#endif
#ifndef TESSERACT_NET_SIBLING_COUNTS
      if (prev_ei != std::numeric_limits<size_t>::max()) {
        for (int d : edets[prev_ei]) {
          int fired = detectors[d] ? 1 : -1;
          for (int oei : active_d2e[d]) {
            next_detector_cost_tuples[oei].detectors_count += fired;
          }
        }
      }
#ifdef TESSERACT_LAZY_CHILD_COUNTS
      prev_ei = std::numeric_limits<size_t>::max();
#else
      prev_ei = ei;
#endif

#endif

      next_detectors = detectors;
      next_detector_cost_tuples[ei].error_blocked = 1;

      double next_cost = node.cost + errors[ei].likelihood_cost;
      size_t next_num_dets = node.num_dets;

      for (int d : edets[ei]) {
        next_detectors[d] = !next_detectors[d];
        int fired = next_detectors[d] ? 1 : -1;
        next_num_dets += fired;
#ifndef TESSERACT_LAZY_CHILD_COUNTS
        for (int oei : active_d2e[d]) {
          next_detector_cost_tuples[oei].detectors_count += fired;
        }
#endif
      }

      if (next_num_dets > max_num_dets) continue;

      if (config.no_revisit_dets && visited_detectors[next_num_dets].find(next_detectors) !=
                                        visited_detectors[next_num_dets].end())
        continue;

#ifdef TESSERACT_NET_SIBLING_COUNTS
      // Counts still represent the last accepted child. Rejected siblings only
      // add blockers; neither beam nor visited checks read these child counts.
      const auto& current_support = edets[ei];
      if (prev_ei != std::numeric_limits<size_t>::max()) {
        for (int d : edets[prev_ei]) {
          if (std::find(current_support.begin(), current_support.end(), d) != current_support.end()) {
#ifdef TESSERACT_NET_SIBLING_STATS
            ++net_common_detectors;
#endif
            continue;
          }
          const int fired = detectors[d] ? 1 : -1;
          for (int oei : active_d2e[d]) next_detector_cost_tuples[oei].detectors_count += fired;
#ifdef TESSERACT_NET_SIBLING_STATS
          net_actual_updates += active_d2e[d].size();
#endif
        }
      }
      for (int d : current_support) {
        if (prev_ei != std::numeric_limits<size_t>::max() &&
            std::find(edets[prev_ei].begin(), edets[prev_ei].end(), d) != edets[prev_ei].end()) continue;
        const int fired = next_detectors[d] ? 1 : -1;
        for (int oei : active_d2e[d]) next_detector_cost_tuples[oei].detectors_count += fired;
#ifdef TESSERACT_NET_SIBLING_STATS
        net_actual_updates += active_d2e[d].size();
#endif
      }
      prev_ei = ei;
#ifdef TESSERACT_NET_SIBLING_STATS
      ++net_children;
      net_baseline_apply += incidence_count(ei);
      net_baseline_previous = ei;
#endif
#elif defined(TESSERACT_LAZY_CHILD_COUNTS)
      // Beam/visited rejection reads only the syndrome. Keep cumulative
      // blocking unchanged, but materialize counts only for surviving children.
      for (int d : edets[ei]) {
        const int fired = next_detectors[d] ? 1 : -1;
        for (int oei : active_d2e[d]) {
          next_detector_cost_tuples[oei].detectors_count += fired;
        }
      }
      prev_ei = ei;
#endif

#if defined(TESSERACT_VERIFY_COMPACT_STATE) || defined(TESSERACT_VERIFY_SPLIT_STORAGE)
      verify_trial_state(node.error_chain_idx, ei, min_detector,
                         next_detectors, next_detector_cost_tuples);
#endif
      for (int d : edets[ei]) {
        if (detectors[d]) {
          if (detector_cost_cache[d] == -1) {
            detector_cost_cache[d] = trial_detcost(d, detector_cost_tuples);
          }
          next_cost -= detector_cost_cache[d];
        } else {
          next_cost += trial_detcost(d, next_detector_cost_tuples);
        }
      }

      for (int od : eneighbors[ei]) {
        if (!detectors[od] || !next_detectors[od]) continue;
        if (detector_cost_cache[od] == -1) {
          detector_cost_cache[od] = trial_detcost(od, detector_cost_tuples);
        }
        next_cost -= detector_cost_cache[od];
        next_cost += trial_detcost(od, next_detector_cost_tuples);
      }

      if (next_cost == INF) continue;

      // Create the error chain node for this candidate.
      error_chain_arena.emplace_back();
      auto& next_node = error_chain_arena.back();
      next_node.error_index = ei;
      next_node.min_detector = min_detector;
      next_node.parent_idx = node.error_chain_idx;

      pq.push({next_cost, next_num_dets, node.depth + 1, (int64_t)(error_chain_arena.size() - 1)});
      ++num_pq_pushed;

      if (num_pq_pushed > config.pqlimit) {
        if (config.verbose) {
          std::cout << "setting low confidence flag" << std::endl;
        }
        low_confidence_flag = true;
        return;
      }
    }
  }

  if (!pq.empty()) {
    throw std::runtime_error("Priority queue should be empty after decoding failure.");
  }
  if (config.verbose) {
    std::cout << "Decoding failed to converge within beam limit." << std::endl;
  }
  low_confidence_flag = true;
  };  // decode_trial
#ifdef TESSERACT_COMPACT_TUPLES
  struct CompactTuple {
    uint8_t error_blocked;
    uint8_t detectors_count;
  };
  static_assert(sizeof(CompactTuple) == 2);
  bool compact_safe = true;
  std::vector<uint8_t> compact_input_seen(num_detectors, 0);
  for (size_t d : detections) {
    if (d >= num_detectors || compact_input_seen[d]) { compact_safe = false; break; }
    compact_input_seen[d] = 1;
  }
  if (compact_safe) {
    auto* plan=batch_detail::active_batch_plan;
    if (plan && (plan->owner != this || config.sparsify_errors || config.verbose ||
                 config.create_visualization || &active_d2e != &d2e ||
                 !batch_detail::batch_environment_supported())) plan=nullptr;
    if (plan && plan->environment != batch_detail::batch_environment()) {
      plan->invalidate();
      plan->environment=batch_detail::batch_environment();
    }
    if (plan && plan->compact_graph_eligible.has_value()) {
      compact_safe=*plan->compact_graph_eligible;
      ++plan->compact_hits;
    } else {
      for (size_t ei = 0; ei < num_errors; ++ei) {
        if (errors[ei].symptom.detectors.size() > 16) { compact_safe = false; break; }
      }
      if (plan) { plan->compact_graph_eligible=compact_safe; ++plan->compact_builds; }
    }
  }
  if (compact_safe) {
#ifdef TESSERACT_SPLIT_COMPACT_STORAGE
    decode_trial.template operator()<CompactTuple, SplitCompactStorage<CompactTuple>>();
#else
    decode_trial.template operator()<CompactTuple>();
#endif
    return;
  }
#endif
  decode_trial.template operator()<DetectorCostTuple>();

}

double TesseractDecoder::cost_from_errors(const std::vector<size_t>& predicted_errors) const {
  double total_cost = 0;
  for (size_t dem_error_index : predicted_errors) {
    size_t error_index = dem_error_to_error.at(dem_error_index);
    if (error_index == std::numeric_limits<size_t>::max()) {
      throw std::invalid_argument("error index does not map to a retained decoder error");
    }
    total_cost += errors[error_index].likelihood_cost;
  }
  return total_cost;
}

std::vector<int> TesseractDecoder::get_flipped_observables(
    const std::vector<size_t>& predicted_errors) const {
  std::unordered_set<int> flipped_observables_set;

  // Iterate over all errors and compute the mask.
  // We use a set to perform an XOR-like sum.
  // If an observable is already in the set, we remove it (XORing with itself).
  // If it's not, we add it.
  for (size_t dem_error_index : predicted_errors) {
    size_t error_index = dem_error_to_error.at(dem_error_index);
    if (error_index == std::numeric_limits<size_t>::max()) {
      throw std::invalid_argument("error index does not map to a retained decoder error");
    }
    for (int obs_index : errors[error_index].symptom.observables) {
      if (flipped_observables_set.count(obs_index)) {
        flipped_observables_set.erase(obs_index);
      } else {
        flipped_observables_set.insert(obs_index);
      }
    }
  }

  // Convert the set to a vector and return it.
  std::vector<int> flipped_observables(flipped_observables_set.begin(),
                                       flipped_observables_set.end());
  // Sort observables
  std::sort(flipped_observables.begin(), flipped_observables.end());
  return flipped_observables;
}

std::vector<int> TesseractDecoder::decode(const std::vector<uint64_t>& detections) {
  decode_to_errors(detections);
  return get_flipped_observables(predicted_errors_buffer);
}

DecodeResult TesseractDecoder::decode_result(const std::vector<uint64_t>& detections) {
  decode_to_errors(detections);

  DecodeResult result;
  result.predictions = get_flipped_observables(predicted_errors_buffer);
  result.predicted_errors = predicted_errors_buffer;
  result.predicted_errors_populated = true;
  result.low_confidence = low_confidence_flag;
  result.total_cost = cost_from_errors(predicted_errors_buffer);
  return result;
}

DecodeResult TesseractDecoder::decode_result(
    const std::vector<uint64_t>& detections,
    const std::vector<ErrorProbabilityUpdate>& probability_updates) {
  if (probability_updates.empty()) {
    return decode_result(detections);
  }
  batch_detail::SuspendBatchPlan suspend_batch;
  ErrorProbabilityRollback rollback = apply_error_probability_updates(probability_updates);
  ScopeExit restore_probabilities([&] { restore_error_probabilities(rollback); });
  return decode_result(detections);
}

void TesseractDecoder::decode_shots(std::vector<stim::SparseShot>& shots,
                                    std::vector<std::vector<int>>& obs_predicted) {
  batch_detail::BatchPlanScope batch_scope(this);
  obs_predicted.resize(shots.size());
  for (size_t i = 0; i < shots.size(); ++i) {
    obs_predicted[i] = decode(shots[i].hits);
  }
}

void TesseractDecoder::build_sparse_d2e(const std::vector<uint64_t>& detections) {
  std::vector<uint8_t> shot_dets(num_detectors, 0);
  for (uint64_t d : detections) {
    if (d < num_detectors) {
      shot_dets[d] = 1;
    }
  }

  std::fill(sparse_error_active.begin(), sparse_error_active.end(), 0);

  for (int ei : sparsify_mandatory_errors) {
    sparse_error_active[ei] = 1;
  }

  struct OptionalErrorCandidate {
    int error_index;
    int overlap;
    int degree;
    double likelihood_cost;
  };

  std::vector<OptionalErrorCandidate> candidates;
  candidates.reserve(sparsify_optional_errors.size());

  for (int ei : sparsify_optional_errors) {
    int overlap = 0;
    for (int d : errors[ei].symptom.detectors) {
      if (shot_dets[d]) {
        overlap++;
      }
    }
    if (overlap > 0) {
      candidates.push_back({ei, overlap, static_cast<int>(errors[ei].symptom.detectors.size()),
                            errors[ei].likelihood_cost});
    }
  }

  std::sort(candidates.begin(), candidates.end(),
            [](const OptionalErrorCandidate& a, const OptionalErrorCandidate& b) {
              if (a.overlap != b.overlap) {
                return a.overlap > b.overlap;
              }
              if (a.degree != b.degree) {
                return a.degree < b.degree;
              }
              if (a.likelihood_cost != b.likelihood_cost) {
                return a.likelihood_cost < b.likelihood_cost;
              }
              return a.error_index < b.error_index;
            });

  size_t limit = std::min(static_cast<size_t>(config.sparsify_reactivate_limit), candidates.size());
  for (size_t i = 0; i < limit; ++i) {
    sparse_error_active[candidates[i].error_index] = 1;
  }

  for (size_t d = 0; d < num_detectors; ++d) {
    sparse_d2e[d].clear();
    for (int ei : d2e[d]) {
      if (sparse_error_active[ei]) {
        sparse_d2e[d].push_back(ei);
      }
    }
  }
}

}  // namespace tesseract_decoder


namespace tesseract_decoder::batch_detail {
void verify_batch_plan_visible(const BatchPlanState* expected) {
  if (active_batch_plan != expected || !batch_environment_supported()) {
    std::cerr << "batch_visibility expected=" << expected << " actual=" << active_batch_plan
              << " frame=" << active_batch_frame << " rounding=" << std::fegetround()
              << " environment=" << batch_environment()
              << " supported=" << batch_environment_supported() << '\n';
    throw std::logic_error("decoder TU batch activation/environment mismatch");
  }
}
}
