#include "src/batch_plan.h"
#include "src/tesseract.h"
#include "src/utils.h"

#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <algorithm>
#include <unordered_map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace {

template <typename T>
void write_vector(const std::filesystem::path& path, const std::vector<T>& values) {
  static_assert(std::is_trivially_copyable_v<T>);
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("failed to open " + path.string());
  out.write(reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(T)));
  if (!out) throw std::runtime_error("failed to write " + path.string());
}

template <typename Rows, typename Value>
void flatten_rows(const Rows& rows, std::vector<uint64_t>& offsets, std::vector<Value>& values) {
  offsets.clear();
  values.clear();
  offsets.reserve(rows.size() + 1);
  offsets.push_back(0);
  for (const auto& row : rows) {
    for (auto value : row) {
      if constexpr (std::is_signed_v<decltype(value)>) {
        if (value < 0) throw std::runtime_error("negative graph index");
      }
      if (static_cast<uint64_t>(value) > std::numeric_limits<Value>::max()) {
        throw std::runtime_error("graph index does not fit export type");
      }
      values.push_back(static_cast<Value>(value));
    }
    offsets.push_back(values.size());
  }
}

struct EneighborExportStats {
  uint64_t detectors;
  uint64_t word_groups;
};

template <typename T>
void write_row(std::ofstream& out, const std::vector<T>& values,
               const std::filesystem::path& path) {
  static_assert(std::is_trivially_copyable_v<T>);
  if (values.empty()) return;
  out.write(reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(T)));
  if (!out) throw std::runtime_error("failed to write " + path.string());
}

// Emit both the legacy detector CSR and the representation used by the CTA
// kernel.  Rows are written immediately, so export memory is bounded by one
// row instead of the complete (potentially multi-gigabyte) neighbor graph.
// Strict row ordering makes enumeration of set bits in (word, mask) order
// exactly reproduce the legacy detector order.
template <typename Rows>
EneighborExportStats write_eneighbors(
    const std::filesystem::path& output_dir, const Rows& rows,
    uint64_t num_detectors) {
  const auto raw_path = output_dir / "eneighbors.u32";
  const auto word_path = output_dir / "eneighbor_words.u32";
  const auto mask_path = output_dir / "eneighbor_masks.u64";
  std::ofstream raw_out(raw_path, std::ios::binary | std::ios::trunc);
  std::ofstream word_out(word_path, std::ios::binary | std::ios::trunc);
  std::ofstream mask_out(mask_path, std::ios::binary | std::ios::trunc);
  if (!raw_out) throw std::runtime_error("failed to open " + raw_path.string());
  if (!word_out) throw std::runtime_error("failed to open " + word_path.string());
  if (!mask_out) throw std::runtime_error("failed to open " + mask_path.string());

  std::vector<uint64_t> raw_offsets;
  std::vector<uint64_t> word_offsets;
  raw_offsets.reserve(rows.size() + 1);
  word_offsets.reserve(rows.size() + 1);
  raw_offsets.push_back(0);
  word_offsets.push_back(0);

  std::vector<uint32_t> raw_row;
  std::vector<uint32_t> word_row;
  std::vector<uint64_t> mask_row;
  for (const auto& row : rows) {
    raw_row.clear();
    word_row.clear();
    mask_row.clear();
    raw_row.reserve(row.size());

    bool have_previous = false;
    uint64_t previous = 0;
    uint32_t current_word = 0;
    uint64_t current_mask = 0;
    for (auto value : row) {
      if constexpr (std::is_signed_v<decltype(value)>) {
        if (value < 0) throw std::runtime_error("negative eneighbor detector");
      }
      const uint64_t detector = static_cast<uint64_t>(value);
      if (detector >= num_detectors ||
          detector > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("eneighbor detector does not fit export model");
      }
      if (have_previous && detector <= previous) {
        throw std::runtime_error(
            "eneighbor rows must be strictly sorted and unique");
      }
      previous = detector;
      raw_row.push_back(static_cast<uint32_t>(detector));

      const uint32_t word = static_cast<uint32_t>(detector >> 6);
      const uint64_t bit = uint64_t{1} << (detector & 63);
      if (!have_previous) {
        current_word = word;
        current_mask = bit;
      } else if (word == current_word) {
        current_mask |= bit;
      } else {
        word_row.push_back(current_word);
        mask_row.push_back(current_mask);
        current_word = word;
        current_mask = bit;
      }
      have_previous = true;
    }
    if (have_previous) {
      word_row.push_back(current_word);
      mask_row.push_back(current_mask);
    }

    write_row(raw_out, raw_row, raw_path);
    write_row(word_out, word_row, word_path);
    write_row(mask_out, mask_row, mask_path);
    raw_offsets.push_back(raw_offsets.back() + raw_row.size());
    word_offsets.push_back(word_offsets.back() + word_row.size());
  }

  raw_out.close();
  word_out.close();
  mask_out.close();
  if (!raw_out) throw std::runtime_error("failed to close " + raw_path.string());
  if (!word_out) throw std::runtime_error("failed to close " + word_path.string());
  if (!mask_out) throw std::runtime_error("failed to close " + mask_path.string());
  write_vector(output_dir / "eneighbor_offsets.u64", raw_offsets);
  write_vector(output_dir / "eneighbor_word_offsets.u64", word_offsets);
  return {raw_offsets.back(), word_offsets.back()};
}

void write_json(const std::filesystem::path& path, const nlohmann::json& value) {
  std::ofstream out(path, std::ios::trunc);
  if (!out) throw std::runtime_error("failed to open " + path.string());
  out << value.dump(2) << '\n';
  if (!out) throw std::runtime_error("failed to write " + path.string());
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 7) {
    std::cerr << "usage: export_model circuit shots seed beam pqlimit output_dir\n";
    return 2;
  }

  const std::filesystem::path circuit_path = argv[1];
  const size_t shot_count = std::stoull(argv[2]);
  const uint64_t seed = std::stoull(argv[3]);
  const size_t beam = std::stoull(argv[4]);
  const size_t pqlimit = std::stoull(argv[5]);
  const std::filesystem::path output_dir = argv[6];
  std::filesystem::create_directories(output_dir);

  FILE* file = std::fopen(circuit_path.string().c_str(), "r");
  if (!file) throw std::runtime_error("failed to open circuit " + circuit_path.string());
  auto circuit = stim::Circuit::from_file(file);
  std::fclose(file);
  const auto dem = stim::ErrorAnalyzer::circuit_to_detector_error_model(
      circuit, false, true, true, 1, false, false);

  tesseract_decoder::TesseractConfig config{dem};
  config.det_beam = static_cast<int>(beam);
  config.pqlimit = pqlimit;
  tesseract_decoder::TesseractDecoder decoder(config);

  std::vector<double> weights;
  std::vector<uint32_t> error_to_dem;
  std::vector<std::vector<uint32_t>> supports(decoder.num_errors);
  std::vector<std::vector<uint32_t>> observables(decoder.num_errors);
  weights.reserve(decoder.num_errors);
  error_to_dem.reserve(decoder.num_errors);
  for (size_t error = 0; error < decoder.num_errors; ++error) {
    weights.push_back(decoder.errors[error].likelihood_cost);
    if (decoder.error_to_dem_error[error] > std::numeric_limits<uint32_t>::max()) {
      throw std::runtime_error("DEM error index does not fit uint32");
    }
    error_to_dem.push_back(static_cast<uint32_t>(decoder.error_to_dem_error[error]));
    for (int detector : decoder.edets[error]) {
      if (detector < 0) throw std::runtime_error("negative detector index");
      supports[error].push_back(static_cast<uint32_t>(detector));
    }
    for (int observable : decoder.errors[error].symptom.observables) {
      if (observable < 0) throw std::runtime_error("negative observable index");
      observables[error].push_back(static_cast<uint32_t>(observable));
    }
  }

  std::vector<uint64_t> offsets;
  std::vector<uint32_t> values;
  flatten_rows(supports, offsets, values);
  write_vector(output_dir / "edet_offsets.u64", offsets);
  write_vector(output_dir / "edets.u32", values);
  const size_t incidence_count = values.size();

  flatten_rows(decoder.d2e, offsets, values);
  write_vector(output_dir / "d2e_offsets.u64", offsets);
  write_vector(output_dir / "d2e.u32", values);

  const auto eneighbor_stats =
      write_eneighbors(output_dir, decoder.eneighbors, decoder.num_detectors);

  flatten_rows(observables, offsets, values);
  write_vector(output_dir / "observable_offsets.u64", offsets);
  write_vector(output_dir / "observables.u32", values);

  std::vector<uint32_t> detector_order;
  for (size_t detector : decoder.config.detector_orders.at(0).get_order()) {
    if (detector > std::numeric_limits<uint32_t>::max()) {
      throw std::runtime_error("detector order index does not fit uint32");
    }
    detector_order.push_back(static_cast<uint32_t>(detector));
  }
  write_vector(output_dir / "weights.f64", weights);
  write_vector(output_dir / "error_to_dem.u32", error_to_dem);
  write_vector(output_dir / "detector_order.u32", detector_order);

  // Export the same exhaustively validated weight/count ordering used by the
  // production ranked heuristic.  Device code consumes these integer ranks
  // but returns the exact CPU-computed quotient for the winning type.
  constexpr uint32_t rank_stride = 17;
  constexpr uint32_t no_rank = std::numeric_limits<uint32_t>::max();
  // The packed GPU rows reserve UINT16_MAX as the invalid-rank sentinel.  The
  // original prototype used one byte, which rejected valid models (including
  // d=11 midout color code) once the global exact rank exceeded 254.
  constexpr uint32_t packed_no_rank = std::numeric_limits<uint16_t>::max();
  struct RankType {
    double weight;
    uint32_t count;
    uint32_t group;
  };
  bool rank_safe = pqlimit >= 1024;
  std::unordered_map<uint64_t, uint32_t> group_map;
  std::vector<double> group_weights;
  std::vector<uint32_t> group_degrees;
  std::vector<uint32_t> rank_group;
  std::vector<uint32_t> rank_lower;
  rank_group.reserve(decoder.num_errors);
  rank_lower.assign(decoder.num_errors, no_rank);
  if (rank_safe) {
    for (size_t error = 0; error < decoder.num_errors; ++error) {
      const double weight = decoder.error_costs[error].likelihood_cost;
      const size_t degree = decoder.errors[error].symptom.detectors.size();
      if (!(weight > 0) || !std::isfinite(weight) ||
          !std::isfinite(weight * double(no_rank)) || degree >= rank_stride) {
        rank_safe = false;
        break;
      }
      auto [it, inserted] =
          group_map.emplace(std::bit_cast<uint64_t>(weight), group_weights.size());
      if (inserted) {
        group_weights.push_back(weight);
        group_degrees.push_back(0);
      }
      const uint32_t group = it->second;
      if (group > std::numeric_limits<uint8_t>::max()) {
        rank_safe = false;
        break;
      }
      group_degrees[group] = std::max(group_degrees[group], static_cast<uint32_t>(degree));
      rank_group.push_back(group);
    }
  }
  std::vector<uint32_t> count_ranks;
  std::vector<double> count_values;
  uint32_t maximum_rank = 0;
  bool rank_rows_monotonic = false;
  if (rank_safe) {
    std::vector<RankType> types;
    for (uint32_t group = 0; group < group_weights.size(); ++group) {
      for (uint32_t count = 1; count <= group_degrees[group]; ++count) {
        types.push_back({group_weights[group], count, group});
      }
    }
    // Sort by the exact mathematical quotient represented by each binary64
    // weight.  Sorting by a rounded binary64 division can collapse two nearby
    // quotients that the production cross-product comparison distinguishes
    // (this occurs in the d=11 midout color graph).  The integer mantissa
    // comparison is a strict weak ordering; the exhaustive binary64-product
    // check below still rejects any model where rounded CPU comparisons do
    // not admit the resulting ranks.
    const auto exact_ratio_less = [](const RankType& a, const RankType& b) {
      const auto parts = [](double value) {
        const uint64_t bits = std::bit_cast<uint64_t>(value);
        const uint64_t fraction = bits & ((uint64_t{1} << 52) - 1);
        const uint32_t biased = static_cast<uint32_t>((bits >> 52) & 0x7ff);
        if (biased == 0) return std::pair{fraction, -1074};
        return std::pair{fraction | (uint64_t{1} << 52),
                         static_cast<int>(biased) - 1023 - 52};
      };
      const auto [a_mantissa, a_exponent] = parts(a.weight);
      const auto [b_mantissa, b_exponent] = parts(b.weight);
      uint64_t lhs = a_mantissa * b.count;
      uint64_t rhs = b_mantissa * a.count;
      const int lhs_top = static_cast<int>(std::bit_width(lhs)) - 1 + a_exponent;
      const int rhs_top = static_cast<int>(std::bit_width(rhs)) - 1 + b_exponent;
      if (lhs_top != rhs_top) return lhs_top < rhs_top;
      if (a_exponent > b_exponent) {
        lhs <<= a_exponent - b_exponent;
      } else {
        rhs <<= b_exponent - a_exponent;
      }
      return lhs < rhs;
    };
    std::sort(types.begin(), types.end(), exact_ratio_less);
    count_ranks.assign(group_weights.size() * rank_stride, no_rank);
    count_values.assign(group_weights.size() * rank_stride, 0);
    uint32_t rank = 0;
    for (size_t i = 0; i < types.size(); ++i) {
      const auto& type = types[i];
      if (i && types[i - 1].weight * type.count !=
                   type.weight * types[i - 1].count) {
        ++rank;
      }
      count_ranks[type.group * rank_stride + type.count] = rank;
      count_values[type.group * rank_stride + type.count] =
          type.weight / type.count + decoder.config.det_penalty;
    }
    maximum_rank = rank;
    if (rank >= packed_no_rank) rank_safe = false;
    for (const auto& a : types) {
      for (const auto& b : types) {
        if ((a.weight * b.count < b.weight * a.count) !=
            (count_ranks[a.group * rank_stride + a.count] <
             count_ranks[b.group * rank_stride + b.count])) {
          rank_safe = false;
          break;
        }
      }
      if (!rank_safe) break;
    }
    if (rank_safe) {
      for (size_t error = 0; error < decoder.num_errors; ++error) {
        const uint32_t degree = decoder.errors[error].symptom.detectors.size();
        rank_lower[error] = count_ranks[rank_group[error] * rank_stride + degree];
      }
      rank_rows_monotonic = true;
      for (const auto& row : decoder.d2e) {
        uint32_t previous = 0;
        bool first = true;
        for (int error : row) {
          const uint32_t lower = rank_lower[error];
          if (!first && lower < previous) rank_rows_monotonic = false;
          previous = lower;
          first = false;
        }
      }
    }
  }
  if (!rank_safe) {
    rank_group.clear();
    rank_lower.clear();
    count_ranks.clear();
    count_values.clear();
  }
  write_vector(output_dir / "rank_group.u32", rank_group);
  write_vector(output_dir / "rank_lower.u32", rank_lower);
  write_vector(output_dir / "count_ranks.u32", count_ranks);
  write_vector(output_dir / "count_values.f64", count_values);

  std::vector<stim::SparseShot> shots;
  tesseract_decoder::sample_shots(seed, circuit, shot_count, shots);
  std::vector<std::vector<uint32_t>> shot_detections(shots.size());
  std::vector<std::vector<uint32_t>> shot_observables(shots.size());
  for (size_t shot = 0; shot < shots.size(); ++shot) {
    for (uint64_t detector : shots[shot].hits) {
      if (detector > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("shot detector index does not fit uint32");
      }
      shot_detections[shot].push_back(static_cast<uint32_t>(detector));
    }
    for (size_t observable = 0; observable < circuit.count_observables(); ++observable) {
      if (shots[shot].obs_mask[observable]) {
        shot_observables[shot].push_back(static_cast<uint32_t>(observable));
      }
    }
  }
  flatten_rows(shot_detections, offsets, values);
  write_vector(output_dir / "shot_offsets.u64", offsets);
  write_vector(output_dir / "shot_detections.u32", values);
  flatten_rows(shot_observables, offsets, values);
  write_vector(output_dir / "truth_offsets.u64", offsets);
  write_vector(output_dir / "truth_observables.u32", values);

  const double decode_seconds = 0;
  write_json(output_dir / "reference.json", nlohmann::json::array());
  nlohmann::json metadata = {
      {"format", 1},
      {"circuit", circuit_path.generic_string()},
      {"shots", shots.size()},
      {"seed", seed},
      {"beam", beam},
      {"pqlimit", pqlimit},
      {"num_detectors", decoder.num_detectors},
      {"num_errors", decoder.num_errors},
      {"num_observables", decoder.num_observables},
      {"incidences", incidence_count},
      {"eneighbors", eneighbor_stats.detectors},
      {"eneighbor_word_groups", eneighbor_stats.word_groups},
      {"det_penalty", decoder.config.det_penalty},
      {"no_revisit_dets", decoder.config.no_revisit_dets},
      {"merge_errors", decoder.config.merge_errors},
      {"detector_orders", decoder.config.detector_orders.size()},
      {"rank_safe", rank_safe},
      {"rank_stride", rank_stride},
      {"rank_groups", group_weights.size()},
      {"rank_maximum", maximum_rank},
      {"rank_storage_bits", 16},
      {"rank_rows_monotonic", rank_rows_monotonic},
      {"cpu_reference_seconds", decode_seconds},
  };
  write_json(output_dir / "metadata.json", metadata);
  std::cout << metadata.dump() << '\n';
  return 0;
}
