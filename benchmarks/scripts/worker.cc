#ifdef FINAL_PRODUCTION
#include "src/tesseract.h"
#include "src/batch_plan.h"
#else
#include "benchmarks/baseline/tesseract.h"
#endif
#include "src/utils.h"
#include <nlohmann/json.hpp>
#include <bit>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>

template<class T> std::vector<T> read(const std::string& p) {
  std::ifstream f(p, std::ios::binary|std::ios::ate);
  if (!f) throw std::runtime_error("missing input: "+p);
  auto bytes=f.tellg();
  if (bytes<0 || size_t(bytes)%sizeof(T)) throw std::runtime_error("invalid input: "+p);
  std::vector<T> v(size_t(bytes)/sizeof(T)); f.seekg(0);
  f.read(reinterpret_cast<char*>(v.data()),bytes);
  if (!f) throw std::runtime_error("short input: "+p);
  return v;
}
int main(int argc,char** argv) {
  try {
    if(argc!=5) throw std::runtime_error("usage: worker circuit model beam budget");
    auto setup=std::chrono::steady_clock::now();
    FILE* f=fopen(argv[1],"r"); if(!f) throw std::runtime_error("missing circuit");
    auto circuit=stim::Circuit::from_file(f); fclose(f);
    auto dem=stim::ErrorAnalyzer::circuit_to_detector_error_model(circuit,false,true,true,1,false,false);
    tesseract_decoder::TesseractConfig config{dem};
    config.det_beam=std::stoi(argv[3]); config.pqlimit=std::stoull(argv[4]);
    tesseract_decoder::TesseractDecoder decoder(config);
#ifdef FINAL_PRODUCTION
    tesseract_decoder::batch_detail::BatchPlanState plan(&decoder);
#endif
    std::string dir=argv[2];
    auto offsets=read<uint64_t>(dir+"/shot_offsets.u64");
    auto detections=read<uint32_t>(dir+"/shot_detections.u32");
    if(offsets.empty() || offsets.front()!=0 || offsets.back()!=detections.size()) throw std::runtime_error("invalid shot CSR");
    std::cout<<nlohmann::json({{"ready",true},{"setup_seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-setup).count()}}).dump()<<std::endl;
    std::string line;
    while(std::getline(std::cin,line)) {
      auto command=nlohmann::json::parse(line);
      if(command.value("quit",false)) break;
      auto indices=command.at("indices").get<std::vector<size_t>>();
      bool full=command.value("full",false);
      nlohmann::json output=nlohmann::json::array();
      std::vector<tesseract_decoder::DecodeResult> results; results.reserve(indices.size());
      std::vector<std::vector<uint64_t>> hits; hits.reserve(indices.size());
      for(auto i:indices) {
        if(i+1>=offsets.size() || offsets[i]>offsets[i+1] || offsets[i+1]>detections.size()) throw std::runtime_error("shot out of bounds");
        hits.emplace_back(detections.begin()+offsets[i],detections.begin()+offsets[i+1]);
      }
      auto start=std::chrono::steady_clock::now();
      {
#ifdef FINAL_PRODUCTION
        tesseract_decoder::batch_detail::BatchPlanActivation activation(plan);
#endif
        for(auto& hit:hits) results.push_back(decoder.decode_result(hit));
      }
      double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
      for(auto& r:results) {
        nlohmann::json row={{"low_confidence",r.low_confidence},{"predictions",r.predictions}};
        if(full) { row["predicted_errors_populated"]=r.predicted_errors_populated; row["total_cost_bits"]=std::bit_cast<uint64_t>(r.total_cost); row["predicted_errors"]=r.predicted_errors; }
        output.push_back(std::move(row));
      }
      std::cout<<nlohmann::json({{"decisions",output},{"decode_seconds",seconds}}).dump()<<std::endl;
    }
  } catch(const std::exception& e) { std::cerr<<e.what()<<std::endl; return 1; }
}
