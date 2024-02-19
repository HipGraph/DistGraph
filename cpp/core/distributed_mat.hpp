#pragma once

namespace distblas::core {

class DistributedMat {

public:
  unique_ptr<vector<vector<Tuple<VALUE_TYPE>>>> sparse_data_collector;

  unique_ptr<vector<INDEX_TYPE>> sparse_data_counter;

  unique_ptr<vector<vector<VALUE_TYPE>>> dense_collector;

  bool  hash_spgemm;

  DistributedMat() = default;
  DistributedMat(const DistributedMat&) = delete;
  DistributedMat& operator=(const DistributedMat&) = delete;

};

}
