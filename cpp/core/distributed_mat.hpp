#pragma once

namespace distblas::core {

class DistributedMat {

public:
  unique_ptr<vector<vector<Tuple<VALUE_TYPE>>>> sparse_data_collector;

  unique_ptr<vector<INDEX_TYPE>> sparse_data_counter;

  unique_ptr<vector<vector<VALUE_TYPE>>> dense_collector;

  bool  hash_spgemm;

  DistributedMat() = default;

  DistributedMat(const DistributedMat& other)
      : sparse_data_collector(nullptr),
        sparse_data_counter(nullptr),
        dense_collector(nullptr),
        hash_spgemm(other.hash_spgemm) {
    // Perform deep copy or other necessary operations
    if (other.sparse_data_collector) {
      sparse_data_collector = make_unique<vector<vector<Tuple<VALUE_TYPE>>>>(
          *(other.sparse_data_collector));
    }

    if (other.sparse_data_counter) {
      sparse_data_counter = make_unique<vector<INDEX_TYPE>>(
          *(other.sparse_data_counter));
    }

    if (other.dense_collector) {
      dense_collector = make_unique<vector<vector<VALUE_TYPE>>>(
          *(other.dense_collector));
    }
  }

};

}
