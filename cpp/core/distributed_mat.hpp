#pragma once
#include "common.h"
#include "csr_local.hpp"
namespace distblas::core {

class DistributedMat {

public:
  unique_ptr<vector<vector<Tuple<VALUE_TYPE>>>> sparse_data_collector;

  unique_ptr<vector<INDEX_TYPE>> sparse_data_counter;

  unique_ptr<vector<vector<VALUE_TYPE>>> dense_collector;

  unique_ptr<CSRLocal<VALUE_TYPE>> csr_local_data;

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

  void initialize_CSR_from_dense_collector(INDEX_TYPE proc_row_width,INDEX_TYPE gCols, vector<vector<VALUE_TYPE>> *dense_collector=nullptr, bool reset_dense=true){
    unique_ptr<vector<Tuple<VALUE_TYPE>>> coords_ptr= make_unique<vector<Tuple<VALUE_TYPE>>>(vector<Tuple<VALUE_TYPE>>());

    if (dense_collector==nullptr) {
      dense_collector = this->dense_collector.get();
    }
#pragma omp parallel for
    for(auto i=0;i<dense_collector->size();i++) {
      vector<Tuple<VALUE_TYPE>> coords_local;
      for (auto j = 0; j < (*dense_collector)[i].size(); j++) {
        if ((*dense_collector)[i][j] != 0) {
          Tuple<VALUE_TYPE> t;
          t.col = j;
          t.row = i;
          t.value = (*dense_collector)[i][j];
          coords_local.push_back(t);
          if (reset_dense){
            (*dense_collector)[i][j]=0;
          }
        }
      }
#pragma omp critical
      (*coords_ptr).insert((*coords_ptr).end(), coords_local.begin(), coords_local.end());
    }
    csr_local_data = make_unique<CSRLocal<VALUE_TYPE>>(proc_row_width, gCols, (*coords_ptr).size(),coords_ptr->data(), (*coords_ptr).size(), false);
  }



  void initialize_CSR_from_sparse_collector() {
    csr_local_data = make_unique<CSRLocal<VALUE_TYPE>>(sparse_data_collector.get());
  }

};

}
