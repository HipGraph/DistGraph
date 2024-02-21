#pragma once
#include "../net/process_3D_grid.hpp"
#include <algorithm>
#include <memory>
#include <set>
#include "common.h"
#include "csr_local.hpp"
#include "distributed_mat.hpp"


using namespace distblas::net;

namespace distblas::core {

template <typename INDEX_TYPE, typename VALUE_TYPE> class SparseTile:public DistributedMat {

private:
public:
  int id;
  INDEX_TYPE row_starting_index;
  INDEX_TYPE row_end_index;
  INDEX_TYPE col_start_index;
  INDEX_TYPE col_end_index;

  Process3DGrid *grid;

  unordered_map<INDEX_TYPE, unordered_map<int, bool>> id_to_proc_mapping;
  unordered_set<INDEX_TYPE> col_id_set;
  unordered_set<INDEX_TYPE> row_id_set;
  int mode = 1; // mode=0 local and mode=1 remote

  uint64_t total_transferrable_datacount=0;
  uint64_t total_receivable_datacount=0;

  static double tile_width_fraction;

  unique_ptr<CSRLocal<VALUE_TYPE>> csr_local_data;

  unique_ptr<vector<distblas::core::SparseCacheEntry<VALUE_TYPE>>> dataCachePtr;

//  shared_ptr<vector<vector<Tuple<VALUE_TYPE>>>> sparse_data_collector;
//
//  shared_ptr<vector<INDEX_TYPE>> sparse_data_counter;
//
//  shared_ptr<vector<vector<VALUE_TYPE>>> dense_collector;
//
//  bool  hash_spgemm=false;

  SparseTile(Process3DGrid *grid, int id, INDEX_TYPE row_starting_index,
             INDEX_TYPE row_end_index, INDEX_TYPE col_start_index,
             INDEX_TYPE col_end_index)
      : DistributedMat(), grid(grid), id(id),
        row_starting_index(row_starting_index), row_end_index(row_end_index),
        col_start_index(col_start_index), col_end_index(col_end_index) {}

  SparseTile(Process3DGrid *grid, bool hash_spgemm)
      : DistributedMat(), grid(grid) {
    this->hash_spgemm =hash_spgemm;
  }

  SparseTile<INDEX_TYPE, VALUE_TYPE>::SparseTile(const SparseTile& other)
      : DistributedMat(other), id(other.id),
        row_starting_index(other.row_starting_index),
        row_end_index(other.row_end_index),
        col_start_index(other.col_start_index),
        col_end_index(other.col_end_index),
        grid(other.grid),
        id_to_proc_mapping(other.id_to_proc_mapping),
        col_id_set(other.col_id_set),
        row_id_set(other.row_id_set),
        mode(other.mode),
        total_transferrable_datacount(other.total_transferrable_datacount),
        total_receivable_datacount(other.total_receivable_datacount),
        csr_local_data(nullptr),  // Create a new object
        dataCachePtr(nullptr)    // Create a new object
  {}

  void insert(INDEX_TYPE col_index) { col_id_set.insert(col_index); }
  void insert_row_index(INDEX_TYPE row_index) { row_id_set.insert(row_index); }



  void initialize_output_DS_if(int comparing_mode){
    if (mode==comparing_mode){
      auto len = row_end_index- row_starting_index;
      dataCachePtr = make_unique<vector<SparseCacheEntry<VALUE_TYPE>>>(len,SparseCacheEntry<VALUE_TYPE>());
      if (this->hash_spgemm) {
        this->sparse_data_counter = make_unique<vector<INDEX_TYPE>>(len,0);
        this->sparse_data_collector = make_unique<vector<vector<Tuple<VALUE_TYPE>>>>(len, vector<Tuple<VALUE_TYPE>>());
      }else {
//        this->dense_collector = make_shared<vector<vector<VALUE_TYPE>>>(len,vector<VALUE_TYPE>(proc_col_width,0));
      }

    }
  }
  void initialize_hashtables(){
    auto len = row_end_index- row_starting_index;
#pragma  omp parallel for
    for(auto i=0;i<len;i++){
      auto count = (*this->sparse_data_counter)[i];
      auto resize_count = pow(2,log2(count)+1);
      (*this->sparse_data_collector)[i].clear();
      Tuple<VALUE_TYPE> t;
      t.row=i;
      t.col=-1;
      t.value=0;
      (*this->sparse_data_collector)[i].resize(resize_count,t);
      (*this->sparse_data_counter)[i]=0;
    }
  }


  void initialize_CSR_from_sparse_collector() {
    if (sparse_data_collector == nullptr){
      cout<<" rank "<<grid->rank_in_col<<" sparse_data_collector passing nullptr "<<"mode"<<mode<<"spgemm"<<this->hash_spgemm<<endl;
    }

    if (sparse_data_collector.get() == nullptr) {
      cout << " rank " << grid->rank_in_col
           << "sparse_data_collector->get() nullptr "
           << "mode" << mode << "spgemm" << this->hash_spgemm << endl;
    }
    csr_local_data = make_unique<CSRLocal<VALUE_TYPE>>(sparse_data_collector.get());

  }

  CSRHandle  fetch_remote_data(INDEX_TYPE global_key) {
    CSRHandle *handle = (csr_local_data)->handler.get();
    CSRHandle new_handler;
    INDEX_TYPE  local_key = global_key-row_starting_index;

    if (handle == nullptr ){
      cout<<" oh my god "<<endl;
    }

    int count=1;
//    int count = handle->rowStart[local_key + 1]-handle->rowStart[local_key];
    new_handler.row_idx.resize(1,global_key);
    if(count>0){
      new_handler.col_idx.resize(count);
      new_handler.values.resize(count);
//      copy(handle->col_idx.begin(),handle->col_idx.begin()+ count, new_handler.col_idx.begin());
//      copy(handle->values.begin(),handle->values.begin()+ count,new_handler.values.begin());
    }
    return new_handler;
  }


  static int get_tile_id(int batch_id, INDEX_TYPE col_index,
                         INDEX_TYPE proc_col_width, int rank) {
    int tiles_per_process_row = static_cast<int>(1/(tile_width_fraction));
    auto tile_width =
        (proc_col_width % tiles_per_process_row) == 0 ? (proc_col_width/tiles_per_process_row): ((proc_col_width/tiles_per_process_row) + 1);
    return static_cast<int>(col_index/tile_width);
  }

  static INDEX_TYPE get_tile_width(INDEX_TYPE proc_col_width) {
    int tiles_per_process_row = static_cast<int>(1 / (tile_width_fraction));
    return (proc_col_width % tiles_per_process_row) == 0
               ? (proc_col_width/tiles_per_process_row)
               : ((proc_col_width/tiles_per_process_row) + 1);
  }

  static int get_tiles_per_process_row() {
    return static_cast<int>(1/(tile_width_fraction));
  }

};

template <typename INDEX_TYPE, typename VALUE_TYPE>
double SparseTile<INDEX_TYPE, VALUE_TYPE>::tile_width_fraction = 0.5;
} // namespace distblas::core
