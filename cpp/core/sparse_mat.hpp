/**
 * This class implements the distributed sparse graph.
 */
#pragma once
#include "common.h"
#include "csr_local.hpp"
#include "distributed_mat.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <parallel/algorithm>
#include <unordered_set>
#include <vector>
#include "common.h"
#include "../net/process_3D_grid.hpp"

using namespace std;
using namespace distblas::net;

namespace distblas::core {

/**
 * This class represents the Sparse Matrix
 */
template <typename T> class SpMat : public DistributedMat {

private:
public:
  uint64_t gRows, gCols, gNNz;
  vector<Tuple<T>> coords;
  int batch_size;
  int proc_col_width, proc_row_width;
  bool transpose = false;
  bool col_partitioned = false;
  unique_ptr<CSRLocal<T>> csr_local_data;
  Process3DGrid *grid;

  unique_ptr<vector<unordered_map<uint64_t, SparseCacheEntry<T>>>>
      tempCachePtr;

  /**
   * Constructor for Sparse Matrix representation of  Adj matrix
   * @param coords  (src, dst, value) Tuple vector loaded as input
   * @param gRows   total number of Rows in Distributed global Adj matrix
   * @param gCols   total number of Cols in Distributed global Adj matrix
   * @param gNNz     total number of NNz in Distributed global Adj matrix
   */
  SpMat(Process3DGrid *grid, vector<Tuple<T>> &coords, uint64_t &gRows, uint64_t &gCols,
        uint64_t &gNNz, int &batch_size, int &proc_row_width,
        int &proc_col_width, bool transpose, bool col_partitioned) {
    this->gRows = gRows;
    this->gCols = gCols;
    this->gNNz = gNNz;
    this->coords = coords;
    this->batch_size = batch_size;
    this->proc_col_width = proc_col_width;
    this->proc_row_width = proc_row_width;
    this->transpose = transpose;
    this->col_partitioned = col_partitioned;
    this->grid = grid;
    this->tempCachePtr = std::make_unique<std::vector<std::unordered_map<uint64_t,SparseCacheEntry<T>>>>(grid->col_world_size);
  }

  SpMat(Process3DGrid *grid) {
    this->grid = grid;
  }

  /**
   * Initialize the CSR from coords data structure
   */
  void initialize_CSR_blocks() {

#pragma omp parallel for
    for (uint64_t i = 0; i < coords.size(); i++) {
      if (col_partitioned) {
        coords[i].col %= proc_col_width;
      } else {
        coords[i].row %= proc_row_width;
      }
    }
    Tuple<T> *coords_ptr = coords.data();

    if (col_partitioned) {
      // This is used to find sending indices
      csr_local_data =
          make_unique<CSRLocal<T>>(gRows, proc_col_width, coords.size(),
                                   coords_ptr, coords.size(), transpose);
    } else {
      // This is used to find receiving indices and computations
      csr_local_data =
          make_unique<CSRLocal<T>>(proc_row_width, gCols, coords.size(),
                                   coords_ptr, coords.size(), transpose);
    }
  }

  // if batch_id<0 it will fetch all the batches
  void fill_col_ids(int batch_id, int starting_proc, int end_proc, vector<unordered_set<uint64_t>> &proc_to_id_mapping,
                    unordered_map<uint64_t, unordered_map<int,bool>> &id_to_proc_mapping, bool mode) {

    if (mode == 0) {

      fill_col_ids_for_pulling(batch_id,starting_proc,end_proc, proc_to_id_mapping,id_to_proc_mapping);
    } else {
      fill_col_ids_for_pushing(batch_id, starting_proc,end_proc,proc_to_id_mapping,id_to_proc_mapping);
    }
  }

  /*
   * This method computes all indicies for pull based approach
   */
  void fill_col_ids_for_pulling(int batch_id, int starting_proc, int end_proc, vector<unordered_set<uint64_t>> &proc_to_id_mapping,
                                unordered_map<uint64_t, unordered_map<int,bool>> &id_to_proc_mapping) {

    int rank= grid->rank_in_col;
    int world_size = grid->col_world_size;

    distblas::core::CSRHandle *handle = (csr_local_data.get())->handler.get();

    vector<int> procs;
    for (int i = starting_proc; i < end_proc; i++) {
      int  target   = (col_partitioned)? (rank + i) % world_size: (rank >= i) ? (rank - i) % world_size : (world_size - i + rank) % world_size;
      procs.push_back(target);
    }


    if (col_partitioned) {
      for (int r = 0 ; r < procs.size(); r++) {
        uint64_t starting_index = batch_id * batch_size + proc_row_width * procs[r];
        auto end_index =
            std::min(std::min((starting_index+batch_size),static_cast<uint64_t>((procs[r] + 1) * proc_row_width)), gRows);

        for (int i = starting_index; i < end_index; i++) {
          if (rank != procs[r] and (handle->rowStart[i + 1] - handle->rowStart[i]) > 0) {
            for (auto j = handle->rowStart[i]; j < handle->rowStart[i + 1];j++) {
              auto col_val = handle->col_idx[j];
              { proc_to_id_mapping[procs[r]].insert(col_val);
                id_to_proc_mapping[col_val][procs[r]] = true;
              }
            }
          }
        }
      }
    } else if (transpose) {
      for (int r = 0 ; r < procs.size(); r++) {
        uint64_t starting_index = proc_col_width * procs[r];
        auto end_index =
            std::min(static_cast<uint64_t>((procs[r] + 1) * proc_col_width), gCols);
        for (int i = starting_index; i < end_index; i++) {
          if (rank != procs[r] and
              (handle->rowStart[i + 1] - handle->rowStart[i]) > 0) {
            for (auto j = handle->rowStart[i]; j < handle->rowStart[i + 1]; j++) {
              auto col_val = handle->col_idx[j];
              uint64_t dst_start = batch_id * batch_size;
              uint64_t dst_end_index = std::min((batch_id + 1) * batch_size, proc_row_width);
              if (col_val >= dst_start and col_val < dst_end_index) {
                { proc_to_id_mapping[procs[r]].insert(i);
                }
              }
            }
          }
        }
      }
    }
  }

  /*
   * This method computes all indicies for push based approach
   */
  void fill_col_ids_for_pushing(int batch_id,int starting_proc, int end_proc, vector<unordered_set<uint64_t>> &proc_to_id_mapping,
                                unordered_map<uint64_t, unordered_map<int,bool>> &id_to_proc_mapping) {
    int rank= grid->rank_in_col;
    int world_size = grid->col_world_size;

    distblas::core::CSRHandle *handle = (csr_local_data.get())->handler.get();

    auto batches = (proc_row_width / batch_size);

    if (!(proc_row_width % batch_size == 0)) {
      batches = (proc_row_width / batch_size) + 1;
    }

    vector<int> procs;
    for (int i = starting_proc; i < end_proc; i++) {
      int  target  = (col_partitioned)? (rank + i) % world_size: (rank >= i) ? (rank - i) % world_size : (world_size - i + rank) % world_size;
      procs.push_back(target);
    }

    if (col_partitioned) {
      // calculation of sender col_ids
      for (int r = 0 ; r < procs.size(); r++) {
        uint64_t starting_index = proc_row_width * procs[r];
        auto end_index = std::min(static_cast<uint64_t>((procs[r] + 1) * proc_row_width), gRows) -1;

        auto eligible_col_id_start =
            (batch_id >= 0) ? batch_id * batch_size : 0;
        auto eligible_col_id_end =
            (batch_id >= 0)
                ? std::min(static_cast<uint64_t>((batch_id + 1) * batch_size),
                           static_cast<uint64_t>(proc_col_width))
                : proc_col_width;

        for (auto i = starting_index; i <= (end_index); i++) {

          if (rank != procs[r] and
              (handle->rowStart[i + 1] - handle->rowStart[i]) > 0) {
            for (auto j = handle->rowStart[i]; j < handle->rowStart[i + 1];
                 j++) {
              auto col_val = handle->col_idx[j];
              if (col_val >= eligible_col_id_start and
                  col_val < eligible_col_id_end) {
                // calculation of sender col_ids
                { proc_to_id_mapping[procs[r]].insert(col_val);
                  id_to_proc_mapping[col_val][procs[r]] = true;
                }
              }
            }
          }
        }
      }
    } else if (transpose) {
      // calculation of receiver col_ids
      for (int r = 0 ; r < procs.size(); r++) {
        uint64_t starting_index =
            (batch_id >= 0) ? batch_id * batch_size + proc_col_width * procs[r]
                            : proc_col_width *  procs[r];
        auto end_index =
            (batch_id >= 0)
                ? std::min(
                      starting_index + batch_size,
                      std::min(static_cast<uint64_t>(( procs[r] + 1) * proc_col_width),
                               gCols)) -
                      1
                : std::min(static_cast<uint64_t>(( procs[r] + 1) * proc_col_width),
                           gCols) -
                      1;
        for (auto i = starting_index; i <= (end_index); i++) {
          if (rank !=  procs[r] and (handle->rowStart[i + 1] - handle->rowStart[i]) > 0 ) {
            proc_to_id_mapping[procs[r]].insert(i);
          }
        }
      }
    }
  }


  template<size_t dimension>
  void fetch_local_data(uint64_t local_key, SpTuple<T,dimension>& sp_tuple) {
     CSRHandle *handle = (csr_local_data.get())->handler.get();
     vector<Tuple<T>> result;
     if(handle->rowStart[local_key + 1]-handle->rowStart[local_key]>0){
       int count = handle->rowStart[local_key + 1]-handle->rowStart[local_key];
       result.resize(count);
       sp_tuple.offset = count;
       #pragma omp parallel for
       for (auto j = handle->rowStart[local_key]; j < handle->rowStart[local_key + 1];j++) {
         int index = j-handle->rowStart[local_key];
         sp_tuple.rows[index]=(col_partitioned)?local_key:local_key+proc_row_width * grid->rank_in_col;
         sp_tuple.cols[index]=handle->col_idx[j];
         sp_tuple.values[index]=handle->values[j];
       }
     }
  }


  void insert_cache(int rank, uint64_t key, int batch_id, int iteration,
                    Tuple<T> tuple) {

    if ((*this->tempCachePtr)[rank].find(key) != (*this->tempCachePtr)[rank].end()){
      (*this->tempCachePtr)[rank][key].tuples.push_back(tuple);
    } else {
      SparseCacheEntry<T> entry;
      entry.inserted_batch_id = batch_id;
      entry.inserted_itr = iteration;
      entry.tuples.push_back(tuple);
      (*this->tempCachePtr)[rank][key]=entry;
    }
  }

  auto fetch_data_vector_from_cache( vector<Tuple<T>>& entries,int rank, uint64_t key) {

    // Access the array using the provided rank and key

    auto arrayMap = (*tempCachePtr)[rank];
    auto it = arrayMap.find(key);

    if (it != arrayMap.end()) {
      auto temp = it->second;
      entries =  temp.value;
    }else {
      throw std::runtime_error("cannot find the given key");
    }
  }



  void print_coords(bool trans) {
    int rank= grid->rank_in_col;
    int world_size = grid->col_world_size;
    string output_path =
        "coords" + to_string(rank) + "trans" + to_string(trans) + ".txt";
    char stats[500];
    strcpy(stats, output_path.c_str());
    ofstream fout(stats, std::ios_base::app);

    for (int i = 0; i < coords.size(); i++) {
      fout << coords[i].row << " " << coords[i].value << " " << endl;
    }
  }

  ~SpMat() {}
};

} // namespace distblas::core
