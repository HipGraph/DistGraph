/**
 * This class implements the distributed sparse graph.
 */
#pragma once
#include "../net/process_3D_grid.hpp"
#include "common.h"
#include "csr_local.hpp"
#include "distributed_mat.hpp"
#include "sparse_mat_tile.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <math.h>
#include <memory>
#include <mpi.h>
#include <parallel/algorithm>
#include <set>
#include <unordered_set>
#include <vector>

using namespace std;
using namespace distblas::net;

namespace distblas::core {

/**
 * This class represents the Sparse Matrix
 */

template <typename VALUE_TYPE> class SpMat : public DistributedMat {

private:
  void initialize_CSR_from_tuples() {
#pragma omp parallel for
    for (INDEX_TYPE i = 0; i < coords.size(); i++) {
      if (col_partitioned) {
        coords[i].col %= proc_col_width;
      } else {
        coords[i].row %= proc_row_width;
      }
    }
    Tuple<VALUE_TYPE> *coords_ptr = coords.data();

    if (col_partitioned) {
      // This is used to find sending indices
      this->csr_local_data = make_unique<CSRLocal<VALUE_TYPE>>(
          gRows, proc_col_width, coords.size(), coords_ptr, coords.size(),
          transpose);
    } else {
      // This is used to find receiving indices and computations
      this->csr_local_data = make_unique<CSRLocal<VALUE_TYPE>>(
          proc_row_width, gCols, coords.size(), coords_ptr, coords.size(),
          transpose);
    }
  }

  //  void initialize_CSR_from_sparse_collector() {
  //   this->csr_local_data =
  //   make_unique<CSRLocal<VALUE_TYPE>>(sparse_data_collector.get());
  //  }

  void find_col_ids_for_pulling_with_tiling(
      int batch_id, int starting_proc, int end_proc,
      vector<vector<unordered_map<INDEX_TYPE, unordered_map<int, bool>>>>
          *id_to_proc_mapping,
      vector<vector<vector<SparseTile<INDEX_TYPE, VALUE_TYPE>>>> *tile_map,
      string semring = "+", SpMat<VALUE_TYPE> *input_data = nullptr) {
    int rank = grid->rank_in_col;
    int world_size = grid->col_world_size;

    distblas::core::CSRHandle *handle =
        (this->csr_local_data.get())->handler.get();

    vector<int> procs;
    for (int i = starting_proc; i < end_proc; i++) {
      int target = (col_partitioned) ? (rank + i) % world_size
                   : (rank >= i)     ? (rank - i) % world_size
                                     : (world_size - i + rank) % world_size;
      procs.push_back(target);
    }
    if (col_partitioned) {
      for (int r = 0; r < procs.size(); r++) {
        INDEX_TYPE starting_index =
            batch_id * batch_size + proc_row_width * procs[r];
        auto end_index = std::min(
            std::min((starting_index + batch_size),
                     static_cast<INDEX_TYPE>((procs[r] + 1) * proc_row_width)),
            gRows);

        for (int i = starting_index; i < end_index; i++) {
          if (rank != procs[r] and
              (handle->rowStart[i + 1] - handle->rowStart[i]) > 0) {
            vector<unordered_set<INDEX_TYPE>> unique_per_row(
                SparseTile<INDEX_TYPE,
                           VALUE_TYPE>::get_tiles_per_process_row());
            for (auto j = handle->rowStart[i]; j < handle->rowStart[i + 1];
                 j++) {
              auto col_val = handle->col_idx[j];
              {
                int tile_id = SparseTile<INDEX_TYPE, VALUE_TYPE>::get_tile_id(
                    batch_id, col_val, proc_col_width, procs[r]);
                (*tile_map)[batch_id][procs[r]][tile_id].insert(col_val);
                if (semring == "+" and input_data != nullptr) {
                  CSRHandle *input_handle =
                      input_data->csr_local_data->handler.get();
                  if (input_handle->rowStart[col_val + 1] -
                          input_handle->rowStart[col_val] >
                      0) {
                    for (auto h = input_handle->rowStart[col_val];
                         h < input_handle->rowStart[col_val + 1]; h++) {
                      unique_per_row[tile_id].insert(input_handle->col_idx[h]);
                    }
                  }
                }
                (*id_to_proc_mapping)[batch_id][tile_id][col_val][procs[r]] =
                    true;
              }
            }
            for (int tile = 0;
                 tile < SparseTile<INDEX_TYPE,
                                   VALUE_TYPE>::get_tiles_per_process_row();
                 tile++) {
              (*tile_map)[batch_id][procs[r]][tile]
                  .total_receivable_datacount += unique_per_row[tile].size();
            }
          }
        }
      }
    } else if (transpose) {
      for (int r = 0; r < procs.size(); r++) {
        INDEX_TYPE starting_index = proc_col_width * procs[r];
        auto end_index = std::min(
            static_cast<INDEX_TYPE>((procs[r] + 1) * proc_col_width), gCols);
        for (auto i = starting_index; i < end_index; i++) {
          if (rank != procs[r] and
              (handle->rowStart[i + 1] - handle->rowStart[i]) > 0) {

            int tile_id = SparseTile<INDEX_TYPE, VALUE_TYPE>::get_tile_id(
                batch_id, (i - starting_index), proc_col_width, procs[r]);
            for (auto j = handle->rowStart[i]; j < handle->rowStart[i + 1];
                 j++) {
              auto col_val = handle->col_idx[j];
              INDEX_TYPE dst_start = batch_id * batch_size;
              INDEX_TYPE dst_end_index =
                  std::min((batch_id + 1) * batch_size, proc_row_width);
              if (col_val >= dst_start and col_val < dst_end_index) {
                {
                  (*tile_map)[batch_id][procs[r]][tile_id].insert(i);
                  (*tile_map)[batch_id][procs[r]][tile_id].insert_row_index(
                      col_val);
                }
              }
            }
          }
        }
      }
    }
  }

  //  void find_col_ids_for_pushing_with_tiling(int batch_id, int starting_proc,
  //  int end_proc,
  //                                            vector<vector<vector<SparseTile<INDEX_TYPE,VALUE_TYPE>>>>*
  //                                            tile_map,
  //                                            unordered_map<INDEX_TYPE,
  //                                            unordered_map<int,bool>>
  //                                            &id_to_proc_mapping) {
  //
  //
  //  }

  /*
   * This method computes all indicies for pull based approach
   */
  void find_col_ids_for_pulling(
      int batch_id, int starting_proc, int end_proc,
      vector<unordered_set<INDEX_TYPE>> &proc_to_id_mapping,
      unordered_map<INDEX_TYPE, unordered_map<int, bool>> &id_to_proc_mapping) {

    int rank = grid->rank_in_col;
    int world_size = grid->col_world_size;

    distblas::core::CSRHandle *handle =
        (this->csr_local_data.get())->handler.get();

    vector<int> procs;
    for (int i = starting_proc; i < end_proc; i++) {
      int target = (col_partitioned) ? (rank + i) % world_size
                   : (rank >= i)     ? (rank - i) % world_size
                                     : (world_size - i + rank) % world_size;
      procs.push_back(target);
    }

    if (col_partitioned) {
      for (int r = 0; r < procs.size(); r++) {
        INDEX_TYPE starting_index =
            batch_id * batch_size + proc_row_width * procs[r];
        auto end_index = std::min(
            std::min((starting_index + batch_size),
                     static_cast<INDEX_TYPE>((procs[r] + 1) * proc_row_width)),
            gRows);

        for (int i = starting_index; i < end_index; i++) {
          if (rank != procs[r] and
              (handle->rowStart[i + 1] - handle->rowStart[i]) > 0) {
            for (auto j = handle->rowStart[i]; j < handle->rowStart[i + 1];
                 j++) {
              auto col_val = handle->col_idx[j];
              {
                proc_to_id_mapping[procs[r]].insert(col_val);
                id_to_proc_mapping[col_val][procs[r]] = true;
              }
            }
          }
        }
      }
    } else if (transpose) {
      for (int r = 0; r < procs.size(); r++) {
        INDEX_TYPE starting_index = proc_col_width * procs[r];
        auto end_index = std::min(
            static_cast<INDEX_TYPE>((procs[r] + 1) * proc_col_width), gCols);
        for (int i = starting_index; i < end_index; i++) {
          if (rank != procs[r] and
              (handle->rowStart[i + 1] - handle->rowStart[i]) > 0) {
            for (auto j = handle->rowStart[i]; j < handle->rowStart[i + 1];
                 j++) {
              auto col_val = handle->col_idx[j];
              INDEX_TYPE dst_start = batch_id * batch_size;
              INDEX_TYPE dst_end_index =
                  std::min((batch_id + 1) * batch_size, proc_row_width);
              if (col_val >= dst_start and col_val < dst_end_index) {
                { proc_to_id_mapping[procs[r]].insert(i); }
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
  void find_col_ids_for_pushing(
      int batch_id, int starting_proc, int end_proc,
      vector<unordered_set<INDEX_TYPE>> &proc_to_id_mapping,
      unordered_map<INDEX_TYPE, unordered_map<int, bool>> &id_to_proc_mapping) {
    int rank = grid->rank_in_col;
    int world_size = grid->col_world_size;

    distblas::core::CSRHandle *handle =
        (this->csr_local_data.get())->handler.get();

    auto batches = (proc_row_width / batch_size);

    if (!(proc_row_width % batch_size == 0)) {
      batches = (proc_row_width / batch_size) + 1;
    }

    vector<int> procs;
    for (int i = starting_proc; i < end_proc; i++) {
      int target = (col_partitioned) ? (rank + i) % world_size
                   : (rank >= i)     ? (rank - i) % world_size
                                     : (world_size - i + rank) % world_size;
      procs.push_back(target);
    }

    if (col_partitioned) {
      // calculation of sender col_ids
      for (int r = 0; r < procs.size(); r++) {
        INDEX_TYPE starting_index = proc_row_width * procs[r];
        auto end_index =
            std::min(static_cast<INDEX_TYPE>((procs[r] + 1) * proc_row_width),
                     gRows) -
            1;

        auto eligible_col_id_start =
            (batch_id >= 0) ? batch_id * batch_size : 0;
        auto eligible_col_id_end =
            (batch_id >= 0)
                ? std::min(static_cast<INDEX_TYPE>((batch_id + 1) * batch_size),
                           static_cast<INDEX_TYPE>(proc_col_width))
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
                {
                  proc_to_id_mapping[procs[r]].insert(col_val);
                  id_to_proc_mapping[col_val][procs[r]] = true;
                }
              }
            }
          }
        }
      }
    } else if (transpose) {
      // calculation of receiver col_ids
      for (int r = 0; r < procs.size(); r++) {
        INDEX_TYPE starting_index =
            (batch_id >= 0) ? batch_id * batch_size + proc_col_width * procs[r]
                            : proc_col_width * procs[r];
        auto end_index =
            (batch_id >= 0)
                ? std::min(starting_index + batch_size,
                           std::min(static_cast<INDEX_TYPE>((procs[r] + 1) *
                                                            proc_col_width),
                                    gCols)) -
                      1
                : std::min(
                      static_cast<INDEX_TYPE>((procs[r] + 1) * proc_col_width),
                      gCols) -
                      1;
        for (auto i = starting_index; i <= (end_index); i++) {
          if (rank != procs[r] and
              (handle->rowStart[i + 1] - handle->rowStart[i]) > 0) {
            proc_to_id_mapping[procs[r]].insert(i);
          }
        }
      }
    }
  }

public:
  INDEX_TYPE gRows, gCols, gNNz;
  vector<Tuple<VALUE_TYPE>> coords;
  INDEX_TYPE batch_size;
  INDEX_TYPE proc_col_width, proc_row_width;
  bool transpose = false;
  bool col_partitioned = false;
  Process3DGrid *grid;

  unique_ptr<vector<unordered_map<INDEX_TYPE, SparseCacheEntry<VALUE_TYPE>>>>
      tempCachePtr;

  /**
   * Constructor for Sparse Matrix representation of  Adj matrix
   * @param coords  (src, dst, value) Tuple vector loaded as input
   * @param gRows   total number of Rows in Distributed global Adj matrix
   * @param gCols   total number of Cols in Distributed global Adj matrix
   * @param gNNz     total number of NNz in Distributed global Adj matrix
   */
  SpMat(Process3DGrid *grid, vector<Tuple<VALUE_TYPE>> &coords,
        INDEX_TYPE &gRows, INDEX_TYPE &gCols, INDEX_TYPE &gNNz, int &batch_size,
        int &proc_row_width, int &proc_col_width, bool transpose,
        bool col_partitioned)
      : DistributedMat() {
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
    this->tempCachePtr = std::make_unique<std::vector<
        std::unordered_map<INDEX_TYPE, SparseCacheEntry<VALUE_TYPE>>>>(
        grid->col_world_size);
  }

  SpMat(Process3DGrid *grid) : DistributedMat() {
    this->grid = grid;
    this->tempCachePtr = std::make_unique<std::vector<
        std::unordered_map<INDEX_TYPE, SparseCacheEntry<VALUE_TYPE>>>>(
        grid->col_world_size);
  }

  SpMat(Process3DGrid *grid, int &proc_row_width,  int &proc_col_width,
        bool hash_spgemm)
      : DistributedMat() {
    this->grid = grid;
    this->tempCachePtr = std::make_unique<std::vector<
        std::unordered_map<INDEX_TYPE, SparseCacheEntry<VALUE_TYPE>>>>(
        grid->col_world_size);
    this->proc_col_width = proc_col_width;
    this->proc_row_width = proc_row_width;
    this->batch_size = proc_row_width;
    if (hash_spgemm) {
      this->sparse_data_collector =
          make_unique<vector<vector<Tuple<VALUE_TYPE>>>>(
              proc_row_width, vector<Tuple<VALUE_TYPE>>());

      this->sparse_data_counter =
          make_unique<vector<INDEX_TYPE>>(proc_row_width, 0);
      this->hash_spgemm = true;
    } else {
      this->dense_collector = make_unique<vector<vector<VALUE_TYPE>>>(
          proc_row_width, vector<VALUE_TYPE>(proc_col_width, 0));
    }
  }

  /**
   * Initialize the CSR from coords data structure
   */
  void initialize_CSR_blocks() {

    if (coords.size() > 0) {
      initialize_CSR_from_tuples();
    } else if (hash_spgemm and sparse_data_collector->size() > 0) {
      this->initialize_CSR_from_sparse_collector();
    } else if (dense_collector->size() > 0) {
      this->initialize_CSR_from_dense_collector(this->proc_row_width,
                                                this->proc_col_width);
    }
  }

  void build_computable_represention() {
    if (this->csr_local_data != nullptr and
        this->csr_local_data->handler != nullptr) {
      distblas::core::CSRHandle *handle = this->csr_local_data->handler.get();
      if (this->hash_spgemm) {
        // TODO: implement hash spgemm
      } else {
        auto rows = handle->rowStart.size() - 1;
        auto cols = this->proc_col_width;
        this->dense_collector = make_unique<vector<vector<VALUE_TYPE>>>(
            rows, vector<VALUE_TYPE>(cols, 0));
#pragma omp parallel for
        for (auto i = 0; i < handle->rowStart.size() - 1; i++) {
          for (auto j = handle->rowStart[i]; j < handle->rowStart[i + 1]; j++) {
            auto d = handle->col_idx[j];
            auto value = handle->values[j];
            if (d < cols) {
              (*dense_collector)[i][d] = value;
            }
          }
        }
      }
    }
  }

  void initialize_hashtables() {
#pragma omp parallel for
    for (auto i = 0; i < proc_row_width; i++) {
      auto count = (*sparse_data_counter)[i];
      auto resize_count = pow(2, log2(count) + 1);
      (*sparse_data_collector)[i].clear();
      Tuple<VALUE_TYPE> t;
      t.row = i;
      t.col = -1;
      t.value = 0;
      (*sparse_data_collector)[i].resize(resize_count, t);
      (*sparse_data_counter)[i] = 0;
    }
  }

  // if batch_id<0 it will fetch all the batches
  void find_col_ids(
      int batch_id, int starting_proc, int end_proc,
      vector<unordered_set<INDEX_TYPE>> &proc_to_id_mapping,
      unordered_map<INDEX_TYPE, unordered_map<int, bool>> &id_to_proc_mapping,
      bool mode) {

    if (mode == 0) {
      find_col_ids_for_pulling(batch_id, starting_proc, end_proc,
                               proc_to_id_mapping, id_to_proc_mapping);
    } else {
      find_col_ids_for_pushing(batch_id, starting_proc, end_proc,
                               proc_to_id_mapping, id_to_proc_mapping);
    }
  }

  void find_col_ids_with_tiling(
      int batch_id, int starting_proc, int end_proc,
      vector<vector<vector<distblas::core::SparseTile<INDEX_TYPE, VALUE_TYPE>>>>
          *proc_to_id_mapping,
      vector<vector<unordered_map<INDEX_TYPE, unordered_map<int, bool>>>>
          *id_to_proc_mapping,
      bool mode, string semring = "+",
      SpMat<VALUE_TYPE> *input_data = nullptr) {

    if (mode == 0) {
      find_col_ids_for_pulling_with_tiling(
          batch_id, starting_proc, end_proc, id_to_proc_mapping,
          proc_to_id_mapping, semring, input_data);
    } else {
      //      find_col_ids_for_pushing_with_tiling(batch_id,
      //      starting_proc,end_proc,proc_to_id_mapping,id_to_proc_mapping);
    }
  }

  CSRHandle fetch_local_data(INDEX_TYPE local_key, bool embedding = false) {
    CSRHandle new_handler;
    INDEX_TYPE global_key = (col_partitioned) ? local_key: local_key + proc_row_width * grid->rank_in_col;
    new_handler.row_idx.resize(1, global_key);
    if (embedding) {
      if (this->hash_spgemm) {
      } else {
         for(auto i=0;i<(*this->dense_collector)[local_key].size();i++){
           if ((*this->dense_collector)[local_key][i]!=0){
             new_handler.col_idx.push_back(i);
             new_handler.values.push_back(i);
           }
         }
      }
    } else {
      CSRHandle *handle = (this->csr_local_data.get())->handler.get();

      int count = handle->rowStart[local_key + 1] - handle->rowStart[local_key];

      if (handle->rowStart[local_key + 1] - handle->rowStart[local_key] > 0) {
        new_handler.col_idx.resize(count);
        new_handler.values.resize(count);
        copy(handle->col_idx.begin(), handle->col_idx.begin() + count,
             new_handler.col_idx.begin());
        copy(handle->values.begin(), handle->values.begin() + count,
             new_handler.values.begin());
      }
    }
    return new_handler;
  }

  void get_transferrable_datacount(
      vector<vector<vector<SparseTile<INDEX_TYPE, VALUE_TYPE>>>> *tile_map,
      int total_batches, bool col_id_set, bool indices_only) {

    CSRHandle *handle = (this->csr_local_data.get())->handler.get();
    int tiles_per_process =
        SparseTile<INDEX_TYPE, VALUE_TYPE>::get_tiles_per_process_row();

    auto itr = total_batches * grid->col_world_size * tiles_per_process;
#pragma omp parallel for
    for (auto in = 0; in < itr; in++) {
      auto i = in / (grid->col_world_size * tiles_per_process);
      auto j = (in / tiles_per_process) % grid->col_world_size;
      auto k = in % tiles_per_process;
      INDEX_TYPE total_count = 0;
      SparseTile<INDEX_TYPE, VALUE_TYPE> &tile = (*tile_map)[i][j][k];
      if (col_id_set and !indices_only) {
        for (auto it = tile.col_id_set.begin(); it != tile.col_id_set.end();
             ++it) {
          total_count += handle->rowStart[(*it) + 1] - handle->rowStart[(*it)];
        }
        (*tile_map)[i][j][k].total_transferrable_datacount = total_count;

      } else if (col_id_set and indices_only) {

      } else if (!indices_only) {
        for (auto it = tile.row_id_set.begin(); it != tile.row_id_set.end();
             ++it) {
          total_count += handle->rowStart[(*it) + 1] - handle->rowStart[(*it)];
        }
        (*tile_map)[i][j][k].total_transferrable_datacount = total_count;
      } else {
        (*tile_map)[i][j][k].total_transferrable_datacount =
            (*tile_map)[i][j][k].row_id_set.size();
      }
    }
  }

  void purge_cache() {
    for (int i = 0; i < grid->col_world_size; i++) {
      (*this->tempCachePtr)[i].clear();
      //      std::unordered_map<INDEX_TYPE,
      //      distblas::core::SparseCacheEntry<VALUE_TYPE>>().swap(
      //          (*this->tempCachePtr)[i]);
    }
    this->tempCachePtr = std::make_unique<std::vector<
        std::unordered_map<INDEX_TYPE, SparseCacheEntry<VALUE_TYPE>>>>(
        grid->col_world_size);
  }

  auto fetch_data_vector_from_cache(vector<Tuple<VALUE_TYPE>> &entries,
                                    int rank, INDEX_TYPE key) {

    // Access the array using the provided rank and key

    auto arrayMap = (*tempCachePtr)[rank];
    auto it = arrayMap.find(key);

    if (it != arrayMap.end()) {
      auto temp = it->second;
      entries = temp.value;
    } else {
      throw std::runtime_error("cannot find the given key");
    }
  }

  void print_coords(bool trans) {
    int rank = grid->rank_in_col;
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
