#pragma once
#include "../core/common.h"
#include "../core/dense_mat.hpp"
#include "../core/sparse_mat.hpp"
#include "process_3D_grid.hpp"
#include <chrono>
#include <iostream>
#include <mpi.h>
#include <thread>
#include <unordered_map>
#include <vector>

using namespace distblas::core;

namespace distblas::net {

/**
 * This class represents the data transfer related operations across processes
 * based on internal data connectivity patterns. This implementation is tightly
 * coupled with 1D partitioning of Sparse Matrix and 1D partitioning of Dense
 * Matrix.
 */

template <typename INDEX_TYPE, typename VALUE_TYPE>
class DataComm {

public:
  SpMat<VALUE_TYPE> *sp_local_receiver;
  SpMat<VALUE_TYPE> *sp_local_sender;
  DenseMat<INDEX_TYPE, VALUE_TYPE> *dense_local;
  SpMat<VALUE_TYPE> *sparse_local;
  Process3DGrid *grid;
  vector<int> sdispls;
  vector<int> sendcounts;
  vector<int> rdispls;
  vector<int> receivecounts;
  vector<unordered_set<INDEX_TYPE>> receive_col_ids_list;
  vector<unordered_set<INDEX_TYPE>> send_col_ids_list;
  unordered_map<INDEX_TYPE, unordered_map<int, bool>> send_indices_to_proc_map;
  unordered_map<INDEX_TYPE, unordered_map<int, bool>>
      receive_indices_to_proc_map;

  int batch_id;

  int embedding_dim=sp_tuple_max_dim;

  double alpha;
  DataComm(distblas::core::SpMat<VALUE_TYPE> *sp_local_receiver,
           distblas::core::SpMat<VALUE_TYPE> *sp_local_sender,
           DenseMat<INDEX_TYPE, VALUE_TYPE> *dense_local,
           Process3DGrid *grid, int batch_id, double alpha) {
    this->sp_local_receiver = sp_local_receiver;
    this->sp_local_sender = sp_local_sender;
    this->dense_local = dense_local;
    this->grid = grid;
    this->sdispls = vector<int>(grid->world_size, 0);
    this->sendcounts = vector<int>(grid->world_size, 0);
    this->rdispls = vector<int>(grid->world_size, 0);
    this->receivecounts = vector<int>(grid->world_size, 0);
    this->send_counts_cyclic = vector<int>(grid->world_size, 0);
    this->receive_counts_cyclic = vector<int>(grid->world_size, 0);
    this->sdispls_cyclic = vector<int>(grid->world_size, 0);
    this->rdispls_cyclic = vector<int>(grid->world_size, 0);
    this->receive_col_ids_list =
        vector<unordered_set<INDEX_TYPE>>(grid->world_size);
    this->send_col_ids_list =
        vector<unordered_set<INDEX_TYPE>>(grid->world_size);
    this->batch_id = batch_id;
    this->alpha = alpha;
    this->embedding_dim=dense_local->cols;
  }

  DataComm(SpMat<VALUE_TYPE> *sp_local_receiver,
           SpMat<VALUE_TYPE> *sp_local_sender, SpMat<VALUE_TYPE> *sparse_local,
           Process3DGrid *grid, int batch_id, double alpha)
      : sp_local_receiver(sp_local_receiver), sp_local_sender(sp_local_sender),
        sparse_local(sparse_local), grid(grid), alpha(alpha),
        batch_id(batch_id) {
    this->sdispls = vector<int>(grid->world_size, 0);
    this->sendcounts = vector<int>(grid->world_size, 0);
    this->rdispls = vector<int>(grid->world_size, 0);
    this->receivecounts = vector<int>(grid->world_size, 0);
    this->send_counts_cyclic = vector<int>(grid->world_size, 0);
    this->receive_counts_cyclic = vector<int>(grid->world_size, 0);
    this->sdispls_cyclic = vector<int>(grid->world_size, 0);
    this->rdispls_cyclic = vector<int>(grid->world_size, 0);
    this->receive_col_ids_list =
        vector<unordered_set<INDEX_TYPE>>(grid->world_size);
    this->send_col_ids_list =
        vector<unordered_set<INDEX_TYPE>>(grid->world_size);
  }

  vector<int> receive_counts_cyclic;
  vector<int> rdispls_cyclic;
  vector<int> send_counts_cyclic;
  vector<int> sdispls_cyclic;

  MPI_Request request = MPI_REQUEST_NULL;

  ~DataComm() {}

  void onboard_data() {

    int total_send_count = 0;
    // processing chunks
    // calculating receiving data cols

    if (alpha == 0) {
      // This represents the case for pulling

      this->sp_local_receiver->find_col_ids(batch_id, 0, grid->col_world_size,
                                            receive_col_ids_list,
                                            receive_indices_to_proc_map, 0);
      // calculating sending data cols
      this->sp_local_sender->find_col_ids(batch_id, 0, grid->col_world_size,
                                          send_col_ids_list,
                                          send_indices_to_proc_map, 0);
    } else if (alpha == 1.0) {
      // This represents the case for pushing
      this->sp_local_receiver->find_col_ids(batch_id, 0, grid->col_world_size,
                                            receive_col_ids_list,
                                            receive_indices_to_proc_map, 1);

      // calculating sending data cols
      this->sp_local_sender->find_col_ids(batch_id, 0, grid->col_world_size,
                                          send_col_ids_list,
                                          send_indices_to_proc_map, 1);
    } else if (alpha > 0 and alpha < 1.0) {

      // This represents the case for pull and pushing
      int end_process = get_end_proc(1, alpha, grid->col_world_size);

      this->sp_local_receiver->find_col_ids(batch_id, 1, end_process,
                                            receive_col_ids_list,
                                            receive_indices_to_proc_map, 1);

      // calculating sending data cols
      this->sp_local_sender->find_col_ids(batch_id, 1, end_process,
                                          send_col_ids_list,
                                          send_indices_to_proc_map, 1);

      if (batch_id >= 0) {
        this->sp_local_receiver->find_col_ids(
            batch_id, end_process, grid->col_world_size, receive_col_ids_list,
            receive_indices_to_proc_map, 0);

        // calculating sending data cols
        this->sp_local_sender->find_col_ids(
            batch_id, end_process, grid->col_world_size, send_col_ids_list,
            send_indices_to_proc_map, 0);
      }

    } else {
      cout << "  alpha needs to be in the range of [0,1]" << endl;
    }

    for (int i = 0; i < grid->world_size; i++) {
      receivecounts[i] = receive_col_ids_list[i].size();
      sendcounts[i] = send_col_ids_list[i].size();
    }
  }

    inline void transfer_dense_data(int iteration, int batch_id,int starting_proc, int end_proc, bool temp_cache) {

        //Buffer used for send MPI operations data
        auto sendbuf_cyclic  = make_unique<std::vector<DataTuple<VALUE_TYPE>>>().get();
        auto receivebuf = make_unique<std::vector<DataTuple<VALUE_TYPE>>>().get();

        int total_receive_count = 0;
        int total_send_count = 0;

        auto data_buffer_ptr = make_unique<vector<vector<DataTuple<VALUE_TYPE>>>>();
        data_buffer_ptr->resize(grid->col_world_size);

        send_counts_cyclic = vector<int>(grid->col_world_size, 0);
        receive_counts_cyclic = vector<int>(grid->col_world_size, 0);
        sdispls_cyclic = vector<int>(grid->col_world_size, 0);
        rdispls_cyclic = vector<int>(grid->col_world_size, 0);

        vector<int> sending_procs;
        vector<int> receiving_procs;

        for (int i = starting_proc; i < end_proc; i++) {
            int sending_rank = (grid->rank_in_col + i) % grid->col_world_size;
            int receiving_rank =
                    (grid->rank_in_col >= i)
                    ? (grid->rank_in_col - i) % grid->col_world_size
                    : (grid->col_world_size - i + grid->rank_in_col) %
                      grid->col_world_size;
            sending_procs.push_back(sending_rank);
            receiving_procs.push_back(receiving_rank);
            (*data_buffer_ptr)[sending_rank] = vector<DataTuple<VALUE_TYPE>>();
        }

        for (const auto &pair : DataComm<INDEX_TYPE, VALUE_TYPE>::send_indices_to_proc_map) {
            auto col_id = pair.first;
            vector<VALUE_TYPE> stdArray(this->dense_local->cols);
            (this->dense_local)->fetch_local_data(stdArray.data(),col_id);
            for (int i = 0; i < sending_procs.size(); i++) {
                if (pair.second.count(sending_procs[i]) > 0) {
                    if (send_counts_cyclic[sending_procs[i]] == 0) {
                        DataTuple<VALUE_TYPE> current;
                        current.rows[0] =
                                2; // rows first two indices are already taken for metadata
                        current.rows[1] = 0;
                        (*data_buffer_ptr)[sending_procs[i]].push_back(current);
                        total_send_count++;
                        send_counts_cyclic[sending_procs[i]]++;
                    }

                    DataTuple<VALUE_TYPE> latest = (*data_buffer_ptr)[sending_procs[i]][send_counts_cyclic[sending_procs[i]] - 1];
                    auto row_index_offset = latest.rows[0];
                    auto col_index_offset = latest.rows[1];
                    if (row_index_offset >= row_max or col_index_offset >= sp_tuple_max_dim) {
                        DataTuple<VALUE_TYPE> current;
                        current.rows[0] =
                                2; // rows first two indices are already taken for metadata
                        current.rows[1] = 0;
                        (*data_buffer_ptr)[sending_procs[i]].push_back(current);
                        total_send_count++;
                        send_counts_cyclic[sending_procs[i]]++;
                        latest =
                                (*data_buffer_ptr)[sending_procs[i]]
                                [send_counts_cyclic[sending_procs[i]] - 1];
                        row_index_offset = latest.rows[0];
                        col_index_offset = latest.rows[1];
                    }

                    INDEX_TYPE offset = stdArray.size();
                    // start filling from offset position
                    INDEX_TYPE pending_col_pos = sp_tuple_max_dim - col_index_offset;
                    INDEX_TYPE num_of_copying_data = min(offset, pending_col_pos);
                    INDEX_TYPE remaining_data_items = offset - num_of_copying_data;

                    latest.rows[row_index_offset] = col_id;
                    latest.rows[row_index_offset + 1] = num_of_copying_data;
                    latest.rows[0] = row_index_offset + 2;
                    latest.rows[1] = latest.rows[1] + num_of_copying_data;

                    if (num_of_copying_data > 0) {
                        copy(stdArray.begin(),
                             stdArray.begin() + num_of_copying_data,
                             latest.values.begin() + col_index_offset);

                    }
                    (*data_buffer_ptr)[sending_procs[i]][send_counts_cyclic[sending_procs[i]] - 1] = latest;
                    if (remaining_data_items > 0) {
                        DataTuple<VALUE_TYPE> current;
                        current.rows[0] =
                                2; // rows first two indices are already taken for metadata
                        current.rows[1] = 0;
                        (*data_buffer_ptr)[sending_procs[i]].push_back(current);
                        total_send_count++;
                        send_counts_cyclic[sending_procs[i]]++;
                        latest =
                                (*data_buffer_ptr)[sending_procs[i]]
                                [send_counts_cyclic[sending_procs[i]] - 1];
                        row_index_offset = latest.rows[0];
                        col_index_offset = latest.rows[1];
                        latest.rows[row_index_offset] = col_id;
                        latest.rows[row_index_offset + 1] = remaining_data_items;
                        latest.rows[0] = row_index_offset + 2;
                        latest.rows[1] = latest.rows[1] + remaining_data_items;

                        copy(stdArray.begin() + num_of_copying_data - 1,
                             stdArray.begin() + num_of_copying_data - 1 +
                             remaining_data_items,
                             latest.values.begin());
                        (*data_buffer_ptr)[sending_procs[i]]
                        [send_counts_cyclic[sending_procs[i]] - 1] =
                                latest;
                    }
                }
            }
        }
        cout<<" exited from  data comm "<<grid->rank_in_col<<endl;
        (*sendbuf_cyclic).resize(total_send_count);
        for (int i = 0; i < grid->col_world_size; i++) {
            sdispls_cyclic[i] =(i > 0) ? sdispls_cyclic[i - 1] + send_counts_cyclic[i - 1]: sdispls_cyclic[i];
            copy((*data_buffer_ptr)[i].begin(), (*data_buffer_ptr)[i].end(),(*sendbuf_cyclic).begin() + sdispls_cyclic[i]);
        }

        cout<<" first MPI_Alltoall reached "<<grid->rank_in_col<<endl;
        MPI_Alltoall(send_counts_cyclic.data(), 1, MPI_INT,receive_counts_cyclic.data(), 1, MPI_INT, grid->col_world);


        for (int i = 0; i < grid->col_world_size; i++) {
            rdispls_cyclic[i] =(i > 0) ? rdispls_cyclic[i - 1] + receive_counts_cyclic[i - 1]: rdispls_cyclic[i];
            total_receive_count += receive_counts_cyclic[i];
        }

        if (total_receive_count > 0) {
            receivebuf->resize(total_receive_count);
        }

        add_perf_stats(total_receive_count, "Data transfers");

        auto t = start_clock();
        MPI_Alltoallv((*sendbuf_cyclic).data(), send_counts_cyclic.data(),
                      sdispls_cyclic.data(), DENSETUPLE, (*receivebuf).data(),
                      receive_counts_cyclic.data(), rdispls_cyclic.data(),
                      DENSETUPLE, grid->col_world);
        stop_clock_and_add(t, "Communication Time");
        cout<<" populate_dense_cache "<<grid->rank_in_col<<endl;
        this->populate_dense_cache(sendbuf_cyclic, receivebuf, iteration, batch_id,temp_cache);
    }


  inline void transfer_sparse_data(
      vector<SpTuple<VALUE_TYPE, sp_tuple_max_dim>> *sendbuf_cyclic,
      vector<SpTuple<VALUE_TYPE, sp_tuple_max_dim>> *receivebuf, int iteration,
      int batch_id, int starting_proc, int end_proc) {

    int total_receive_count = 0;
    shared_ptr<vector<vector<SpTuple<VALUE_TYPE, sp_tuple_max_dim>>>>
        data_buffer_ptr = make_shared<
            vector<vector<SpTuple<VALUE_TYPE, sp_tuple_max_dim>>>>();
    data_buffer_ptr->resize(grid->col_world_size);

    int total_send_count = 0;
    send_counts_cyclic = vector<int>(grid->col_world_size, 0);
    receive_counts_cyclic = vector<int>(grid->col_world_size, 0);
    sdispls_cyclic = vector<int>(grid->col_world_size, 0);
    rdispls_cyclic = vector<int>(grid->col_world_size, 0);

    vector<int> sending_procs;
    vector<int> receiving_procs;

    for (int i = starting_proc; i < end_proc; i++) {
      int sending_rank = (grid->rank_in_col + i) % grid->col_world_size;
      int receiving_rank =
          (grid->rank_in_col >= i)
              ? (grid->rank_in_col - i) % grid->col_world_size
              : (grid->col_world_size - i + grid->rank_in_col) %
                    grid->col_world_size;
      sending_procs.push_back(sending_rank);
      receiving_procs.push_back(receiving_rank);
      (*data_buffer_ptr)[sending_rank] =
          vector<SpTuple<VALUE_TYPE, sp_tuple_max_dim>>();
    }

    for (const auto &pair : DataComm<INDEX_TYPE, VALUE_TYPE>::send_indices_to_proc_map) {
      auto col_id = pair.first;
      CSRHandle sparse_tuple = (this->sparse_local)->fetch_local_data(col_id);
      for (int i = 0; i < sending_procs.size(); i++) {
        if (pair.second.count(sending_procs[i]) > 0) {

          if (send_counts_cyclic[sending_procs[i]] == 0) {
            SpTuple<VALUE_TYPE, sp_tuple_max_dim> current;
            current.rows[0] =
                2; // rows first two indices are already taken for metadata
            current.rows[1] = 0;
            (*data_buffer_ptr)[sending_procs[i]].push_back(current);
            total_send_count++;
            send_counts_cyclic[sending_procs[i]]++;
          }

          SpTuple<VALUE_TYPE, sp_tuple_max_dim> latest = (*data_buffer_ptr)[sending_procs[i]][send_counts_cyclic[sending_procs[i]] - 1];
          auto row_index_offset = latest.rows[0];
          auto col_index_offset = latest.rows[1];
          if (row_index_offset >= row_max or col_index_offset >= sp_tuple_max_dim) {
            SpTuple<VALUE_TYPE, sp_tuple_max_dim> current;
            current.rows[0] =
                2; // rows first two indices are already taken for metadata
            current.rows[1] = 0;
            (*data_buffer_ptr)[sending_procs[i]].push_back(current);
            total_send_count++;
            send_counts_cyclic[sending_procs[i]]++;
            latest =
                (*data_buffer_ptr)[sending_procs[i]]
                                  [send_counts_cyclic[sending_procs[i]] - 1];
            row_index_offset = latest.rows[0];
            col_index_offset = latest.rows[1];
          }

          INDEX_TYPE offset = sparse_tuple.col_idx.size();
          // start filling from offset position
          INDEX_TYPE pending_col_pos = sp_tuple_max_dim - col_index_offset;
          INDEX_TYPE num_of_copying_data = min(offset, pending_col_pos);
          INDEX_TYPE remaining_data_items = offset - num_of_copying_data;

          latest.rows[row_index_offset] = sparse_tuple.row_idx[0];
          latest.rows[row_index_offset + 1] = num_of_copying_data;
          latest.rows[0] = row_index_offset + 2;
          latest.rows[1] = latest.rows[1] + num_of_copying_data;

          if (num_of_copying_data > 0) {
            copy(sparse_tuple.col_idx.begin(),
                 sparse_tuple.col_idx.begin() + num_of_copying_data,
                 latest.cols.begin() + col_index_offset);
            copy(sparse_tuple.values.begin(),
                 sparse_tuple.values.begin() + num_of_copying_data,
                 latest.values.begin() + col_index_offset);
          }
          (*data_buffer_ptr)[sending_procs[i]][send_counts_cyclic[sending_procs[i]] - 1] = latest;
          if (remaining_data_items > 0) {
            SpTuple<VALUE_TYPE, sp_tuple_max_dim> current;
            current.rows[0] =
                2; // rows first two indices are already taken for metadata
            current.rows[1] = 0;
            (*data_buffer_ptr)[sending_procs[i]].push_back(current);
            total_send_count++;
            send_counts_cyclic[sending_procs[i]]++;
            latest =
                (*data_buffer_ptr)[sending_procs[i]]
                                  [send_counts_cyclic[sending_procs[i]] - 1];
            row_index_offset = latest.rows[0];
            col_index_offset = latest.rows[1];
            latest.rows[row_index_offset] = sparse_tuple.row_idx[0];
            latest.rows[row_index_offset + 1] = remaining_data_items;
            latest.rows[0] = row_index_offset + 2;
            latest.rows[1] = latest.rows[1] + remaining_data_items;

            copy(sparse_tuple.col_idx.begin() + num_of_copying_data - 1,
                 sparse_tuple.col_idx.begin() + num_of_copying_data - 1 +
                     remaining_data_items,
                 latest.cols.begin());
            copy(sparse_tuple.values.begin() + num_of_copying_data - 1,
                 sparse_tuple.values.begin() + num_of_copying_data - 1 +
                     remaining_data_items,
                 latest.values.begin());
            (*data_buffer_ptr)[sending_procs[i]]
                              [send_counts_cyclic[sending_procs[i]] - 1] =
                                  latest;
          }
        }
      }
    }
    MPI_Barrier(grid->col_world);

    (*sendbuf_cyclic).resize(total_send_count);
    for (int i = 0; i < grid->col_world_size; i++) {
      sdispls_cyclic[i] =
          (i > 0) ? sdispls_cyclic[i - 1] + send_counts_cyclic[i - 1]
                  : sdispls_cyclic[i];
      copy((*data_buffer_ptr)[i].begin(), (*data_buffer_ptr)[i].end(),
           (*sendbuf_cyclic).begin() + sdispls_cyclic[i]);
    }
//    auto t = start_clock();
    MPI_Alltoall(send_counts_cyclic.data(), 1, MPI_INT,
                 receive_counts_cyclic.data(), 1, MPI_INT, grid->col_world);
//    stop_clock_and_add(t, "Communication Time");

    for (int i = 0; i < grid->col_world_size; i++) {
      rdispls_cyclic[i] =
          (i > 0) ? rdispls_cyclic[i - 1] + receive_counts_cyclic[i - 1]
                  : rdispls_cyclic[i];
      total_receive_count += receive_counts_cyclic[i];
    }

    if (total_receive_count > 0) {
      receivebuf->resize(total_receive_count);
    }

    add_perf_stats(total_receive_count, "Data transfers");

   auto t = start_clock();
    MPI_Alltoallv((*sendbuf_cyclic).data(), send_counts_cyclic.data(),
                  sdispls_cyclic.data(), SPARSETUPLE, (*receivebuf).data(),
                  receive_counts_cyclic.data(), rdispls_cyclic.data(),
                  SPARSETUPLE, grid->col_world);
    stop_clock_and_add(t, "Communication Time");
    this->populate_sparse_cache(sendbuf_cyclic, receivebuf, iteration, batch_id);
  }

  void transfer_dense_data(vector<INDEX_TYPE> &col_ids, int iteration, int batch_id) {
    for (int i = 0; i < col_ids.size(); i++) {
      int owner_rank = col_ids[i] / (this->sp_local_receiver)->proc_row_width;
      if (owner_rank == grid->rank_in_col) {
          for (int j = 0; j < grid->col_world_size; j++) {
              if (j != grid->rank_in_col) {
                  DataComm<INDEX_TYPE, VALUE_TYPE>::send_indices_to_proc_map[col_ids[i]][j] = true;
              }
          }
      }
    }

    this->transfer_dense_data(iteration,batch_id,0,grid->col_world_size-1,true);
  }

  inline void populate_sparse_cache(
      vector<SpTuple<VALUE_TYPE, sp_tuple_max_dim>> *sendbuf,
      vector<SpTuple<VALUE_TYPE, sp_tuple_max_dim>> *receivebuf, int iteration,
      int batch_id, bool force_delete=false) {

    #pragma omp parallel for
    for (int i = 0; i < this->grid->col_world_size; i++) {
      INDEX_TYPE base_index = this->rdispls_cyclic[i];
      INDEX_TYPE count = this->receive_counts_cyclic[i];

      for (INDEX_TYPE j = base_index; j < base_index + count; j++) {
        SpTuple<VALUE_TYPE, sp_tuple_max_dim> sp_tuple = (*receivebuf)[j];
        auto row_offset = sp_tuple.rows[0];
        auto offset_so_far = 0;
        for (auto k = 2; k < row_offset; k = k + 2) {
          auto key = sp_tuple.rows[k];
          auto copying_count = sp_tuple.rows[k + 1];
          if (force_delete or (*(this->sparse_local)->tempCachePtr)[i].find(key) ==
              (*(this->sparse_local)->tempCachePtr)[i].end() or ((*(this->sparse_local)->tempCachePtr)[i][key].inserted_itr!=iteration
               or (*(this->sparse_local)->tempCachePtr)[i][key].inserted_batch_id)!=batch_id or ((*(this->sparse_local)->tempCachePtr)[i][key].force_delete)) {
            SparseCacheEntry<VALUE_TYPE> sp_entry;
            sp_entry.inserted_itr = iteration;
            sp_entry.inserted_batch_id = batch_id;
            sp_entry.cols = vector<INDEX_TYPE>();
            sp_entry.values = vector<VALUE_TYPE>();
            if (force_delete){
              sp_entry.force_delete =true;
            }
            (*(this->sparse_local)->tempCachePtr)[i][key] = sp_entry;
          }
          if (copying_count > 0) {
            SparseCacheEntry<VALUE_TYPE> cache_entry =
                (*(this->sparse_local)->tempCachePtr)[i][key];
            auto entry_offset = cache_entry.cols.size();
            cache_entry.cols.resize(entry_offset + copying_count);
            cache_entry.values.resize(entry_offset + copying_count);
            copy(sp_tuple.cols.begin() + offset_so_far,
                 sp_tuple.cols.begin() + offset_so_far + copying_count,
                 cache_entry.cols.begin() + entry_offset);
            copy(sp_tuple.values.begin() + offset_so_far,
                 sp_tuple.values.begin() + offset_so_far + copying_count,
                 cache_entry.values.begin() + entry_offset);
            offset_so_far += copying_count;
            (*(this->sparse_local)->tempCachePtr)[i][key] = cache_entry;
          }
        }
      }
    }
    receivebuf->clear();
    receivebuf->shrink_to_fit();
    sendbuf->clear();
    sendbuf->shrink_to_fit();
  }


    inline void populate_dense_cache(
            vector<DataTuple<VALUE_TYPE>> *sendbuf,
            vector<DataTuple<VALUE_TYPE>> *receivebuf,  int iteration,
            int batch_id, bool temp) {

        #pragma omp parallel for
        for (int i = 0; i < this->grid->col_world_size; i++) {
            auto &cache = temp?(*(this->dense_local)->tempCachePtr)[i]:(*(this->dense_local)->cachePtr)[i];

            INDEX_TYPE base_index = this->rdispls_cyclic[i];
            INDEX_TYPE count = this->receive_counts_cyclic[i];

            for (INDEX_TYPE j = base_index; j < base_index + count; j++) {
                DataTuple<VALUE_TYPE> sp_tuple = (*receivebuf)[j];
                auto row_offset = sp_tuple.rows[0];
                auto offset_so_far = 0;
                for (auto k = 2; k < row_offset; k = k + 2) {
                    auto key = sp_tuple.rows[k];
                    auto copying_count = sp_tuple.rows[k + 1];
                    if (cache.find(key) == cache.end() or (cache[key].inserted_itr!=iteration
                                                                                           or cache[key].inserted_batch_id)!=batch_id ) {
                        CacheEntry<VALUE_TYPE> sp_entry;
                        sp_entry.inserted_itr = iteration;
                        sp_entry.inserted_batch_id = batch_id;
                        cache[key] = sp_entry;
                    }
                    if (copying_count > 0) {
                        CacheEntry<VALUE_TYPE> cache_entry =cache[key];
                        auto entry_offset = cache_entry.values.size();
                        cache_entry.values.resize(entry_offset + copying_count);
                        copy(sp_tuple.values.begin() + offset_so_far,
                             sp_tuple.values.begin() + offset_so_far + copying_count,
                             cache_entry.values.begin() + entry_offset);
                        offset_so_far += copying_count;
                        cache[key] = cache_entry;
                    }
                }
            }
        }
        receivebuf->clear();
        receivebuf->shrink_to_fit();
        sendbuf->clear();
        sendbuf->shrink_to_fit();
    }

};

} // namespace distblas::net
