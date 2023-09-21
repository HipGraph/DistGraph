#pragma once
#include "../core/common.h"
#include "../core/dense_mat.hpp"
#include "../core/sparse_mat.hpp"
#include "process_3D_grid.hpp"
#include <iostream>
#include <mpi.h>
#include <unordered_map>
#include <vector>

using namespace distblas::core;

namespace distblas::net {

/**
 * This class represents the data transfer related operations across processes
 * based on internal data connectivity patterns.
 */

template <typename SPT, typename DENT, size_t embedding_dim> class DataComm {

private:
  distblas::core::SpMat<SPT> *sp_local_receiver;
  distblas::core::SpMat<SPT> *sp_local_sender;
  distblas::core::DenseMat<SPT, DENT, embedding_dim> *dense_local;
  Process3DGrid *grid;
  vector<int> sdispls;
  vector<int> sendcounts;
  vector<int> rdispls;
  vector<int> receivecounts;
  vector<int> send_counts_cyclic;
  vector<int> receive_counts_cyclic;
  vector<int> sdispls_cyclic;
  vector<int> rdispls_cyclic;
  vector<vector<uint64_t>> receive_col_ids_list;
  vector<vector<uint64_t>> send_col_ids_list;
  unordered_map<uint64_t, vector<int>> send_indices_to_proc_map;

  // related to cache misses
  unique_ptr<vector<DataTuple<DENT, embedding_dim>>> sending_missing_cols_ptr;
  unique_ptr<vector<DataTuple<DENT, embedding_dim>>> receive_missing_cols_ptr;




  int batch_id;

  double alpha;

  //  DataTuple<DENT, embedding_dim> *receivebuf;

public:
  DataComm(distblas::core::SpMat<SPT> *sp_local_receiver,
           distblas::core::SpMat<SPT> *sp_local_sender,
           DenseMat<SPT, DENT, embedding_dim> *dense_local, Process3DGrid *grid,
           int batch_id, double alpha) {
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
    this->receive_col_ids_list = vector<vector<uint64_t>>(grid->world_size);
    this->send_col_ids_list = vector<vector<uint64_t>>(grid->world_size);
    this->batch_id = batch_id;
    this->alpha = alpha;
//    if (batch_id >= 0) {
//      auto base_index = batch_id * sp_local_sender->batch_size;
//      for (int i = 0; i < sp_local_sender->batch_size; i++) {
//        send_indices_to_proc_map.emplace(base_index + i,
//                                         vector<int>(grid->world_size, 0));
//      }
//    } else {
      for (int i = 0; i < sp_local_sender->proc_row_width; i++) {
        send_indices_to_proc_map.emplace(i, vector<int>(grid->world_size, 0));
      }
    }

    sending_missing_cols_ptr =
        unique_ptr<vector<DataTuple<DENT, embedding_dim>>>(
            new vector<DataTuple<DENT, embedding_dim>>());

    receive_missing_cols_ptr =
        unique_ptr<vector<DataTuple<DENT, embedding_dim>>>(
            new vector<DataTuple<DENT, embedding_dim>>());

  }

  // storing cache misses sending metadata
  std::unordered_map<int, unique_ptr<DataComm<SPT, DENT, embedding_dim>>> data_comm_cache_misses_update;

  ~DataComm() {}

  void onboard_data() {

    int total_send_count = 0;
    // processing chunks
    // calculating receiving data cols
    this->sp_local_receiver->fill_col_ids(batch_id, receive_col_ids_list,alpha);

//    // calculating sending data cols
    this->sp_local_sender->fill_col_ids(batch_id, send_col_ids_list, alpha);

    // This needs to be changed
    for (int i = 0; i < grid->world_size; i++) {

      std::unordered_set<uint64_t> unique_set_receiv(
          receive_col_ids_list[i].begin(), receive_col_ids_list[i].end());

      std::unordered_set<uint64_t> unique_set_send(send_col_ids_list[i].begin(),
                                                   send_col_ids_list[i].end());



      if (unique_set_receiv.size()>0) {
        receive_col_ids_list[i] = vector<uint64_t>(unique_set_receiv.begin(),
                                                   unique_set_receiv.end());

        receivecounts[i] = receive_col_ids_list[i].size();
      }

      if (unique_set_send.size()>0) {
        send_col_ids_list[i] =
            vector<uint64_t>(unique_set_send.begin(), unique_set_send.end());

        sendcounts[i] = send_col_ids_list[i].size();
      }

      for (int j = 0; j < send_col_ids_list[i].size(); j++) {
        uint64_t local_key = send_col_ids_list[i][j];
        send_indices_to_proc_map[local_key][i] = 1;
      }
    }
  }

  void transfer_data(std::vector<DataTuple<DENT, embedding_dim>> *receivebuf,
                     bool synchronous, MPI_Request &request, int iteration,
                     int batch_id, int starting_proc, int end_proc, bool temp_cache) {
    int total_receive_count = 0;
    vector<int> offset_vector(grid->world_size, 0);

      int total_send_count = 0;
      send_counts_cyclic = vector<int>(grid->world_size, 0);
      receive_counts_cyclic = vector<int>(grid->world_size, 0);
      sdispls_cyclic = vector<int>(grid->world_size, 0);
      rdispls_cyclic = vector<int>(grid->world_size, 0);

      vector<int> sending_procs;
      vector<int> receiving_procs;

      for (int i = starting_proc; i < end_proc; i++) {
        int sending_rank = (grid->global_rank + i) % grid->world_size;
        int receiving_rank =
            (grid->global_rank >= i)
                ? (grid->global_rank - i) % grid->world_size
                : (grid->world_size - i + grid->global_rank) % grid->world_size;
        sending_procs.push_back(sending_rank);
        receiving_procs.push_back(receiving_rank);
      }

      for (int i = 0; i < sending_procs.size(); i++) {
        send_counts_cyclic[sending_procs[i]] = sendcounts[sending_procs[i]];
        receive_counts_cyclic[receiving_procs[i]] =
            receivecounts[receiving_procs[i]];
        total_send_count += send_counts_cyclic[sending_procs[i]];
        total_receive_count += receive_counts_cyclic[receiving_procs[i]];
      }

      for (int i = 0; i < grid->world_size; i++) {
        sdispls_cyclic[i] =
            (i > 0) ? sdispls_cyclic[i - 1] + send_counts_cyclic[i - 1]
                    : sdispls_cyclic[i];
        rdispls_cyclic[i] =
            (i > 0) ? rdispls_cyclic[i - 1] + receive_counts_cyclic[i - 1]
                    : rdispls_cyclic[i];
      }
      unique_ptr<std::vector<DataTuple<DENT, embedding_dim>>> sendbuf_cyclic =
          unique_ptr<std::vector<DataTuple<DENT, embedding_dim>>>(
              new vector<DataTuple<DENT, embedding_dim>>());
      if (total_send_count > 0) {
        sendbuf_cyclic->resize(total_send_count);
        for (const auto &pair : send_indices_to_proc_map) {
          auto col_id = pair.first;
          bool already_fetched = false;
          vector<int> proc_list = pair.second;
          std::array<DENT, embedding_dim> dense_vector;
          for (int i = 0; i < sending_procs.size(); i++) {
            if (proc_list[sending_procs[i]] == 1) {
              if (!already_fetched) {
                dense_vector = (this->dense_local)->fetch_local_data(col_id);
                already_fetched = true;
              }
              int offset = sdispls_cyclic[sending_procs[i]];
              int index = offset_vector[sending_procs[i]] + offset;
              (*sendbuf_cyclic)[index].col =
                  col_id + (this->sp_local_sender->proc_col_width *
                            this->grid->global_rank);
              (*sendbuf_cyclic)[index].value = dense_vector;
              offset_vector[sending_procs[i]]++;
            }
          }
        }
      }
      receivebuf->resize(total_receive_count);

      add_datatransfers(total_receive_count, "Data transfers");

      if (synchronous) {
        MPI_Alltoallv((*sendbuf_cyclic).data(), send_counts_cyclic.data(),
                      sdispls_cyclic.data(), DENSETUPLE, (*receivebuf).data(),
                      receive_counts_cyclic.data(), rdispls_cyclic.data(),
                      DENSETUPLE, MPI_COMM_WORLD);
        MPI_Request dumy;
        this->populate_cache(receivebuf, dumy, true, iteration, batch_id, temp_cache);
      } else {
        MPI_Ialltoallv((*sendbuf_cyclic).data(), send_counts_cyclic.data(),
                       sdispls_cyclic.data(), DENSETUPLE, (*receivebuf).data(),
                       receive_counts_cyclic.data(), rdispls_cyclic.data(),
                       DENSETUPLE, MPI_COMM_WORLD, &request);
      }
      sendbuf_cyclic->clear();
      sendbuf_cyclic->shrink_to_fit();
//    }
  }

  void transfer_data(vector<uint64_t> &col_ids, int iteration, int batch_id) {

    vector<vector<uint64_t>> receive_col_ids_list(grid->world_size);
    vector<uint64_t> send_col_ids_list;

    int total_send_count = 0;
    int total_receive_count = 0;

    for (int i = 0; i < col_ids.size(); i++) {
      int owner_rank = col_ids[i] / (this->sp_local_receiver)->proc_row_width;
      if (owner_rank == grid->global_rank) {
        send_col_ids_list.push_back(col_ids[i]);
      } else {
        receive_col_ids_list[owner_rank].push_back(col_ids[i]);
      }
    }

    for (int i = 0; i < grid->world_size; i++) {
      int send_size = send_col_ids_list.size();
      if (i != grid->global_rank) {
        sendcounts[i] = send_size;
      } else {
        sendcounts[i] = 0;
      }
      receive_counts_cyclic[i] = receive_col_ids_list[i].size();
    }
    sdispls[0] = 0;
    rdispls_cyclic[0] = 0;
    for (int i = 0; i < grid->world_size; i++) {

      sdispls[i] = 0;
      rdispls_cyclic[i] =
          (i > 0) ? rdispls_cyclic[i - 1] + receive_counts_cyclic[i - 1]
                  : rdispls_cyclic[i];

      total_send_count = total_send_count + sendcounts[i];
      total_receive_count = total_receive_count + receive_counts_cyclic[i];
    }

    unique_ptr<std::vector<DataTuple<DENT, embedding_dim>>> sendbuf =
        unique_ptr<std::vector<DataTuple<DENT, embedding_dim>>>(
            new vector<DataTuple<DENT, embedding_dim>>());

    sendbuf->resize(total_send_count);

    unique_ptr<std::vector<DataTuple<DENT, embedding_dim>>> receivebuf_ptr =
        unique_ptr<std::vector<DataTuple<DENT, embedding_dim>>>(
            new vector<DataTuple<DENT, embedding_dim>>());

    receivebuf_ptr.get()->resize(total_receive_count);

    for (int j = 0; j < send_col_ids_list.size(); j++) {
      int local_key =
          send_col_ids_list[j] -
          (grid->global_rank) * (this->sp_local_receiver)->proc_row_width;
      std::array<DENT, embedding_dim> val_arr =
          (this->dense_local)->fetch_local_data(local_key);
      for (int i = 0; i < grid->world_size; i++) {
        int index = sdispls[i] + j;
        (*sendbuf)[index].col = send_col_ids_list[j];
        (*sendbuf)[index].value = val_arr;
      }
    }

    MPI_Alltoallv((*sendbuf).data(), sendcounts.data(), sdispls.data(),
                  DENSETUPLE, (*receivebuf_ptr.get()).data(),
                  receive_counts_cyclic.data(), rdispls_cyclic.data(),
                  DENSETUPLE, MPI_COMM_WORLD);
    MPI_Request dumy;
    this->populate_cache(receivebuf_ptr.get(), dumy, true, iteration, batch_id,
                         true); // we should not do this
    sendbuf->clear();
    sendbuf->shrink_to_fit();
    //    delete[] sendbuf;
  }

  void transfer_data(vector<vector<uint64_t>> *cache_misses, int iteration, int batch_id, int starting_proc, int end_proc) {

    if (iteration == 0) {
      int total_send_count = 0;
      int total_receive_count = 0;

      vector<int> sending_procs;

      for (int i = starting_proc; i < end_proc; i++) {
        int sending_rank = (grid->global_rank + i) % grid->world_size;
        int receiving_rank =
            (grid->global_rank >= i)
                ? (grid->global_rank - i) % grid->world_size
                : (grid->world_size - i + grid->global_rank) % grid->world_size;
        sending_procs.push_back(sending_rank);
      }

      std::sort((sending_procs).begin(), (sending_procs).end());

      for (int i = 0; i < sending_procs.size(); i++) {
        receive_counts_cyclic[sending_procs[i]] =
            (*cache_misses)[sending_procs[i]].size();
        total_send_count += receive_counts_cyclic[sending_procs[i]];
      }

      receive_missing_cols_ptr->resize(total_send_count);

      for (int i = 0; i < grid->world_size; i++) {
        rdispls_cyclic[i] =
            (i > 0) ? rdispls_cyclic[i - 1] + receive_counts_cyclic[i - 1]
                    : rdispls_cyclic[i];
        int base_index = rdispls_cyclic[i];
        for (int k = 0; k < receive_counts_cyclic[i]; k++) {
          int index = base_index + k;
          DataTuple<DENT, embedding_dim> temp;
          temp.col = static_cast<uint64_t>((*cache_misses)[i][k]);
          (*receive_missing_cols_ptr)[index] = temp;
        }
      }

      // sending number of misses for each rank
      MPI_Alltoall(receive_counts_cyclic.data(), 1, MPI_INT,
                   send_counts_cyclic.data(), 1, MPI_INT, MPI_COMM_WORLD);

      for (int i = 0; i < grid->world_size; i++) {
        total_receive_count += send_counts_cyclic[i];
        sdispls_cyclic[i] =
            (i > 0) ? sdispls_cyclic[i - 1] + send_counts_cyclic[i - 1]
                    : sdispls_cyclic[i];
      }

      sending_missing_cols_ptr->resize(total_receive_count);

      // sending actual Ids
      MPI_Alltoallv((*receive_missing_cols_ptr).data(),
                    receive_counts_cyclic.data(), rdispls_cyclic.data(),
                    DENSETUPLE, (*sending_missing_cols_ptr).data(),
                    send_counts_cyclic.data(), sdispls_cyclic.data(),
                    DENSETUPLE, MPI_COMM_WORLD);

      add_datatransfers(total_send_count, "Data transfers");

      for (int i = 0; i < grid->world_size; i++) {
        int base_index = sdispls_cyclic[i];
        //      #pragma omp parallel for
        for (int j = 0; j < send_counts_cyclic[i]; j++) {
          DataTuple<DENT, embedding_dim> t =
              (*sending_missing_cols_ptr)[base_index + j];
          uint64_t global_id = t.col;
          uint64_t local_id =
              t.col -
              grid->global_rank * this->sp_local_receiver->proc_row_width;
          std::array<DENT, embedding_dim> val_arr =
              (this->dense_local)->fetch_local_data(local_id);
          t.value = val_arr;
          (*sending_missing_cols_ptr)[base_index + j] = t;
        }
      }

      MPI_Alltoallv((*sending_missing_cols_ptr).data(),
                    send_counts_cyclic.data(), sdispls_cyclic.data(),
                    DENSETUPLE, (*receive_missing_cols_ptr).data(),
                    receive_counts_cyclic.data(), rdispls_cyclic.data(),
                    DENSETUPLE, MPI_COMM_WORLD);

      for (int i = 0; i < this->grid->world_size; i++) {
        int base_index = rdispls_cyclic[i];
        int count = receive_counts_cyclic[i];
        for (int j = base_index; j < base_index + count; j++) {
          DataTuple<DENT, embedding_dim> t = (*receive_missing_cols_ptr)[j];
          (this->dense_local)
              ->insert_cache(i, t.col, batch_id, iteration, t.value, true);
        }
      }
      //      sending_missing_cols_ptr->clear();
      //      sending_missing_cols_ptr->shrink_to_fit();
      receive_missing_cols_ptr->clear();
      receive_missing_cols_ptr->shrink_to_fit();
    } else {

      int total_receive_count = 0;

      for (int i = 0; i < grid->world_size; i++) {
        int base_index = sdispls_cyclic[i];
        for (int j = 0; j < send_counts_cyclic[i]; j++) {
          DataTuple<DENT, embedding_dim> t =
              (*sending_missing_cols_ptr)[base_index + j];
          uint64_t global_id = t.col;
          uint64_t local_id =
              t.col - grid->global_rank * this->sp_local_receiver->proc_row_width;
          std::array<DENT, embedding_dim> val_arr = (this->dense_local)->fetch_local_data(local_id);
          t.value = val_arr;
          (*sending_missing_cols_ptr)[base_index + j] = t;
        }
        total_receive_count += receive_counts_cyclic[i];
      }

      receive_missing_cols_ptr->resize(total_receive_count);

      MPI_Alltoallv((*sending_missing_cols_ptr).data(),
                    send_counts_cyclic.data(), sdispls_cyclic.data(),
                    DENSETUPLE, (*receive_missing_cols_ptr).data(),
                    receive_counts_cyclic.data(), rdispls_cyclic.data(),
                    DENSETUPLE, MPI_COMM_WORLD);
      for (int i = 0; i < this->grid->world_size; i++) {
        int base_index = rdispls_cyclic[i];
        int count = receive_counts_cyclic[i];
        for (int j = base_index; j < base_index + count; j++) {
          DataTuple<DENT, embedding_dim> t = (*receive_missing_cols_ptr)[j];
          (this->dense_local)
              ->insert_cache(i, t.col, batch_id, iteration, t.value, true);
        }
      }
      receive_missing_cols_ptr->clear();
      receive_missing_cols_ptr->shrink_to_fit();
    }
  }

  void populate_cache(std::vector<DataTuple<DENT, embedding_dim>> *receivebuf,
                      MPI_Request &request, bool synchronous, int iteration,
                      int batch_id, bool temp) {
    if (!synchronous) {
      MPI_Status status;
      MPI_Wait(&request, &status);
    }

    for (int i = 0; i < this->grid->world_size; i++) {
      int base_index =  this->rdispls_cyclic[i];
      int count = this->receive_counts_cyclic[i];

      for (int j = base_index; j < base_index + count; j++) {
        DataTuple<DENT, embedding_dim> t = (*receivebuf)[j];
        (this->dense_local)->insert_cache(i, t.col, batch_id, iteration, t.value, temp);
      }
    }
    receivebuf->clear();
    receivebuf->shrink_to_fit();
  }

  void cross_validate_batch_from_metadata(int batch_id) {
    int total_nodes = this->sp_local_receiver->gCols /
                      this->sp_local_receiver->block_col_width;
    for (int i = 0; i < total_nodes; i++) {
      vector<uint64_t> col_ids;
      this->sp_local_receiver->fill_col_ids(batch_id, i, col_ids, false, true);
      for (int j = 0; j < col_ids.size(); j++) {
        uint64_t global_col_id = col_ids[j];
        uint64_t local_col_id =
            global_col_id -
            static_cast<uint64_t>(
                ((this->grid)->global_rank * (this->sp_local)->proc_row_width));
        bool fetch_from_cache = false;

        int owner_rank = static_cast<int>(
            global_col_id / (this->sp_local_receiver)->proc_row_width);
        if (owner_rank != (this->grid)->global_rank) {
          fetch_from_cache = true;
        }
        if (fetch_from_cache) {
          if (!(this->dense_local)->searchForKey(global_col_id)) {
            cout << " Assert not found my_rank " << grid->global_rank
                 << "  target_rank " << owner_rank << " id " << global_col_id
                 << "batch Id" << batch_id << endl;
          }
        }
      }
    }
  }
};
} // namespace distblas::net
