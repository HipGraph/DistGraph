#pragma once
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
  vector<vector<uint64_t>> receive_col_ids_list;
  vector<vector<uint64_t>> send_col_ids_list;
  DataTuple<DENT, embedding_dim> *sendbuf;
  unordered_map<uint64_t, vector<int>> send_indices_to_proc_map;
  int batch_id;

  //  DataTuple<DENT, embedding_dim> *receivebuf;

public:
  DataComm(distblas::core::SpMat<SPT> *sp_local_receiver,
           distblas::core::SpMat<SPT> *sp_local_sender,
           DenseMat<SPT, DENT, embedding_dim> *dense_local, Process3DGrid *grid,
           int batch_id) {
    this->sp_local_receiver = sp_local_receiver;
    this->sp_local_sender = sp_local_sender;
    this->dense_local = dense_local;
    this->grid = grid;
    this->sdispls = vector<int>(grid->world_size, 0);
    this->sendcounts = vector<int>(grid->world_size, 0);
    this->rdispls = vector<int>(grid->world_size, 0);
    this->receivecounts = vector<int>(grid->world_size, 0);
    this->receive_col_ids_list = vector<vector<uint64_t>>(grid->world_size);
    this->send_col_ids_list = vector<vector<uint64_t>>(grid->world_size);
    this->batch_id = batch_id;
    if (batch_id >= 0) {
      auto base_index = batch_id * sp_local_sender->batch_size;
      for (int i = 0; i < sp_local_sender->batch_size; i++) {
        send_indices_to_proc_map.emplace(base_index + i,
                                        vector<int>(grid->world_size, 0));
      }
    } else {
      for (int i = 0; i < sp_local_sender->proc_row_width; i++) {
        send_indices_to_proc_map.emplace(i, vector<int>(grid->world_size, 0));
      }
    }
  }

  ~DataComm() {}

  void onboard_data() {

    int total_send_count = 0;
    // processing chunks
    // calculating receiving data cols
    this->sp_local_receiver->fill_col_ids(batch_id, receive_col_ids_list);

    // calculating sending data cols
    this->sp_local_sender->fill_col_ids(batch_id, send_col_ids_list);
    for (int i = 0; i < grid->world_size; i++) {
      std::unordered_set<uint64_t> unique_set_receiv(
          receive_col_ids_list[i].begin(), receive_col_ids_list[i].end());
      receive_col_ids_list[i] =
          vector<uint64_t>(unique_set_receiv.begin(), unique_set_receiv.end());

      receivecounts[i] = receive_col_ids_list[i].size();

      std::unordered_set<uint64_t> unique_set_send(send_col_ids_list[i].begin(),
                                                   send_col_ids_list[i].end());
      send_col_ids_list[i] =
          vector<uint64_t>(unique_set_send.begin(), unique_set_send.end());

      sendcounts[i] = send_col_ids_list[i].size();
      total_send_count += sendcounts[i];
      sdispls[i] = (i > 0) ? sdispls[i - 1] + sendcounts[i - 1] : sdispls[i];
      rdispls[i] = (i > 0) ? rdispls[i - 1] + receivecounts[i - 1] : rdispls[i];
//      cout<<" rank "<<grid->global_rank<<" sending data   "<<sendcounts[i]<<"to rank "<<i <<" receving data "<<receivecounts[i]<< " from rank "<<i<<endl;
      for (int j = 0; j < send_col_ids_list[i].size(); j++) {
        uint64_t local_key = send_col_ids_list[i][j];
        send_indices_to_proc_map[local_key][i] = 1;
      }
    }
    if (total_send_count>0){
      sendbuf = new DataTuple<DENT, embedding_dim>[total_send_count];
    }
  }

  void transfer_data(std::vector<DataTuple<DENT, embedding_dim>> *receivebuf,
                     bool synchronous, MPI_Request &request) {
    int total_receive_count = 0;
    vector<int> offset_vector(grid->world_size, 0);
    for (const auto &pair : send_indices_to_proc_map) {
      auto col_id = pair.first;
      bool already_fetched = false;
      vector<int> proc_list = pair.second;
      std::array<DENT, embedding_dim> dense_vector;
      for (int i = 0; i < proc_list.size(); i++) {
        if (proc_list[i] == 1) {
          if (!already_fetched) {
            dense_vector = (this->dense_local)->fetch_local_data(col_id);
            already_fetched = true;
          }
          int offset = sdispls[i];
          int index = offset_vector[i] + offset;
          sendbuf[index].col =
              col_id + (this->sp_local_sender->proc_col_width *this->grid->global_rank);
          sendbuf[index].value =dense_vector;
          offset_vector[i]++;
        }
      }
    }
    cout<<" global rank "<<grid->global_rank<<" dense vector loading completed"<<endl;
    for (int i = 0; i < grid->world_size; i++) {
      total_receive_count += receivecounts[i];
    }

    receivebuf->resize(total_receive_count);

    if (synchronous) {
      MPI_Alltoallv(sendbuf, sendcounts.data(), sdispls.data(), DENSETUPLE,
                    (*receivebuf).data(), receivecounts.data(), rdispls.data(),
                    DENSETUPLE, MPI_COMM_WORLD);
      cout<<" global rank "<<grid->global_rank<<" synchronuse MPI_Alltoallv completed"<<endl;
      MPI_Request dumy;
      this->populate_cache(receivebuf, dumy, true);
      cout<<" global rank "<<grid->global_rank<<" populate_cache completed"<<endl;

    } else {
      MPI_Ialltoallv(sendbuf, sendcounts.data(), sdispls.data(), DENSETUPLE,
                     (*receivebuf).data(), receivecounts.data(), rdispls.data(),
                     DENSETUPLE, MPI_COMM_WORLD, &request);
    }
  }

  void transfer_data(vector<uint64_t> &col_ids) {

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
      receivecounts[i] = receive_col_ids_list[i].size();
    }
    sdispls[0] = 0;
    rdispls[0] = 0;
    for (int i = 0; i < grid->world_size; i++) {

      sdispls[i] = 0;
      rdispls[i] = (i > 0) ? rdispls[i - 1] + receivecounts[i - 1] : rdispls[i];

      total_send_count = total_send_count + sendcounts[i];
      total_receive_count = total_receive_count + receivecounts[i];
    }

    DataTuple<DENT, embedding_dim> *sendbuf =
        new DataTuple<DENT, embedding_dim>[total_send_count];

    vector<DataTuple<DENT, embedding_dim>> *receivebuf =
        new vector<DataTuple<DENT, embedding_dim>>(total_receive_count);


    for (int i = 0; i < grid->world_size; i++) {
      for (int j = 0; j < send_col_ids_list.size(); j++) {
        int index = sdispls[i] + j;
        ((sendbuf)[index]).col = send_col_ids_list[j];
        int local_key = ((sendbuf)[index]).col -
                        (grid->global_rank) * (this->sp_local_receiver)->proc_row_width;
        sendbuf[index].value = (this->dense_local)->fetch_local_data(local_key);
      }

    }

    MPI_Alltoallv(sendbuf, sendcounts.data(), sdispls.data(), DENSETUPLE,
                  receivebuf, receivecounts.data(), rdispls.data(),
                   DENSETUPLE, MPI_COMM_WORLD);
    MPI_Request dumy;
    cout<<" grid global "<<grid->global_rank<< " populating cache"<<endl;
    this->populate_cache(receivebuf, dumy, true);

//    delete[] sendbuf;
//    delete[] receivebuf;
  }
  void populate_cache(std::vector<DataTuple<DENT, embedding_dim>> *receivebuf,
                      MPI_Request &request, bool synchronous) {
    if (!synchronous) {
      MPI_Status status;
      MPI_Wait(&request, &status);
    }

    //    #pragma omp parallel for
    for (int i = 0; i < this->grid->world_size; i++) {
      int base_index = this->rdispls[i];
      int count = this->receivecounts[i];
      cout<<" rank "<<grid->global_rank<<" accessing base index "<<base_index<<" count "<<count<<" for rank "<<i<<endl;
      for (int j = base_index; j < base_index + count; j++) {
        cout<<" rank "<<grid->global_rank<<" accessing "<<j<<endl;
        DataTuple<DENT, embedding_dim> t = (*receivebuf)[j];
        cout<<" rank "<<grid->global_rank<<" accessing "<<j<<" success and inserting into cache"<<endl;
        (this->dense_local)->insert_cache(i, t.col, t.value);
      }
    }
  }

  void cross_validate_batch_from_metadata(int batch_id) {
    int total_nodes = this->sp_local_receiver->gCols / this->sp_local_receiver->block_col_width;
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

        int owner_rank =
            static_cast<int>(global_col_id / (this->sp_local_receiver)->proc_row_width);
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
