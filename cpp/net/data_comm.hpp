#pragma once
#include "../core/dense_mat.hpp"
#include "../core/sparse_mat.hpp"
#include "process_3D_grid.hpp"
#include <iostream>
#include <mpi.h>
#include <vector>

using namespace distblas::core;

namespace distblas::net {

/**
 * This class represents the data transfer related operations across processes
 * based on internal data connectivity patterns.
 */
template <typename SPT, typename DENT, size_t embedding_dim> class DataComm {

private:
  distblas::core::SpMat<SPT> *sp_local;
  distblas::core::SpMat<SPT> *sp_local_trans;
  distblas::core::DenseMat<DENT, embedding_dim> *dense_local;
  Process3DGrid *grid;
  vector<int> sdispls;
  vector<int> sendcounts;
  vector<int> rdispls;
  vector<int> receivecounts;
  DataTuple<DENT, embedding_dim> *receivebuf;

public:
  DataComm(distblas::core::SpMat<SPT> *sp_local,
           distblas::core::SpMat<SPT> *sp_local_trans,
           DenseMat<DENT, embedding_dim> *dense_local, Process3DGrid *grid) {
    this->sp_local = sp_local;
    this->sp_local_trans = sp_local_trans;
    this->dense_local = dense_local;
    this->grid = grid;
    this->sdispls = vector<int>(grid->world_size, 0);
    this->sendcounts = vector<int>(grid->world_size, 0);
    this->rdispls = vector<int>(grid->world_size, 0);
    this->receivecounts = vector<int>(grid->world_size, 0);
  }

  ~DataComm() {
    if (receivebuf != nullptr) {
      delete[] receivebuf;
    }
    cout << "successfully executed" << endl;
  }

  MPI_Request &async_transfer(int batch_id, bool fetch_all, bool verify) {

    int total_nodes = this->sp_local->gCols / this->sp_local->block_col_width;
    int total_nodes_trans =
        this->sp_local_trans->gRows / this->sp_local_trans->block_row_width;
    int no_of_nodes_per_proc_list =
        (this->sp_local->proc_col_width / this->sp_local->block_col_width);
    int no_of_nodes_per_proc_list_trans =
        (this->sp_local_trans->proc_row_width /
         this->sp_local_trans->block_row_width);

    int no_of_lists =
        (this->sp_local->proc_row_width / this->sp_local->block_row_width);

    int no_of_lists_trans = (this->sp_local_trans->proc_col_width /
                             this->sp_local_trans->block_col_width);

    vector<vector<uint64_t>> receive_col_ids_list(grid->world_size);
    vector<vector<uint64_t>> send_col_ids_list(grid->world_size);

    int total_send_count = 0;
    int total_receive_count = 0;

    // processing initial communication
    if (fetch_all and batch_id == 0) {

      // calculating receiving data cols
      for (int i = 0; i < no_of_lists; i++) {
        int working_rank = 0;

        for (int j = 0; j < total_nodes; j++) {
          if (j > 0 and j % no_of_nodes_per_proc_list == 0) {
            ++working_rank;
          }
          if (working_rank != grid->global_rank) {
            vector<uint64_t> col_ids;
            this->sp_local->fill_col_ids(i, j, col_ids, false, true);
            receive_col_ids_list[working_rank].insert(
                receive_col_ids_list[working_rank].end(), col_ids.begin(),
                col_ids.end());
            std::unordered_set<MKL_INT> unique_set(
                receive_col_ids_list[working_rank].begin(),
                receive_col_ids_list[working_rank].end());
            receive_col_ids_list[working_rank] =
                vector<uint64_t>(unique_set.begin(), unique_set.end());
          }
          receivecounts[working_rank] =
              receive_col_ids_list[working_rank].size();
        }
      }

      // calculating sending data cols

      for (int i = 0; i < no_of_lists_trans; i++) {
        int working_rank = 0;

        for (int j = 0; j < total_nodes_trans; j++) {
          if (j > 0 and j % no_of_nodes_per_proc_list_trans == 0) {
            ++working_rank;
          }
          if (working_rank != grid->global_rank) {
            vector<uint64_t> col_ids;
            this->sp_local_trans->fill_col_ids(j, i, col_ids, true, true);
            send_col_ids_list[working_rank].insert(
                send_col_ids_list[working_rank].end(), col_ids.begin(),
                col_ids.end());
            std::unordered_set<MKL_INT> unique_set(
                send_col_ids_list[working_rank].begin(),
                send_col_ids_list[working_rank].end());
            send_col_ids_list[working_rank] =
                vector<uint64_t>(unique_set.begin(), unique_set.end());
            //
          }
          sendcounts[working_rank] = send_col_ids_list[working_rank].size();
        }
      }

      for (int i = 0; i < grid->world_size; i++) {

        sdispls[i] = (i > 0) ? sdispls[i - 1] + sendcounts[i - 1] : sdispls[i];
        rdispls[i] =
            (i > 0) ? rdispls[i - 1] + receivecounts[i - 1] : rdispls[i];

        total_send_count = total_send_count + sendcounts[i];
        total_receive_count = total_receive_count + receivecounts[i];
      }

    } else {
      // processing chunks
      // calculating receiving data cols

      int offset = batch_id;
      for (int i = 0; i < no_of_lists; i++) {
        int working_rank = 0;
        for (int j = 0; j < total_nodes; j++) {
          if (j > 0 and j % no_of_nodes_per_proc_list == 0) {
            ++working_rank;
          }

          if (j == working_rank * no_of_nodes_per_proc_list + offset) {
            if (working_rank != grid->global_rank) {
              vector<uint64_t> col_ids;
              this->sp_local->fill_col_ids(i, j, col_ids, false, true);

              receive_col_ids_list[working_rank].insert(
                  receive_col_ids_list[working_rank].end(), col_ids.begin(),
                  col_ids.end());
              std::unordered_set<MKL_INT> unique_set(
                  receive_col_ids_list[working_rank].begin(),
                  receive_col_ids_list[working_rank].end());
              receive_col_ids_list[working_rank] =
                  vector<uint64_t>(unique_set.begin(), unique_set.end());
            }

            receivecounts[working_rank] =
                receive_col_ids_list[working_rank].size();
          }
        }
      }

      // calculating sending data cols
      int working_rank = 0;
      for (int j = 0; j < total_nodes_trans; j++) {
        if (j > 0 and j % no_of_nodes_per_proc_list_trans == 0) {
          ++working_rank;
        }
        if (working_rank != grid->global_rank) {
          vector<uint64_t> col_ids;
          this->sp_local_trans->fill_col_ids(j, batch_id, col_ids, true, true);

          send_col_ids_list[working_rank].insert(
              send_col_ids_list[working_rank].end(), col_ids.begin(),
              col_ids.end());
          std::unordered_set<MKL_INT> unique_set(
              send_col_ids_list[working_rank].begin(),
              send_col_ids_list[working_rank].end());
          send_col_ids_list[working_rank] =
              vector<uint64_t>(unique_set.begin(), unique_set.end());
        }
        sendcounts[working_rank] = send_col_ids_list[working_rank].size();
      }

      for (int i = 0; i < grid->world_size; i++) {

        sdispls[i] = (i > 0) ? sdispls[i - 1] + sendcounts[i - 1] : sdispls[i];
        rdispls[i] =
            (i > 0) ? rdispls[i - 1] + receivecounts[i - 1] : rdispls[i];

        total_send_count = total_send_count + sendcounts[i];
        total_receive_count = total_receive_count + receivecounts[i];
      }
    }

    DataTuple<DENT, embedding_dim> *sendbuf =
        new DataTuple<DENT, embedding_dim>[total_send_count];
    receivebuf = new DataTuple<DENT, embedding_dim>[total_receive_count];
    DataTuple<DENT, embedding_dim> *receivebufverify;
    if (verify) {
      receivebufverify =
          new DataTuple<DENT, embedding_dim>[total_receive_count];
    }

    for (int i = 0; i < grid->world_size; i++) {
      vector<uint64_t> sending_vec = send_col_ids_list[i];
      vector<uint64_t> receiving_vec = receive_col_ids_list[i];

#pragma omp parallel
      for (int j = 0; j < sending_vec.size(); j++) {
        int index = sdispls[i] + j;
        sendbuf[index].col = sending_vec[j];
        int local_key = sendbuf[index].col -
                        (grid->global_rank) * (this->sp_local)->proc_row_width;
        sendbuf[index].value = (this->dense_local)->fetch_local_data(local_key);
      }

      if (verify) {
        for (int j = 0; j < receiving_vec.size(); j++) {
          int index = rdispls[i] + j;
          receivebufverify[index].col = receiving_vec[j];
        }
      }
    }

    MPI_Request request;
    MPI_Ialltoallv(sendbuf, sendcounts.data(), sdispls.data(), DENSETUPLE,
                   receivebuf, receivecounts.data(), rdispls.data(), DENSETUPLE,
                   MPI_COMM_WORLD, &request);

    if (verify) {
      MPI_Status status;
      MPI_Wait(&request, &status);

      for (int i = 0; i < grid->world_size; i++) {
        int base_index = rdispls[i];
        int base_index_send = sdispls[i];
        int size = receivecounts[i];
        int size_send = sendcounts[i];
        for (int k = 0; k < size; k++) {
          int index = rdispls[i] + k;
          bool matched = false;
          for (int m = rdispls[i]; m < rdispls[i] + receivecounts[i]; m++) {
            if (receivebufverify[m].col == receivebuf[index].col) {
              matched = true;
            }
          }
          if (!matched) {
            cout << " rank " << grid->global_rank << "cannot verify value "
                 << receivebuf[index].col << endl;
          }
        }
      }
      delete[] receivebufverify;
    }
    delete[] sendbuf;
    return request;
  }

  void populate_cache(MPI_Request &request) {
    MPI_Status status;
    MPI_Wait(&request, &status);
    if (status.MPI_ERROR == MPI_SUCCESS) {

      // TODO parallaize
      for (int i = 0; i < this->grid->world_size; i++) {
        int base_index = this->rdispls[i];

        int count = this->receivecounts[i];
        cout<<" rank "<<grid->global_rank<<" baseindex "<<base_index<<" working rank "
             <<i<<" count "<<count<<endl;
        for (int j = base_index; j < base_index + count; j++) {
          DataTuple<DENT, embedding_dim> t = receivebuf[j];
          (this->dense_local)->insert_cache(i, t.col, t.value);
        }
      }
    }
  }
};
} // namespace distblas::net
