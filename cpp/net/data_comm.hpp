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
template <typename SPT, typename DENT> class DataComm {

private:
  distblas::core::SpMat<SPT> *sp_local;
  distblas::core::SpMat<SPT> *sp_local_trans;
  distblas::core::DenseMat *dense_local;
  Process3DGrid *grid;

public:
  DataComm(distblas::core::SpMat<SPT> *sp_local,
           distblas::core::SpMat<SPT> *sp_local_trans, DenseMat *dense_local,
           Process3DGrid *grid) {
    this->sp_local = sp_local;
    this->sp_local_trans = sp_local_trans;
    this->dense_local = dense_local;
    this->grid = grid;
  }

  void invoke(int batch_id, bool fetch_all) {

    int total_nodes = this->sp_local->gCols / this->sp_local->block_col_width;
    int total_nodes_trans =
        this->sp_local_trans->gRows / this->sp_local->block_row_width;
    int no_of_nodes_per_proc_list =
        (this->sp_local->proc_col_width / this->sp_local->block_col_width);
    int no_of_nodes_per_proc_list_trans =
        (this->sp_local_trans->proc_row_width /
         this->sp_local_trans->block_row_width);

    int no_of_lists =
        (this->sp_local->proc_row_width / this->sp_local->block_row_width);

    int no_of_lists_trans = (this->sp_local_trans->proc_col_width /
                             this->sp_local_trans->block_col_width);

    cout << " rank " << grid->global_rank << "total nodes" << total_nodes << " no_of_nodes_per_proc_list "
         << no_of_nodes_per_proc_list << " no_od_lists " << no_of_lists << endl;

    cout << " rank " << grid->global_rank <<"total nodes trans " << total_nodes_trans <<  " no_of_nodes_per_proc_list_trans "
         << no_of_nodes_per_proc_list_trans << " no_of_lists_trans "
         << no_of_lists_trans << endl;

    int *sdispls = new int[grid->world_size];
    int *sendcounts = new int[grid->world_size];
    int *rdispls = new int[grid->world_size];
    int *receivecounts = new int[grid->world_size];

    std::fill_n(sdispls, grid->world_size, 0);
    std::fill_n(rdispls, grid->world_size, 0);
    std::fill_n(sendcounts, grid->world_size, 0);
    std::fill_n(receivecounts, grid->world_size, 0);

    DataTuple<DENT> *sendbuf;
    DataTuple<DENT> *receivebuf;

    vector<vector<uint64_t>> receive_col_ids_list(grid->world_size);
    vector<vector<uint64_t>> send_col_ids_list(grid->world_size);

    int total_send_count = 0;
    int total_receive_count = 0;

    if (fetch_all and batch_id == 0) {

      // calculating receiving data cols
      for (int i = 0; i < no_of_lists; i++) {
        int working_rank = 0;

        for (int j = 0; j < total_nodes; j++) {
          if (j > 0 and j % no_of_nodes_per_proc_list == 0) {
            ++working_rank;
          }
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
          sendcounts[working_rank] = send_col_ids_list[working_rank].size();
        }
      }

      for (int i = 1; i < grid->world_size; i++) {
        sdispls[i] = sdispls[i - 1] + sendcounts[i];
        rdispls[i] = rdispls[i - 1] + receivecounts[i];

        total_send_count = total_send_count + sendcounts[i];
        total_receive_count = total_receive_count + receivecounts[i];
      }

      sendbuf = new DataTuple<DENT>[total_send_count];
      receivebuf = new DataTuple<DENT>[total_receive_count];

    } else {
    }

    cout << " rank " << grid->global_rank << " send count " << total_send_count
         << endl;
    cout << " rank " << grid->global_rank << " receive count "
         << total_receive_count << endl;

    for (int i = 0; i < grid->world_size; i++) {
      vector<uint64_t> sending_vec = send_col_ids_list[i];
      vector<uint64_t> receiving_vec = receive_col_ids_list[i];

      cout << " rank " << grid->global_rank << " sending to dst " << i
           << " count " << sendcounts[i] << endl;
      cout << " rank " << grid->global_rank << " receving  from src " << i
           << " count " << receivecounts[i] << endl;
      for (int j = 0; j < sending_vec.size(); j++) {
        int index = sdispls[i] + j;
        sendbuf[index].col = sending_vec[j];
      }

      for (int j = 0; j < receiving_vec.size(); j++) {
        int index = rdispls[i] + j;
        receivebuf[index].col = receiving_vec[j];
      }
    }
  }
};
} // namespace distblas::net
