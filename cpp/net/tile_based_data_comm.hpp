#pragma once
#include "../core/common.h"
#include "../core/sparse_mat_tile.hpp"
#include "data_comm.hpp"
#include <math.h>
#include <memory>
#include "../core/dense_mat.hpp"

using namespace distblas::core;
using namespace std;

namespace distblas::net {

template <typename INDEX_TYPE, typename VALUE_TYPE, size_t embedding_dim>
class TileDataComm : public DataComm<INDEX_TYPE, VALUE_TYPE, embedding_dim> {

private:
  shared_ptr<
      vector<vector<unordered_map<INDEX_TYPE, unordered_map<int, bool>>>>>
      send_indices_proc_map;
  shared_ptr<
      vector<vector<unordered_map<INDEX_TYPE, unordered_map<int, bool>>>>>
      receive_indices_proc_map;

  DistributedMat* state_holder;

  int total_batches;

  double tile_width_fraction;
  int tiles_per_process_row;

  bool hash_spgemm = true;

  bool embedding = false;

  double merge_cost_factor = 1.0;

public:
  shared_ptr<vector<vector<vector<SparseTile<INDEX_TYPE, VALUE_TYPE>>>>>
      receiver_proc_tile_map;
  shared_ptr<vector<vector<vector<SparseTile<INDEX_TYPE, VALUE_TYPE>>>>>
      sender_proc_tile_map;

  TileDataComm(distblas::core::SpMat<VALUE_TYPE> *sp_local_receiver,
               distblas::core::SpMat<VALUE_TYPE> *sp_local_sender,
               distblas::core::SpMat<VALUE_TYPE> *sparse_local,
               Process3DGrid *grid, double alpha, int total_batches,
               double tile_width_fraction, bool hash_spgemm = true,
               bool embedding = false, double merge_cost_factor = 1.0,
               DistributedMat* state_holder=nullptr)
      : DataComm<INDEX_TYPE, VALUE_TYPE, embedding_dim>(
            sp_local_receiver, sp_local_sender, sparse_local, grid, -1, alpha) {
    tiles_per_process_row = static_cast<int>(1 / (tile_width_fraction));
    this->total_batches = total_batches;
    this->tile_width_fraction = tile_width_fraction;
    this->hash_spgemm = hash_spgemm;
    this->embedding = embedding;
    this->merge_cost_factor = merge_cost_factor;
    this->state_holder= state_holder;
    SparseTile<INDEX_TYPE, VALUE_TYPE>::tile_width_fraction =
        tile_width_fraction;
    receiver_proc_tile_map =
        make_shared<vector<vector<vector<SparseTile<INDEX_TYPE, VALUE_TYPE>>>>>(
            total_batches,
            vector<vector<SparseTile<INDEX_TYPE, VALUE_TYPE>>>(
                grid->col_world_size,
                vector<SparseTile<INDEX_TYPE, VALUE_TYPE>>(
                    tiles_per_process_row,
                    SparseTile<INDEX_TYPE, VALUE_TYPE>(grid, false))));
    sender_proc_tile_map =
        make_shared<vector<vector<vector<SparseTile<INDEX_TYPE, VALUE_TYPE>>>>>(
            total_batches,
            vector<vector<SparseTile<INDEX_TYPE, VALUE_TYPE>>>(
                grid->col_world_size,
                vector<SparseTile<INDEX_TYPE, VALUE_TYPE>>(
                    tiles_per_process_row,
                    SparseTile<INDEX_TYPE, VALUE_TYPE>(grid, hash_spgemm))));

    auto tiles_per_process =
        SparseTile<INDEX_TYPE, VALUE_TYPE>::get_tiles_per_process_row();
    send_indices_proc_map = make_shared<
        vector<vector<unordered_map<INDEX_TYPE, unordered_map<int, bool>>>>>(
        total_batches,
        vector<unordered_map<INDEX_TYPE, unordered_map<int, bool>>>(
            tiles_per_process));
    receive_indices_proc_map = make_shared<
        vector<vector<unordered_map<INDEX_TYPE, unordered_map<int, bool>>>>>(
        total_batches,
        vector<unordered_map<INDEX_TYPE, unordered_map<int, bool>>>(
            tiles_per_process));

    auto total_tiles =
        total_batches * this->grid->col_world_size * tiles_per_process;

    add_perf_stats(total_tiles, "Total Tiles");

    if (alpha == 0) {
#pragma omp parallel for
      for (int i = 0; i < total_batches; i++) {
        INDEX_TYPE row_starting_index_receiver =
            i * sp_local_receiver->batch_size;
        auto row_end_index_receiver =
            std::min(std::min(((i + 1) * sp_local_receiver->batch_size),
                              sp_local_receiver->proc_row_width),
                     sp_local_receiver->gRows);

        for (int j = 0; j < grid->col_world_size; j++) {
          INDEX_TYPE row_starting_index_sender =
              i * sp_local_receiver->batch_size +
              sp_local_receiver->proc_row_width * j;
          INDEX_TYPE row_end_index_sender = std::min(
              std::min(
                  (row_starting_index_sender + sp_local_receiver->batch_size),
                  static_cast<INDEX_TYPE>((j + 1) *
                                          sp_local_receiver->proc_row_width)),
              sp_local_receiver->gRows);
          for (int k = 0; k < tiles_per_process_row; k++) {
            auto tile_width =
                SparseTile<INDEX_TYPE, VALUE_TYPE>::get_tile_width(
                    sp_local_receiver->proc_col_width);
            INDEX_TYPE col_starting_index_receiver =
                k * tile_width + sp_local_receiver->proc_col_width * j;

            INDEX_TYPE col_end_index_receiver = std::min(
                std::min((col_starting_index_receiver + tile_width),
                         static_cast<INDEX_TYPE>(
                             (j + 1) * sp_local_receiver->proc_col_width)),
                sp_local_receiver->gCols);

            INDEX_TYPE col_starting_index_sender = k * tile_width;
            auto col_end_index_sender =
                std::min((col_starting_index_sender + tile_width),
                         sp_local_receiver->proc_col_width);

            (*receiver_proc_tile_map)[i][j][k].id = k;
            (*receiver_proc_tile_map)[i][j][k].row_starting_index =
                row_starting_index_receiver;
            (*receiver_proc_tile_map)[i][j][k].row_end_index =
                row_end_index_receiver;
            (*receiver_proc_tile_map)[i][j][k].col_start_index =
                col_starting_index_receiver;
            (*receiver_proc_tile_map)[i][j][k].col_end_index =
                col_end_index_receiver;
            (*receiver_proc_tile_map)[i][j][k].dimension = embedding_dim;

            (*sender_proc_tile_map)[i][j][k].id = k;
            (*sender_proc_tile_map)[i][j][k].row_starting_index =
                row_starting_index_sender;
            (*sender_proc_tile_map)[i][j][k].row_end_index =
                row_end_index_sender;
            (*sender_proc_tile_map)[i][j][k].col_start_index =
                col_starting_index_sender;
            (*sender_proc_tile_map)[i][j][k].col_end_index =
                col_end_index_sender;
            (*sender_proc_tile_map)[i][j][k].dimension = embedding_dim;
          }
        }
      }
    }
  }

  ~TileDataComm() {}

  void onboard_data(bool enable_remote_compute=true) {
    if (this->alpha == 0) {
      for (int i = 0; i < total_batches; i++) {
        this->sp_local_receiver->find_col_ids_with_tiling(
            i, 0, this->grid->col_world_size, receiver_proc_tile_map.get(),
            receive_indices_proc_map.get(), 0);
        // calculating sending data cols
        this->sp_local_sender->find_col_ids_with_tiling(
            i, 0, this->grid->col_world_size, sender_proc_tile_map.get(),
            send_indices_proc_map.get(), 0, "+", this->sparse_local);
      }
      // This represents the case for pulling
      this->sparse_local->get_transferrable_datacount(
          sender_proc_tile_map.get(), total_batches, true, false);

      if (embedding) {
        this->sparse_local->get_transferrable_datacount(
            receiver_proc_tile_map.get(), total_batches, false, false);
      }

      int tiles_per_process = SparseTile<INDEX_TYPE, VALUE_TYPE>::get_tiles_per_process_row();
      auto itr = total_batches * this->grid->col_world_size * tiles_per_process;

      auto per_process_messages = total_batches * tiles_per_process;

      unique_ptr<vector<TileTuple<INDEX_TYPE>>> send_tile_meta = make_unique<vector<TileTuple<INDEX_TYPE>>>(itr);

      unique_ptr<vector<TileTuple<INDEX_TYPE>>> receive_tile_meta = make_unique<vector<TileTuple<INDEX_TYPE>>>(itr);

      #pragma omp parallel for
      for (auto in = 0; in < itr; in++) {
        auto i = in / (this->grid->col_world_size * tiles_per_process);
        auto j = (in / tiles_per_process) % this->grid->col_world_size;
        auto k = in % tiles_per_process;

        auto offset = j * per_process_messages;
        auto index = offset + i * tiles_per_process + k;
        TileTuple<INDEX_TYPE> t;
        t.batch_id = i;
        t.tile_id = k;
        t.count =
            (*sender_proc_tile_map)[i][j][k].total_transferrable_datacount; // individual data vectors transffered
        t.send_merge_count =
            (*sender_proc_tile_map)[i][j][k].total_receivable_datacount; // remote computation total data vectors
        (*send_tile_meta)[index] = t;
        if (t.count > t.send_merge_count and !embedding and enable_remote_compute and j != this->grid->rank_in_col) {
          (*sender_proc_tile_map)[i][j][k].mode = 0;
        }
      }

//      auto t = start_clock();
      MPI_Alltoall((*send_tile_meta).data(), per_process_messages, TILETUPLE,
                   (*receive_tile_meta).data(), per_process_messages, TILETUPLE,
                   this->grid->col_world);
//      stop_clock_and_add(t, "Communication Time");
//      add_perf_stats(sizeof(TileTuple<INDEX_TYPE>)*per_process_messages*this->grid->col_world_size, "Data transfers");

      send_tile_meta->clear();
      send_tile_meta->resize(itr);

#pragma omp parallel for
      for (auto in = 0; in < itr; in++) {
        auto i = in / (this->grid->col_world_size * tiles_per_process);
        auto j = (in / tiles_per_process) % this->grid->col_world_size;
        auto k = in % tiles_per_process;
        auto offset = j * per_process_messages;
        auto index = offset + i * tiles_per_process + k;
        TileTuple<INDEX_TYPE> t = (*receive_tile_meta)[index];
        if (embedding and (t.batch_id == i and t.tile_id == k)) {
          TileTuple<INDEX_TYPE> st;
          st.batch_id = i;
          st.tile_id = k;
          auto remote_cost = (merge_cost_factor*t.send_merge_count) +  (*receiver_proc_tile_map)[i][j][k].total_transferrable_datacount;
          if (t.count <= remote_cost or !enable_remote_compute) {
            (*receiver_proc_tile_map)[i][j][k].mode = 0;
            st.mode = 1;
          } else {
            (*receiver_proc_tile_map)[i][j][k].initialize_dataCache(); // initialize data cache to receive//// remote computed data
            st.mode = 0;
          }
          (*send_tile_meta)[index] = st;
        } else {
          if (t.batch_id == i and t.tile_id == k) {
            (*receiver_proc_tile_map)[i][j][k].total_receivable_datacount =
                t.count;
            (*receiver_proc_tile_map)[i][j][k].total_transferrable_datacount =
                t.send_merge_count;
            if (t.count <= t.send_merge_count or !enable_remote_compute or j==this->grid->rank_in_col) {
              (*receiver_proc_tile_map)[i][j][k].mode = 0;
            } else {
              (*receiver_proc_tile_map)[i][j][k].initialize_dataCache(); // initialize data cache to receive remote computed data
            }
          }
        }
      }

      (receive_tile_meta)->clear();
      (receive_tile_meta)->resize(itr);

      if (embedding) {
//        MPI_Barrier(this->grid->col_world);
//        auto t = start_clock();
        MPI_Alltoall((*send_tile_meta).data(), per_process_messages, TILETUPLE,
                     (*receive_tile_meta).data(), per_process_messages,
                     TILETUPLE, this->grid->col_world);
//        stop_clock_and_add(t, "Communication Time");
//        add_perf_stats(sizeof(TileTuple<INDEX_TYPE>)*per_process_messages*this->grid->col_world_size, "Data transfers");

#pragma omp parallel for
        for (auto in = 0; in < itr; in++) {
          auto i = in / (this->grid->col_world_size * tiles_per_process);
          auto j = (in / tiles_per_process) % this->grid->col_world_size;
          auto k = in % tiles_per_process;
          auto offset = j * per_process_messages;
          auto index = offset + i * tiles_per_process + k;
          TileTuple<INDEX_TYPE> st = (*receive_tile_meta)[index];
          if (st.batch_id == i and st.tile_id == k){
              if (st.mode == 0) {
                (*sender_proc_tile_map)[i][j][k].mode = 0;
                (*sender_proc_tile_map)[i][j][k].initialize_dataCache();
              }
            }
        }
      }
    }
  }

  inline void transfer_sparse_data(
      vector<SpTuple<VALUE_TYPE, sp_tuple_max_dim>> *sendbuf_cyclic,
      vector<SpTuple<VALUE_TYPE, sp_tuple_max_dim>> *receivebuf, int iteration,
      int batch_id, int starting_proc, int end_proc, int start_tile,
      int end_tile, bool embedding=false, DistributedMat* state_holder=nullptr) {
    auto t = start_clock();

    int total_receive_count = 0;
    shared_ptr<vector<vector<SpTuple<VALUE_TYPE, sp_tuple_max_dim>>>>
        data_buffer_ptr = make_shared<
            vector<vector<SpTuple<VALUE_TYPE, sp_tuple_max_dim>>>>();
    data_buffer_ptr->resize(this->grid->col_world_size);

    int total_send_count = 0;
    this->send_counts_cyclic = vector<int>(this->grid->col_world_size, 0);
    this->receive_counts_cyclic = vector<int>(this->grid->col_world_size, 0);
    this->sdispls_cyclic = vector<int>(this->grid->col_world_size, 0);
    this->rdispls_cyclic = vector<int>(this->grid->col_world_size, 0);

    vector<int> sending_procs;
    vector<int> receiving_procs;

    for (int i = starting_proc; i < end_proc; i++) {
      int sending_rank =
          (this->grid->rank_in_col + i) % this->grid->col_world_size;
      int receiving_rank =
          (this->grid->rank_in_col >= i)
              ? (this->grid->rank_in_col - i) % this->grid->col_world_size
              : (this->grid->col_world_size - i + this->grid->rank_in_col) %
                    this->grid->col_world_size;
      sending_procs.push_back(sending_rank);
      receiving_procs.push_back(receiving_rank);
      (*data_buffer_ptr)[sending_rank] =
          vector<SpTuple<VALUE_TYPE, sp_tuple_max_dim>>();
    }

    for (int tile = start_tile; tile < end_tile; tile++) {
      for (const auto &pair : (*send_indices_proc_map)[batch_id][tile]) {
        auto col_id = pair.first;
        CSRHandle sparse_tuple = (this->sparse_local)->fetch_local_data(col_id, embedding,static_cast<VALUE_TYPE>(INT_MIN));
        if (sparse_tuple.col_idx.size() > 0) {
          for (int i = 0; i < sending_procs.size(); i++) {
            if (pair.second.count(sending_procs[i]) > 0 and
                (*sender_proc_tile_map)[batch_id][sending_procs[i]][tile].mode == 1) {
              if (this->send_counts_cyclic[sending_procs[i]] == 0) {
                SpTuple<VALUE_TYPE, sp_tuple_max_dim> current;
                current.rows[0] =2; // rows first two indices are already taken for metadata
                current.rows[1] = 0;
                (*data_buffer_ptr)[sending_procs[i]].push_back(current);
                total_send_count++;
                this->send_counts_cyclic[sending_procs[i]]++;
              }

              SpTuple<VALUE_TYPE, sp_tuple_max_dim> latest = (*data_buffer_ptr)
                  [sending_procs[i]]
                  [this->send_counts_cyclic[sending_procs[i]] - 1];
              auto row_index_offset = latest.rows[0];
              auto col_index_offset = latest.rows[1];
              if (row_index_offset >= row_max-2 or col_index_offset >= sp_tuple_max_dim) {
                SpTuple<VALUE_TYPE, sp_tuple_max_dim> current;
                current.rows[0] =2; // rows first two indices are already taken for metadata
                current.rows[1] = 0;
                (*data_buffer_ptr)[sending_procs[i]].push_back(current);
                total_send_count++;
                this->send_counts_cyclic[sending_procs[i]]++;
                latest = (*data_buffer_ptr)[sending_procs[i]][this->send_counts_cyclic[sending_procs[i]] - 1];
                row_index_offset = latest.rows[0];
                col_index_offset = latest.rows[1];
              }

              INDEX_TYPE offset = sparse_tuple.col_idx.size();
              // start filling from offset position
              INDEX_TYPE pending_col_pos = sp_tuple_max_dim - col_index_offset;
              INDEX_TYPE num_of_copying_data = min(offset, pending_col_pos);
              INDEX_TYPE remaining_data_items = offset - num_of_copying_data;
              offset =num_of_copying_data;
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
                add_perf_stats(num_of_copying_data, "Data transfers");
              }
              (*data_buffer_ptr)[sending_procs[i]][this->send_counts_cyclic[sending_procs[i]] -1] = latest;
              do {
                if (remaining_data_items > 0) {

                   num_of_copying_data = min(remaining_data_items, static_cast<INDEX_TYPE>(sp_tuple_max_dim));
                   remaining_data_items = remaining_data_items - num_of_copying_data;

                  SpTuple<VALUE_TYPE, sp_tuple_max_dim> current;
                  current.rows[0] = 2; // rows first two indices are already taken for metadata
                  current.rows[1] = 0;
                  (*data_buffer_ptr)[sending_procs[i]].push_back(current);
                  total_send_count++;
                  this->send_counts_cyclic[sending_procs[i]]++;
                  latest = (*data_buffer_ptr)
                      [sending_procs[i]]
                      [this->send_counts_cyclic[sending_procs[i]] - 1];
                  row_index_offset = latest.rows[0];
                  col_index_offset = latest.rows[1];
                  latest.rows[row_index_offset] = sparse_tuple.row_idx[0];
                  latest.rows[row_index_offset + 1] = num_of_copying_data;
                  latest.rows[0] = row_index_offset + 2;
                  latest.rows[1] = latest.rows[1] + num_of_copying_data;

                  copy(sparse_tuple.col_idx.begin() + offset - 1,
                       sparse_tuple.col_idx.begin() + offset - 1 +
                           num_of_copying_data,
                       latest.cols.begin());
                  copy(sparse_tuple.values.begin() + offset - 1,
                       sparse_tuple.values.begin() + offset - 1 +
                           num_of_copying_data,
                       latest.values.begin());
                  (*data_buffer_ptr)
                      [sending_procs[i]]
                      [this->send_counts_cyclic[sending_procs[i]] - 1] = latest;
                  add_perf_stats(num_of_copying_data, "Data transfers");
                  offset +=num_of_copying_data;
                }
              }while(remaining_data_items>0);
            }
          }
        }
      }
    }
    (*sendbuf_cyclic).resize(total_send_count);
    for (int i = 0; i < this->grid->col_world_size; i++) {
      this->sdispls_cyclic[i] = (i > 0) ? this->sdispls_cyclic[i - 1] +
                                              this->send_counts_cyclic[i - 1]
                                        : this->sdispls_cyclic[i];
      copy((*data_buffer_ptr)[i].begin(), (*data_buffer_ptr)[i].end(),
           (*sendbuf_cyclic).begin() + this->sdispls_cyclic[i]);
    }

//    auto t = start_clock();
    MPI_Alltoall(this->send_counts_cyclic.data(), 1, MPI_INT,
                 this->receive_counts_cyclic.data(), 1, MPI_INT,
                 this->grid->col_world);
//    stop_clock_and_add(t, "Communication Time");
//    add_perf_stats(sizeof(int)*this->grid->col_world_size, "Data transfers");
    for (int i = 0; i < this->grid->col_world_size; i++) {
      this->rdispls_cyclic[i] = (i > 0) ? this->rdispls_cyclic[i - 1] +
                                              this->receive_counts_cyclic[i - 1]
                                        : this->rdispls_cyclic[i];
      total_receive_count += this->receive_counts_cyclic[i];
    }

    if (total_receive_count > 0) {
      receivebuf->resize(total_receive_count);
    }

    MPI_Barrier(this->grid->col_world);
    stop_clock_and_add(t, "Communication Data Loading");
     t = start_clock();
    MPI_Alltoallv((*sendbuf_cyclic).data(), this->send_counts_cyclic.data(),
                  this->sdispls_cyclic.data(), SPARSETUPLE,
                  (*receivebuf).data(), this->receive_counts_cyclic.data(),
                  this->rdispls_cyclic.data(), SPARSETUPLE,
                  this->grid->col_world);
    stop_clock_and_add(t, "Communication Time");
//    add_perf_stats(total_receive_count*sizeof(SpTuple<VALUE_TYPE,sp_tuple_max_dim>), "Data transfers");
     t = start_clock();
    this->populate_sparse_cache(sendbuf_cyclic, receivebuf, iteration,batch_id);
    stop_clock_and_add(t, "Communicated Data Store");
  }

  void transfer_sparse_data(vector<INDEX_TYPE> &col_ids, int iteration, int batch_id) {

    vector<vector<INDEX_TYPE>> receive_col_ids_list(this->grid->col_world_size);
    vector<INDEX_TYPE> send_col_ids_list;

    int total_send_count = 0;
    int total_receive_count = 0;

    this->send_counts_cyclic = vector<int>(this->grid->col_world_size, 0);
    this->receive_counts_cyclic = vector<int>(this->grid->col_world_size, 0);
    this->sdispls_cyclic = vector<int>(this->grid->col_world_size, 0);
    this->rdispls_cyclic = vector<int>(this->grid->col_world_size, 0);

    for (int i = 0; i < col_ids.size(); i++) {
      int owner_rank = col_ids[i] / (this->sp_local_receiver)->proc_row_width;
      if (owner_rank == this->grid->rank_in_col) {
        send_col_ids_list.push_back(col_ids[i]);
      } else {
        receive_col_ids_list[owner_rank].push_back(col_ids[i]);
      }
    }

    for (int i = 0; i < this->grid->col_world_size; i++) {
      total_send_count = send_col_ids_list.size();
      if (i != this->grid->rank_in_col) {
        this->send_counts_cyclic[i] = total_send_count;
      } else {
        this->send_counts_cyclic[i] = 0;
      }
      this->receive_counts_cyclic[i] = receive_col_ids_list[i].size();
    }

    this->sdispls_cyclic[0] = 0;
    this->rdispls_cyclic[0] = 0;
    for (int i = 0; i < this->grid->col_world_size; i++) {

      this->sdispls_cyclic[i] = 0;
      this->rdispls_cyclic[i] =
          (i > 0) ? this->rdispls_cyclic[i - 1] + this->receive_counts_cyclic[i - 1]
                  : this->rdispls_cyclic[i];
      total_receive_count = total_receive_count + this->receive_counts_cyclic[i];
    }

    unique_ptr<std::vector<SpTuple<VALUE_TYPE, sp_tuple_max_dim>>> sendbuf =
        unique_ptr<std::vector<SpTuple<VALUE_TYPE, sp_tuple_max_dim>>>(
            new vector<SpTuple<VALUE_TYPE, sp_tuple_max_dim>>());

    unique_ptr<std::vector<SpTuple<VALUE_TYPE, sp_tuple_max_dim>>>
        receivebuf_ptr =
            unique_ptr<std::vector<SpTuple<VALUE_TYPE, sp_tuple_max_dim>>>(
                new vector<SpTuple<VALUE_TYPE, sp_tuple_max_dim>>());

    receivebuf_ptr.get()->resize(total_receive_count);

    for (int j = 0; j < send_col_ids_list.size(); j++) {
      int local_key = send_col_ids_list[j] - (this->grid->rank_in_col) * (this->sp_local_receiver)->proc_row_width;
      CSRHandle sparse_tuple  =(this->sparse_local)->fetch_local_data(local_key,true, static_cast<VALUE_TYPE>(INT_MIN));

      SpTuple<VALUE_TYPE, sp_tuple_max_dim> current;
      INDEX_TYPE num_of_copying_data = sparse_tuple.col_idx.size();
      current.rows[0] =2; // rows first two indices are already taken for metadata
      current.rows[1] = 0;
      current.rows[current.rows[0]] = sparse_tuple.row_idx[0];
      current.rows[current.rows[0]+1] = num_of_copying_data;
      copy(sparse_tuple.col_idx.begin() ,
           sparse_tuple.col_idx.begin() + num_of_copying_data ,
           current.cols.begin());
      copy(sparse_tuple.values.begin(),
           sparse_tuple.values.begin() + num_of_copying_data,
           current.values.begin());
      add_perf_stats(num_of_copying_data, "Data transfers");
      current.rows[0] =  current.rows[0]+2;
      current.rows[1] =current.rows[1]+num_of_copying_data;
      (*sendbuf).push_back(current);
    }
    MPI_Barrier(this->grid->col_world);
    auto t = start_clock();
    MPI_Alltoallv((*sendbuf).data(), this->send_counts_cyclic.data(), this->sdispls_cyclic.data(),
                  SPARSETUPLE, (*receivebuf_ptr.get()).data(),
                  this->receive_counts_cyclic.data(), this->rdispls_cyclic.data(),
                  SPARSETUPLE, this->grid->col_world);
    stop_clock_and_add(t, "Communication Time");
    MPI_Request dumy;
    this->populate_sparse_cache(sendbuf.get(), receivebuf_ptr.get(),  iteration,batch_id,true); // we should not do this
  }

  inline void transfer_remotely_computable_data(
      vector<SpTuple<VALUE_TYPE, sp_tuple_max_dim>> *sendbuf_cyclic,
      vector<SpTuple<VALUE_TYPE, sp_tuple_max_dim>> *receivebuf, int iteration,
      int batch_id, int starting_proc, int end_proc, int start_tile,
      int end_tile, bool embedding=false) {

    int total_receive_count = 0;
    unique_ptr<vector<vector<SpTuple<VALUE_TYPE, sp_tuple_max_dim>>>>
        data_buffer_ptr = make_unique<
            vector<vector<SpTuple<VALUE_TYPE, sp_tuple_max_dim>>>>();
    data_buffer_ptr->resize(this->grid->col_world_size);

    int total_send_count = 0;
    this->send_counts_cyclic = vector<int>(this->grid->col_world_size, 0);
    this->receive_counts_cyclic = vector<int>(this->grid->col_world_size, 0);
    this->sdispls_cyclic = vector<int>(this->grid->col_world_size, 0);
    this->rdispls_cyclic = vector<int>(this->grid->col_world_size, 0);

    vector<int> sending_procs;
    vector<int> receiving_procs;

    for (int i = starting_proc; i < end_proc; i++) {
      int sending_rank =
          (this->grid->rank_in_col + i) % this->grid->col_world_size;
      int receiving_rank =
          (this->grid->rank_in_col >= i)
              ? (this->grid->rank_in_col - i) % this->grid->col_world_size
              : (this->grid->col_world_size - i + this->grid->rank_in_col) %
                    this->grid->col_world_size;
      sending_procs.push_back(sending_rank);
      receiving_procs.push_back(receiving_rank);
      (*data_buffer_ptr)[sending_rank] =
          vector<SpTuple<VALUE_TYPE, sp_tuple_max_dim>>();
    }

    for (int i = 0; i < sending_procs.size(); i++) {
      for (int tile = start_tile; tile < end_tile; tile++) {
        SparseTile<INDEX_TYPE,VALUE_TYPE>& spTile =  (*receiver_proc_tile_map)[batch_id][sending_procs[i]][tile];
        if (spTile.mode ==1) {
          for (auto it=spTile.row_id_set.begin(); it!= spTile.row_id_set.end();++it) {
            auto index = *it;
            CSRHandle sparse_tuple =(this->sparse_local)->fetch_local_data(index,embedding,static_cast<VALUE_TYPE>(INT_MIN));

            if (sparse_tuple.col_idx.size() > 0) {
              if (this->send_counts_cyclic[sending_procs[i]] == 0) {
                SpTuple<VALUE_TYPE, sp_tuple_max_dim> current;
                current.rows[0] =
                    2; // rows first two indices are already taken for metadata
                current.rows[1] = 0;
                (*data_buffer_ptr)[sending_procs[i]].push_back(current);
                total_send_count++;
                this->send_counts_cyclic[sending_procs[i]]++;
              }
              SpTuple<VALUE_TYPE, sp_tuple_max_dim> latest = (*data_buffer_ptr)
                  [sending_procs[i]]
                  [this->send_counts_cyclic[sending_procs[i]] - 1];

              auto row_index_offset = latest.rows[0];
              auto col_index_offset = latest.rows[1];
              if (row_index_offset >= row_max - 3 or
                  col_index_offset >= sp_tuple_max_dim) {
                SpTuple<VALUE_TYPE, sp_tuple_max_dim> current;
                current.rows[0] =
                    2; // rows first two indices are already taken for metadata
                current.rows[1] = 0;
                (*data_buffer_ptr)[sending_procs[i]].push_back(current);
                total_send_count++;
                this->send_counts_cyclic[sending_procs[i]]++;
                latest = (*data_buffer_ptr)
                    [sending_procs[i]]
                    [this->send_counts_cyclic[sending_procs[i]] - 1];
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
              latest.rows[row_index_offset + 2] = static_cast<INDEX_TYPE>(tile);
              latest.rows[0] = row_index_offset + 3;
              latest.rows[1] = latest.rows[1] + num_of_copying_data;

              if (num_of_copying_data > 0) {
                copy(sparse_tuple.col_idx.begin(),
                     sparse_tuple.col_idx.begin() + num_of_copying_data,
                     latest.cols.begin() + col_index_offset);
                copy(sparse_tuple.values.begin(),
                     sparse_tuple.values.begin() + num_of_copying_data,
                     latest.values.begin() + col_index_offset);
              }
              (*data_buffer_ptr)[sending_procs[i]]
                                [this->send_counts_cyclic[sending_procs[i]] -
                                 1] = latest;
              if (remaining_data_items > 0) {
                SpTuple<VALUE_TYPE, sp_tuple_max_dim> current;
                current.rows[0] =
                    2; // rows first two indices are already taken for metadata
                current.rows[1] = 0;
                (*data_buffer_ptr)[sending_procs[i]].push_back(current);
                total_send_count++;
                this->send_counts_cyclic[sending_procs[i]]++;
                latest = (*data_buffer_ptr)
                    [sending_procs[i]]
                    [(*data_buffer_ptr)[sending_procs[i]].size() - 1];
                row_index_offset = latest.rows[0];
                col_index_offset = latest.rows[1];
                latest.rows[row_index_offset] = sparse_tuple.row_idx[0];
                latest.rows[row_index_offset + 1] = remaining_data_items;
                latest.rows[row_index_offset + 2] =
                    static_cast<INDEX_TYPE>(tile);
                latest.rows[0] = row_index_offset + 3;
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
                                  [this->send_counts_cyclic[sending_procs[i]] -
                                   1] = latest;
              }
            }
          }
        }
      }
    }
    if (total_send_count>0) {
      (*sendbuf_cyclic).resize(total_send_count);

      for (int i = 0; i < this->grid->col_world_size; i++) {
        this->sdispls_cyclic[i] = (i > 0) ? this->sdispls_cyclic[i - 1] +
                                                this->send_counts_cyclic[i - 1]
                                          : this->sdispls_cyclic[i];
        copy((*data_buffer_ptr)[i].begin(), (*data_buffer_ptr)[i].end(),
             (*sendbuf_cyclic).begin() + this->sdispls_cyclic[i]);
      }
    }
//    MPI_Barrier(this->grid->col_world);
//    auto t = start_clock();
    MPI_Alltoall(this->send_counts_cyclic.data(), 1, MPI_INT,
                 this->receive_counts_cyclic.data(), 1, MPI_INT,
                 this->grid->col_world);
//    stop_clock_and_add(t, "Communication Time");
//    add_perf_stats(sizeof(int)*this->grid->col_world_size, "Data transfers");

    for (int i = 0; i < this->grid->col_world_size; i++) {
      this->rdispls_cyclic[i] = (i > 0) ? this->rdispls_cyclic[i - 1] +
                                              this->receive_counts_cyclic[i - 1]
                                        : this->rdispls_cyclic[i];
      total_receive_count += this->receive_counts_cyclic[i];
    }

    if (total_receive_count > 0) {
      receivebuf->resize(total_receive_count);
    }


    //
    MPI_Barrier(this->grid->col_world);
   auto  t = start_clock();
    MPI_Alltoallv((*sendbuf_cyclic).data(), this->send_counts_cyclic.data(),
                  this->sdispls_cyclic.data(), SPARSETUPLE,
                  (*receivebuf).data(), this->receive_counts_cyclic.data(),
                  this->rdispls_cyclic.data(), SPARSETUPLE,
                  this->grid->col_world);
    stop_clock_and_add(t, "Communication Time");
//    add_perf_stats(total_receive_count*sizeof(SpTuple<VALUE_TYPE,sp_tuple_max_dim>), "Data transfers");
    this->store_remotely_computed_data(sendbuf_cyclic, receivebuf, iteration,batch_id,false);
  }





  inline void receive_remotely_computed_data(
      vector<SpTuple<VALUE_TYPE, sp_tuple_max_dim>> *sendbuf_cyclic,
      vector<SpTuple<VALUE_TYPE, sp_tuple_max_dim>> *receivebuf, int iteration,
      int batch_id, int starting_proc, int end_proc, int start_tile,
      int end_tile) {

    int total_receive_count = 0;
    unique_ptr<vector<vector<SpTuple<VALUE_TYPE, sp_tuple_max_dim>>>>
        data_buffer_ptr = make_unique<
            vector<vector<SpTuple<VALUE_TYPE, sp_tuple_max_dim>>>>();
    data_buffer_ptr->resize(this->grid->col_world_size);

    int total_send_count = 0;
    this->send_counts_cyclic = vector<int>(this->grid->col_world_size, 0);
    this->receive_counts_cyclic = vector<int>(this->grid->col_world_size, 0);
    this->sdispls_cyclic = vector<int>(this->grid->col_world_size, 0);
    this->rdispls_cyclic = vector<int>(this->grid->col_world_size, 0);

    vector<int> sending_procs;
    vector<int> receiving_procs;

    for (int i = starting_proc; i < end_proc; i++) {
      int sending_rank =
          (this->grid->rank_in_col + i) % this->grid->col_world_size;
      int receiving_rank =
          (this->grid->rank_in_col >= i)
              ? (this->grid->rank_in_col - i) % this->grid->col_world_size
              : (this->grid->col_world_size - i + this->grid->rank_in_col) %
                    this->grid->col_world_size;
      sending_procs.push_back(sending_rank);
      receiving_procs.push_back(receiving_rank);
      (*data_buffer_ptr)[sending_rank] =
          vector<SpTuple<VALUE_TYPE, sp_tuple_max_dim>>();
    }

    for (int i = 0; i < sending_procs.size(); i++) {
      for (int tile = start_tile; tile < end_tile; tile++) {
       SparseTile<INDEX_TYPE,VALUE_TYPE>& spTile =  (*sender_proc_tile_map)[batch_id][sending_procs[i]][tile];
        if (spTile.mode ==0) {
          spTile.initialize_CSR_blocks();
          for (INDEX_TYPE index =spTile.row_starting_index;index < spTile.row_end_index;++index) {
            CSRHandle sparse_tuple =spTile.fetch_remote_data(index);

            if (sparse_tuple.col_idx.size() > 0) {
              if (this->send_counts_cyclic[sending_procs[i]] == 0) {
                SpTuple<VALUE_TYPE, sp_tuple_max_dim> current;
                current.rows[0] =
                    2; // rows first two indices are already taken for metadata
                current.rows[1] = 0;
                (*data_buffer_ptr)[sending_procs[i]].push_back(current);
                total_send_count++;
                this->send_counts_cyclic[sending_procs[i]]++;
              }
              SpTuple<VALUE_TYPE, sp_tuple_max_dim> latest = (*data_buffer_ptr)
                  [sending_procs[i]]
                  [this->send_counts_cyclic[sending_procs[i]] - 1];

              auto row_index_offset = latest.rows[0];
              auto col_index_offset = latest.rows[1];
              if (row_index_offset >= row_max - 3 or
                  col_index_offset >= sp_tuple_max_dim) {
                SpTuple<VALUE_TYPE, sp_tuple_max_dim> current;
                current.rows[0] =
                    2; // rows first two indices are already taken for metadata
                current.rows[1] = 0;
                (*data_buffer_ptr)[sending_procs[i]].push_back(current);
                total_send_count++;
                this->send_counts_cyclic[sending_procs[i]]++;
                latest = (*data_buffer_ptr)
                    [sending_procs[i]]
                    [this->send_counts_cyclic[sending_procs[i]] - 1];
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
              latest.rows[row_index_offset + 2] = static_cast<INDEX_TYPE>(tile);
              latest.rows[0] = row_index_offset + 3;
              latest.rows[1] = latest.rows[1] + num_of_copying_data;

              if (num_of_copying_data > 0) {
                copy(sparse_tuple.col_idx.begin(),
                     sparse_tuple.col_idx.begin() + num_of_copying_data,
                     latest.cols.begin() + col_index_offset);
                copy(sparse_tuple.values.begin(),
                     sparse_tuple.values.begin() + num_of_copying_data,
                     latest.values.begin() + col_index_offset);
                add_perf_stats(num_of_copying_data, "Data transfers");
              }
              (*data_buffer_ptr)[sending_procs[i]]
                                [this->send_counts_cyclic[sending_procs[i]] -
                                 1] = latest;
              if (remaining_data_items > 0) {
                SpTuple<VALUE_TYPE, sp_tuple_max_dim> current;
                current.rows[0] =
                    2; // rows first two indices are already taken for metadata
                current.rows[1] = 0;
                (*data_buffer_ptr)[sending_procs[i]].push_back(current);
                total_send_count++;
                this->send_counts_cyclic[sending_procs[i]]++;
                latest = (*data_buffer_ptr)
                    [sending_procs[i]]
                    [(*data_buffer_ptr)[sending_procs[i]].size() - 1];
                row_index_offset = latest.rows[0];
                col_index_offset = latest.rows[1];
                latest.rows[row_index_offset] = sparse_tuple.row_idx[0];
                latest.rows[row_index_offset + 1] = remaining_data_items;
                latest.rows[row_index_offset + 2] =
                    static_cast<INDEX_TYPE>(tile);
                latest.rows[0] = row_index_offset + 3;
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
                                  [this->send_counts_cyclic[sending_procs[i]] -
                                   1] = latest;
                add_perf_stats(remaining_data_items, "Data transfers");
              }
            }
          }
        }
      }
    }
    (*sendbuf_cyclic).resize(total_send_count);
    for (int i = 0; i < this->grid->col_world_size; i++) {
      this->sdispls_cyclic[i] = (i > 0) ? this->sdispls_cyclic[i - 1] +
                                              this->send_counts_cyclic[i - 1]
                                        : this->sdispls_cyclic[i];
      copy((*data_buffer_ptr)[i].begin(), (*data_buffer_ptr)[i].end(),
           (*sendbuf_cyclic).begin() + this->sdispls_cyclic[i]);
    }
//    MPI_Barrier(this->grid->col_world);
//    auto t = start_clock();
    MPI_Alltoall(this->send_counts_cyclic.data(), 1, MPI_INT,
                 this->receive_counts_cyclic.data(), 1, MPI_INT,
                 this->grid->col_world);
//    add_perf_stats(sizeof(int)*this->grid->col_world_size, "Data transfers");
//    stop_clock_and_add(t, "Communication Time");

    for (int i = 0; i < this->grid->col_world_size; i++) {
      this->rdispls_cyclic[i] = (i > 0) ? this->rdispls_cyclic[i - 1] +
                                              this->receive_counts_cyclic[i - 1]
                                        : this->rdispls_cyclic[i];
      total_receive_count += this->receive_counts_cyclic[i];
    }

    if (total_receive_count > 0) {
      receivebuf->resize(total_receive_count);
    }


    //
    MPI_Barrier(this->grid->col_world);
    auto t = start_clock();
    MPI_Alltoallv((*sendbuf_cyclic).data(), this->send_counts_cyclic.data(),
                  this->sdispls_cyclic.data(), SPARSETUPLE,
                  (*receivebuf).data(), this->receive_counts_cyclic.data(),
                  this->rdispls_cyclic.data(), SPARSETUPLE,
                  this->grid->col_world);
    stop_clock_and_add(t, "Communication Time");
//    add_perf_stats(total_receive_count*sizeof(SpTuple<VALUE_TYPE,sp_tuple_max_dim>), "Data transfers");
    this->store_remotely_computed_data(sendbuf_cyclic, receivebuf, iteration,batch_id,true);
  }

  inline void store_remotely_computed_data(
      vector<SpTuple<VALUE_TYPE, sp_tuple_max_dim>> *sendbuf,
      vector<SpTuple<VALUE_TYPE, sp_tuple_max_dim>> *receivebuf, int iteration,
      int batch_id, bool receive_computed_data) {

#pragma omp parallel for
    for (int i = 0; i < this->grid->col_world_size; i++) {
      INDEX_TYPE base_index = this->rdispls_cyclic[i];
      INDEX_TYPE count = this->receive_counts_cyclic[i];
      for (INDEX_TYPE j = base_index; j < base_index + count; j++) {
        auto row_offset = (*receivebuf)[j].rows[0];
        auto offset_so_far = 0;
        for (auto k = 2; k < row_offset; k = k + 3) {
          auto key = (*receivebuf)[j].rows[k];
          auto data_count = (*receivebuf)[j].rows[k + 1];
          auto tile = (*receivebuf)[j].rows[k + 2];
          SparseTile<INDEX_TYPE,VALUE_TYPE>&spTile = receive_computed_data?(*receiver_proc_tile_map)[batch_id][i][tile]:(*sender_proc_tile_map)[batch_id][i][tile];

          if (!receive_computed_data)
            key = key- spTile.row_starting_index;

          SparseCacheEntry<VALUE_TYPE> cache_entry = (*spTile.dataCachePtr)[key];
          auto entry_offset = cache_entry.cols.size();
          cache_entry.cols.resize(entry_offset + data_count);
          cache_entry.values.resize(entry_offset + data_count);
          copy((*receivebuf)[j].cols.begin() + offset_so_far,
               (*receivebuf)[j].cols.begin() + offset_so_far + data_count,
               cache_entry.cols.begin() + entry_offset);
          copy((*receivebuf)[j].values.begin() + offset_so_far,
               (*receivebuf)[j].values.begin() + offset_so_far + data_count,
               cache_entry.values.begin() + entry_offset);
          offset_so_far += data_count;
          (*spTile.dataCachePtr)[key] =cache_entry;
        }
      }
    }
    receivebuf->clear();
    receivebuf->shrink_to_fit();
    sendbuf->clear();
    sendbuf->shrink_to_fit();
  }

  inline void update_local_input(distblas::core::SpMat<VALUE_TYPE>* sparse_input){
    this->sparse_local->csr_local_data = make_unique<CSRLocal<VALUE_TYPE>>(*sparse_input->csr_local_data);
  }
};

} // namespace distblas::net
