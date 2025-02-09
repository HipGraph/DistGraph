#pragma once
#include "../core/sparse_mat_tile.hpp"
#include "../net/tile_based_data_comm.hpp"

using namespace std;
using namespace distblas::core;
using namespace distblas::net;


using namespace distblas::core;

namespace distblas::algo {
template <typename INDEX_TYPE, typename VALUE_TYPE, size_t embedding_dim>
class SpGEMMAlgoWithTiling {

private:
  distblas::core::SpMat<VALUE_TYPE> *sparse_local_output;
  distblas::core::SpMat<VALUE_TYPE> *sparse_local;
  distblas::core::SpMat<VALUE_TYPE> *sp_local_receiver;
  distblas::core::SpMat<VALUE_TYPE> *sp_local_sender;
  distblas::core::SpMat<VALUE_TYPE> *sp_local_native;
  distblas::core::DenseMat<INDEX_TYPE, VALUE_TYPE, embedding_dim> *state_holder;
  Process3DGrid *grid;

  // record temp local output
  unique_ptr<vector<unordered_map<INDEX_TYPE, VALUE_TYPE>>> output_ptr;

  TileDataComm<INDEX_TYPE ,VALUE_TYPE ,embedding_dim> *communicator=nullptr;

  // cache size controlling hyper parameter
  double alpha = 0;

  // hyper parameter controls the  computation and communication overlapping
  double beta = 1.0;

  // hyper parameter controls the switching the sync vs async commiunication
  bool sync = true;

  // hyper parameter controls the col major or row major  data access
  bool col_major = false;

  double tile_width_fraction;

  bool hash_spgemm = false;

public:
  vector<double> timing_info;
  SpGEMMAlgoWithTiling(
      distblas::core::SpMat<VALUE_TYPE> *sp_local_native,
      distblas::core::SpMat<VALUE_TYPE> *sp_local_receiver,
      distblas::core::SpMat<VALUE_TYPE> *sp_local_sender,
      distblas::core::SpMat<VALUE_TYPE> *sparse_local,
      distblas::core::SpMat<VALUE_TYPE> *sparse_local_output,
      Process3DGrid *grid, double alpha, double beta, bool col_major,
      bool sync_comm, double tile_width_fraction, bool hash_spgemm,
      TileDataComm<INDEX_TYPE, VALUE_TYPE, embedding_dim>* communicator=nullptr,
      distblas::core::DenseMat<INDEX_TYPE, VALUE_TYPE, embedding_dim>*state_holder = nullptr)
      : sp_local_native(sp_local_native), sp_local_receiver(sp_local_receiver),
        sp_local_sender(sp_local_sender), sparse_local(sparse_local),
        grid(grid), alpha(alpha), beta(beta), col_major(col_major),
        sync(sync_comm), sparse_local_output(sparse_local_output),
        tile_width_fraction(tile_width_fraction),communicator(communicator),state_holder(state_holder)

  {
    timing_info = vector<double>(sp_local_receiver->proc_row_width,0);
    this->hash_spgemm = hash_spgemm;
  }

  void algo_spgemm(int iterations, int batch_size, VALUE_TYPE lr, bool enable_remote=true) {

//    size_t total_memory = 0;
    int batches = 0;
    int last_batch_size = batch_size;

    if (sp_local_receiver->proc_row_width % batch_size == 0) {
      batches =
          static_cast<int>(sp_local_receiver->proc_row_width / batch_size);
    } else {
      batches =
          static_cast<int>(sp_local_receiver->proc_row_width / batch_size) + 1;
      last_batch_size =
          sp_local_receiver->proc_row_width - batch_size * (batches - 1);
    }

    cout << " rank " << grid->rank_in_col << " total batches " << batches
         << endl;

    // This communicator is being used for negative updates and in alpha > 0 to
    // fetch initial embeddings

    if (communicator==nullptr) {
      communicator =
          unique_ptr<TileDataComm<INDEX_TYPE, VALUE_TYPE, embedding_dim>>(
              new TileDataComm<INDEX_TYPE, VALUE_TYPE, embedding_dim>(
                  sp_local_receiver, sp_local_sender, sparse_local, grid, alpha,
                  batches, tile_width_fraction, hash_spgemm)).get();
      communicator->onboard_data(enable_remote);
    }

    // Buffer used for receive MPI operations data
    unique_ptr<std::vector<SpTuple<VALUE_TYPE, sp_tuple_max_dim>>> update_ptr =
        unique_ptr<std::vector<SpTuple<VALUE_TYPE, sp_tuple_max_dim>>>(
            new vector<SpTuple<VALUE_TYPE, sp_tuple_max_dim>>());

    // Buffer used for send MPI operations data
    unique_ptr<vector<SpTuple<VALUE_TYPE, sp_tuple_max_dim>>> sendbuf_ptr =
        unique_ptr<vector<SpTuple<VALUE_TYPE, sp_tuple_max_dim>>>(
            new vector<SpTuple<VALUE_TYPE, sp_tuple_max_dim>>());

    cout << " rank " << grid->rank_in_col << " on board data starting " << endl;


    cout << " rank " << grid->rank_in_col << " on board data completed "<< endl;

    int total_tiles = SparseTile<INDEX_TYPE, VALUE_TYPE>::get_tiles_per_process_row();

    CSRLocal<VALUE_TYPE> *csr_block =
        (col_major) ? (this->sp_local_receiver)->csr_local_data.get()
                    : (this->sp_local_native)->csr_local_data.get();

    int considering_batch_size = batch_size;

    for (int i = 0; i < iterations; i++) {

      for (int j = 0; j < batches; j++) {
        cout << " rank " << grid->rank_in_col << " batch " << j << endl;
        if (j == batches - 1) {
          considering_batch_size = last_batch_size;
        }
          if ((this->sparse_local_output)->hash_spgemm) {
            this->execute_pull_model_computations(
                sendbuf_ptr.get(), update_ptr.get(), i, j, communicator,
                csr_block, batch_size, considering_batch_size, lr, 1, 0, true,
                true, this->sparse_local_output);

            (this->sparse_local_output)->initialize_hashtables();

            // compute remote computations
            this->calc_t_dist_grad_rowptr(
                (this->sp_local_sender)->csr_local_data.get(), lr, i, j,
                batch_size, considering_batch_size, 2, 0,
                this->grid->col_world_size, true, communicator, nullptr);
          }

          this->execute_pull_model_computations(
              sendbuf_ptr.get(), update_ptr.get(), i, j, communicator,
              csr_block, batch_size, considering_batch_size, lr, 1, 0, true,
              false, this->sparse_local_output);
          if (enable_remote) {
            this->calc_t_dist_grad_rowptr(
                (this->sp_local_sender)->csr_local_data.get(), lr, i, j,
                batch_size, considering_batch_size, 2, 0,
                this->grid->col_world_size, false, communicator, nullptr);

            communicator->receive_remotely_computed_data(
                sendbuf_ptr.get(), update_ptr.get(), i, j, 0,
                this->grid->col_world_size, 0, total_tiles);

            auto t = start_clock();
            this->merge_remote_computations(
                j, batch_size, this->sparse_local_output, communicator);
            stop_clock_and_add(t, "Remote Merge Time");
        }
//        total_memory += get_memory_usage();
      }
      (this->sparse_local)->purge_cache();
    }
    auto t = start_clock();
    (this->sparse_local_output)->initialize_CSR_blocks(false,state_holder);
    stop_clock_and_add(t, "CSR Conversion");

//    total_memory = total_memory / (iterations * batches);
//    add_perf_stats(total_memory, "Memory usage");
//    stop_clock_and_add(t, "Total Time");
  }

  inline void execute_pull_model_computations(
      std::vector<SpTuple<VALUE_TYPE, sp_tuple_max_dim>> *sendbuf,
      std::vector<SpTuple<VALUE_TYPE, sp_tuple_max_dim>> *receivebuf,
      int iteration, int batch,
      TileDataComm<INDEX_TYPE, VALUE_TYPE, embedding_dim> *main_comm,
      CSRLocal<VALUE_TYPE> *csr_block, int batch_size,
      int considering_batch_size, double lr, int comm_initial_start,
      int first_execution_proc, bool communication, bool symbolic,
      DistributedMat *output) {

    int proc_length = get_proc_length(beta, this->grid->col_world_size);
    int prev_start = comm_initial_start;

    auto tiles_per_process =
        SparseTile<INDEX_TYPE, VALUE_TYPE>::get_tiles_per_process_row();
    bool measured=false;

    size_t total_memory = 0;

    for (int k = prev_start; k < this->grid->col_world_size; k += proc_length) {
      int end_process = get_end_proc(k, beta, this->grid->col_world_size);

      MPI_Request req;

      if (communication and (symbolic or !output->hash_spgemm)) {
        auto t = start_clock();
        main_comm->transfer_sparse_data(sendbuf, receivebuf, iteration, batch,k, end_process, 0, tiles_per_process,false);
        stop_clock_and_add(t, "CombinedComm Time");
        if (!measured){
          total_memory += get_memory_usage();
          add_perf_stats(total_memory, "Memory usage");
          measured=true;
        }
      }
      if (k == comm_initial_start) {
        // local computation

        this->calc_t_dist_grad_rowptr(
            csr_block, lr, iteration, batch, batch_size, considering_batch_size,
            0, first_execution_proc, prev_start, symbolic, main_comm, output);


      } else if (k > comm_initial_start) {
        int prev_end_process =
            get_end_proc(prev_start, beta, grid->col_world_size);

        this->calc_t_dist_grad_rowptr(
            csr_block, lr, iteration, batch, batch_size, considering_batch_size,
            1, prev_start, prev_end_process, symbolic, main_comm, output);
      }
      prev_start = k;
    }
    int prev_end_process = get_end_proc(prev_start, beta, grid->col_world_size);

    // updating last remote fetched data vectors
    this->calc_t_dist_grad_rowptr(
        csr_block, lr, iteration, batch, batch_size, considering_batch_size, 1,
        prev_start, prev_end_process, symbolic, main_comm, output);

//     dense_local->invalidate_cache(i, j, true);
  }

  inline void calc_t_dist_grad_rowptr(
      CSRLocal<VALUE_TYPE> *csr_block, VALUE_TYPE lr, int itr, int batch_id,
      int batch_size, int block_size, int mode, int start_process,
      int end_process, bool symbolic,
      TileDataComm<INDEX_TYPE, VALUE_TYPE, embedding_dim> *main_com,
      DistributedMat *output, DistributedMat *state_holder=nullptr) {
    if (mode == 0) { // local computation
      auto source_start_index = batch_id * batch_size;
      auto source_end_index = std::min(
          std::min(static_cast<INDEX_TYPE>((batch_id + 1) * batch_size),
                   this->sp_local_receiver->proc_row_width),
          this->sp_local_receiver->gRows);
      auto dst_start_index =
          this->sp_local_receiver->proc_col_width * this->grid->rank_in_col;
      auto dst_end_index = std::min(
          static_cast<INDEX_TYPE>(this->sp_local_receiver->proc_col_width *
                                  (this->grid->rank_in_col + 1)),
          this->sp_local_receiver->gCols);
      calc_embedding_row_major(source_start_index, source_end_index,
                               dst_start_index, dst_end_index, csr_block, lr,
                               batch_id, batch_size, block_size, symbolic, mode,
                               output);
    } else if (mode == 1) { // remote pull
      for (int r = start_process; r < end_process; r++) {
        if (r != grid->rank_in_col) {
          int computing_rank =
              (grid->rank_in_col >= r)
                  ? (grid->rank_in_col - r) % grid->col_world_size
                  : (grid->col_world_size - r + grid->rank_in_col) %
                        grid->col_world_size;
          int total_tiles =
              SparseTile<INDEX_TYPE, VALUE_TYPE>::get_tiles_per_process_row();
          for (int tile = 0; tile < total_tiles; tile++) {
            if ((*main_com->receiver_proc_tile_map)[batch_id][computing_rank][tile].mode == 0) {
              auto source_start_index =(*main_com->receiver_proc_tile_map)[batch_id][computing_rank][tile].row_starting_index;
              auto source_end_index =(*main_com->receiver_proc_tile_map)[batch_id][computing_rank][tile].row_end_index;
              auto dst_start_index =(*main_com->receiver_proc_tile_map)[batch_id][computing_rank][tile].col_start_index;
              auto dst_end_index =(*main_com->receiver_proc_tile_map)[batch_id][computing_rank][tile].col_end_index;
              calc_embedding_row_major(source_start_index, source_end_index,
                                       dst_start_index, dst_end_index,
                                       csr_block, lr, batch_id, batch_size,
                                       block_size, symbolic, mode, output);
            }
          }
        }
      }
    } else { // execute remote computations
      for (int r = start_process; r < end_process; r++) {
        int computing_rank =
            (grid->rank_in_col >= r)
                ? (grid->rank_in_col - r) % grid->col_world_size
                : (grid->col_world_size - r + grid->rank_in_col) %
                      grid->col_world_size;
        int total_tiles =
            SparseTile<INDEX_TYPE, VALUE_TYPE>::get_tiles_per_process_row();
        for (int tile = 0; tile < total_tiles; tile++) {
          SparseTile<INDEX_TYPE, VALUE_TYPE> &sp_tile =
              (*main_com->sender_proc_tile_map)[batch_id][computing_rank][tile];
          if (sp_tile.mode == 0) {
            auto source_start_index = sp_tile.row_starting_index;
            auto source_end_index = sp_tile.row_end_index;
            auto dst_start_index = sp_tile.col_start_index;
            auto dst_end_index = sp_tile.col_end_index;
            sp_tile.initialize_output_DS_if(0, symbolic);

            calc_embedding_row_major(source_start_index, source_end_index,
                                     dst_start_index, dst_end_index, csr_block,
                                     lr, batch_id, batch_size, block_size,
                                     symbolic, mode, &sp_tile);
            if (symbolic) {
              sp_tile.initialize_hashtables();
            }
            if (itr == 0 and !symbolic) {
              add_perf_stats(1, "Remote Computed Tiles");
            }
          }
        }
      }
    }
  }

  inline void
  calc_embedding_row_major(INDEX_TYPE source_start_index,
                           INDEX_TYPE source_end_index,
                           INDEX_TYPE dst_start_index, INDEX_TYPE dst_end_index,
                           CSRLocal<VALUE_TYPE> *csr_block, VALUE_TYPE lr,
                           int batch_id, int batch_size, int block_size,
                           bool symbolic, int mode, DistributedMat *output) {
    if (csr_block->handler != nullptr) {
      CSRHandle *csr_handle = csr_block->handler.get();

      #pragma omp parallel for schedule(static) // enable for full batch training or
      for (INDEX_TYPE i = source_start_index; i < source_end_index; i++) {

        INDEX_TYPE index = (mode == 0 or mode == 1) ? i : i - source_start_index;
        int max_reach = 0;

        for (INDEX_TYPE j = static_cast<INDEX_TYPE>(csr_handle->rowStart[i]);
             j < static_cast<INDEX_TYPE>(csr_handle->rowStart[i + 1]); j++) {
          auto dst_id = csr_handle->col_idx[j];
          if (dst_id >= dst_start_index and dst_id < dst_end_index) {
            INDEX_TYPE local_dst =
                (mode == 0 or mode == 1)
                    ? dst_id - (this->grid)->rank_in_col *
                                   (this->sp_local_receiver)->proc_col_width
                    : dst_id;
            int target_rank =
                (int)(dst_id / (this->sp_local_receiver)->proc_col_width);
            bool fetch_from_cache = !(target_rank == (this->grid)->rank_in_col or mode == 2);

            vector<INDEX_TYPE> remote_cols;
            vector<VALUE_TYPE> remote_values;

            if (fetch_from_cache) {
              unordered_map<INDEX_TYPE, SparseCacheEntry<VALUE_TYPE>>
                  &arrayMap = (*this->sparse_local->tempCachePtr)[target_rank];
              if (arrayMap.find(dst_id)==arrayMap.end()){
                continue; //this key does not contain any data.
              }
              remote_cols = arrayMap[dst_id].cols;
              remote_values = arrayMap[dst_id].values;
            }

            CSRHandle *handle = ((this->sparse_local)->csr_local_data)->handler.get();

            if (!fetch_from_cache) {
              int count =handle->rowStart[local_dst + 1] - handle->rowStart[local_dst];
              if (symbolic) {
                INDEX_TYPE val =(*(output->sparse_data_counter))[index] + count;
                (*(output->sparse_data_counter))[index] =std::min(val, static_cast<INDEX_TYPE>(embedding_dim));
              } else if (output->hash_spgemm) {
                auto t= start_clock();
                INDEX_TYPE ht_size =(*(output->sparse_data_collector))[index].size();
                for (auto k = handle->rowStart[local_dst];k < handle->rowStart[local_dst + 1]; k++) {
                  auto d = (handle->col_idx[k]);
                  if (state_holder == nullptr or (mode==2 and (*(state_holder->state_metadata))[local_dst][d] == 0)) {
                    INDEX_TYPE hash = (d * hash_scale) & (ht_size - 1);
                    auto value = lr * handle->values[k];
                    int max_count = 10;
                    int count = 0;
                    while (count < max_count) {
                      if ((*(output->sparse_data_collector))[index][hash].col ==d) {
                        (*(output->sparse_data_collector))[index][hash].value = (*(output->sparse_data_collector))[index][hash].value +value;
                        break;
                      } else if ((*(output->sparse_data_collector))[index][hash]
                                     .col == -1) {
                        (*(output->sparse_data_collector))[index][hash].col = d;
                        (*(output->sparse_data_collector))[index][hash].value =
                            value;
                        break;
                      } else {
                        hash = (hash + 100) & (ht_size - 1);
                        count++;
                      }
                    }
                  }
                }
                auto time = stop_clock_get_elapsed(t);
                timing_info[index]+=time;
              } else {

                for (auto k = handle->rowStart[local_dst];k < handle->rowStart[local_dst + 1]; k++) {
                  auto t= start_clock();
                  auto d = (handle->col_idx[k]);
                  (*(output->dense_collector))[index][d] += lr * (handle->values[k]);
                  auto time = stop_clock_get_elapsed(t);
                  timing_info[index]+=time;
                }

              }
            } else {
              int count = remote_cols.size();
              if (symbolic) {
                INDEX_TYPE val =
                    (*(output->sparse_data_counter))[index] + count;
                (*(output->sparse_data_counter))[index] =
                    std::min(val, static_cast<INDEX_TYPE>(embedding_dim));
              } else if (output->hash_spgemm) {
                auto t= start_clock();
                INDEX_TYPE ht_size =
                    (*(output->sparse_data_collector))[index].size();
//                auto t = start_clock();
                for (int m = 0; m < remote_cols.size(); m++) {
                  auto d = remote_cols[m];
                  auto value = lr * remote_values[m];
                  INDEX_TYPE hash = (d * hash_scale) & (ht_size - 1);
                  int max_count = 10;
                  int count = 0;
                  while (count < max_count) {
                    if ((*(output->sparse_data_collector))[index][hash].col ==
                        d) {
                      (*(output->sparse_data_collector))[index][hash].value =
                          (*(output->sparse_data_collector))[index][hash]
                              .value +
                          value;
                      break;
                    } else if ((*(output->sparse_data_collector))[index][hash]
                                   .col == -1) {
                      (*(output->sparse_data_collector))[index][hash].col = d;
                      (*(output->sparse_data_collector))[index][hash].value =
                          value;
                      break;
                    } else {
                      hash = (hash + 100) & (ht_size - 1);
                      count++;
                    }
                  }
                }
                auto time = stop_clock_get_elapsed(t);
                timing_info[index]+=time;
//                stop_clock_and_add(t, "Local SpGEMM");
              } else {

                for (int m = 0; m < remote_cols.size(); m++) {
                  auto t= start_clock();
                  auto d = remote_cols[m];
                  (*(output->dense_collector))[index][d] +=lr * remote_values[m];
                  auto time = stop_clock_get_elapsed(t);
                  timing_info[index]+=time;
                }

              }
            }
          }
        }
      }
    }
  }

  inline void merge_remote_computations(
      int batch_id, INDEX_TYPE batch_size, DistributedMat *output,
      TileDataComm<INDEX_TYPE, VALUE_TYPE, embedding_dim> *main_comm) {
    auto source_start_index = batch_id * batch_size;
    auto source_end_index =
        std::min(std::min(static_cast<INDEX_TYPE>((batch_id + 1) * batch_size),
                          this->sp_local_receiver->proc_row_width),
                 this->sp_local_receiver->gRows);

    vector<vector<vector<SparseTile<INDEX_TYPE, VALUE_TYPE>>>> *tile_map = main_comm->receiver_proc_tile_map.get();
    int tiles_per_process_row = SparseTile<INDEX_TYPE, VALUE_TYPE>::get_tiles_per_process_row();
    #pragma omp parallel for schedule(static)
    for (auto i = source_start_index; i < source_end_index; i++) {
      INDEX_TYPE index = i - source_start_index;
      unordered_map<INDEX_TYPE, VALUE_TYPE> value_map;
      for (int ra = 0; ra < this->grid->col_world_size; ra++) {
        if (ra!=this->grid->rank_in_col) {
          for (int j = 0; j < tiles_per_process_row; j++) {
            SparseTile<INDEX_TYPE, VALUE_TYPE> &sp_tile =(*tile_map)[batch_id][ra][j];
            if (sp_tile.mode == 1) {
              SparseCacheEntry<VALUE_TYPE> newEntry;
              SparseCacheEntry<VALUE_TYPE> &cache_entry =(*(sp_tile.dataCachePtr))[index];
              if (cache_entry.cols.size() > 0) {
                for (int k = 0; k < cache_entry.cols.size(); k++) {
                  auto d = cache_entry.cols[k];
                  if (!this->hash_spgemm) {
                    (*(output->dense_collector))[index][d] += cache_entry.values[k];
                    continue;
                  }
                  value_map[d] += cache_entry.values[k];
                }
              }
              (*(sp_tile.dataCachePtr))[index] = newEntry;
            }
          }
        }
      }
      if (this->hash_spgemm) {
        vector<INDEX_TYPE> available_spots;
        for (auto k = 0; k < (*(output->sparse_data_collector))[index].size();k++) {
          auto d = (*(output->sparse_data_collector))[index][k].col;
          if (d > 0 and value_map.find(d) != value_map.end()) {
            (*(output->sparse_data_collector))[index][k].value +=
                (*(output->sparse_data_collector))[index][k].value +
                value_map[d];
            value_map.erase(d);
          } else {
            available_spots.push_back(k);
          }
        }
        if (value_map.size() > 0) {
          for (int k = 0; k < available_spots.size(); k++) {
            for (auto it = value_map.begin(); it != value_map.end(); ++it) {
              (*(output->sparse_data_collector))[index][available_spots[k]].col = (*it).first;
              (*(output->sparse_data_collector))[index][available_spots[k]].value = (*it).second;
              value_map.erase((*it).first);
              break;
            }
          }
          for (auto it = value_map.begin(); it != value_map.end(); ++it) {
            Tuple<VALUE_TYPE> tuple;
            tuple.col = (*it).first;
            tuple.value = (*it).second;
            (*(output->sparse_data_collector))[index].push_back(tuple);
          }
        }
      }
    }
  }
};
} // namespace distblas::algo
