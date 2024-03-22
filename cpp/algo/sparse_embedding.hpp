#pragma once
#include "../core/sparse_mat_tile.hpp"
#include "../net/tile_based_data_comm.hpp"
#include <queue>


using namespace std;
using namespace distblas::core;
using namespace distblas::net;
using namespace Eigen;

using namespace distblas::core;

namespace distblas::algo {

template <typename INDEX_TYPE, typename VALUE_TYPE>
struct index_value_pair {
  INDEX_TYPE index;
  VALUE_TYPE value;
};

template <typename INDEX_TYPE, typename VALUE_TYPE>
struct MIN_HEAP_OPERATOR {
  bool operator()(const index_value_pair<INDEX_TYPE, VALUE_TYPE>& a, const index_value_pair<INDEX_TYPE, VALUE_TYPE>& b) const {
    return a.value > b.value; // Change to a.value > b.value for max heap
  }
};


template <typename INDEX_TYPE, typename VALUE_TYPE, size_t embedding_dim>
class SparseEmbedding {

private:
  distblas::core::SpMat<VALUE_TYPE> *sparse_local_output;
  distblas::core::SpMat<VALUE_TYPE> *sparse_local;
  distblas::core::SpMat<VALUE_TYPE> *sp_local_receiver;
  distblas::core::SpMat<VALUE_TYPE> *sp_local_sender;
  distblas::core::SpMat<VALUE_TYPE> *sp_local_native;
  Process3DGrid *grid;

  // record temp local output
  unique_ptr<vector<unordered_map<INDEX_TYPE, VALUE_TYPE>>> output_ptr;

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
  SparseEmbedding(distblas::core::SpMat<VALUE_TYPE> *sp_local_native,
                  distblas::core::SpMat<VALUE_TYPE> *sp_local_receiver,
                  distblas::core::SpMat<VALUE_TYPE> *sp_local_sender,
                  distblas::core::SpMat<VALUE_TYPE> *sparse_local_output,
                  Process3DGrid *grid, double alpha, double beta,
                  bool col_major, bool sync_comm, double tile_width_fraction,
                  bool hash_spgemm)
      : sp_local_native(sp_local_native), sp_local_receiver(sp_local_receiver),
        sp_local_sender(sp_local_sender),grid(grid), alpha(alpha),
        beta(beta), col_major(col_major),
        sync(sync_comm), sparse_local_output(sparse_local_output),sparse_local(sparse_local_output),
        tile_width_fraction(tile_width_fraction) {
    this->hash_spgemm = hash_spgemm;
  }

  void algo_sparse_embedding(int iterations, int batch_size,int ns, VALUE_TYPE lr,double density=1.0, bool enable_remote=true) {
    auto t = start_clock();
    size_t total_memory = 0;
    int batches = 0;
    int last_batch_size = batch_size;

//    auto sparse_input = make_unique<distblas::core::SpMat<VALUE_TYPE>>(grid,sp_local_receiver->proc_row_width,embedding_dim,hash_spgemm,true);

    auto t_knn = start_clock();
    auto expected_nnz_per_row = static_cast<int>(embedding_dim*density);
//    this->preserveHighestK(this->sparse_local->dense_collector.get(),expected_nnz_per_row);
    (this->sparse_local)->initialize_CSR_blocks(false,nullptr,static_cast<VALUE_TYPE>(INT_MIN),false);
    stop_clock_and_add(t, "KNN Time");

    if (sp_local_receiver->proc_row_width % batch_size == 0) {
      batches = static_cast<int>(sp_local_receiver->proc_row_width / batch_size);
    } else {
      batches = static_cast<int>(sp_local_receiver->proc_row_width / batch_size) + 1;
      last_batch_size = sp_local_receiver->proc_row_width - batch_size * (batches - 1);
    }

    cout << " rank " << grid->rank_in_col << " total batches " << batches<< endl;

    // This communicator is being used for negative updates and in alpha > 0 to
    // fetch initial embeddings
    auto main_comm =unique_ptr<TileDataComm<INDEX_TYPE, VALUE_TYPE, embedding_dim>>
        (new TileDataComm<INDEX_TYPE, VALUE_TYPE, embedding_dim>(sp_local_receiver, sp_local_sender, sparse_local, grid, alpha,
                batches, tile_width_fraction, hash_spgemm, true));

    // Buffer used for receive MPI operations data
    unique_ptr<std::vector<SpTuple<VALUE_TYPE, sp_tuple_max_dim>>> update_ptr =
        unique_ptr<std::vector<SpTuple<VALUE_TYPE, sp_tuple_max_dim>>>(
            new vector<SpTuple<VALUE_TYPE, sp_tuple_max_dim>>());

    // Buffer used for send MPI operations data
    unique_ptr<vector<SpTuple<VALUE_TYPE, sp_tuple_max_dim>>> sendbuf_ptr =
        unique_ptr<vector<SpTuple<VALUE_TYPE, sp_tuple_max_dim>>>(
            new vector<SpTuple<VALUE_TYPE, sp_tuple_max_dim>>());

//    this->sparse_local->build_computable_represention();
    main_comm.get()->onboard_data(enable_remote);
    cout << " rank " << grid->rank_in_col << " on board data completed "<< endl;

    int total_tiles = SparseTile<INDEX_TYPE, VALUE_TYPE>::get_tiles_per_process_row();

    CSRLocal<VALUE_TYPE> *csr_block =
        (col_major) ? (this->sp_local_receiver)->csr_local_data.get()
                    : (this->sp_local_native)->csr_local_data.get();

    int considering_batch_size = batch_size;

    for (int i = 0; i < iterations; i++) {
      for (int j = 0; j < batches; j++) {
//        cout << " rank " << grid->rank_in_col << " batch " << j << endl;
        int seed = j + i;
        if (j == batches - 1) {
          considering_batch_size = last_batch_size;
        }
        vector<INDEX_TYPE> random_number_vec = generate_random_numbers(
            0, (this->sp_local_receiver)->gRows, seed, ns);
        // One process computations without MPI operations
        if (this->grid->col_world_size == 1) {
          // local computations for 1 process
          this->calc_t_dist_grad_rowptr(
              csr_block, lr, i, j, batch_size, considering_batch_size, 0, 0, 0,
              false, main_comm.get(), this->sparse_local_output);

        } else {
          //          if ((this->sparse_local_output)->hash_spgemm) {
          //            this->execute_pull_model_computations(
          //                sendbuf_ptr.get(), update_ptr.get(), i, j,
          //                main_comm.get(), csr_block, batch_size,
          //                considering_batch_size, lr, 1, 0, true, true,
          //                this->sparse_local_output);
          //
          //            (this->sparse_local_output)->initialize_hashtables();
          //
          //            // compute remote computations
          //            this->calc_t_dist_grad_rowptr(
          //                (this->sp_local_sender)->csr_local_data.get(), lr,
          //                i, j, batch_size, considering_batch_size, 2, 0,
          //                this->grid->col_world_size, true, main_comm.get(),
          //                nullptr);
          //          }
          main_comm->transfer_sparse_data(random_number_vec,i,j);
          this->calc_t_dist_replus_rowptr( random_number_vec,
                                          lr, j, batch_size,
                                          considering_batch_size,this->sparse_local_output);
          this->execute_pull_model_computations(
              sendbuf_ptr.get(), update_ptr.get(), i, j, main_comm.get(),
              csr_block, batch_size, considering_batch_size, lr, 1, 0, true,
              false, this->sparse_local_output,enable_remote);

          if (enable_remote) {
            this->calc_t_dist_grad_rowptr(
                (this->sp_local_sender)->csr_local_data.get(), lr, i, j,
                batch_size, considering_batch_size, 2, 0,
                this->grid->col_world_size, false, main_comm.get(), nullptr);

            main_comm->receive_remotely_computed_data(
                sendbuf_ptr.get(), update_ptr.get(), i, j, 0,
                this->grid->col_world_size, 0, total_tiles);

            this->merge_remote_computations(
                j, batch_size, this->sparse_local_output, main_comm.get());
          }
        }
        total_memory += get_memory_usage();
      }
      if (i<iterations-1) {
        auto t_knn = start_clock();
        this->sparse_local_output->initialize_CSR_blocks(false, nullptr, static_cast<VALUE_TYPE>(INT_MIN), false);
        size_t size_r =this->sparse_local_output->csr_local_data->handler->rowStart.size();
        double output_nnz = this->sparse_local_output->csr_local_data->handler->rowStart[size_r - 1];
        auto output_nnz_per_row = static_cast<int>(output_nnz / this->sp_local_receiver->proc_row_width);
        cout << " rank " << grid->rank_in_col << "iteration " << i<< " expected nnz per row " << expected_nnz_per_row<< " output nnz per row" << output_nnz_per_row << endl;
        if (output_nnz_per_row > expected_nnz_per_row) {
          this->preserveHighestK(this->sparse_local_output->dense_collector.get(),expected_nnz_per_row, static_cast<VALUE_TYPE>(INT_MIN));
        }
        stop_clock_and_add(t_knn, "KNN Time");
      }else if (i==iterations-1) {
        auto t_knn = start_clock();
        this->sparse_local_output->initialize_CSR_blocks(false, nullptr, static_cast<VALUE_TYPE>(INT_MIN), true);
        stop_clock_and_add(t_knn, "KNN Time");
      }
      (this->sparse_local)->purge_cache();
    }
    total_memory = total_memory / (iterations * batches);
    add_perf_stats(total_memory, "Memory usage");
    stop_clock_and_add(t, "Total Time");
  }

  inline void execute_pull_model_computations(
      std::vector<SpTuple<VALUE_TYPE, sp_tuple_max_dim>> *sendbuf,
      std::vector<SpTuple<VALUE_TYPE, sp_tuple_max_dim>> *receivebuf,
      int iteration, int batch,
      TileDataComm<INDEX_TYPE, VALUE_TYPE, embedding_dim> *main_comm,
      CSRLocal<VALUE_TYPE> *csr_block, int batch_size,
      int considering_batch_size, double lr, int comm_initial_start,
      int first_execution_proc, bool communication, bool symbolic,
      DistributedMat *output, bool enable_remote_computation) {

    int proc_length = get_proc_length(beta, this->grid->col_world_size);
    int prev_start = comm_initial_start;

    auto tiles_per_process =
        SparseTile<INDEX_TYPE, VALUE_TYPE>::get_tiles_per_process_row();
    for (int k = prev_start; k < this->grid->col_world_size; k += proc_length) {
      int end_process = get_end_proc(k, beta, this->grid->col_world_size);

      MPI_Request req;

      if (communication and (symbolic or !output->hash_spgemm)) {
        main_comm->transfer_sparse_data(sendbuf, receivebuf, iteration, batch,
                                        k, end_process, 0, tiles_per_process,
                                        true);
        if (enable_remote_computation) {
          main_comm->transfer_remotely_computable_data(
              sendbuf, receivebuf, iteration, batch, k, end_process, 0,
              tiles_per_process, true);
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

    // dense_local->invalidate_cache(i, j, true);
  }

  inline void calc_t_dist_grad_rowptr(
      CSRLocal<VALUE_TYPE> *csr_block, VALUE_TYPE lr, int itr, int batch_id,
      int batch_size, int block_size, int mode, int start_process,
      int end_process, bool symbolic,
      TileDataComm<INDEX_TYPE, VALUE_TYPE, embedding_dim> *main_com,
      DistributedMat *output) {
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
            if ((*main_com
                      ->receiver_proc_tile_map)[batch_id][computing_rank][tile]
                    .mode == 0) {
              auto source_start_index =
                  (*main_com->receiver_proc_tile_map)[batch_id][computing_rank]
                                                     [tile]
                                                         .row_starting_index;
              auto source_end_index =
                  (*main_com->receiver_proc_tile_map)[batch_id][computing_rank]
                                                     [tile]
                                                         .row_end_index;
              auto dst_start_index =
                  (*main_com->receiver_proc_tile_map)[batch_id][computing_rank]
                                                     [tile]
                                                         .col_start_index;
              auto dst_end_index =
                  (*main_com->receiver_proc_tile_map)[batch_id][computing_rank]
                                                     [tile]
                                                         .col_end_index;
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

        INDEX_TYPE index =
            (mode == 0 or mode == 1) ? i : i - source_start_index;
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
            bool fetch_from_cache =
                (target_rank == (this->grid)->rank_in_col or mode == 2) ? false
                                                                        : true;

            vector<INDEX_TYPE> remote_cols;
            vector<VALUE_TYPE> remote_values;

            if (fetch_from_cache) {
              unordered_map<INDEX_TYPE, SparseCacheEntry<VALUE_TYPE>>
                  &arrayMap = (*this->sparse_local->tempCachePtr)[target_rank];
              if (arrayMap.find(dst_id)== arrayMap.end()){
                cout<<" rank "<<this->grid->rank_in_col<<" trying to access "<<dst_id<<" failed "<<endl;
              }
              remote_cols = arrayMap[dst_id].cols;
              remote_values = arrayMap[dst_id].values;
            }

            if (!fetch_from_cache) {
              CSRHandle local_handle = this->sparse_local->fetch_local_data(index,true,static_cast<INDEX_TYPE>(INT_MIN));
              CSRHandle remote_handle = this->sparse_local->fetch_local_data(local_dst,true,static_cast<INDEX_TYPE>(INT_MIN));
                int local_count = (mode==2)?(*(output->dataCachePtr))[index].cols.size():local_handle.col_idx.size();
                int remote_count = remote_handle.col_idx.size();
                int total_count = local_count + remote_count;
                int remote_tracker = 0;
                int remote_tracker_end = remote_count;
                int local_tracker = 0;
                int local_tracker_end = (mode==2)?(*(output->dataCachePtr))[index].cols.size():local_count;
                int count = 0;
                VALUE_TYPE attrc=0;
                vector<INDEX_TYPE> indexes_to_updates;
                vector<VALUE_TYPE> values_to_updates;
                while (count < total_count) {
                  auto local_d = (local_tracker < local_tracker_end)
                                     ? (mode==2)?(*(output->dataCachePtr))[index].cols[local_tracker]:local_handle.col_idx.size();
                                     : INT_MAX;
                  auto remote_d = (remote_tracker < remote_tracker_end)
                                      ? remote_handle.col_idx[remote_tracker]
                                      : INT_MAX;
                  if (local_d == INT_MAX and remote_d == INT_MAX) {
                    break;
                  } else if (remote_d == INT_MAX or local_d < remote_d) {
                    auto local_value = mode==2?(*(output->dataCachePtr))[index].values[local_tracker]:local_handle.values[local_tracker];
                     attrc += local_value * local_value;
                    indexes_to_updates.push_back(local_d);
                    values_to_updates.push_back(local_value);
                    local_tracker++;
                    count++;
                  } else if (local_d == INT_MAX or remote_d < local_d) {
                    auto remote_value = remote_handle.values[remote_tracker];
                     attrc += remote_value * remote_value;
                    indexes_to_updates.push_back(remote_d);
                    values_to_updates.push_back(-1*remote_value);
                    remote_tracker++;
                    count++;
                  } else {
                    auto local_value = mode==2?(*(output->dataCachePtr))[index].values[local_tracker]:local_handle.values[local_tracker];
                    auto remote_value = remote_handle.values[remote_tracker];
                    VALUE_TYPE value = local_value - remote_value;
                     attrc += value * value;
                    indexes_to_updates.push_back(remote_d);
                    values_to_updates.push_back(value);
                    local_tracker++;
                    remote_tracker++;
                    count = count + 2;
                  }
                }
                VALUE_TYPE d1 = -2.0 / (1.0 + attrc);

                for(INDEX_TYPE i=0;i<indexes_to_updates.size();i++){
                  VALUE_TYPE l = scale(values_to_updates[i] * d1);
                  (*(output->dense_collector))[index][indexes_to_updates[i]] += (lr)*l;
                }
            } else {
              int count = remote_cols.size();
              CSRHandle local_handle = this->sparse_local->fetch_local_data(index,true,static_cast<INDEX_TYPE>(INT_MIN));
                int local_count = local_handle.col_idx.size();
                int remote_count = remote_cols.size();
                int total_count = local_count + remote_count;
                int remote_tracker = 0;
                int remote_tracker_end = remote_cols.size();
                int local_tracker = 0;
                int local_tracker_end = local_count;
                int count = 0;
                vector<INDEX_TYPE> indexes_to_updates;
                vector<VALUE_TYPE> values_to_updates;
                VALUE_TYPE attrc=0;
                while (count < total_count) {
                  auto local_d = (local_tracker < local_tracker_end)
                                     ? local_handle.col_idx[local_tracker]
                                     : INT_MAX;
                  auto remote_d = (remote_tracker < remote_tracker_end)
                                      ? remote_cols[remote_tracker]
                                      : INT_MAX;
                  if (local_d == INT_MAX and remote_d == INT_MAX) {
                    break;
                  } else if (remote_d == INT_MAX or local_d < remote_d) {
                    auto local_value = local_handle.values[local_tracker];
                     attrc += local_value * local_value;
                     indexes_to_updates.push_back(local_d);
                     values_to_updates.push_back(local_value);
                    local_tracker++;
                    count++;
                  } else if (local_d == INT_MAX or remote_d < local_d) {
                    auto remote_value = remote_values[remote_tracker];
                     attrc += remote_value * remote_value;
                    indexes_to_updates.push_back(remote_d);
                    values_to_updates.push_back(-1*remote_value);
                    remote_tracker++;
                    count++;
                  } else {
                    auto local_value = local_handle.values[local_tracker];
                    auto remote_value = remote_values[remote_tracker];
                    VALUE_TYPE value = local_value - remote_value;
                     attrc += value * value;
                     indexes_to_updates.push_back(remote_d);
                     values_to_updates.push_back(value);
                    local_tracker++;
                    remote_tracker++;
                    count = count + 2;
                  }
                }
                VALUE_TYPE d1 = -2.0 / (1.0 + attrc);

                for(INDEX_TYPE i=0;i<indexes_to_updates.size();i++){
                  VALUE_TYPE l = scale(values_to_updates[i] * d1);
                  (*(output->dense_collector))[index][indexes_to_updates[i]] += (lr)*l;
                }
              }
            }
          }
        }
      }
  }

  inline void calc_t_dist_replus_rowptr(vector<INDEX_TYPE> &col_ids, VALUE_TYPE lr,
                                        int batch_id, int batch_size,
                                        int block_size, DistributedMat * output) {

    int row_base_index = batch_id * batch_size;

#pragma omp parallel for schedule(static)
    for (int i = 0; i < block_size; i++) {
      INDEX_TYPE row_id = static_cast<INDEX_TYPE>(i + row_base_index);
      for (int j = 0; j < col_ids.size(); j++) {
        INDEX_TYPE global_col_id = col_ids[j];
        bool fetch_from_cache = false;
        INDEX_TYPE local_col_id =
            global_col_id -
            static_cast<INDEX_TYPE>(((grid)->rank_in_col *
                                     (this->sp_local_receiver)->proc_row_width));
        int owner_rank = static_cast<int>(
            global_col_id / (this->sp_local_receiver)->proc_row_width);

        if (owner_rank != (grid)->rank_in_col) {
          fetch_from_cache = true;
        }
        vector<INDEX_TYPE> remote_cols;
        vector<VALUE_TYPE> remote_values;

        if (fetch_from_cache) {
          unordered_map<INDEX_TYPE, SparseCacheEntry<VALUE_TYPE>>
              &arrayMap = (*this->sparse_local->tempCachePtr)[owner_rank];
          if (arrayMap.find(global_col_id)== arrayMap.end()){
            cout<<" rank "<<this->grid->rank_in_col<<" trying to access "<<global_col_id<<" failed "<<endl;
          }
          remote_cols = arrayMap[global_col_id].cols;
          remote_values = arrayMap[global_col_id].values;
        }

        if (fetch_from_cache) {
          CSRHandle local_handle = ((this->sparse_local)->fetch_local_data(row_id,true,static_cast<INDEX_TYPE>(INT_MIN)));
          int local_count =local_handle.col_idx.size();
          int remote_count = remote_cols.size();
          int total_count = local_count + remote_count;
          int remote_tracker = 0;
          int remote_tracker_end = remote_cols.size();
          int local_tracker = 0;
          int local_tracker_end = local_count;
          int count = 0;
          VALUE_TYPE repuls=0;
          vector<INDEX_TYPE> indexs_to_updates;
          vector<VALUE_TYPE> values_to_updates;
          while (count < total_count) {
            auto local_d = (local_tracker < local_tracker_end)
                               ? local_handle.col_idx[local_tracker]
                               : INT_MAX;
            auto remote_d = (remote_tracker < remote_tracker_end)
                                ? remote_cols[remote_tracker]
                                : INT_MAX;
            if (local_d == INT_MAX and remote_d == INT_MAX) {
              break;
            } else if (remote_d == INT_MAX or local_d < remote_d) {
              auto local_value = local_handle.values[local_tracker];
               repuls += local_value * local_value;
              indexs_to_updates.push_back(local_d);
              values_to_updates.push_back(local_value);
              local_tracker++;
              count++;
            } else if (local_d == INT_MAX or remote_d < local_d) {
              auto remote_value = remote_values[remote_tracker];
               repuls += remote_value * remote_value;
              indexs_to_updates.push_back(remote_d);
              values_to_updates.push_back(-1*remote_value);
              remote_tracker++;
              count++;
            } else {
              auto local_value = local_handle.values[local_tracker];
              auto remote_value = remote_values[remote_tracker];
              VALUE_TYPE value = local_value - remote_value;
               repuls += value * value;
              indexs_to_updates.push_back(remote_d);
              values_to_updates.push_back(value);
              local_tracker++;
              remote_tracker++;
              count = count + 2;
            }
          }
          VALUE_TYPE d1 = 2.0 / ((repuls + 0.000001) * (1.0 + repuls));
          for(INDEX_TYPE i=0;i<indexs_to_updates.size();i++){
            VALUE_TYPE l = scale(values_to_updates[i] * d1);
            (*(output->dense_collector))[row_id][indexs_to_updates[i]] += (lr)*l;
          }

        } else {
          CSRHandle handle = ((this->sparse_local)->fetch_local_data(local_col_id,true,static_cast<INDEX_TYPE>(INT_MIN)));
          CSRHandle local_handle = ((this->sparse_local)->fetch_local_data(local_col_id,true,static_cast<INDEX_TYPE>(INT_MIN)));
          int local_count = local_handle.col_idx.size();
          int remote_count = handle.col_idx.size();
          int total_count = local_count + remote_count;
          int remote_tracker = 0;
          int remote_tracker_end = remote_count;
          int local_tracker =0;
          int local_tracker_end = local_handle.col_idx.size();
          int count = 0;
          VALUE_TYPE  repuls=0;
          vector<VALUE_TYPE> values_to_updates;
          vector<INDEX_TYPE> indexs_to_updates;
          while (count < total_count) {
            auto local_d = (local_tracker < local_tracker_end)
                               ? local_handle.col_idx[local_tracker]
                               : INT_MAX;
            auto remote_d = (remote_tracker < remote_tracker_end)
                                ? handle.col_idx[remote_tracker]
                                : INT_MAX;
            if (local_d == INT_MAX and remote_d == INT_MAX) {
              break;
            } else if (remote_d == INT_MAX or local_d < remote_d) {
              auto local_value =local_handle.values[local_tracker];
               repuls += local_value * local_value;
              indexs_to_updates.push_back(local_d);
              values_to_updates.push_back(local_value);
              local_tracker++;
              count++;
            } else if (local_d == INT_MAX or remote_d < local_d) {
              auto remote_value = handle.values[remote_tracker];
               repuls += remote_value * remote_value;
              indexs_to_updates.push_back(remote_d);
              values_to_updates.push_back(-1*remote_value);
              remote_tracker++;
              count++;
            } else {
              auto local_value =local_handle.values[local_tracker];
              auto remote_value = handle.values[remote_tracker];
              VALUE_TYPE value = local_value - remote_value;
              repuls += value * value;
              indexs_to_updates.push_back(remote_d);
              values_to_updates.push_back(value);
              local_tracker++;
              remote_tracker++;
              count = count + 2;
            }
          }
          VALUE_TYPE d1 = 2.0 / ((repuls + 0.000001) * (1.0 + repuls));
          for(INDEX_TYPE i=0;i<indexs_to_updates.size();i++){
            VALUE_TYPE l = scale(values_to_updates[i] * d1);
            (*(output->dense_collector))[row_id][indexs_to_updates[i]] += (lr)*l;
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

    vector<vector<vector<SparseTile<INDEX_TYPE, VALUE_TYPE>>>> *tile_map =
        main_comm->receiver_proc_tile_map.get();
    int tiles_per_process_row =
        SparseTile<INDEX_TYPE, VALUE_TYPE>::get_tiles_per_process_row();
#pragma omp parallel for schedule(static)
    for (auto i = source_start_index; i < source_end_index; i++) {
      INDEX_TYPE index = i - source_start_index;
      unordered_map<INDEX_TYPE, VALUE_TYPE> value_map;
      for (int ra = 0; ra < this->grid->col_world_size; ra++) {
        for (int j = 0; j < tiles_per_process_row; j++) {
          SparseTile<INDEX_TYPE, VALUE_TYPE> &sp_tile =
              (*tile_map)[batch_id][ra][j];
          if (sp_tile.mode == 1) {
            SparseCacheEntry<VALUE_TYPE> newEntry;
            SparseCacheEntry<VALUE_TYPE> &cache_entry =
                (*(sp_tile.dataCachePtr))[index];
            if (cache_entry.cols.size() > 0) {
              for (int k = 0; k < cache_entry.cols.size(); k++) {
                value_map[k] += cache_entry.values[k];
                if (!this->hash_spgemm) {
                  (*(output->dense_collector))[index][k] +=
                      cache_entry.values[k];
                }
              }
            }
            (*(sp_tile.dataCachePtr))[index] = newEntry;
          }
        }
      }
      if (this->hash_spgemm) {
        vector<INDEX_TYPE> available_spots;
        for (auto k = 0; k < (*(output->sparse_data_collector))[index].size();
             k++) {
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
              (*(output->sparse_data_collector))[index][available_spots[k]]
                  .col = (*it).first;
              (*(output->sparse_data_collector))[index][available_spots[k]]
                  .value = (*it).second;
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

  VALUE_TYPE scale(VALUE_TYPE v) {
    if (v > MAX_BOUND)
      return MAX_BOUND;
    else if (v < -MAX_BOUND)
      return -MAX_BOUND;
    else
      return v;
  }


  void preserveHighestK(vector<vector<VALUE_TYPE>> *matrix,  int k, VALUE_TYPE nullify_value) {
    // Check if index is within bounds

    #pragma omp parallel for
    for(auto i=0;i<(*matrix).size();i++) {
      std::priority_queue<index_value_pair<INDEX_TYPE,VALUE_TYPE>,vector<index_value_pair<INDEX_TYPE,VALUE_TYPE>>,MIN_HEAP_OPERATOR<INDEX_TYPE,VALUE_TYPE>> queue;
      for(auto j=0;j<(*matrix)[i].size();j++){
        index_value_pair<INDEX_TYPE,VALUE_TYPE> a;
        a.index = j;
        a.value = (*matrix)[i][j];
        queue.push(a);
      }
      auto nullify_count = embedding_dim - k;
      auto count=0;
      while(count<nullify_count){
        index_value_pair<INDEX_TYPE,VALUE_TYPE> a =  queue.top();
        (*matrix)[i][a.index]=nullify_value;
        queue.pop();
        count++;
      }
    }
  }
};
} // namespace distblas::algo
