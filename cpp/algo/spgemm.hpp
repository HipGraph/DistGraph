#pragma once
#include "algo.hpp"

using namespace std;
using namespace distblas::core;
using namespace distblas::net;
using namespace Eigen;

using namespace distblas::core;

namespace distblas::algo {
template <typename SPT, typename DENT, size_t embedding_dim>
class SpGEMMAlgo {

private:
  distblas::core::SpMat<DENT> *sparse_local_output;
  distblas::core::SpMat<DENT> *sparse_local;
  distblas::core::SpMat<SPT> *sp_local_receiver;
  distblas::core::SpMat<SPT> *sp_local_sender;
  distblas::core::SpMat<SPT> *sp_local_native;
  Process3DGrid *grid;

  std::unordered_map<int, unique_ptr<DataComm<SPT, DENT, embedding_dim>>> data_comm_cache;

  //record temp local output
  unique_ptr<vector<unordered_map<uint64_t,DENT>>> output_ptr;

  //cache size controlling hyper parameter
  double alpha = 0;

  //hyper parameter controls the  computation and communication overlapping
  double beta = 1.0;

  //hyper parameter controls the switching the sync vs async commiunication
  bool sync = true;

  //hyper parameter controls the col major or row major  data access
  bool col_major = false;

public:
  SpGEMMAlgo(distblas::core::SpMat<SPT> *sp_local_native,
             distblas::core::SpMat<SPT> *sp_local_receiver,
             distblas::core::SpMat<SPT> *sp_local_sender,
             distblas::core::SpMat<DENT> *sparse_local,
             distblas::core::SpMat<DENT> *sparse_local_output,
           Process3DGrid *grid, double alpha, double beta, bool col_major, bool sync_comm)
      : sp_local_native(sp_local_native), sp_local_receiver(sp_local_receiver),
        sp_local_sender(sp_local_sender), sparse_local(sparse_local), grid(grid),
        alpha(alpha), beta(beta),col_major(col_major),sync(sync_comm),
        sparse_local_output(sparse_local_output) {

    output_ptr =  make_unique<vector<unordered_map<uint64_t,DENT>>>(sparse_local->proc_row_width);

  }



  void algo_spgemm(int iterations, int batch_size, DENT lr) {
    auto t = start_clock();

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

    cout << " rank " << grid->rank_in_col << " total batches " << batches<< endl;

    // This communicator is being used for negative updates and in alpha > 0 to
    // fetch initial embeddings
    auto full_comm = unique_ptr<DataComm<SPT, DENT, embedding_dim>>(
        new DataComm<SPT, DENT, embedding_dim>(
            sp_local_receiver, sp_local_sender, sparse_local, grid, -1, alpha));
    full_comm.get()->onboard_data();

    // Buffer used for receive MPI operations data
    unique_ptr<std::vector<SpTuple<DENT,sp_tuple_max_dim>>> update_ptr = unique_ptr<std::vector<SpTuple<DENT,sp_tuple_max_dim>>>(new vector<SpTuple<DENT,sp_tuple_max_dim>>());

    //Buffer used for send MPI operations data
    unique_ptr<vector<SpTuple<DENT,sp_tuple_max_dim>>> sendbuf_ptr = unique_ptr<vector<SpTuple<DENT,sp_tuple_max_dim>>>(new vector<SpTuple<DENT,sp_tuple_max_dim>>());


    for (int i = 0; i < batches; i++) {
      auto communicator = unique_ptr<DataComm<SPT, DENT, embedding_dim>>(
          new DataComm<SPT, DENT, embedding_dim>(
              sp_local_receiver, sp_local_sender, sparse_local, grid, i, alpha));
      data_comm_cache.insert(std::make_pair(i, std::move(communicator)));
      data_comm_cache[i].get()->onboard_data();
    }

    cout << " rank " << grid->rank_in_col << " onboard_data completed " << batches << endl;

    // output is accumalated in a dense array for performance.It won't be an issue with tall and skinny
//    vector<unordered_map<int,DENT>> *prevCoordinates = sparse_local_output->sparse_data_collector.get();

    size_t total_memory = 0;

    CSRLocal<SPT> *csr_block =
        (col_major) ? (this->sp_local_receiver)->csr_local_data.get()
                    : (this->sp_local_native)->csr_local_data.get();

    int considering_batch_size = batch_size;

    for (int i = 0; i < iterations; i++) {

      for (int j = 0; j < batches; j++) {

        if (j == batches - 1) {
          considering_batch_size = last_batch_size;
        }


        // One process computations without MPI operations
        if (grid->col_world_size == 1) {
          // local computations for 1 process
          this->calc_t_dist_grad_rowptr(csr_block,  lr, j,
                                        batch_size, considering_batch_size,
                                        true,  0, 0,false);

        } else {

          //  pull model code
            this->execute_pull_model_computations(
                sendbuf_ptr.get(), update_ptr.get(), i, j,
                this->data_comm_cache[j].get(), csr_block, batch_size,
                considering_batch_size, lr,  1,
                true, 0, true, true);
            (sparse_local_output)->initialize_hashtables();
            this->execute_pull_model_computations(
                sendbuf_ptr.get(), update_ptr.get(), i, j,
                this->data_comm_cache[j].get(), csr_block, batch_size,
                considering_batch_size, lr,  1,
                true, 0, true,false);
        }
        total_memory += get_memory_usage();
      }
      (sparse_local)->purge_cache();
    }
    total_memory = total_memory / (iterations * batches);
    add_memory(total_memory, "Memory usage");
    stop_clock_and_add(t, "Total Time");
  }

  inline void execute_pull_model_computations(
      std::vector<SpTuple<DENT,sp_tuple_max_dim>> *sendbuf,
      std::vector<SpTuple<DENT,sp_tuple_max_dim>> *receivebuf, int iteration,
      int batch, DataComm<SPT, DENT, embedding_dim> *data_comm,
      CSRLocal<SPT> *csr_block, int batch_size, int considering_batch_size,
      double lr,  int comm_initial_start, bool local_execution,
      int first_execution_proc, bool communication, bool symbolic) {

    int proc_length = get_proc_length(beta, grid->col_world_size);
    int prev_start = comm_initial_start;

    for (int k = prev_start; k < grid->col_world_size; k += proc_length) {
      int end_process = get_end_proc(k, beta, grid->col_world_size);

      MPI_Request req;

      if (communication and symbolic) {
        data_comm->transfer_sparse_data(sendbuf, receivebuf,  iteration,
                                        batch, k, end_process);
      }
      if (k == comm_initial_start) {
        // local computation
        this->calc_t_dist_grad_rowptr(
            csr_block,  lr, batch, batch_size,
            considering_batch_size, local_execution,
            first_execution_proc, prev_start,symbolic);
      } else if (k > comm_initial_start) {
        int prev_end_process = get_end_proc(prev_start, beta, grid->col_world_size);

        this->calc_t_dist_grad_rowptr(csr_block,  lr, batch,
                                      batch_size, considering_batch_size, false,
                                       prev_start, prev_end_process,symbolic);
      }
      prev_start = k;
    }
    int prev_end_process = get_end_proc(prev_start, beta, grid->col_world_size);

    // updating last remote fetched data vectors
    this->calc_t_dist_grad_rowptr(csr_block,  lr, batch,
                                  batch_size, considering_batch_size,
                                  false,prev_start, prev_end_process,symbolic);
    // dense_local->invalidate_cache(i, j, true);
  }



  inline void calc_t_dist_grad_rowptr(CSRLocal<SPT> *csr_block,
                          DENT lr, int batch_id, int batch_size, int block_size,
                          bool local, int start_process,int end_process, bool symbolic) {

    auto source_start_index = batch_id * batch_size;
    auto source_end_index = std::min((batch_id + 1) * batch_size,
                                     this->sp_local_receiver->proc_row_width) -
                            1;

    auto dst_start_index =
        this->sp_local_receiver->proc_col_width * grid->rank_in_col;
    auto dst_end_index =
        std::min(static_cast<uint64_t>(this->sp_local_receiver->proc_col_width *
                                       (grid->rank_in_col + 1)),
                 this->sp_local_receiver->gCols) -
        1;

    if (local) {
        calc_embedding_row_major(source_start_index, source_end_index,
                                 dst_start_index, dst_end_index, csr_block,
                                  lr, batch_id, batch_size,
                                 block_size,symbolic);
    } else {
      for (int r = start_process; r < end_process; r++) {

        if (r != grid->rank_in_col) {

          int computing_rank = (grid->rank_in_col >= r)
                                   ? (grid->rank_in_col - r) % grid->col_world_size
                                   : (grid->col_world_size - r + grid->rank_in_col) % grid->col_world_size;

          dst_start_index = this->sp_local_receiver->proc_row_width * computing_rank;
          dst_end_index =
              std::min(static_cast<uint64_t>(
                           this->sp_local_receiver->proc_row_width * (computing_rank + 1)),
                       this->sp_local_receiver->gCols) -
              1;

            calc_embedding_row_major(source_start_index, source_end_index,
                                     dst_start_index, dst_end_index, csr_block,
                                      lr, batch_id, batch_size,
                                     block_size,symbolic);
        }
      }
    }
  }

  inline void calc_embedding_row_major(uint64_t source_start_index,
                           uint64_t source_end_index, uint64_t dst_start_index,
                           uint64_t dst_end_index, CSRLocal<SPT> *csr_block,DENT lr, int batch_id,
                           int batch_size, int block_size, bool symbolic) {
    if (csr_block->handler != nullptr) {
      CSRHandle *csr_handle = csr_block->handler.get();


      #pragma omp parallel for schedule(static) // enable for full batch training or // batch size larger than 1000000
      for (uint64_t i = source_start_index; i <= source_end_index; i++) {

        uint64_t index = i - batch_id * batch_size;


        for (uint64_t j = static_cast<uint64_t>(csr_handle->rowStart[i]);
             j < static_cast<uint64_t>(csr_handle->rowStart[i + 1]); j++) {
          auto dst_id = csr_handle->col_idx[j];
          if (dst_id >= dst_start_index and dst_id < dst_end_index) {
            uint64_t local_dst =
                dst_id - (grid)->rank_in_col *
                             (this->sp_local_receiver)->proc_col_width;
            int target_rank =
                (int)(dst_id / (this->sp_local_receiver)->proc_col_width);
            bool fetch_from_cache =
                target_rank == (grid)->rank_in_col ? false : true;

            vector<uint64_t> remote_cols;
            vector<DENT> remote_values;

            if (fetch_from_cache) {
              unordered_map<uint64_t, SparseCacheEntry<DENT>>
                  &arrayMap = (*sparse_local->tempCachePtr)[target_rank];
              remote_cols = arrayMap[dst_id].cols;
              remote_values =arrayMap[dst_id].values;
            }

            CSRHandle *handle = ((sparse_local)->csr_local_data)->handler.get();
            uint64_t ht_size = (*(sparse_local_output->sparse_data_collector))[index].size();
            if (!fetch_from_cache) {
              int count = handle->rowStart[local_dst+1]- handle->rowStart[local_dst];
              if (symbolic) {
                auto val =(*(sparse_local_output->sparse_data_counter))[index] +count;
                (*(sparse_local_output->sparse_data_counter))[index] =std::min(val, embedding_dim);
              }else {
                for (auto k = handle->rowStart[local_dst]; k < handle->rowStart[local_dst + 1]; k++) {
                   auto  d = (handle->col_idx[k]);
                   uint64_t hash = (d*hash_scale) & (ht_size-1);
                   auto value =  lr *handle->values[k];
                   while(1){
                     if ((*(sparse_local_output->sparse_data_collector))[index][hash].first==d){
                       (*(sparse_local_output->sparse_data_collector))[index][hash].second = (*(sparse_local_output->sparse_data_collector))[index][hash].second + value;
                       break;
                     }else if ((*(sparse_local_output->sparse_data_collector))[index][hash].first==-1){
                       (*(sparse_local_output->sparse_data_collector))[index][hash].first = d;
                       (*(sparse_local_output->sparse_data_collector))[index][hash].second =   value;
                       break;
                     }else {
                       hash = (hash+1)& (ht_size-1);
                     }
                   }
                }
              }
            }else{
              int count = remote_cols.size();
              if (symbolic){
                auto val  = (*(sparse_local_output->sparse_data_counter))[index]+ count;
                (*(sparse_local_output->sparse_data_counter))[index] = std::min(val,embedding_dim);
              }else {
                for (int m = 0; m < remote_cols.size(); m++) {
                  auto d = remote_cols[m];
                  auto value =  lr *remote_values[m];
                  uint64_t hash = (d*hash_scale) & (ht_size-1);
                  while (1) {
                    if ((*(sparse_local_output->sparse_data_collector))[index][hash].first == d) {
                      (*(sparse_local_output->sparse_data_collector))[index][hash].second = (*(sparse_local_output->sparse_data_collector))[index][hash].second + value;
                      break;
                    } else if ((*(sparse_local_output->sparse_data_collector))[index][hash].first ==-1) {
                      (*(sparse_local_output->sparse_data_collector))[index][hash].first = d;
                      (*(sparse_local_output->sparse_data_collector))[index][hash].second = value;
                      break;
                    } else {
                      hash =(hash + 1) &(ht_size -1);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
};
} // namespace distblas::algo
