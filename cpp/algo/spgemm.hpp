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
        alpha(alpha), beta(beta),col_major(col_major),sync(sync_comm),sparse_local_output(sparse_local_output) {}



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
    unique_ptr<std::vector<SpTuple<DENT,embedding_dim>>> update_ptr = unique_ptr<std::vector<SpTuple<DENT,embedding_dim>>>(new vector<SpTuple<DENT,embedding_dim>>());

    //Buffer used for send MPI operations data
    unique_ptr<vector<SpTuple<DENT,embedding_dim>>> sendbuf_ptr = unique_ptr<vector<SpTuple<DENT,embedding_dim>>>(new vector<SpTuple<DENT,embedding_dim>>());


    for (int i = 0; i < batches; i++) {
      auto communicator = unique_ptr<DataComm<SPT, DENT, embedding_dim>>(
          new DataComm<SPT, DENT, embedding_dim>(
              sp_local_receiver, sp_local_sender, sparse_local, grid, i, alpha));
      data_comm_cache.insert(std::make_pair(i, std::move(communicator)));
      data_comm_cache[i].get()->onboard_data();
    }

    cout << " rank " << grid->rank_in_col << " onboard_data completed " << batches << endl;

    DENT *prevCoordinates = static_cast<DENT *>(
        ::operator new(sizeof(DENT[batch_size * embedding_dim])));

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
          for (int k = 0; k < batch_size; k += 1) {
            int IDIM = k * embedding_dim;
            for (int d = 0; d < embedding_dim; d++) {
              prevCoordinates[IDIM + d] = 0;
            }
          }
          // local computations for 1 process
          this->calc_t_dist_grad_rowptr(csr_block, prevCoordinates, lr, j,
                                        batch_size, considering_batch_size,
                                        true,  0, 0);

//          this->update_data_matrix_rowptr(prevCoordinates, j, batch_size);

        } else {

          //  pull model code
            this->execute_pull_model_computations(
                sendbuf_ptr.get(), update_ptr.get(), i, j,
                this->data_comm_cache[j].get(), csr_block, batch_size,
                considering_batch_size, lr, prevCoordinates, 1,
                true, 0, true);
//            this->update_data_matrix_rowptr(prevCoordinates, j, batch_size);

//            for (int k = 0; k < batch_size; k += 1) {
//              int IDIM = k * embedding_dim;
//              for (int d = 0; d < embedding_dim; d++) {
//                prevCoordinates[IDIM + d] = 0;
//              }
//            }
        }
        total_memory += get_memory_usage();
      }
    }
    total_memory = total_memory / (iterations * batches);
    add_memory(total_memory, "Memory usage");
    stop_clock_and_add(t, "Total Time");
  }

  inline void execute_pull_model_computations(
      std::vector<SpTuple<DENT,embedding_dim>> *sendbuf,
      std::vector<SpTuple<DENT,embedding_dim>> *receivebuf, int iteration,
      int batch, DataComm<SPT, DENT, embedding_dim> *data_comm,
      CSRLocal<SPT> *csr_block, int batch_size, int considering_batch_size,
      double lr, DENT *prevCoordinates, int comm_initial_start, bool local_execution,
      int first_execution_proc, bool communication) {

    int proc_length = get_proc_length(beta, grid->col_world_size);
    int prev_start = comm_initial_start;

    for (int k = prev_start; k < grid->col_world_size; k += proc_length) {
      int end_process = get_end_proc(k, beta, grid->col_world_size);

      MPI_Request req;

      if (communication) {
        auto t = start_clock();
        data_comm->transfer_sparse_data(sendbuf, receivebuf,  iteration,
                                        batch, k, end_process);
        stop_clock_and_add(t, "Compute  Local");
      }
      if (k == comm_initial_start) {
        // local computation
        this->calc_t_dist_grad_rowptr(
            csr_block, prevCoordinates, lr, batch, batch_size,
            considering_batch_size, local_execution,
            first_execution_proc, prev_start);
      } else if (k > comm_initial_start) {
        int prev_end_process = get_end_proc(prev_start, beta, grid->col_world_size);

        this->calc_t_dist_grad_rowptr(csr_block, prevCoordinates, lr, batch,
                                      batch_size, considering_batch_size, false,
                                       prev_start, prev_end_process);
      }
      prev_start = k;
    }
    int prev_end_process = get_end_proc(prev_start, beta, grid->col_world_size);

    // updating last remote fetched data vectors
    auto t = start_clock();
    this->calc_t_dist_grad_rowptr(csr_block, prevCoordinates, lr, batch,
                                  batch_size, considering_batch_size,
                                  false,prev_start, prev_end_process);
    stop_clock_and_add(t, "Compute  Remote");
    // dense_local->invalidate_cache(i, j, true);
  }



  inline void calc_t_dist_grad_rowptr(CSRLocal<SPT> *csr_block, DENT *prevCoordinates,
                          DENT lr, int batch_id, int batch_size, int block_size,
                          bool local, int start_process,int end_process) {

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
//      if (col_major) {
//        calc_embedding(source_start_index, source_end_index, dst_start_index,
//                       dst_end_index, csr_block, prevCoordinates, lr, batch_id,
//                       batch_size, block_size, fetch_from_temp_cache);
//      } else {
        calc_embedding_row_major(source_start_index, source_end_index,
                                 dst_start_index, dst_end_index, csr_block,
                                 prevCoordinates, lr, batch_id, batch_size,
                                 block_size);
//      }
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

//          if (col_major) {
//            calc_embedding(source_start_index, source_end_index,
//                           dst_start_index, dst_end_index, csr_block,
//                           prevCoordinates, lr, batch_id, batch_size,
//                           block_size, fetch_from_temp_cache);
//          } else {
            calc_embedding_row_major(source_start_index, source_end_index,
                                     dst_start_index, dst_end_index, csr_block,
                                     prevCoordinates, lr, batch_id, batch_size,
                                     block_size);
//          }
        }
      }
    }
  }

//  inline void calc_embedding(uint64_t source_start_index,
//                             uint64_t source_end_index,
//                             uint64_t dst_start_index, uint64_t dst_end_index,
//                             CSRLocal<SPT> *csr_block, DENT *prevCoordinates,
//                             DENT lr, int batch_id, int batch_size,
//                             int block_size, bool temp_cache) {
//    if (csr_block->handler != nullptr) {
//      CSRHandle *csr_handle = csr_block->handler.get();
//
//#pragma omp parallel for schedule(static)
//      for (uint64_t i = dst_start_index; i <= dst_end_index; i++) {
//
//        uint64_t local_dst = i - (grid)->rank_in_col *
//                                     (this->sp_local_receiver)->proc_row_width;
//        int target_rank = (int)(i / (this->sp_local_receiver)->proc_row_width);
//        bool fetch_from_cache =
//            target_rank == (grid)->rank_in_col ? false : true;
//
//
//        bool matched = false;
//        std::array<DENT, embedding_dim> array_ptr;
//        bool col_inserted = false;
//        for (uint64_t j = static_cast<uint64_t>(csr_handle->rowStart[i]);
//             j < static_cast<uint64_t>(csr_handle->rowStart[i + 1]); j++) {
//          if (csr_handle->col_idx[j] >= source_start_index and
//              csr_handle->col_idx[j] <= source_end_index) {
//            auto source_id = csr_handle->col_idx[j];
//            auto index = source_id - batch_id * batch_size;
//
//            if (!matched) {
//              if (fetch_from_cache) {
//                unordered_map<uint64_t, CacheEntry<DENT, embedding_dim>>
//                    &arrayMap =
//                        (temp_cache)
//                            ? (*this->dense_local->tempCachePtr)[target_rank]
//                            : (*this->dense_local->cachePtr)[target_rank];
//                array_ptr = arrayMap[i].value;
//              }
//              matched = true;
//            }
//            for (int d = 0; d < embedding_dim; d++) {
//              if (!fetch_from_cache) {
//                prevCoordinates[index * embedding_dim + d] += lr *(this->dense_local)
//                                                                       ->nCoordinates[local_dst * embedding_dim + d];
//              } else {
//                prevCoordinates[index * embedding_dim + d] += lr *(array_ptr[d]);
//              }
//            }
//          }
//        }
//      }
//    }
//  }

  inline void calc_embedding_row_major(uint64_t source_start_index,
                           uint64_t source_end_index, uint64_t dst_start_index,
                           uint64_t dst_end_index, CSRLocal<SPT> *csr_block,
                           DENT *prevCoordinates, DENT lr, int batch_id,
                           int batch_size, int block_size) {
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

            vector<Tuple<DENT>> remote_tuples;

            if (fetch_from_cache) {
              unordered_map<uint64_t, SparseCacheEntry<DENT>>
                  &arrayMap = (*sparse_local->tempCachePtr)[target_rank];
              remote_tuples = arrayMap[dst_id].tuples;
            }

            CSRHandle *handle = ((sparse_local)->csr_local_data)->handler.get();
            if (!fetch_from_cache) {
              for (auto k = handle->rowStart[local_dst]; k < handle->rowStart[local_dst + 1]; k++) {
                auto d = handle->col_idx[k];
                prevCoordinates[index * embedding_dim + d] += lr *handle->values[k];
              }
            }else{
              for(Tuple<DENT> t: remote_tuples){
                auto d = t.col;
                prevCoordinates[index * embedding_dim + d] += lr *t.value;
              }
            }
          }
        }
      }
    }
  }


  inline void update_data_matrix_rowptr(DENT *prevCoordinates, int batch_id,
                                        int batch_size) {

    int row_base_index = batch_id * batch_size;
    int end_row = std::min((batch_id + 1) * batch_size,
                           ((this->sp_local_receiver)->proc_row_width));

#pragma omp parallel for schedule(static)
    for (int i = 0; i < (end_row - row_base_index); i++) {
      for (int d = 0; d < embedding_dim; d++) {
        (this->dense_local_output)
            ->nCoordinates[(row_base_index + i) * embedding_dim + d] =
            prevCoordinates[i * embedding_dim + d];
      }
    }
  }
};
} // namespace distblas::algo
