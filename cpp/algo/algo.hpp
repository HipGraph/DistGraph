#pragma once
#include "../core/common.h"
#include "../core/csr_local.hpp"
#include "../core/dense_mat.hpp"
#include "../core/json.hpp"
#include "../core/sparse_mat.hpp"
#include "../net/data_comm.hpp"
#include "../net/process_3D_grid.hpp"
#include <Eigen/Dense>
#include <chrono>
#include <math.h>
#include <memory>
#include <mpi.h>
#include <random>
#include <unordered_map>

using namespace std;
using namespace distblas::core;
using namespace distblas::net;
using namespace Eigen;

namespace distblas::algo {
template <typename SPT, typename DENT, size_t embedding_dim>

class EmbeddingAlgo {

protected:
  DenseMat<SPT, DENT, embedding_dim> *dense_local;
  distblas::core::SpMat<SPT> *sp_local_receiver;
  distblas::core::SpMat<SPT> *sp_local_sender;
  distblas::core::SpMat<SPT> *sp_local_native;
  Process3DGrid *grid;
  DENT MAX_BOUND, MIN_BOUND;
  std::unordered_map<int, unique_ptr<DataComm<SPT, DENT, embedding_dim>>>
      data_comm_cache;

  // cache size controlling hyper parameter
  double alpha = 1.0;

  double beta = 1.0;

  bool sync = false;

  bool col_major = false;

public:
  EmbeddingAlgo(distblas::core::SpMat<SPT> *sp_local_native,
                distblas::core::SpMat<SPT> *sp_local_receiver,
                distblas::core::SpMat<SPT> *sp_local_sender,
                DenseMat<SPT, DENT, embedding_dim> *dense_local,
                Process3DGrid *grid, double alpha, double beta, DENT MAX_BOUND,
                DENT MIN_BOUND)
      : sp_local_native(sp_local_native), sp_local_receiver(sp_local_receiver),
        sp_local_sender(sp_local_sender), dense_local(dense_local), grid(grid),
        alpha(alpha), beta(beta), MAX_BOUND(MAX_BOUND), MIN_BOUND(MIN_BOUND) {}

  DENT scale(DENT v) {
    if (v > MAX_BOUND)
      return MAX_BOUND;
    else if (v < -MAX_BOUND)
      return -MAX_BOUND;
    else
      return v;
  }

  void algo_force2_vec_ns(int iterations, int batch_size, int ns, DENT lr) {
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

    cout << " rank " << this->grid->global_rank << " total batches " << batches
         << endl;

    // This communicator is being used for negative updates and in alpha > 0 to
    // fetch initial embeddings
    auto full_comm = unique_ptr<DataComm<SPT, DENT, embedding_dim>>(
        new DataComm<SPT, DENT, embedding_dim>(
            sp_local_receiver, sp_local_sender, dense_local, grid, -1, alpha));
    full_comm.get()->onboard_data();

    // first batch onboarding
    unique_ptr<std::vector<DataTuple<DENT, embedding_dim>>> update_ptr =
        unique_ptr<std::vector<DataTuple<DENT, embedding_dim>>>(
            new vector<DataTuple<DENT, embedding_dim>>());

    unique_ptr<std::vector<DataTuple<DENT, embedding_dim>>> sendbuf_ptr =
        unique_ptr<std::vector<DataTuple<DENT, embedding_dim>>>(
            new vector<DataTuple<DENT, embedding_dim>>());

    if (alpha > 0 and this->grid->world_size > 1) {
      MPI_Request fetch_batch_full;
      int alpha_proc_end = get_end_proc(1, alpha, grid->world_size);
      full_comm.get()->transfer_data(sendbuf_ptr.get(), update_ptr.get(), true,
                                     &fetch_batch_full, 0, 0, 1, alpha_proc_end,
                                     false);
    }

    vector<MPI_Request *> mpi_requests(batches);

    for (int i = 0; i < batches; i++) {
      auto communicator = unique_ptr<DataComm<SPT, DENT, embedding_dim>>(
          new DataComm<SPT, DENT, embedding_dim>(
              sp_local_receiver, sp_local_sender, dense_local, grid, i, alpha));
      data_comm_cache.insert(std::make_pair(i, std::move(communicator)));
      data_comm_cache[i].get()->onboard_data();
    }

    cout << " rank " << this->grid->global_rank << " onboard_data completed "
         << batches << endl;

    DENT *prevCoordinates = static_cast<DENT *>(
        ::operator new(sizeof(DENT[batch_size * embedding_dim])));

    size_t total_memory = 0;

    CSRLocal<SPT> *csr_block =
        (col_major) ? (this->sp_local_receiver)->csr_local_data.get()
                    : (this->sp_local_native)->csr_local_data.get();

    int considering_batch_size = batch_size;

    for (int i = 0; i < iterations; i++) {

      if (alpha > 0 and this->grid->world_size > 1 and i == 0) {

        for (int k = 0; k < batch_size; k += 1) {
          int IDIM = k * embedding_dim;
          for (int d = 0; d < embedding_dim; d++) {
            prevCoordinates[IDIM + d] = 0;
          }
        }

        int last_proc = this->execute_push_model_computations(
            sendbuf_ptr.get(), update_ptr.get(), 0, 0, batches,
            this->data_comm_cache[0].get(), csr_block, batch_size,
            last_batch_size, considering_batch_size, lr, prevCoordinates, 0, 0,
            considering_batch_size, false);

        if (alpha < 1.0) {
          int prev_start = get_end_proc(1, alpha, this->grid->world_size);
          this->execute_pull_model_computations(
              sendbuf_ptr.get(), update_ptr.get(), 0, 0,
              this->data_comm_cache[0].get(), csr_block, batch_size,
              considering_batch_size, lr, prevCoordinates, prev_start, false,
              last_proc, true);
        }
      }

      for (int j = 0; j < batches; j++) {
        int seed = j + i;

        if (j == batches - 1) {
          considering_batch_size = last_batch_size;
        }

        // negative samples generation
        vector<uint64_t> random_number_vec = generate_random_numbers(
            0, (this->sp_local_receiver)->gRows, seed, ns);

        if (this->grid->world_size == 1) {
          for (int k = 0; k < batch_size; k += 1) {
            int IDIM = k * embedding_dim;
            for (int d = 0; d < embedding_dim; d++) {
              prevCoordinates[IDIM + d] = 0;
            }
          }
          // local computations for 1 process
          this->calc_t_dist_grad_rowptr(csr_block, prevCoordinates, lr, j,
                                        batch_size, considering_batch_size,
                                        true, false, 0, 0, false);

          this->calc_t_dist_replus_rowptr(prevCoordinates, random_number_vec,
                                          lr, j, batch_size,
                                          considering_batch_size);

          this->update_data_matrix_rowptr(prevCoordinates, j, batch_size);

        } else {
          full_comm.get()->transfer_data(random_number_vec, i, j);
          this->calc_t_dist_replus_rowptr(prevCoordinates, random_number_vec,
                                          lr, j, batch_size,
                                          considering_batch_size);

          //  pull model code
          if (alpha == 0) {

            this->execute_pull_model_computations(
                sendbuf_ptr.get(), update_ptr.get(), i, j,
                this->data_comm_cache[j].get(), csr_block, batch_size,
                considering_batch_size, lr, prevCoordinates, 1,
                true, 0, true);

            this->update_data_matrix_rowptr(prevCoordinates, j, batch_size);

            for (int k = 0; k < batch_size; k += 1) {
              int IDIM = k * embedding_dim;
              for (int d = 0; d < embedding_dim; d++) {
                prevCoordinates[IDIM + d] = 0;
              }
            }

          } else {

            this->update_data_matrix_rowptr(prevCoordinates, j, batch_size);

            if (!(i == iterations - 1 and j == batches - 1)) {
              // clear up data
              for (int k = 0; k < batch_size; k += 1) {
                int IDIM = k * embedding_dim;
                for (int d = 0; d < embedding_dim; d++) {
                  prevCoordinates[IDIM + d] = 0;
                }
              }
              int next_batch_id = (j + 1) % batches;
              int next_iteration = (next_batch_id == 0) ? i + 1 : i;
              int next_considering_batch_size = (next_batch_id == batches - 1)
                                                    ? last_batch_size
                                                    : considering_batch_size;

              int last_proc = this->execute_push_model_computations(
                  sendbuf_ptr.get(), update_ptr.get(), i, j, batches,
                  this->data_comm_cache[j].get(), csr_block, batch_size,
                  last_batch_size, considering_batch_size, lr, prevCoordinates,
                  next_batch_id, next_iteration, next_considering_batch_size,
                  true);

              if (alpha < 1.0) {
                //                int proc_length = get_proc_length(beta,
                //                this->grid->world_size);
                int prev_start = get_end_proc(1, alpha, this->grid->world_size);
                this->execute_pull_model_computations(
                    sendbuf_ptr.get(), update_ptr.get(), next_iteration,
                    next_batch_id, this->data_comm_cache[next_batch_id].get(),
                    csr_block, batch_size, considering_batch_size, lr,
                    prevCoordinates, prev_start, false, last_proc, true);

                //                for (int k = prev_start; k <
                //                this->grid->world_size; k += proc_length) {
                //                  MPI_Request misses_update_request;
                //                  int end_process = get_end_proc(k, beta,
                //                  this->grid->world_size);
                //
                //                  this->data_comm_cache[next_batch_id].get()->transfer_data(update_ptr.get(),
                //                  false, misses_update_request,next_iteration,
                //                  next_batch_id, k, end_process, true);
                //
                //                  if (k == prev_start) {
                //                    // remote computation for first batch
                //                    this->calc_t_dist_grad_rowptr(csr_block,
                //                    prevCoordinates, lr, next_batch_id,
                //                        batch_size, considering_batch_size,
                //                        false, true, prev_start_proc,
                //                        prev_start, false);
                //
                //                  } else if (k > prev_start) {
                //                    // updating last remote fetched data
                //                    vectors int prev_end_process =
                //                    get_end_proc(prev_start, this->beta,
                //                                                        this->grid->world_size);
                //                    this->calc_t_dist_grad_rowptr(
                //                        csr_block, prevCoordinates, lr,
                //                        next_batch_id, batch_size,
                //                        considering_batch_size, false, true,
                //                        prev_start, prev_end_process, true);
                //
                //// dense_local->invalidate_cache(next_iteration,
                /// next_batch_id, / true);
                //                  }
                //
                //                  this->data_comm_cache[next_batch_id].get()->populate_cache(update_ptr.get(),
                //                  misses_update_request, false,next_iteration,
                //                  next_batch_id, true); prev_start = k;
                //                }
              }
            }
          }
        }
        total_memory += get_memory_usage();
      }
    }
    total_memory = total_memory / (iterations * batches);
    add_memory(total_memory, "Memory usage");
    stop_clock_and_add(t, "Total Time");
    mpi_requests.clear();
  }

  inline void execute_pull_model_computations(
      std::vector<DataTuple<DENT, embedding_dim>> *sendbuf,
      std::vector<DataTuple<DENT, embedding_dim>> *receivebuf, int iteration,
      int batch, DataComm<SPT, DENT, embedding_dim> *data_comm,
      CSRLocal<SPT> *csr_block, int batch_size, int considering_batch_size,
      double lr, DENT *prevCoordinates, int comm_initial_start, bool local_execution,
      int first_execution_proc, bool communication) {

    int proc_length = get_proc_length(beta, grid->world_size);
    int prev_start = comm_initial_start;

    for (int k = prev_start; k < grid->world_size; k += proc_length) {
      int end_process = get_end_proc(k, beta, grid->world_size);

      MPI_Request req;

      if (communication) {
        data_comm->transfer_data(sendbuf, receivebuf, sync, &req, iteration,batch, k, end_process, true);
      }

      if (!sync and communication) {
        MPI_Ialltoallv(
            (*sendbuf).data(), data_comm->send_counts_cyclic.data(),
            data_comm->sdispls_cyclic.data(), DENSETUPLE, (*receivebuf).data(),
            data_comm->receive_counts_cyclic.data(),
            data_comm->rdispls_cyclic.data(), DENSETUPLE, MPI_COMM_WORLD, &req);
      }

      if (k == comm_initial_start) {
        if (grid->global_rank == 0)
          cout << "executing local"<< endl;
        // local computation
        this->calc_t_dist_grad_rowptr(
            csr_block, prevCoordinates, lr, batch, batch_size,
            considering_batch_size, local_execution, col_major,
            first_execution_proc, prev_start, local_execution);

      } else if (k > comm_initial_start) {
        int prev_end_process = get_end_proc(prev_start, beta, grid->world_size);

        this->calc_t_dist_grad_rowptr(csr_block, prevCoordinates, lr, batch,
                                      batch_size, considering_batch_size, false,
                                      col_major, prev_start, prev_end_process,
                                      true);
      }

      if (!sync and communication) {
        data_comm->populate_cache(sendbuf, receivebuf, &req, sync, iteration, batch, true);
      }

      prev_start = k;
    }
    int prev_end_process = get_end_proc(prev_start, beta, grid->world_size);
    if (grid->global_rank == 0)
      cout << "prev_start" << prev_start << " prev_end_process "
           << prev_end_process << "iterations" << iteration << endl;
    // updating last remote fetched data vectors
    this->calc_t_dist_grad_rowptr(csr_block, prevCoordinates, lr, batch,
                                  batch_size, considering_batch_size, false,
                                  col_major, prev_start, prev_end_process,
                                  true);

    // dense_local->invalidate_cache(i, j, true);
  }

  inline int execute_push_model_computations(
      std::vector<DataTuple<DENT, embedding_dim>> *sendbuf,
      std::vector<DataTuple<DENT, embedding_dim>> *receivebuf, int iteration,
      int batch, int batches, DataComm<SPT, DENT, embedding_dim> *data_comm,
      CSRLocal<SPT> *csr_block, int batch_size, int last_batch_size,
      int considering_batch_size, double lr, DENT *prevCoordinates,
      int next_batch_id, int next_iteration, int next_considering_batch_size,
      bool communication) {

    int prev_start_proc = 0;
    int alpha_proc_length = get_proc_length(alpha, grid->world_size);

    int alpha_cyc_len = get_proc_length(beta, alpha_proc_length);
    int alpha_cyc_end = get_end_proc(1, beta, alpha_proc_length);

    for (int k = 1; k < alpha_proc_length; k += alpha_cyc_len) {
      MPI_Request request_batch_update;

      if (communication) {
        data_comm->transfer_data(sendbuf, receivebuf, sync,
                                 &request_batch_update, iteration, batch, k,
                                 (k + alpha_cyc_len), false);
      }
      if (!sync and communication) {
        MPI_Ialltoallv((*sendbuf).data(), data_comm->send_counts_cyclic.data(),
                       data_comm->sdispls_cyclic.data(), DENSETUPLE,
                       (*receivebuf).data(),
                       data_comm->receive_counts_cyclic.data(),
                       data_comm->rdispls_cyclic.data(), DENSETUPLE,
                       MPI_COMM_WORLD, &request_batch_update);
      }

      if (k == 1) {
        // local computation for first batch
        this->calc_t_dist_grad_rowptr(
            csr_block, prevCoordinates, lr, next_batch_id, batch_size,
            next_considering_batch_size, true, col_major, 0, 0, false);
      } else if (k > 1) {
        this->calc_t_dist_grad_rowptr(csr_block, prevCoordinates, lr,
                                      next_batch_id, batch_size,
                                      next_considering_batch_size, false,
                                      col_major, prev_start_proc, k, false);
      }

      if (!sync and communication) {
        data_comm->populate_cache(sendbuf, receivebuf, &request_batch_update,
                                  sync, iteration, batch, false);
      }

      prev_start_proc = k;
    }
    if (alpha == 1.0) {
      this->calc_t_dist_grad_rowptr(
          csr_block, prevCoordinates, lr, next_batch_id, batch_size,
          next_considering_batch_size, false, col_major, prev_start_proc,
          alpha_cyc_end, false);
      return alpha_cyc_end;
    }
    return prev_start_proc;
    //              dense_local->invalidate_cache(i, j, false);
  }

  inline void
  calc_t_dist_grad_rowptr(CSRLocal<SPT> *csr_block, DENT *prevCoordinates,
                          DENT lr, int batch_id, int batch_size, int block_size,
                          bool local, bool col_major, int start_process,
                          int end_process, bool fetch_from_temp_cache) {

    auto source_start_index = batch_id * batch_size;
    auto source_end_index = std::min((batch_id + 1) * batch_size,
                                     this->sp_local_receiver->proc_row_width) -
                            1;

    auto dst_start_index =
        this->sp_local_receiver->proc_col_width * this->grid->global_rank;
    auto dst_end_index =
        std::min(static_cast<uint64_t>(this->sp_local_receiver->proc_col_width *
                                       (this->grid->global_rank + 1)),
                 this->sp_local_receiver->gCols) -
        1;

    if (local) {
      if (col_major) {
        calc_embedding(source_start_index, source_end_index, dst_start_index,
                       dst_end_index, csr_block, prevCoordinates, lr, batch_id,
                       batch_size, block_size, fetch_from_temp_cache);
      } else {
        calc_embedding_row_major(source_start_index, source_end_index,
                                 dst_start_index, dst_end_index, csr_block,
                                 prevCoordinates, lr, batch_id, batch_size,
                                 block_size, fetch_from_temp_cache);
      }
    } else {
      for (int r = start_process; r < end_process; r++) {
        if (r != grid->global_rank) {
          dst_start_index = this->sp_local_receiver->proc_row_width * r;
          dst_end_index =
              std::min(static_cast<uint64_t>(
                           this->sp_local_receiver->proc_row_width * (r + 1)),
                       this->sp_local_receiver->gCols) -
              1;

          if (col_major) {
            calc_embedding(source_start_index, source_end_index,
                           dst_start_index, dst_end_index, csr_block,
                           prevCoordinates, lr, batch_id, batch_size,
                           block_size, fetch_from_temp_cache);
          } else {
            calc_embedding_row_major(source_start_index, source_end_index,
                                     dst_start_index, dst_end_index, csr_block,
                                     prevCoordinates, lr, batch_id, batch_size,
                                     block_size, fetch_from_temp_cache);
          }
        }
      }
    }
  }

  inline void calc_embedding(uint64_t source_start_index,
                             uint64_t source_end_index,
                             uint64_t dst_start_index, uint64_t dst_end_index,
                             CSRLocal<SPT> *csr_block, DENT *prevCoordinates,
                             DENT lr, int batch_id, int batch_size,
                             int block_size, bool temp_cache) {
    if (csr_block->handler != nullptr) {
      CSRHandle *csr_handle = csr_block->handler.get();

#pragma omp parallel for schedule(static)
      for (uint64_t i = dst_start_index; i <= dst_end_index; i++) {

        //        cout<<" rank "<< (this->grid)->global_rank<<"
        //        dst_start_index
        //        "<<dst_start_index<<" dst_end_index "<<dst_end_index<<endl;

        uint64_t local_dst = i - (this->grid)->global_rank *
                                     (this->sp_local_receiver)->proc_row_width;
        int target_rank = (int)(i / (this->sp_local_receiver)->proc_row_width);
        bool fetch_from_cache =
            target_rank == (this->grid)->global_rank ? false : true;

        //        if (fetch_from_cache) cout<<" rank "<<
        //        (this->grid)->global_rank<<" target rank "<<target_rank<<"
        //        for i "<<i<<endl;

        bool matched = false;
        std::optional<std::array<DENT, embedding_dim>> array_ptr;
        bool col_inserted = false;
        for (uint64_t j = static_cast<uint64_t>(csr_handle->rowStart[i]);
             j < static_cast<uint64_t>(csr_handle->rowStart[i + 1]); j++) {
          if (csr_handle->col_idx[j] >= source_start_index and
              csr_handle->col_idx[j] <= source_end_index) {
            DENT forceDiff[embedding_dim];
            auto source_id = csr_handle->col_idx[j];
            auto index = source_id - batch_id * batch_size;

            if (!matched) {
              if (fetch_from_cache) {
                unordered_map<uint64_t, CacheEntry<DENT, embedding_dim>>
                    &arrayMap =
                        (temp_cache)
                            ? (*this->dense_local->tempCachePtr)[target_rank]
                            : (*this->dense_local->cachePtr)[target_rank];
                array_ptr = arrayMap[i].value;
              }
              matched = true;
            }
            DENT attrc = 0;
            for (int d = 0; d < embedding_dim; d++) {
              if (!fetch_from_cache) {
                forceDiff[d] =
                    (this->dense_local)
                        ->nCoordinates[source_id * embedding_dim + d] -
                    (this->dense_local)
                        ->nCoordinates[local_dst * embedding_dim + d];
              } else {
                forceDiff[d] =
                    (this->dense_local)
                        ->nCoordinates[source_id * embedding_dim + d] -
                    (array_ptr.value()[d]);
              }
              attrc += forceDiff[d] * forceDiff[d];
            }
            DENT d1 = -2.0 / (1.0 + attrc);

            for (int d = 0; d < embedding_dim; d++) {
              DENT l = scale(forceDiff[d] * d1);
              prevCoordinates[index * embedding_dim + d] =
                  prevCoordinates[index * embedding_dim + d] + (lr)*l;
            }
          }
        }
      }
    }
  }

  inline void
  calc_embedding_row_major(uint64_t source_start_index,
                           uint64_t source_end_index, uint64_t dst_start_index,
                           uint64_t dst_end_index, CSRLocal<SPT> *csr_block,
                           DENT *prevCoordinates, DENT lr, int batch_id,
                           int batch_size, int block_size, bool temp_cache) {
    if (csr_block->handler != nullptr) {
      CSRHandle *csr_handle = csr_block->handler.get();

#pragma omp parallel for schedule(static) // enable for full batch training or
                                          // batch size larger than 1000000
      for (uint64_t i = source_start_index; i <= source_end_index; i++) {

        uint64_t index = i - batch_id * batch_size;

        for (uint64_t j = static_cast<uint64_t>(csr_handle->rowStart[i]);
             j < static_cast<uint64_t>(csr_handle->rowStart[i + 1]); j++) {
          auto dst_id = csr_handle->col_idx[j];
          if (dst_id >= dst_start_index and dst_id < dst_end_index) {
            uint64_t local_dst =
                dst_id - (this->grid)->global_rank *
                             (this->sp_local_receiver)->proc_col_width;
            int target_rank =
                (int)(dst_id / (this->sp_local_receiver)->proc_col_width);
            bool fetch_from_cache =
                target_rank == (this->grid)->global_rank ? false : true;

            DENT forceDiff[embedding_dim];
            std::array<DENT, embedding_dim> array_ptr;
            if (fetch_from_cache) {
              unordered_map<uint64_t, CacheEntry<DENT, embedding_dim>>
                  &arrayMap =
                      (temp_cache)
                          ? (*this->dense_local->tempCachePtr)[target_rank]
                          : (*this->dense_local->cachePtr)[target_rank];
              array_ptr = arrayMap[dst_id].value;
              // If not in cache we should fetch that from remote for limited
              // cache
            }
            DENT attrc = 0;
            for (int d = 0; d < embedding_dim; d++) {
              if (!fetch_from_cache) {
                forceDiff[d] =
                    (this->dense_local)->nCoordinates[i * embedding_dim + d] -
                    (this->dense_local)
                        ->nCoordinates[local_dst * embedding_dim + d];
              } else {
                forceDiff[d] =
                    (this->dense_local)->nCoordinates[i * embedding_dim + d] -
                    array_ptr[d];
              }
              attrc += forceDiff[d] * forceDiff[d];
            }
            DENT d1 = -2.0 / (1.0 + attrc);

            for (int d = 0; d < embedding_dim; d++) {
              DENT l = scale(forceDiff[d] * d1);
              prevCoordinates[index * embedding_dim + d] =
                  prevCoordinates[index * embedding_dim + d] + (lr)*l;
            }
          }
        }
      }
    }
  }

  inline void calc_t_dist_replus_rowptr(DENT *prevCoordinates,
                                        vector<uint64_t> &col_ids, DENT lr,
                                        int batch_id, int batch_size,
                                        int block_size) {

    int row_base_index = batch_id * batch_size;

#pragma omp parallel for schedule(static)
    for (int i = 0; i < block_size; i++) {
      uint64_t row_id = static_cast<uint64_t>(i + row_base_index);
      DENT forceDiff[embedding_dim];
      for (int j = 0; j < col_ids.size(); j++) {
        uint64_t global_col_id = col_ids[j];
        uint64_t local_col_id =
            global_col_id -
            static_cast<uint64_t>(((this->grid)->global_rank *
                                   (this->sp_local_receiver)->proc_row_width));
        bool fetch_from_cache = false;

        int owner_rank = static_cast<int>(
            global_col_id / (this->sp_local_receiver)->proc_row_width);

        if (owner_rank != (this->grid)->global_rank) {
          fetch_from_cache = true;
        }

        if (fetch_from_cache) {
          DENT repuls = 0;
          //          std::array<DENT, embedding_dim> colvec;
          //          (this->dense_local)->fetch_data_vector_from_cache(colvec,owner_rank,
          //          global_col_id, true);
          unordered_map<uint64_t, CacheEntry<DENT, embedding_dim>> &arrayMap =
              (*this->dense_local->tempCachePtr)[owner_rank];
          std::array<DENT, embedding_dim> &colvec =
              arrayMap[global_col_id].value;

          for (int d = 0; d < embedding_dim; d++) {
            forceDiff[d] =
                (this->dense_local)->nCoordinates[row_id * embedding_dim + d] -
                colvec[d];
            repuls += forceDiff[d] * forceDiff[d];
          }

          DENT d1 = 2.0 / ((repuls + 0.000001) * (1.0 + repuls));
          for (int d = 0; d < embedding_dim; d++) {
            forceDiff[d] = scale(forceDiff[d] * d1);
            prevCoordinates[i * embedding_dim + d] += (lr)*forceDiff[d];
          }

        } else {
          DENT repuls = 0;
          for (int d = 0; d < embedding_dim; d++) {
            forceDiff[d] =
                (this->dense_local)->nCoordinates[row_id * embedding_dim + d] -
                (this->dense_local)
                    ->nCoordinates[local_col_id * embedding_dim + d];
            repuls += forceDiff[d] * forceDiff[d];
          }
          DENT d1 = 2.0 / ((repuls + 0.000001) * (1.0 + repuls));
          for (int d = 0; d < embedding_dim; d++) {
            forceDiff[d] = scale(forceDiff[d] * d1);
            prevCoordinates[i * embedding_dim + d] += (lr)*forceDiff[d];
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
        (this->dense_local)
            ->nCoordinates[(row_base_index + i) * embedding_dim + d] +=
            prevCoordinates[i * embedding_dim + d];
      }
    }
  }
};
} // namespace distblas::algo
