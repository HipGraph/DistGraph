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

    vector<MPI_Request *> mpi_requests(batches);

    for (int i = 0; i < batches; i++) {
      cout << " rank " << this->grid->global_rank
           << " on boarding data for batch  " << i << " out of " << batches
           << endl;
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

    CSRLocal<SPT> *csr_block = (this->sp_local_receiver)->csr_local_data.get();

    int considering_batch_size = batch_size;

    for (int i = 0; i < iterations; i++) {
      if (this->grid->global_rank == 0)
        cout << " rank " << grid->global_rank << " iteration " << i << endl;

      if (alpha > 0) {
        for (int k = 0; k < batch_size; k += 1) {
          int IDIM = k * embedding_dim;
          for (int d = 0; d < embedding_dim; d++) {
            prevCoordinates[IDIM + d] = 0;
          }
        }
        int prev_start_proc = 0;
        int alpha_cyc_start = 1;
        int alpha_proc_length = get_end_proc(1, alpha, grid->world_size);
        if (i == 0) {
          stop_clock_and_add(t, "Computation Time");
          t = start_clock();
          MPI_Request fetch_batch;

          int alpha_cyc_len = get_proc_length(beta, alpha_proc_length);
          int alpha_cyc_end = get_end_proc(1, beta, alpha_proc_length);

          for (int k = alpha_cyc_end; k <= alpha_proc_length; k += alpha_cyc_len) {
            update_ptr.get()->clear();
            full_comm.get()->transfer_data(update_ptr.get(), false, fetch_batch,0, 0, alpha_cyc_start, k, false);
            stop_clock_and_add(t, "Communication Time");
            t = start_clock();

            if (k == alpha_cyc_end) {
              // local computation for first batch
              this->calc_t_dist_grad_rowptr(csr_block, prevCoordinates, lr, 0,
                                            batch_size, considering_batch_size,
                                            true, true, 0, 0, false);
            } else if (k > alpha_cyc_end) {
              this->calc_t_dist_grad_rowptr(csr_block, prevCoordinates, lr, 0,
                                            batch_size, considering_batch_size,
                                            false, true, prev_start_proc,
                                            alpha_cyc_start, false);
            }
            stop_clock_and_add(t, "Computation Time");
            t = start_clock();
            full_comm.get()->populate_cache(update_ptr.get(), fetch_batch, false,0, 0, false);
            stop_clock_and_add(t, "Communication Time");
            t = start_clock();
            prev_start_proc = alpha_cyc_start;
            alpha_cyc_start = k;
          }
          if (alpha==1.0) {
            // remote computation for first batch
            this->calc_t_dist_grad_rowptr(csr_block, prevCoordinates, lr, 0,
                                          batch_size, considering_batch_size,
                                          false, true, prev_start_proc,
                                          alpha_cyc_start, false);
          } else if (this->alpha < 1.0) {
            int proc_length = get_proc_length(this->beta, this->grid->world_size);
            int beta_prev_start = get_end_proc(1, this->alpha, this->grid->world_size);
            for (int k = beta_prev_start; k < this->grid->world_size; k += proc_length) {
              MPI_Request misses_update_request;
              int end_process = get_end_proc(k, this->beta, this->grid->world_size);
              stop_clock_and_add(t, "Computation Time");
              t = start_clock();
              update_ptr.get()->clear();
              this->data_comm_cache[0].get()->transfer_data(update_ptr.get(), false,
                                                            misses_update_request,
                                                            i, 0, k,end_process, true);
              stop_clock_and_add(t, "Communication Time");
              t = start_clock();
              if (k == beta_prev_start) {
                // remote computation for first batch
                this->calc_t_dist_grad_rowptr(
                    csr_block, prevCoordinates, lr, 0, batch_size,
                    considering_batch_size, false, true, prev_start_proc,
                    beta_prev_start, false);

              } else if (k > beta_prev_start) {
                // updating last remote fetched data vectors
                int prev_end_process = get_end_proc(beta_prev_start, this->beta,
                                                    this->grid->world_size);
                this->calc_t_dist_grad_rowptr(
                    csr_block, prevCoordinates, lr, 0, batch_size,
                    considering_batch_size, false, true, beta_prev_start,
                    prev_end_process, true);
                dense_local->invalidate_cache(i, 0, true);
              }
              stop_clock_and_add(t, "Computation Time");
              t = start_clock();
              this->data_comm_cache[0].get()->populate_cache(update_ptr.get(), misses_update_request, false, i, 0, true);
              stop_clock_and_add(t, "Communication Time");
              t = start_clock();
              beta_prev_start = k;
            }
          }
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

        if (this->grid->world_size > 1) {
          MPI_Barrier(MPI_COMM_WORLD);
          stop_clock_and_add(t, "Computation Time");
          t = start_clock();
          full_comm.get()->transfer_data(random_number_vec, i, j);
          stop_clock_and_add(t, "Communication Time");
          t = start_clock();
        }

        this->calc_t_dist_replus_rowptr(prevCoordinates, random_number_vec, lr,
                                        j, batch_size, considering_batch_size);

        dense_local->invalidate_cache(i, j, true);

        //  pull model code
        if (alpha == 0) {
          int proc_length = get_proc_length(beta, grid->world_size);
          int prev_start = 0;
          for (int k = 1; k < grid->world_size; k += proc_length) {

            MPI_Request request_batch_update_cyclic;
            int end_process = get_end_proc(k, beta, grid->world_size);
            stop_clock_and_add(t, "Computation Time");

            t = start_clock();
            this->data_comm_cache[j].get()->transfer_data(
                update_ptr.get(), false, request_batch_update_cyclic, i, j, k,
                end_process, true);

            stop_clock_and_add(t, "Communication Time");
            t = start_clock();
            if (k == 1) {
              // local computation
              this->calc_t_dist_grad_rowptr(csr_block, prevCoordinates, lr, j,
                                            batch_size, considering_batch_size,
                                            true, true, 0, 0, false);

            } else if (k > 1) {
              int prev_end_process =
                  get_end_proc(prev_start, beta, grid->world_size);
              this->calc_t_dist_grad_rowptr(csr_block, prevCoordinates, lr, j,
                                            batch_size, considering_batch_size,
                                            false, true, prev_start,
                                            prev_end_process, true);
              dense_local->invalidate_cache(i, j, true);
            }
            stop_clock_and_add(t, "Computation Time");
            t = start_clock();

            data_comm_cache[j].get()->populate_cache(
                update_ptr.get(), request_batch_update_cyclic, false, i, j,
                true);

            prev_start = k;
            update_ptr.get()->clear();
            stop_clock_and_add(t, "Communication Time");
            t = start_clock();
          }
          int prev_end_process =
              get_end_proc(prev_start, beta, grid->world_size);

          // updating last remote fetched data vectors
          this->calc_t_dist_grad_rowptr(csr_block, prevCoordinates, lr, j,
                                        batch_size, considering_batch_size,
                                        false, true, prev_start,
                                        prev_end_process, true);

          dense_local->invalidate_cache(i, j, true);
          update_ptr.get()->resize(0);

          this->update_data_matrix_rowptr(prevCoordinates, j, batch_size);
          for (int k = 0; k < batch_size; k += 1) {
            int IDIM = k * embedding_dim;
            for (int d = 0; d < embedding_dim; d++) {
              prevCoordinates[IDIM + d] = 0;
            }
          }

        } else {
          this->update_data_matrix_rowptr(prevCoordinates, j, batch_size);

          if (!(i== iterations-1 and j == batches -1)) {
            // clear up data
            for (int k = 0; k < batch_size; k += 1) {
              int IDIM = k * embedding_dim;
              for (int d = 0; d < embedding_dim; d++) {
                prevCoordinates[IDIM + d] = 0;
              }
            }
            int next_batch_id = (j + 1) % batches;
            int next_iteration = (next_batch_id == 0) ? i + 1 : i;

            int prev_start_proc = 0;
            int alpha_cyc_start = 1;
            int alpha_proc_length = get_end_proc(1, alpha, grid->world_size);

            int alpha_cyc_len = get_proc_length(beta, alpha_proc_length);
            int alpha_cyc_end = get_end_proc(1, beta, alpha_proc_length);

            for (int k = alpha_cyc_end; k <= alpha_proc_length;
                 k += alpha_cyc_len) {
              MPI_Request request_batch_update;
              update_ptr.get()->clear();
              stop_clock_and_add(t, "Computation Time");
              t = start_clock();
              data_comm_cache[j].get()->transfer_data(
                  update_ptr.get(), false, request_batch_update, i, j,
                  alpha_cyc_start, k, false);
              stop_clock_and_add(t, "Communication Time");
              t = start_clock();
              if (k == alpha_cyc_end) {
                // local computation for first batch
                this->calc_t_dist_grad_rowptr(
                    csr_block, prevCoordinates, lr, next_batch_id, batch_size,
                    considering_batch_size, true, true, 0, 0, false);
              } else if (k > alpha_cyc_end) {
                this->calc_t_dist_grad_rowptr(
                    csr_block, prevCoordinates, lr, next_batch_id, batch_size,
                    considering_batch_size, false, true, prev_start_proc,
                    alpha_cyc_start, false);
              }
              if (this->grid->world_size > 1) {
                stop_clock_and_add(t, "Computation Time");
                t = start_clock();
                data_comm_cache[j].get()->populate_cache(
                    update_ptr.get(), request_batch_update, false, i, j, false);
                stop_clock_and_add(t, "Communication Time");
                t = start_clock();
              }
              prev_start_proc = alpha_cyc_start;
              alpha_cyc_start = k;
            }
            if (alpha == 1.0) {
              this->calc_t_dist_grad_rowptr(
                  csr_block, prevCoordinates, lr, next_batch_id, batch_size,
                  considering_batch_size, false, true, prev_start_proc,
                  alpha_cyc_start, false);
            }
            dense_local->invalidate_cache(i, j, false);

            if (alpha < 1.0) {
              int proc_length =
                  get_proc_length(this->beta, this->grid->world_size);
              int prev_start =
                  get_end_proc(1, this->alpha, this->grid->world_size);
              for (int k = prev_start; k < this->grid->world_size;
                   k += proc_length) {
                MPI_Request misses_update_request;
                int end_process =
                    get_end_proc(k, this->beta, this->grid->world_size);
                stop_clock_and_add(t, "Computation Time");
                t = start_clock();
                update_ptr.get()->clear();
                this->data_comm_cache[next_batch_id].get()->transfer_data(
                    update_ptr.get(), false, misses_update_request,
                    next_iteration, next_batch_id, k, end_process, true);
                stop_clock_and_add(t, "Communication Time");
                t = start_clock();
                if (k == prev_start) {
                  // remote computation for first batch
                  this->calc_t_dist_grad_rowptr(
                      csr_block, prevCoordinates, lr, next_batch_id, batch_size,
                      considering_batch_size, false, true, prev_start_proc,
                      prev_start, false);

                } else if (k > prev_start) {
                  // updating last remote fetched data vectors
                  int prev_end_process = get_end_proc(prev_start, this->beta,
                                                      this->grid->world_size);
                  this->calc_t_dist_grad_rowptr(
                      csr_block, prevCoordinates, lr, next_batch_id, batch_size,
                      considering_batch_size, false, true, prev_start,
                      prev_end_process, true);
                  dense_local->invalidate_cache(next_iteration, next_batch_id,
                                                true);
                }
                stop_clock_and_add(t, "Computation Time");
                t = start_clock();
                this->data_comm_cache[next_batch_id].get()->populate_cache(
                    update_ptr.get(), misses_update_request, false,
                    next_iteration, next_batch_id, true);
                stop_clock_and_add(t, "Communication Time");
                t = start_clock();
                prev_start = k;
              }
            }
          }
        }
        total_memory += get_memory_usage();
      }
    }
    total_memory = total_memory / (iterations * batches);
    add_memory(total_memory, "Memory usage");
    stop_clock_and_add(t, "Computation Time");
    mpi_requests.clear();
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
                                 block_size);
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
                                     block_size);
          }
        }
      }
    }
  }

  inline void
  calc_t_dist_grad_for_cache_misses(vector<vector<Tuple<DENT>>> *cache_misses,
                                    DENT *prevCoordinates, int iteration,
                                    int batch_id, int batch_size, double lr,
                                    int starting_proc, int end_proc) {

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

    //#pragma omp parallel for schedule(static)
    for (int i = 0; i < sending_procs.size(); i++) {
      for (int k = 0; k < (*cache_misses)[sending_procs[i]].size(); k++) {
        uint64_t col_id = (*cache_misses)[sending_procs[i]][k].col;
        uint64_t source_id = (*cache_misses)[sending_procs[i]][k].row;
        auto index = source_id - batch_id * batch_size;
        DENT forceDiff[embedding_dim];
        DENT attrc = 0;
        DENT *array_ptr =
            (this->dense_local)
                ->fetch_data_vector_from_cache(sending_procs[i], col_id, true);
        for (int d = 0; d < embedding_dim; d++) {
          forceDiff[d] =
              (this->dense_local)->nCoordinates[source_id * embedding_dim + d] -
              array_ptr[d];

          attrc += forceDiff[d] * forceDiff[d];
        }
        DENT d1 = -2.0 / (1.0 + attrc);

        for (int d = 0; d < embedding_dim; d++) {
          DENT l = scale(forceDiff[d] * d1);
          prevCoordinates[index * embedding_dim + d] =
              prevCoordinates[index * embedding_dim + d] + (lr)*l;
        }
      }
      (*cache_misses)[sending_procs[i]].clear();
      (*cache_misses)[sending_procs[i]].shrink_to_fit();
    }
    dense_local->invalidate_cache(iteration, batch_id, true);
  }

  inline void calc_embedding(uint64_t source_start_index,
                             uint64_t source_end_index,
                             uint64_t dst_start_index, uint64_t dst_end_index,
                             CSRLocal<SPT> *csr_block, DENT *prevCoordinates,
                             DENT lr, int batch_id, int batch_size,
                             int block_size, bool temp_cache) {
    if (csr_block->handler != nullptr) {
      CSRHandle *csr_handle = csr_block->handler.get();

      //#pragma omp parallel for schedule(static)
      for (uint64_t i = dst_start_index; i <= dst_end_index; i++) {

        uint64_t local_dst = i - (this->grid)->global_rank *
                                     (this->sp_local_receiver)->proc_row_width;
        int target_rank = (int)(i / (this->sp_local_receiver)->proc_row_width);
        bool fetch_from_cache =
            target_rank == (this->grid)->global_rank ? false : true;
        bool matched = false;
        DENT *array_ptr = nullptr;
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
                array_ptr = (this->dense_local)
                                ->fetch_data_vector_from_cache(target_rank, i,
                                                               temp_cache);

                if (array_ptr == nullptr) {
                  continue;
                }
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

  inline void
  calc_embedding_row_major(uint64_t source_start_index,
                           uint64_t source_end_index, uint64_t dst_start_index,
                           uint64_t dst_end_index, CSRLocal<SPT> *csr_block,
                           DENT *prevCoordinates, DENT lr, int batch_id,
                           int batch_size, int block_size) {
    if (csr_block->handler != nullptr) {
      CSRHandle *csr_handle = csr_block->handler.get();

#pragma omp parallel for schedule(static) // enable for full batch training or
                                          // batch size larger than 1000000
      for (uint64_t i = source_start_index; i <= source_end_index; i++) {

        uint64_t index = i - batch_id * batch_size;
        for (uint64_t j = static_cast<uint64_t>(csr_handle->rowStart[i]);
             j < static_cast<uint64_t>(csr_handle->rowStart[i + 1]); j++) {
          auto dst_id = csr_handle->col_idx[j];

          if (dst_id >= dst_start_index and dst_id <= dst_end_index) {
            uint64_t local_dst =
                dst_id - (this->grid)->global_rank *
                             (this->sp_local_receiver)->proc_col_width;
            int target_rank =
                (int)(dst_id / (this->sp_local_receiver)->proc_col_width);
            bool fetch_from_cache =
                target_rank == (this->grid)->global_rank ? false : true;
            bool matched = false;
            DENT forceDiff[embedding_dim];
            DENT *array_ptr = nullptr;
            if (fetch_from_cache) {
              array_ptr = (this->dense_local)
                              ->fetch_data_vector_from_cache(target_rank,
                                                             dst_id, false);
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

    //#pragma omp parallel for schedule(static)
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
          DENT *colvec = (this->dense_local)
                             ->fetch_data_vector_from_cache(
                                 owner_rank, global_col_id, true);
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
    //    #pragma omp parallel for schedule(static)
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
