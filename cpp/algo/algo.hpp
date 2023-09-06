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

private:
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

public:
  EmbeddingAlgo(distblas::core::SpMat<SPT> *sp_local_native,
                distblas::core::SpMat<SPT> *sp_local_receiver,
                distblas::core::SpMat<SPT> *sp_local_sender,
                DenseMat<SPT, DENT, embedding_dim> *dense_local,
                Process3DGrid *grid, double alpha, DENT MAX_BOUND,
                DENT MIN_BOUND) {
    this->grid = grid;
    this->dense_local = dense_local;
    this->sp_local_sender = sp_local_sender;
    this->sp_local_receiver = sp_local_receiver;
    this->sp_local_native = sp_local_native;
    this->MAX_BOUND = MAX_BOUND;
    this->MIN_BOUND = MIN_BOUND;
    this->alpha = alpha;
  }

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

    auto negative_update_com = unique_ptr<DataComm<SPT, DENT, embedding_dim>>(
        new DataComm<SPT, DENT, embedding_dim>(
            sp_local_receiver, sp_local_sender, dense_local, grid, -1, alpha));

    unique_ptr<std::vector<DataTuple<DENT, embedding_dim>>> fetch_all_ptr =
        unique_ptr<std::vector<DataTuple<DENT, embedding_dim>>>(
            new vector<DataTuple<DENT, embedding_dim>>());

    MPI_Request fetch_all;
    if (alpha > 0) {
      negative_update_com.get()->onboard_data();
      cout << " rank " << this->grid->global_rank << " onboard_data completed "
           << batches << endl;
      stop_clock_and_add(t, "Computation Time");

      t = start_clock();
      negative_update_com.get()->transfer_data(fetch_all_ptr.get(), false,
                                               fetch_all, 0, 0);
      stop_clock_and_add(t, "Communication Time");
      t = start_clock();
    }

    for (int i = 0; i < batches; i++) {
      auto communicator = unique_ptr<DataComm<SPT, DENT, embedding_dim>>(
          new DataComm<SPT, DENT, embedding_dim>(
              sp_local_receiver, sp_local_sender, dense_local, grid, i, alpha));
      data_comm_cache.insert(std::make_pair(i, std::move(communicator)));
      data_comm_cache[i].get()->onboard_data();
    }

    if (alpha > 0) {
      stop_clock_and_add(t, "Computation Time");
      t = start_clock();
      negative_update_com.get()->populate_cache(fetch_all_ptr.get(), fetch_all,
                                                false, 0, 0);
      stop_clock_and_add(t, "Communication Time");
      t = start_clock();
    }

    DENT *prevCoordinates = static_cast<DENT *>(
        ::operator new(sizeof(DENT[batch_size * embedding_dim])));

    unique_ptr<std::vector<DataTuple<DENT, embedding_dim>>> update_ptr =
        unique_ptr<std::vector<DataTuple<DENT, embedding_dim>>>(
            new vector<DataTuple<DENT, embedding_dim>>());

    unique_ptr<vector<vector<Tuple<DENT>>>> cache_misses_ptr =
        unique_ptr<vector<vector<Tuple<DENT>>>>(
            new vector<vector<Tuple<DENT>>>(grid->world_size));

    vector<MPI_Request> mpi_requests(iterations * batches);
    stop_clock_and_add(t, "Computation Time");
    t = start_clock();
    size_t total_memory = 0;
    for (int i = 0; i < iterations; i++) {
      if (this->grid->global_rank == 0)
        cout << " iteration " << i << endl;

      for (int j = 0; j < batches; j++) {

        int seed = j + i;
        for (int k = 0; k < batch_size; k += 1) {
          int IDIM = k * embedding_dim;
          for (int d = 0; d < embedding_dim; d++) {
            prevCoordinates[IDIM + d] = 0;
          }
        }

        int considering_batch_size = batch_size;

        if (j == batches - 1) {
          considering_batch_size = last_batch_size;
        }

        CSRLocal<SPT> *csr_block =
            (this->sp_local_receiver)->csr_local_data.get();
        //        CSRLocal<SPT> *csr_block_native =
        //            (this->sp_local_native)->csr_local_data.get();

        if (alpha == 0) {
          update_ptr.get()->clear();
          MPI_Request request_batch_update;
          stop_clock_and_add(t, "Computation Time");
          t = start_clock();
          data_comm_cache[j].get()->transfer_data(update_ptr.get(), false,
                                                  request_batch_update, i, j);
          mpi_requests[i * batches + j] = request_batch_update;
          stop_clock_and_add(t, "Communication Time");
          t = start_clock();
        }

        this->calc_t_dist_grad_rowptr(csr_block, prevCoordinates, lr, j,
                                      batch_size, considering_batch_size, true,
                                      true, cache_misses_ptr.get());

        if (this->grid->world_size > 1) {
          stop_clock_and_add(t, "Computation Time");
          t = start_clock();
          if (!(i == 0 and j == 0) and alpha > 0) {
            data_comm_cache[j].get()->populate_cache(
                update_ptr.get(), mpi_requests[i * batches + j - 1], false, i,
                j);
          } else if (alpha == 0) {
            data_comm_cache[j].get()->populate_cache(
                update_ptr.get(), mpi_requests[i * batches + j], false, i, j);
          }
          stop_clock_and_add(t, "Communication Time");
          t = start_clock();
        }

        total_memory += get_memory_usage();

        this->calc_t_dist_grad_rowptr(csr_block, prevCoordinates, lr, j,
                                      batch_size, considering_batch_size, false,
                                      true, cache_misses_ptr.get());

        if (alpha > 0 and alpha < 1.0) {
          MPI_Barrier(MPI_COMM_WORLD);
          stop_clock_and_add(t, "Computation Time");
          t = start_clock();
          data_comm_cache[j].get()->transfer_data(cache_misses_ptr.get(), i, j);
          stop_clock_and_add(t, "Communication Time");
          t = start_clock();
          this->calc_t_dist_grad_for_cache_misses(
              cache_misses_ptr.get(), prevCoordinates, j, batch_size, lr);
        }

        // negative samples generation
        vector<uint64_t> random_number_vec = generate_random_numbers(
            0, (this->sp_local_receiver)->gRows, seed, ns);

        if (this->grid->world_size > 1) {
          MPI_Barrier(MPI_COMM_WORLD);
          stop_clock_and_add(t, "Computation Time");
          t = start_clock();
          negative_update_com.get()->transfer_data(random_number_vec, i, j);
          stop_clock_and_add(t, "Communication Time");
          t = start_clock();
        }

        this->calc_t_dist_replus_rowptr(prevCoordinates, random_number_vec, lr,
                                        j, batch_size, considering_batch_size);

        this->update_data_matrix_rowptr(prevCoordinates, j, batch_size);

        if (this->grid->world_size > 1 and
            !(i == iterations - 1 and j == batches - 1) and alpha > 0) {
          update_ptr.get()->clear();
          MPI_Request request_batch_update;
          stop_clock_and_add(t, "Computation Time");
          t = start_clock();
          data_comm_cache[j].get()->transfer_data(update_ptr.get(), false,
                                                  request_batch_update, i, j);
          mpi_requests[i * batches + j] = request_batch_update;
          stop_clock_and_add(t, "Communication Time");
          t = start_clock();
        }
        dense_local->invalidate_cache(i, j);

        if (i == iterations / 2) {
          size_t mem = get_memory_usage();
          add_memory(mem, "Memory usage");
        }
      }
    }
    total_memory = total_memory / (iterations * batches);
    add_memory(total_memory, "Memory usage");
    stop_clock_and_add(t, "Computation Time");
  }

  inline void
  calc_t_dist_grad_rowptr(CSRLocal<SPT> *csr_block, DENT *prevCoordinates,
                          DENT lr, int batch_id, int batch_size, int block_size,
                          bool local, bool col_major,
                          vector<vector<Tuple<DENT>>> *cache_misses) {

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
                       batch_size, block_size, cache_misses);
      } else {
        calc_embedding_row_major(source_start_index, source_end_index,
                                 dst_start_index, dst_end_index, csr_block,
                                 prevCoordinates, lr, batch_id, batch_size,
                                 block_size);
      }
    } else {
      for (int r = 0; r < grid->world_size; r++) {
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
                           block_size, cache_misses);
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
                                    DENT *prevCoordinates, int batch_id,
                                    int batch_size, double lr) {
    for (int i = 0; i < grid->world_size; i++) {
#pragma omp parallel for
      for (int k = 0; k < (*cache_misses)[i].size(); k++) {
        uint64_t col_id = (*cache_misses)[i][k].col;
        uint64_t source_id = (*cache_misses)[i][k].row;
        auto index = source_id - batch_id * batch_size;
        DENT forceDiff[embedding_dim];
        DENT attrc = 0;
        DENT *array_ptr =
            (this->dense_local)->fetch_data_vector_from_cache(i, col_id);
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
      (*cache_misses)[i].clear();
    }
  }

  inline void calc_embedding(uint64_t source_start_index,
                             uint64_t source_end_index,
                             uint64_t dst_start_index, uint64_t dst_end_index,
                             CSRLocal<SPT> *csr_block, DENT *prevCoordinates,
                             DENT lr, int batch_id, int batch_size,
                             int block_size,
                             vector<vector<Tuple<DENT>>> *cache_misses) {
    if (csr_block->handler != nullptr) {
      CSRHandle *csr_handle = csr_block->handler.get();

#pragma omp parallel for schedule(static)
      for (uint64_t i = dst_start_index; i <= dst_end_index; i++) {

        uint64_t local_dst = i - (this->grid)->global_rank *
                                     (this->sp_local_receiver)->proc_row_width;
        int target_rank = (int)(i / (this->sp_local_receiver)->proc_row_width);
        bool fetch_from_cache =
            target_rank == (this->grid)->global_rank ? false : true;
        bool matched = false;
        DENT *array_ptr = nullptr;
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
                                ->fetch_data_vector_from_cache(target_rank, i);

                if (array_ptr == nullptr) {
                  Tuple<DENT> cacheRef;
                  cacheRef.row = source_id;
                  cacheRef.col = i;
#pragma omp critical
                  { (*cache_misses)[target_rank].push_back(cacheRef); }
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
              array_ptr =
                  (this->dense_local)
                      ->fetch_data_vector_from_cache(target_rank, dst_id);
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
          DENT *colvec =
              (this->dense_local)
                  ->fetch_data_vector_from_cache(owner_rank, global_col_id);
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
