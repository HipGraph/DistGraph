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

using json = nlohmann::json;
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
  Process3DGrid *grid;
  DENT MAX_BOUND, MIN_BOUND;
  std::unordered_map<int, unique_ptr<DataComm<SPT, DENT, embedding_dim>>>
      data_comm_cache;

  // Related to performance counting
  vector<string> perf_counter_keys;
  map<string, int> call_count;
  map<string, double> total_time;

public:
  EmbeddingAlgo(distblas::core::SpMat<SPT> *sp_local_receiver,
                distblas::core::SpMat<SPT> *sp_local_sender,
                DenseMat<SPT, DENT, embedding_dim> *dense_local,
                Process3DGrid *grid, DENT MAX_BOUND, DENT MIN_BOUND) {
    this->grid = grid;
    this->dense_local = dense_local;
    this->sp_local_sender = sp_local_sender;
    this->sp_local_receiver = sp_local_receiver;
    this->MAX_BOUND = MAX_BOUND;
    this->MIN_BOUND = MIN_BOUND;

    perf_counter_keys = {"Computation Time", "Communication Time"};
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
      // TODO:Error prone
      last_batch_size =
          sp_local_receiver->proc_row_width - batch_size * (batches - 1);
    }

    cout << " rank " << this->grid->global_rank << " total batches " << batches
         << endl;

    auto negative_update_com = unique_ptr<DataComm<SPT, DENT, embedding_dim>>(
        new DataComm<SPT, DENT, embedding_dim>(
            sp_local_receiver, sp_local_sender, dense_local, grid, -1));

    unique_ptr<std::vector<DataTuple<DENT, embedding_dim>>> fetch_all_ptr =
        unique_ptr<std::vector<DataTuple<DENT, embedding_dim>>>(
            new vector<DataTuple<DENT, embedding_dim>>());

    MPI_Request fetch_all;
//    negative_update_com.get()->onboard_data();
    cout << " rank " << this->grid->global_rank << " onboard_data completed "
         << batches << endl;
    stop_clock_and_add(t, "Computation Time");

    t = start_clock();
//    negative_update_com.get()->transfer_data(fetch_all_ptr.get(), false,
//                                             fetch_all);
    stop_clock_and_add(t, "Communication Time");

    t = start_clock();
    for (int i = 0; i < batches; i++) {
      auto communicator = unique_ptr<DataComm<SPT, DENT, embedding_dim>>(
          new DataComm<SPT, DENT, embedding_dim>(
              sp_local_receiver, sp_local_sender, dense_local, grid, 1));
      data_comm_cache.insert(std::make_pair(i, std::move(communicator)));
      data_comm_cache[i].get()->onboard_data();
      negative_update_com.get()->transfer_data(fetch_all_ptr.get(), false,
                                                   fetch_all);
    }
    stop_clock_and_add(t, "Computation Time");
    t = start_clock();
//    negative_update_com.get()->populate_cache(fetch_all_ptr.get(), fetch_all,
//                                              false);
    stop_clock_and_add(t, "Communication Time");
    t = start_clock();
    DENT *prevCoordinates = static_cast<DENT *>(
        ::operator new(sizeof(DENT[batch_size * embedding_dim])));

    unique_ptr<std::vector<DataTuple<DENT, embedding_dim>>> update_ptr =
        unique_ptr<std::vector<DataTuple<DENT, embedding_dim>>>(
            new vector<DataTuple<DENT, embedding_dim>>());

    vector<MPI_Request> mpi_requests(iterations * batches);

    for (int i = 0; i < 1; i++) {
      cout << " global rank " << grid->global_rank << endl;
      for (int j = 0; j < batches; j++) {

        int seed = j + i;

        for (int i = 0; i < batch_size; i += 1) {
          int IDIM = i * embedding_dim;
          for (int d = 0; d < embedding_dim; d++) {
            prevCoordinates[IDIM + d] = 0;
          }
        }

        // negative samples generation
        vector<uint64_t> random_number_vec =
            generate_random_numbers(0, (this->sp_local_receiver)->gRows, seed, ns);

        if (this->grid->world_size > 1) {
          stop_clock_and_add(t, "Computation Time");
          t = start_clock();
//          negative_update_com.get()->transfer_data(random_number_vec);
          stop_clock_and_add(t, "Communication Time");
          t = start_clock();
        }
        int considering_batch_size = batch_size;

        if (j == batches - 1) {
          considering_batch_size = last_batch_size;
        }
//        this->calc_t_dist_replus_rowptr(prevCoordinates, random_number_vec, lr,
//                                        j, batch_size, considering_batch_size);

        CSRLocal<SPT> *csr_block = (this->sp_local_receiver)->csr_local_data.get();


//        this->calc_t_dist_grad_rowptr(csr_block, prevCoordinates, lr, j,
//                                      batch_size, considering_batch_size, true);


        if (this->grid->world_size > 1) {
          if (!(i == 0 and j == 0)) {
            stop_clock_and_add(t, "Computation Time");
            t = start_clock();
            data_comm_cache[j].get()->populate_cache(
                update_ptr.get(), mpi_requests[i * batches + j - 1], false);
            stop_clock_and_add(t, "Communication Time");
            t = start_clock();
          }

//          this->calc_t_dist_grad_rowptr(csr_block, prevCoordinates, lr, j,
//                                        batch_size, considering_batch_size,
//                                        false);
          stop_clock_and_add(t, "Computation Time");
        }
//        this->update_data_matrix_rowptr(prevCoordinates, j, batch_size);
        update_ptr.get()->clear();

        if (this->grid->world_size > 1) {
          MPI_Request request_batch_update;
          stop_clock_and_add(t, "Computation Time");
          t = start_clock();
//          data_comm_cache[j].get()->transfer_data(update_ptr.get(), false,
//                                                  request_batch_update);
//          mpi_requests[i * batches + j] = request_batch_update;
          if (i == iterations - 1 and j == batches - 1) {
//            data_comm_cache[j].get()->populate_cache(update_ptr.get(),
//                                                     request_batch_update, false);
          }
          stop_clock_and_add(t, "Communication Time");
        }
      }
    }
    if (this->grid->world_size == 1) {
      stop_clock_and_add(t, "Computation Time");
    }
  }

  inline void calc_t_dist_grad_rowptr(CSRLocal<SPT> *csr_block,
                                      DENT *prevCoordinates, DENT lr,
                                      int batch_id, int batch_size,
                                      int block_size, bool local) {

    auto source_start_index = batch_id * batch_size;
    auto source_end_index = std::min((batch_id + 1) * batch_size,this->sp_local_receiver->proc_row_width) -1;

    auto dst_start_index = this->sp_local_receiver->proc_row_width * this->grid->global_rank;
    auto dst_end_index = std::min(static_cast<uint64_t>(this->sp_local_receiver->proc_row_width *(this->grid->global_rank + 1)),this->sp_local_receiver->gCols) -1;

    if (local) {
      calc_embedding(source_start_index, source_end_index, dst_start_index,
                     dst_end_index, csr_block, prevCoordinates, lr, batch_id,
                     batch_size, block_size);
    } else {
      for (int r = 0; r < grid->world_size; r++) {
        if (r != grid->global_rank) {
          dst_start_index = this->sp_local_receiver->proc_row_width * r;
          dst_end_index =
              std::min(static_cast<uint64_t>(this->sp_local_receiver->proc_row_width * (r + 1)),
                       this->sp_local_receiver->gCols) -
              1;
          calc_embedding(source_start_index, source_end_index, dst_start_index,
                         dst_end_index, csr_block, prevCoordinates, lr,
                         batch_id, batch_size, block_size);
        }
      }
    }
  }

  inline void calc_embedding(uint64_t source_start_index,
                             uint64_t source_end_index,
                             uint64_t dst_start_index, uint64_t dst_end_index,
                             CSRLocal<SPT> *csr_block, DENT *prevCoordinates,
                             DENT lr, int batch_id, int batch_size,
                             int block_size) {
    if (csr_block->handler != nullptr) {
      CSRHandle *csr_handle = csr_block->handler.get();

#pragma omp parallel for schedule(static)
      for (uint64_t i = dst_start_index; i <= dst_end_index; i++) {

        DENT forceDiff[embedding_dim];

        uint64_t local_dst = i - (this->grid)->global_rank *
                                     (this->sp_local_receiver)->proc_row_width;
        int target_rank = (int)(i / (this->sp_local_receiver)->proc_row_width);
        bool fetch_from_cache =
            target_rank == (this->grid)->global_rank ? false : true;
        std::array<DENT, embedding_dim> colvec;

        bool matched = false;
        for (uint64_t j = static_cast<uint64_t>(csr_handle->rowStart[i]);
             j < static_cast<uint64_t>(csr_handle->rowStart[i + 1]); j++) {
          if (csr_handle->col_idx[j] >= source_start_index and csr_handle->col_idx[j] <= source_end_index) {
            auto source_id = csr_handle->col_idx[j];
            auto index = source_id - batch_id * batch_size;

            if (!matched) {
              if (fetch_from_cache) {
                colvec = (this->dense_local)
                             ->fetch_data_vector_from_cache(target_rank, i);
                // If not in cache we should fetch that from remote for limited
                // cache
              } else {
                colvec = (this->dense_local)->fetch_local_data(local_dst);
              }
            }
            DENT attrc = 0;
            for (int d = 0; d < embedding_dim; d++) {
              forceDiff[d] = (this->dense_local)
                                 ->nCoordinates[source_id * embedding_dim + d] -
                             colvec[d];
              attrc += forceDiff[d] * forceDiff[d];
            }
            DENT d1 = -2.0 / (1.0 + attrc);
            for (int d = 0; d < embedding_dim; d++) {
              forceDiff[d] = scale(forceDiff[d] * d1);
              prevCoordinates[index * embedding_dim + d] += (lr)*forceDiff[d];
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
            static_cast<uint64_t>(
                ((this->grid)->global_rank * (this->sp_local_receiver)->proc_row_width));
        bool fetch_from_cache = false;

        int owner_rank =
            static_cast<int>(global_col_id / (this->sp_local_receiver)->proc_row_width);
        if (owner_rank != (this->grid)->global_rank) {
          fetch_from_cache = true;
        }
        if (fetch_from_cache) {
          DENT repuls = 0;
          std::array<DENT, embedding_dim> colvec =
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
#pragma omp parallel for schedule(static)
    for (int i = 0; i < (end_row - row_base_index); i++) {
      for (int d = 0; d < embedding_dim; d++) {
        (this->dense_local)
            ->nCoordinates[(row_base_index + i) * embedding_dim + d] +=
            prevCoordinates[i * embedding_dim + d];
      }
    }
  }

  void reset_performance_timers() {
    for (auto it = perf_counter_keys.begin(); it != perf_counter_keys.end();
         it++) {
      call_count[*it] = 0;
      total_time[*it] = 0.0;
    }
  }

  void stop_clock_and_add(my_timer_t &start, string counter_name) {
    if (find(perf_counter_keys.begin(), perf_counter_keys.end(),
             counter_name) != perf_counter_keys.end()) {
      call_count[counter_name]++;
      total_time[counter_name] += stop_clock_get_elapsed(start);
    } else {
      cout << "Error, performance counter " << counter_name
           << " not registered." << endl;
      exit(1);
    }
  }

  void print_performance_statistics() {
    // This is going to assume that all timing starts and ends with a barrier,
    // so that all processors enter and leave the call at the same time. Also,
    // I'm taking an average over several calls by all processors; might want to
    // compute the variance as well.
    if (grid->global_rank == 0) {
      cout << endl;
      cout << "================================" << endl;
      cout << "==== Performance Statistics ====" << endl;
      cout << "================================" << endl;
      //      print_algorithm_info();
    }

    cout << this->json_perf_statistics().dump(4);

    if (grid->global_rank == 0) {
      cout << "=================================" << endl;
    }
  }

  json json_perf_statistics() {
    json j_obj;

    for (auto it = perf_counter_keys.begin(); it != perf_counter_keys.end();
         it++) {
      double val = total_time[*it];

      MPI_Allreduce(MPI_IN_PLACE, &val, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

      // We also have the call count for each statistic timed
      val /= grid->world_size;

      if (grid->global_rank == 0) {
        j_obj[*it] = val;
      }
    }
    return j_obj;
  }

  my_timer_t start_clock() { return std::chrono::steady_clock::now(); }

  double stop_clock_get_elapsed(my_timer_t &start) {
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;
    return diff.count();
  }
};
} // namespace distblas::algo
