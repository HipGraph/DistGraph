#pragma once
#include "../core/common.h"
#include "../core/csr_local.hpp"
#include "../core/dense_mat.hpp"
#include "../core/sparse_mat.hpp"
#include "../core/json.hpp"
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
      batches = static_cast<int>(sp_local_receiver->proc_row_width / batch_size);
    } else {
      batches = static_cast<int>(sp_local_receiver->proc_row_width / batch_size) + 1;
      // TODO:Error prone
      last_batch_size = sp_local_receiver->proc_row_width - batch_size * (batches - 1);
    }

    cout << " rank " << this->grid->global_rank << " total batches " << batches << endl;

    auto negative_update_com = unique_ptr<DataComm<SPT, DENT, embedding_dim>>(
        new DataComm<SPT, DENT, embedding_dim>(sp_local_receiver, sp_local_sender, dense_local, grid,-1));

    MPI_Request fetch_all;
    negative_update_com.get()->onboard_data();
    stop_clock_and_add(t, "Computation Time");

    t = start_clock();
    negative_update_com.get()->transfer_data(fetch_all_ptr.get(), false,fetch_all);
    stop_clock_and_add(t, "Communication Time");

//    t = start_clock();
//    for (int i = 0; i < batches; i++) {
//      auto communicator = unique_ptr<DataComm<SPT, DENT, embedding_dim>>(
//          new DataComm<SPT, DENT, embedding_dim>(
//              sp_local_metadata, sp_local_trans, dense_local, grid));
//      data_comm_cache.insert(std::make_pair(i, std::move(communicator)));
//      data_comm_cache[i].get()->onboard_data(i);
//    }
//    stop_clock_and_add(t, "Computation Time");
//    t = start_clock();
//    negative_update_com.get()->populate_cache(fetch_all_ptr.get(), fetch_all,
//                                              false);
//    stop_clock_and_add(t, "Communication Time");
//    MPI_Barrier(MPI_COMM_WORLD); //MPI Barrier
//    t = start_clock();
//    DENT *prevCoordinates = static_cast<DENT *>(
//        ::operator new(sizeof(DENT[batch_size * embedding_dim])));
//
//    unique_ptr<std::vector<DataTuple<DENT, embedding_dim>>>
//        results_negative_ptr =
//            unique_ptr<std::vector<DataTuple<DENT, embedding_dim>>>(
//                new vector<DataTuple<DENT, embedding_dim>>());
//
//    unique_ptr<std::vector<DataTuple<DENT, embedding_dim>>> update_ptr =
//        unique_ptr<std::vector<DataTuple<DENT, embedding_dim>>>(
//            new vector<DataTuple<DENT, embedding_dim>>());
//    vector<MPI_Request> mpi_requests(iterations*batches);
//    for (int i = 0; i < iterations; i++) {
//      cout<<" global rank "<<grid->global_rank<<endl;
//      for (int j = 0; j < batches; j++) {
//
//        int seed = j + i;
//
//        for (int i = 0; i < batch_size; i += 1) {
//          int IDIM = i * embedding_dim;
//          for (int d = 0; d < embedding_dim; d++) {
//            prevCoordinates[IDIM + d] = 0;
//          }
//        }
//
//        // negative samples generation
//        vector<uint64_t> random_number_vec =
//            generate_random_numbers(0, (this->sp_local)->gRows, seed, ns);
//
//        MPI_Request request_negative_update;
//
//        if (this->grid->world_size>1){
//          results_negative_ptr.get()->clear();
//          stop_clock_and_add(t, "Computation Time");
//          t = start_clock();
//          negative_update_com.get()->transfer_data(
//              random_number_vec, false, results_negative_ptr.get(), request_negative_update);
//          stop_clock_and_add(t, "Communication Time");
//          t = start_clock();
//        }
//
//        CSRLinkedList<SPT> *batch_list = (this->sp_local)->get_batch_list(0);
//
//        auto head = batch_list->getHeadNode();
//        CSRLocal<SPT> *csr_block_local = (head.get())->data.get();
//        CSRLocal<SPT> *csr_block_remote = nullptr;
//
//        if (this->grid->world_size > 1) {
//          auto remote = (head.get())->next;
//          csr_block_remote = (remote.get())->data.get();
//        }
//
//        int working_rank = 0;
//        bool fetch_remote =
//            (working_rank == ((this->grid)->global_rank)) ? false : true;
//
//        int considering_batch_size = batch_size;
//
//        if (j == batches - 1) {
//          considering_batch_size = last_batch_size;
//        }
//
//        this->calc_t_dist_grad_rowptr(csr_block_local, prevCoordinates, lr, j,
//                                      batch_size, considering_batch_size);
//
//        if (this->grid->world_size > 1) {
//          if (!(i==0 and j==0) ) {
//            stop_clock_and_add(t, "Computation Time");
//            t = start_clock();
//            data_comm_cache[j].get()->populate_cache(
//                update_ptr.get(), mpi_requests[i*batches+j-1], false);
//            stop_clock_and_add(t, "Communication Time");
//            t = start_clock();
//          }
//
//          this->calc_t_dist_grad_rowptr(csr_block_remote, prevCoordinates, lr,
//                                        j, batch_size, considering_batch_size);
//          stop_clock_and_add(t, "Computation Time");
//          t = start_clock();
//          negative_update_com.get()->populate_cache(results_negative_ptr.get(),
//                                                    request_negative_update, false);
//          stop_clock_and_add(t, "Communication Time");
//          t = start_clock();
//        }
//
//        this->calc_t_dist_replus_rowptr(prevCoordinates, random_number_vec, lr,
//                                        j, batch_size, considering_batch_size);
//        this->update_data_matrix_rowptr(prevCoordinates, j, batch_size);
//        update_ptr.get()->clear();
//
//        if (this->grid->world_size > 1) {
//          MPI_Request request_batch_update;
//          stop_clock_and_add(t, "Computation Time");
//          t = start_clock();
//          data_comm_cache[j].get()->transfer_data(update_ptr.get(), false,
//                                                  false, request_batch_update);
//          mpi_requests[i*batches+j]=request_batch_update;
//          if (i== iterations-1 and j==batches-1) {
//            data_comm_cache[j].get()->populate_cache(
//                update_ptr.get(), request_batch_update, false);
//          }
//          stop_clock_and_add(t, "Communication Time");
//        }
//      }
//    }
//    if (this->grid->world_size == 0) {
//      stop_clock_and_add(t, "Computation Time");
//    }
  }

  inline void calc_t_dist_grad_rowptr(CSRLocal<SPT> *csr_block,
                                      DENT *prevCoordinates, DENT lr,
                                      int batch_id, int batch_size,
                                      int block_size) {

    auto row_base_index = batch_id * batch_size;

    if (csr_block->handler != nullptr) {
      CSRHandle *csr_handle = csr_block->handler.get();

#pragma omp parallel for schedule(static)
      for (uint64_t i = row_base_index; i < row_base_index + block_size; i++) {
        uint64_t row_id = i;
        int ind = i - row_base_index;

        DENT forceDiff[embedding_dim];
//        #pragma forceinline
//        #pragma omp simd
        for (uint64_t j = static_cast<uint64_t>(csr_handle->rowStart[i]);
             j < static_cast<uint64_t>(csr_handle->rowStart[i + 1]); j++) {

          uint64_t global_col_id = static_cast<uint64_t>(csr_handle->values[j]);

          uint64_t local_col =
              global_col_id -
              (this->grid)->global_rank * (this->sp_local)->proc_row_width;
          int target_rank =
              (int)(global_col_id / (this->sp_local)->proc_row_width);
          bool fetch_from_cache =
              target_rank == (this->grid)->global_rank ? false : true;

          if (fetch_from_cache) {

            std::array<DENT, embedding_dim> colvec =
                (this->dense_local)
                    ->fetch_data_vector_from_cache(target_rank, global_col_id);
            DENT attrc = 0;
            for (int d = 0; d < embedding_dim; d++) {
              forceDiff[d] = (this->dense_local)
                                 ->nCoordinates[row_id * embedding_dim + d] -
                             colvec[d];
              attrc += forceDiff[d] * forceDiff[d];
            }
            DENT d1 = -2.0 / (1.0 + attrc);
            for (int d = 0; d < embedding_dim; d++) {
              forceDiff[d] = scale(forceDiff[d] * d1);
              prevCoordinates[ind * embedding_dim + d] += (lr)*forceDiff[d];
            }

          } else {

            DENT attrc = 0;
            for (int d = 0; d < embedding_dim; d++) {
              forceDiff[d] = (this->dense_local)
                                 ->nCoordinates[row_id * embedding_dim + d] -
                             (this->dense_local)
                                 ->nCoordinates[local_col * embedding_dim + d];

              attrc += forceDiff[d] * forceDiff[d];
            }
            DENT d1 = -2.0 / (1.0 + attrc);
            for (int d = 0; d < embedding_dim; d++) {
              forceDiff[d] = scale(forceDiff[d] * d1);
              prevCoordinates[ind * embedding_dim + d] += (lr)*forceDiff[d];
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
//      #pragma forceinline
//      #pragma omp simd
      for (int j = 0; j < col_ids.size(); j++) {
        uint64_t global_col_id = col_ids[j];
        uint64_t local_col_id =
            global_col_id -
            static_cast<uint64_t>(
                ((this->grid)->global_rank * (this->sp_local)->proc_row_width));
        bool fetch_from_cache = false;

        int owner_rank =
            static_cast<int>(global_col_id / (this->sp_local)->proc_row_width);
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
                           ((this->sp_local)->proc_row_width));
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

  my_timer_t start_clock() {
    return std::chrono::steady_clock::now();
  }

  double stop_clock_get_elapsed(my_timer_t &start) {
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;
    return diff.count();
  }
};
} // namespace distblas::embedding
