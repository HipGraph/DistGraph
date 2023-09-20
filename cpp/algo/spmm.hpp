#pragma once
#include "algo.hpp"

using namespace std;
using namespace distblas::core;
using namespace distblas::net;
using namespace Eigen;

using namespace distblas::core;

namespace distblas::algo {
template <typename SPT, typename DENT, size_t embedding_dim>
class SpMMAlgo : public EmbeddingAlgo<SPT, DENT, embedding_dim> {

private:
  DenseMat<SPT, DENT, embedding_dim> *dense_local_output;

public:
  SpMMAlgo(distblas::core::SpMat<SPT> *sp_local_native,
           distblas::core::SpMat<SPT> *sp_local_receiver,
           distblas::core::SpMat<SPT> *sp_local_sender,
           DenseMat<SPT, DENT, embedding_dim> *dense_local_input,
           DenseMat<SPT, DENT, embedding_dim> *dense_local_output,
           Process3DGrid *grid, double alpha, double beta, DENT MAX_BOUND,
           DENT MIN_BOUND)
      : EmbeddingAlgo<SPT, DENT, embedding_dim>(
            sp_local_native, sp_local_receiver, sp_local_sender,
            dense_local_input, grid, alpha, beta, MAX_BOUND, MIN_BOUND) {
    this->dense_local_output = dense_local_output;
  }

  void algo_spmm(int iterations, int batch_size, DENT lr) {
    auto t = start_clock();
    int batches = 1;

    cout << " rank " << this->grid->global_rank << " total batches " << batches
         << endl;

    // first batch onboarding
    unique_ptr<std::vector<DataTuple<DENT, embedding_dim>>> update_ptr =
        unique_ptr<std::vector<DataTuple<DENT, embedding_dim>>>(
            new vector<DataTuple<DENT, embedding_dim>>());

    MPI_Request fetch_batch;
    vector<MPI_Request*> mpi_requests(iterations);

    mpi_requests[0] = &fetch_batch;
    auto communicator = unique_ptr<DataComm<SPT, DENT, embedding_dim>>(
        new DataComm<SPT, DENT, embedding_dim>(
            this->sp_local_receiver, this->sp_local_sender, this->dense_local,
            this->grid, 0, this->alpha));

    this->data_comm_cache.insert(std::make_pair(0, std::move(communicator)));
    this->data_comm_cache[0].get()->onboard_data();



    if (this->alpha > 0) {
      stop_clock_and_add(t, "Computation Time");
      t = start_clock();

      int proc_length = get_proc_length(this->alpha, this->grid->world_size);
      this->data_comm_cache[0].get()->transfer_data(update_ptr.get(), false,
                                                    (*mpi_requests[0]), 0, 0, 1,
                                                    proc_length, false);

      stop_clock_and_add(t, "Communication Time");
      t = start_clock();
    }

    cout << " rank " << this->grid->global_rank << " onboard_data completed "
         << batches << endl;

    DENT *prevCoordinates = static_cast<DENT *>(
        ::operator new(sizeof(DENT[batch_size * embedding_dim])));

    size_t total_memory = 0;

    CSRLocal<SPT> *csr_block = (this->sp_local_receiver)->csr_local_data.get();

    int considering_batch_size = batch_size;

    for (int i = 0; i < iterations; i++) {
      total_memory += get_memory_usage();
      if (this->grid->global_rank == 0)
        cout << " rank " << this->grid->global_rank << " iteration " << i
             << endl;

      if (this->alpha > 0) {

        for (int k = 0; k < batch_size; k += 1) {
          int IDIM = k * embedding_dim;
          for (int d = 0; d < embedding_dim; d++) {
            prevCoordinates[IDIM + d] = 0;
          }
        }


        // local computation for first batch
        this->calc_t_dist_grad_rowptr(csr_block, prevCoordinates, lr, 0,
                                      batch_size, considering_batch_size, true,
                                      true, 0, this->grid->world_size, false);

        stop_clock_and_add(t, "Computation Time");
        t = start_clock();
        this->data_comm_cache[0].get()->populate_cache(update_ptr.get(), (*mpi_requests[i]), false, i, 0, false);
        stop_clock_and_add(t, "Communication Time");
        t = start_clock();

        if (this->alpha < 1.0) {

          int proc_length = get_proc_length(this->beta, this->grid->world_size);
          int prev_start = get_end_proc(1, this->alpha, this->grid->world_size);
          size_t temp_mem = 0;
          for (int k = prev_start; k < this->grid->world_size; k += proc_length) {
            MPI_Request misses_update_request;
            int end_process =
                get_end_proc(k, this->beta, this->grid->world_size);
            stop_clock_and_add(t, "Computation Time");

            t = start_clock();
            update_ptr.get()->clear();
            this->data_comm_cache[0].get()->transfer_data(update_ptr.get(), false, misses_update_request, i, 0, k,end_process, true);
            stop_clock_and_add(t, "Communication Time");
            t = start_clock();
            temp_mem += get_memory_usage();
            if (k == prev_start) {
              // remote computation for first batch
              this->calc_t_dist_grad_rowptr(csr_block, prevCoordinates, lr, 0,batch_size,
                                            considering_batch_size,false, true, 0,
                                            this->grid->world_size, false);

            } else if (k > prev_start) {
              // updating last remote fetched data vectors
              int prev_end_process =
                  get_end_proc(prev_start, this->beta, this->grid->world_size);
              this->calc_t_dist_grad_rowptr(csr_block, prevCoordinates, lr, 0,
                                            batch_size, considering_batch_size,
                                            false, true, prev_start,
                                            prev_end_process, true);
              this->dense_local->invalidate_cache(i, 0, true);
            }
            stop_clock_and_add(t, "Computation Time");


            t = start_clock();
            this->data_comm_cache[0].get()->populate_cache(update_ptr.get(), misses_update_request, false, i, 0, true);
            stop_clock_and_add(t, "Communication Time");
            t = start_clock();
            prev_start = k;
          }

          temp_mem = temp_mem/(prev_start+1);
          total_memory +=temp_mem;

        } else {
          // remote computation for first batch
          this->calc_t_dist_grad_rowptr(
              csr_block, prevCoordinates, lr, 0, batch_size,
              considering_batch_size, false, true, 0,
              this->grid->world_size,  false);
        }

        this->update_data_matrix_rowptr(prevCoordinates, 0, batch_size);

        if (this->grid->world_size > 1 and i < iterations - 1) {
          update_ptr.get()->clear();
          int end_process =
              get_end_proc(1, this->alpha, this->grid->world_size);
          stop_clock_and_add(t, "Computation Time");

          t = start_clock();
          MPI_Request request_batch_update;
          mpi_requests[i + 1] = &request_batch_update;
          this->data_comm_cache[0].get()->transfer_data(update_ptr.get(), false, (*mpi_requests[i + 1]), i, 0, 1,
              end_process, false);
          stop_clock_and_add(t, "Communication Time");
          t = start_clock();
          this->dense_local->invalidate_cache(i, 0, false);
        }

      } else if (this->alpha == 0) {
        int proc_length = get_proc_length(this->beta, this->grid->world_size);
        int prev_start = 0;
        size_t temp_mem = 0;
        for (int k = 1; k < this->grid->world_size; k += proc_length) {

          MPI_Request request_batch_update_cyclic;
          int end_process = get_end_proc(k, this->beta, this->grid->world_size);
          stop_clock_and_add(t, "Computation Time");

          t = start_clock();
          this->data_comm_cache[0].get()->transfer_data(
              update_ptr.get(), false, request_batch_update_cyclic, i, 0, k,
              end_process, true);
          stop_clock_and_add(t, "Communication Time");
          t = start_clock();
          temp_mem += get_memory_usage();
          if (k == 1) {
            // local computation
            this->calc_t_dist_grad_rowptr(
                csr_block, prevCoordinates, lr, 0, batch_size,
                considering_batch_size, true, true,
                0, 0, false);

          } else if (k > 1) {
            int prev_end_process =
                get_end_proc(prev_start, this->beta, this->grid->world_size);

            this->calc_t_dist_grad_rowptr(
                csr_block, prevCoordinates, lr, 0, batch_size,
                considering_batch_size, false, true,
                prev_start, prev_end_process, true);

            this->dense_local->invalidate_cache(i, 0, true);
          }
          stop_clock_and_add(t, "Computation Time");
          t = start_clock();

          this->data_comm_cache[0].get()->populate_cache(
              update_ptr.get(), request_batch_update_cyclic, false, i, 0, true);

          prev_start = k;
          update_ptr.get()->clear();
          stop_clock_and_add(t, "Communication Time");
          t = start_clock();
        }

        temp_mem = temp_mem/(prev_start+1);
        total_memory +=temp_mem;

        int prev_end_process =
            get_end_proc(prev_start, this->beta, this->grid->world_size);

        // updating last remote fetched data vectors
        this->calc_t_dist_grad_rowptr(
            csr_block, prevCoordinates, lr, 0, batch_size,
            considering_batch_size, false, true,
            prev_start, prev_end_process, true);

        this->dense_local->invalidate_cache(i, 0, true);
        update_ptr.get()->resize(0);

        this->update_data_matrix_rowptr(prevCoordinates, 0, batch_size);
      }
      total_memory += get_memory_usage();
    }

    total_memory = total_memory / (iterations * batches * 3);
    add_memory(total_memory, "Memory usage");
    stop_clock_and_add(t, "Computation Time");
  }

  inline void calc_embedding(uint64_t source_start_index,
                             uint64_t source_end_index,
                             uint64_t dst_start_index, uint64_t dst_end_index,
                             CSRLocal<SPT> *csr_block, DENT *prevCoordinates,
                             DENT lr, int batch_id, int batch_size,
                             int block_size, bool temp_cache) {
    if (csr_block->handler != nullptr) {
      CSRHandle *csr_handle = csr_block->handler.get();

      //      cout<<"executing sppmm"<<endl;
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
                matched = true;
              }
            }

            if (fetch_from_cache) {
              for (int d = 0; d < embedding_dim; d++) {
                prevCoordinates[index * embedding_dim + d] += lr * array_ptr[d];
              }
            } else {
              for (int d = 0; d < embedding_dim; d++) {
                prevCoordinates[index * embedding_dim + d] +=
                    (lr) * (this->dense_local)
                               ->nCoordinates[local_dst * embedding_dim + d];
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
    //    #pragma omp parallel for schedule(static)
    for (int i = 0; i < (end_row - row_base_index); i++) {
      for (int d = 0; d < embedding_dim; d++) {
        (this->dense_local_output)
            ->nCoordinates[(row_base_index + i) * embedding_dim + d] +=
            prevCoordinates[i * embedding_dim + d];
      }
    }
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
      calc_embedding(source_start_index, source_end_index, dst_start_index,
                     dst_end_index, csr_block, prevCoordinates, lr, batch_id,
                     batch_size, block_size, fetch_from_temp_cache);
    } else {
      for (int r = start_process; r < end_process; r++) {
        if (r != this->grid->global_rank) {
          dst_start_index = this->sp_local_receiver->proc_row_width * r;
          dst_end_index =
              std::min(static_cast<uint64_t>(
                           this->sp_local_receiver->proc_row_width * (r + 1)),
                       this->sp_local_receiver->gCols) -
              1;

          calc_embedding(source_start_index, source_end_index, dst_start_index,
                         dst_end_index, csr_block, prevCoordinates, lr,
                         batch_id, batch_size, block_size,
                         fetch_from_temp_cache);
        }
      }
    }
  }
};
} // namespace distblas::algo
