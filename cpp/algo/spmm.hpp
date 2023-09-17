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
    int batches = 0;
    int last_batch_size = batch_size;

    if (this->sp_local_receiver->proc_row_width % batch_size == 0) {
      batches =
          static_cast<int>(this->sp_local_receiver->proc_row_width / batch_size);
    } else {
      batches =
          static_cast<int>(this->sp_local_receiver->proc_row_width / batch_size) + 1;
      last_batch_size =
          this->sp_local_receiver->proc_row_width - batch_size * (batches - 1);
    }

    cout << " rank " << this->grid->global_rank << " total batches " << batches
         << endl;


    // first batch onboarding
    unique_ptr<std::vector<DataTuple<DENT, embedding_dim>>> fetch_all_ptr =
        unique_ptr<std::vector<DataTuple<DENT, embedding_dim>>>(
            new vector<DataTuple<DENT, embedding_dim>>());

    vector<MPI_Request *> mpi_requests(batches);

    for (int i = 0; i < batches; i++) {
      MPI_Request fetch_batch;
      MPI_Request fetch_batch_next;
      fetch_all_ptr.get()->clear();

      if (i == 0) {
        auto communicator = unique_ptr<DataComm<SPT, DENT, embedding_dim>>(
            new DataComm<SPT, DENT, embedding_dim>(this->sp_local_receiver,
                                                   this->sp_local_sender, this->dense_local,
                                                   this->grid, i, this->alpha));
        this->data_comm_cache.insert(std::make_pair(i, std::move(communicator)));
        this->data_comm_cache[i].get()->onboard_data();
        if (this->alpha > 0) {
          stop_clock_and_add(t, "Computation Time");
          t = start_clock();
          mpi_requests[i] = &fetch_batch;
          this->data_comm_cache[i].get()->transfer_data(
              fetch_all_ptr.get(), false, (*mpi_requests[i]), 0, i, 0, 0);
          stop_clock_and_add(t, "Communication Time");
          t = start_clock();
        }
      }

      if (batches > 1 and i < batches - 1) {
        auto communicator = unique_ptr<DataComm<SPT, DENT, embedding_dim>>(
            new DataComm<SPT, DENT, embedding_dim>(this->sp_local_receiver,
                                                   this->sp_local_sender, this->dense_local,
                                                   this->grid, i + 1, this->alpha));
        this->data_comm_cache.insert(std::make_pair(i + 1, std::move(communicator)));
        this->data_comm_cache[i + 1].get()->onboard_data();
      }

      if (this->alpha > 0) {
        stop_clock_and_add(t, "Computation Time");
        t = start_clock();
        this->data_comm_cache[i].get()->populate_cache(
            fetch_all_ptr.get(), (*mpi_requests[i]), false, 0, i, false);
        if (batches > 1 and i < batches - 1) {
          mpi_requests[i + 1] = &fetch_batch_next;
          this->data_comm_cache[i + 1].get()->transfer_data(
              fetch_all_ptr.get(), false, (*mpi_requests[i + 1]), 0, i, 0, 0);
        }
        stop_clock_and_add(t, "Communication Time");
        t = start_clock();
      }
    }

    cout << " rank " << this->grid->global_rank << " onboard_data completed "
         << batches << endl;

    DENT *prevCoordinates = static_cast<DENT *>(
        ::operator new(sizeof(DENT[batch_size * embedding_dim])));

    unique_ptr<std::vector<DataTuple<DENT, embedding_dim>>> update_ptr =
        unique_ptr<std::vector<DataTuple<DENT, embedding_dim>>>(
            new vector<DataTuple<DENT, embedding_dim>>());

    unique_ptr<vector<vector<Tuple<DENT>>>> cache_misses_ptr =
        unique_ptr<vector<vector<Tuple<DENT>>>>(
            new vector<vector<Tuple<DENT>>>(this->grid->world_size));

    unique_ptr<vector<vector<uint64_t>>> cache_misses_col_ptr =
        unique_ptr<vector<vector<uint64_t>>>(
            new vector<vector<uint64_t>>(this->grid->world_size));

    size_t total_memory = 0;

    CSRLocal<SPT> *csr_block = (this->sp_local_receiver)->csr_local_data.get();

    int considering_batch_size = batch_size;


//    for (int i = 0; i < iterations; i++) {
//      if (this->grid->global_rank == 0)
//        cout << " rank " << this->grid->global_rank << " iteration " << i << endl;
//
//      for (int j = 0; j < batches; j++) {
//
//        int seed = j + i;
//
//        for (int k = 0; k < batch_size; k += 1) {
//          int IDIM = k * embedding_dim;
//          for (int d = 0; d < embedding_dim; d++) {
//            prevCoordinates[IDIM + d] = 0;
//          }
//        }
//
//        int considering_batch_size = batch_size;
//
//        if (j == batches - 1) {
//          considering_batch_size = last_batch_size;
//        }
//
//        CSRLocal<SPT> *csr_block =
//            (this->sp_local_receiver)->csr_local_data.get();
//
//        if (this->alpha == 0) {
//          int proc_length = get_proc_length(this->beta, this->grid->world_size);
//          int prev_start = 0;
//          for (int k = 1; k < this->grid->world_size; k += proc_length) {
//
//
//            MPI_Request request_batch_update_cyclic;
//            int end_process = get_end_proc(k, this->beta, this->grid->world_size);
//            stop_clock_and_add(t, "Computation Time");
//            t = start_clock();
//            this->data_comm_cache[j].get()->transfer_data(update_ptr.get(), false,
//                                                    request_batch_update_cyclic,
//                                                    i, j, k, end_process);
//
//            stop_clock_and_add(t, "Communication Time");
//            t = start_clock();
//            if (k == 1) {
//              // local computation
//              this->calc_t_dist_grad_rowptr(
//                  csr_block, prevCoordinates, lr, j, batch_size,
//                  considering_batch_size, true, true, cache_misses_ptr.get(),
//                  cache_misses_col_ptr.get(), 0, 0, false);
//            } else if (k > 1) {
//              int prev_end_process =
//                  get_end_proc(prev_start, this->beta, this->grid->world_size);
//              this->calc_t_dist_grad_rowptr(csr_block, prevCoordinates, lr, j,
//                                            batch_size, considering_batch_size,
//                                            false, true, cache_misses_ptr.get(),
//                                            cache_misses_col_ptr.get(),
//                                            prev_start, prev_end_process, true);
//              this->dense_local->invalidate_cache(i, j, true);
//            }
//            stop_clock_and_add(t, "Computation Time");
//            t = start_clock();
//
//            this->data_comm_cache[j].get()->populate_cache(
//                update_ptr.get(), request_batch_update_cyclic, false, i, j,
//                true);
//
//            prev_start = k;
//            update_ptr.get()->clear();
//            stop_clock_and_add(t, "Communication Time");
//            t = start_clock();
//          }
//          int prev_end_process = get_end_proc(prev_start, this->beta, this->grid->world_size);
//
//          this->calc_t_dist_grad_rowptr(
//              csr_block, prevCoordinates, lr, j, batch_size,
//              considering_batch_size, false, true, cache_misses_ptr.get(),
//              cache_misses_col_ptr.get(), prev_start, prev_end_process, true);
//
//          this->dense_local->invalidate_cache(i, j, true);
//          update_ptr.get()->resize(0);
//
//        } else if (this->alpha > 0) {
//          // local computation
//          this->calc_t_dist_grad_rowptr(
//              csr_block, prevCoordinates, lr, j, batch_size,
//              considering_batch_size, true, true, cache_misses_ptr.get(),
//              cache_misses_col_ptr.get(), 0, 0, false);
//
//          if (this->grid->world_size > 1) {
//            stop_clock_and_add(t, "Computation Time");
//            t = start_clock();
//            if (!(i == 0 and j == 0)) {
//              this->data_comm_cache[j].get()->populate_cache(
//                  update_ptr.get(), mpi_requests[i * batches + j - 1], false, i,
//                  j, false);
//            }
//            stop_clock_and_add(t, "Communication Time");
//            t = start_clock();
//          }
//
//          //remote computation
//          this->calc_t_dist_grad_rowptr(
//              csr_block, prevCoordinates, lr, j, batch_size,
//              considering_batch_size, false, true, cache_misses_ptr.get(),
//              cache_misses_col_ptr.get(), 0, this->grid->world_size, false);
//
//          if (this->alpha < 1.0) {
//            MPI_Barrier(MPI_COMM_WORLD);
//            stop_clock_and_add(t, "Computation Time");
//            int proc_length = get_proc_length(this->beta, this->grid->world_size);
//            int prev_start = 0;
//            for (int k = 1; k < this->grid->world_size; k += proc_length) {
//              int end_process = get_end_proc(k, this->beta, this->grid->world_size);
//              cout << "rank " << this->grid->global_rank << " processing  " << k
//                   << " out of " << this->grid->world_size << " with proc length "
//                   << proc_length << " end process " << end_process << endl;
//
//              t = start_clock();
////              this->data_comm_cache[j].get()->transfer_data(
////                  cache_misses_col_ptr.get(), i, j, k, end_process);
//              stop_clock_and_add(t, "Communication Time");
//              t = start_clock();
//              this->calc_t_dist_grad_for_cache_misses(
//                  cache_misses_ptr.get(), prevCoordinates, i, j, batch_size, lr,
//                  k, end_process);
//            }
//          }
//        }
//        total_memory += get_memory_usage();
//
//        if (this->alpha<1.0) {
//          this->dense_local->invalidate_cache(i, j, true);
//        }
//
//        this->update_data_matrix_rowptr(prevCoordinates, j, batch_size);
//
//        if (this->grid->world_size > 1 and !(i == iterations - 1 and j == batches - 1) and this->alpha > 0) {
//          update_ptr.get()->clear();
//          MPI_Request request_batch_update;
//          stop_clock_and_add(t, "Computation Time");
//          t = start_clock();
//          this->data_comm_cache[j].get()->transfer_data(
//              update_ptr.get(), false, request_batch_update, i, j, 0, 0);
//          mpi_requests[i * batches + j] = request_batch_update;
//          stop_clock_and_add(t, "Communication Time");
//          t = start_clock();
//          this->dense_local->invalidate_cache(i, j, false);
//        }
//      }
//    }

    for (int i = 0; i < iterations; i++) {
      if (this->grid->global_rank == 0)
        cout << " rank " << this->grid->global_rank << " iteration " << i << endl;

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
                                      true, cache_misses_ptr.get(),
                                      cache_misses_col_ptr.get(), 0, 0, false);

        // remote computation for first batch
        this->calc_t_dist_grad_rowptr(
            csr_block, prevCoordinates, lr, 0, batch_size,
            considering_batch_size, false, true, cache_misses_ptr.get(),
            cache_misses_col_ptr.get(), 0, this->grid->world_size, false);

        if (this->alpha < 1.0) {
          int proc_length = get_proc_length(this->beta, this->grid->world_size);
          int prev_start = 0;

          for (int k = 1; k < this->grid->world_size; k += proc_length) {
            MPI_Request misses_update_request;

            if (i == 0) {
              auto communicator_cache_miss =
                  unique_ptr<DataComm<SPT, DENT, embedding_dim>>(
                      new DataComm<SPT, DENT, embedding_dim>(
                          this->sp_local_receiver, this->sp_local_sender, this->dense_local, this->grid,
                          i, this->alpha));

              this->data_comm_cache[0].get()->data_comm_cache_misses_update.insert(
                  std::make_pair(k, std::move(communicator_cache_miss)));
            }

            int end_process = get_end_proc(k, this->beta, this->grid->world_size);
            stop_clock_and_add(t, "Computation Time");
            t = start_clock();
            this->data_comm_cache[0].get()->data_comm_cache_misses_update[k].get()->transfer_data(cache_misses_col_ptr.get(), i, 0, k, end_process);
            stop_clock_and_add(t, "Communication Time");
            t = start_clock();
            this->calc_t_dist_grad_for_cache_misses(cache_misses_ptr.get(), prevCoordinates, i, 0, batch_size, lr,k, end_process);
          }
        }
      }

      for (int j = 0; j < batches; j++) {
        int seed = j + i;
        if (j == batches - 1) {
          considering_batch_size = last_batch_size;
        }
        //  pull model code
        if (this->alpha == 0) {
          int proc_length = get_proc_length(this->beta, this->grid->world_size);
          int prev_start = 0;
          for (int k = 1; k < this->grid->world_size; k += proc_length) {

            MPI_Request request_batch_update_cyclic;
            int end_process = get_end_proc(k, this->beta, this->grid->world_size);
            stop_clock_and_add(t, "Computation Time");

            t = start_clock();
            this->data_comm_cache[j].get()->transfer_data(
                update_ptr.get(), false, request_batch_update_cyclic, i, j, k,
                end_process);

            stop_clock_and_add(t, "Communication Time");
            t = start_clock();
            if (k == 1) {
              // local computation
              this->calc_t_dist_grad_rowptr(
                  csr_block, prevCoordinates, lr, j, batch_size,
                  considering_batch_size, true, true, cache_misses_ptr.get(),
                  cache_misses_col_ptr.get(), 0, 0, false);

            } else if (k > 1) {
              int prev_end_process =
                  get_end_proc(prev_start, this->beta, this->grid->world_size);
              this->calc_t_dist_grad_rowptr(csr_block, prevCoordinates, lr, j,
                                            batch_size, considering_batch_size,
                                            false, true, cache_misses_ptr.get(),
                                            cache_misses_col_ptr.get(),
                                            prev_start, prev_end_process, true);
              this->dense_local->invalidate_cache(i, j, true);
            }
            stop_clock_and_add(t, "Computation Time");
            t = start_clock();

            this->data_comm_cache[j].get()->populate_cache(
                update_ptr.get(), request_batch_update_cyclic, false, i, j,
                true);

            prev_start = k;
            update_ptr.get()->clear();
            stop_clock_and_add(t, "Communication Time");
            t = start_clock();
          }
          int prev_end_process =
              get_end_proc(prev_start, this->beta, this->grid->world_size);

          // updating last remote fetched data vectors
          this->calc_t_dist_grad_rowptr(
              csr_block, prevCoordinates, lr, j, batch_size,
              considering_batch_size, false, true, cache_misses_ptr.get(),
              cache_misses_col_ptr.get(), prev_start, prev_end_process, true);

          this->dense_local->invalidate_cache(i, j, true);
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

          // clear up data
          for (int k = 0; k < batch_size; k += 1) {
            int IDIM = k * embedding_dim;
            for (int d = 0; d < embedding_dim; d++) {
              prevCoordinates[IDIM + d] = 0;
            }
          }

          MPI_Request request_batch_update;
          if (this->grid->world_size > 1) {
            update_ptr.get()->clear();
            stop_clock_and_add(t, "Computation Time");
            t = start_clock();

            this->data_comm_cache[j].get()->transfer_data(update_ptr.get(), false, request_batch_update, i, j, 0, 0);

            stop_clock_and_add(t, "Communication Time");
            t = start_clock();
            this->dense_local->invalidate_cache(i, j, false);
          }

          if (j < batches - 1) {
            this->calc_t_dist_grad_rowptr(
                csr_block, prevCoordinates, lr, j + 1, batch_size,
                considering_batch_size, true, true, cache_misses_ptr.get(),
                cache_misses_col_ptr.get(), 0, 0, false);
          }

          if (this->grid->world_size > 1) {
            stop_clock_and_add(t, "Computation Time");
            t = start_clock();
            this->data_comm_cache[j].get()->populate_cache(
                update_ptr.get(), request_batch_update, false, i, j, false);
            stop_clock_and_add(t, "Communication Time");
            t = start_clock();
          }

          if (j < batches - 1) {
            this->calc_t_dist_grad_rowptr(
                csr_block, prevCoordinates, lr, j + 1, batch_size,
                considering_batch_size, false, true, cache_misses_ptr.get(),
                cache_misses_col_ptr.get(), 0, this->grid->world_size, false);

            if (this->alpha < 1.0) {

              int proc_length = get_proc_length(this->beta, this->grid->world_size);
              for (int k = 1; k < this->grid->world_size; k += proc_length) {
                if (i == 0) {
                  auto communicator_cache_miss =
                      unique_ptr<DataComm<SPT, DENT, embedding_dim>>(
                          new DataComm<SPT, DENT, embedding_dim>(
                              this->sp_local_receiver, this->sp_local_sender, this->dense_local, this->grid,
                              i, this->alpha));

                  this->data_comm_cache[j].get()->data_comm_cache_misses_update.insert(
                      std::make_pair(k, std::move(communicator_cache_miss)));
                }

                MPI_Request misses_update_request;
                int end_process = get_end_proc(k, this->beta, this->grid->world_size);
                stop_clock_and_add(t, "Computation Time");
                t = start_clock();
                this->data_comm_cache[j].get()->data_comm_cache_misses_update[k].get()->transfer_data(cache_misses_col_ptr.get(), i, j, k,end_process);
                stop_clock_and_add(t, "Communication Time");
                t = start_clock();
                this->calc_t_dist_grad_for_cache_misses(cache_misses_ptr.get(), prevCoordinates, i, j, batch_size,lr,
                                                        k, end_process);
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
  }

  inline void calc_embedding(uint64_t source_start_index, uint64_t source_end_index,
                 uint64_t dst_start_index, uint64_t dst_end_index,
                 CSRLocal<SPT> *csr_block, DENT *prevCoordinates, DENT lr,
                 int batch_id, int batch_size, int block_size,
                 vector<vector<Tuple<DENT>>> *cache_misses,
                 vector<vector<uint64_t>> *cache_misses_col, bool temp_cache) {
    if (csr_block->handler != nullptr) {
      CSRHandle *csr_handle = csr_block->handler.get();

//      cout<<"executing sppmm"<<endl;
//#pragma omp parallel for schedule(static)
      for (uint64_t i = dst_start_index; i <= dst_end_index; i++) {

        uint64_t local_dst = i - (this->grid)->global_rank * (this->sp_local_receiver)->proc_row_width;
        int target_rank = (int) (i/(this->sp_local_receiver)->proc_row_width);
        bool fetch_from_cache = target_rank == (this->grid)->global_rank ? false : true;
        bool matched = false;
        DENT *array_ptr = nullptr;
        bool col_inserted = false;
        for (uint64_t j = static_cast<uint64_t>(csr_handle->rowStart[i]); j < static_cast<uint64_t>(csr_handle->rowStart[i + 1]); j++) {
          if (csr_handle->col_idx[j] >= source_start_index and csr_handle->col_idx[j] <= source_end_index) {
            DENT forceDiff[embedding_dim];
            auto source_id = csr_handle->col_idx[j];
            auto index = source_id - batch_id * batch_size;

            if (!matched) {
              if (fetch_from_cache) {
                array_ptr = (this->dense_local)
                                ->fetch_data_vector_from_cache(target_rank, i,
                                                               temp_cache);

                if (array_ptr == nullptr) {
                  Tuple<DENT> cacheRef;
                  cacheRef.row = source_id;
                  cacheRef.col = i;
//#pragma omp critical
//                  {
                    (*cache_misses)[target_rank].push_back(cacheRef);
                    if (!col_inserted) {
                      (*cache_misses_col)[target_rank].push_back(i);
                      col_inserted = true;
                    }
//                  }
                  continue;
                }
              }
              matched = true;
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

  inline void update_data_matrix_rowptr(DENT *prevCoordinates, int batch_id, int batch_size) {
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

  inline void calc_t_dist_grad_rowptr(
      CSRLocal<SPT> *csr_block, DENT *prevCoordinates, DENT lr, int batch_id,
      int batch_size, int block_size, bool local, bool col_major,
      vector<vector<Tuple<DENT>>> *cache_misses,
      vector<vector<uint64_t>> *cache_misses_col, int start_process,
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
                       batch_size, block_size, cache_misses, cache_misses_col,
                       fetch_from_temp_cache);
    } else {
      for (int r = start_process; r < end_process; r++) {
        (*cache_misses)[r].clear();
        (*cache_misses_col)[r].clear();
        if (r != this->grid->global_rank) {
          dst_start_index = this->sp_local_receiver->proc_row_width * r;
          dst_end_index =
              std::min(static_cast<uint64_t>(
                           this->sp_local_receiver->proc_row_width * (r + 1)),
                       this->sp_local_receiver->gCols) -
              1;

            calc_embedding(source_start_index, source_end_index,
                           dst_start_index, dst_end_index, csr_block,
                           prevCoordinates, lr, batch_id, batch_size,
                           block_size, cache_misses, cache_misses_col,
                           fetch_from_temp_cache);

        }
      }
    }
  }
};
}
