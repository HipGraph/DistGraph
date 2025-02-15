#pragma once

#include "../embedding/algo.hpp"

using namespace std;
using namespace distblas::core;
using namespace distblas::net;


using namespace distblas::core;

namespace distblas::algo {
    template<typename INDEX_TYPE, typename VALUE_TYPE, size_t embedding_dim>
    class SDDMM {

    private:
        distblas::core::SpMat<VALUE_TYPE> *sp_local_output;
        DenseMat<INDEX_TYPE, VALUE_TYPE, embedding_dim> *dense_local_a;
        DenseMat<INDEX_TYPE, VALUE_TYPE, embedding_dim> *dense_local_b;

        distblas::core::SpMat<VALUE_TYPE> *sp_local_receiver;
        distblas::core::SpMat<VALUE_TYPE> *sp_local_sender;
        distblas::core::SpMat<VALUE_TYPE> *sp_local_native;

        Process3DGrid *grid;

        std::unordered_map<int, unique_ptr < DataComm<INDEX_TYPE, VALUE_TYPE, embedding_dim>>>
        data_comm_cache;

        //cache size controlling hyper parameter
        double alpha = 0;

        //hyper parameter controls the  computation and communication overlapping
        double beta = 1.0;

        //hyper parameter controls the switching the sync vs async commiunication
        bool sync = false;

        //hyper parameter controls the col major or row major  data access
        bool col_major = false;

    public:
        vector<double> timing_info;

        SDDMM(distblas::core::SpMat<VALUE_TYPE> *sp_local_native,
              distblas::core::SpMat<VALUE_TYPE> *sp_local_receiver,
              distblas::core::SpMat<VALUE_TYPE> *sp_local_sender,
              DenseMat<INDEX_TYPE, VALUE_TYPE, embedding_dim> *dense_local_a,
              DenseMat<INDEX_TYPE, VALUE_TYPE, embedding_dim> *dense_local_b,
              distblas::core::SpMat<VALUE_TYPE> *sp_local_output,
              Process3DGrid *grid, double alpha, double beta, bool col_major, bool sync_comm)
                : sp_local_native(sp_local_native), sp_local_receiver(sp_local_receiver),
                  sp_local_sender(sp_local_sender), dense_local_a(dense_local_a), dense_local_b(dense_local_b),
                  grid(grid),
                  alpha(alpha), beta(beta), col_major(col_major), sync(sync_comm), sp_local_output(sp_local_output) {
            this->timing_info = vector<double>(sp_local_receiver->proc_row_width, 0);
        }


        void execute(int iterations, int batch_size, VALUE_TYPE lr) {
            auto t = start_clock();

            int batches = 0;
            int last_batch_size = batch_size;

            if (sp_local_receiver->proc_row_width % batch_size == 0) {
                batches = static_cast<int>(sp_local_receiver->proc_row_width / batch_size);
            } else {
                batches = static_cast<int>(sp_local_receiver->proc_row_width / batch_size) + 1;
                last_batch_size = sp_local_receiver->proc_row_width - batch_size * (batches - 1);
            }

            cout << " rank " << grid->rank_in_col << " total batches " << batches << endl;

            // Buffer used for receive MPI operations data
            auto update_ptr = make_unique<std::vector<DataTuple<VALUE_TYPE, embedding_dim>>>();

            //Buffer used for send MPI operations data
            auto sendbuf_ptr = make_unique<std::vector<DataTuple<VALUE_TYPE, embedding_dim>>>();

            for (int i = 0; i < batches; i++) {
                auto communicator = make_unique<DataComm<INDEX_TYPE, VALUE_TYPE, embedding_dim>>(sp_local_receiver,
                                                                                                 sp_local_sender,
                                                                                                 dense_local_b, grid, i,
                                                                                                 alpha);
                data_comm_cache.insert(std::make_pair(i, std::move(communicator)));
                data_comm_cache[i].get()->onboard_data();
            }

            cout << " rank " << grid->rank_in_col << " onboard_data completed " << batches << endl;

            size_t total_memory = 0;
            CSRLocal<VALUE_TYPE> *csr_block = (col_major) ? (this->sp_local_receiver)->csr_local_data.get()
                                                          : (this->sp_local_native)->csr_local_data.get();
            CSRLocal<VALUE_TYPE> *csr_block_output = (this->sp_local_output)->csr_local_data.get();
            int considering_batch_size = batch_size;

            cout << " rank " << grid->rank_in_col << " onboard_data completed " << batches << endl;

            // One process computations without MPI operations
            if (grid->col_world_size == 1) {
                // local computations for 1 process
                this->calc_t_dist_grad_rowptr(csr_block, csr_block_output, lr,0,
                                              batch_size, considering_batch_size,
                                              true, false, 0, 0, false);
            } else {
                //  pull model code
                this->execute_pull_model_computations(
                        sendbuf_ptr.get(), update_ptr.get(), 0, 0,
                        this->data_comm_cache[0].get(), csr_block, batch_size,
                        considering_batch_size, lr, csr_block_output, 1,
                        true, 0, true);
            }

            total_memory += get_memory_usage();

            total_memory = total_memory / (iterations * batches);
            add_perf_stats(total_memory, "Memory usage");
            stop_clock_and_add(t, "Total Time");
        }

        inline void execute_pull_model_computations(
                std::vector<DataTuple<VALUE_TYPE, embedding_dim>> *sendbuf,
                std::vector<DataTuple<VALUE_TYPE, embedding_dim>> *receivebuf, int iteration,
                int batch, DataComm<INDEX_TYPE, VALUE_TYPE, embedding_dim> *data_comm,
                CSRLocal<VALUE_TYPE> *csr_block, int batch_size, int considering_batch_size,
                double lr, CSRLocal<VALUE_TYPE> *csr_block_output, int comm_initial_start, bool local_execution,
                int first_execution_proc, bool communication) {

            int proc_length = get_proc_length(beta, grid->col_world_size);
            int prev_start = comm_initial_start;

            for (int k = prev_start; k < grid->col_world_size; k += proc_length) {
                int end_process = get_end_proc(k, beta, grid->col_world_size);

                MPI_Request req;

                if (communication) {
                    data_comm->transfer_data(sendbuf, receivebuf, sync, &req, iteration, batch, k, end_process, true);
                }

                if (!sync and communication) {
                    MPI_Ialltoallv(
                            (*sendbuf).data(), data_comm->send_counts_cyclic.data(),
                            data_comm->sdispls_cyclic.data(), DENSETUPLE, (*receivebuf).data(),
                            data_comm->receive_counts_cyclic.data(),
                            data_comm->rdispls_cyclic.data(), DENSETUPLE, grid->col_world, &req);
                }

                if (k == comm_initial_start) {
                    // local computation
                    this->calc_t_dist_grad_rowptr(
                            csr_block, csr_block_output, lr, batch, batch_size,
                            considering_batch_size, local_execution, col_major,
                            first_execution_proc, prev_start, local_execution);

                } else if (k > comm_initial_start) {
                    int prev_end_process = get_end_proc(prev_start, beta, grid->col_world_size);

                    this->calc_t_dist_grad_rowptr(csr_block, csr_block_output, lr, batch,
                                                  batch_size, considering_batch_size, false,
                                                  col_major, prev_start, prev_end_process,
                                                  true);
                }

                if (!sync and communication) {
                    data_comm->populate_cache(sendbuf, receivebuf, &req, sync, iteration, batch, true);
                }

                prev_start = k;
            }

            int prev_end_process = get_end_proc(prev_start, beta, grid->col_world_size);

            // updating last remote fetched data vectors
            this->calc_t_dist_grad_rowptr(csr_block, csr_block_output, lr, batch,
                                          batch_size, considering_batch_size, false,
                                          col_major, prev_start, prev_end_process,
                                          true);

            // dense_local->invalidate_cache(i, j, true);
        }


        inline void calc_t_dist_grad_rowptr(CSRLocal<VALUE_TYPE> *csr_block, CSRLocal<VALUE_TYPE> *csr_block_output,
                                            VALUE_TYPE lr, int batch_id, int batch_size, int block_size,
                                            bool local, bool col_major, int start_process,
                                            int end_process, bool fetch_from_temp_cache) {

            auto source_start_index = batch_id * batch_size;
            auto source_end_index = std::min(static_cast<INDEX_TYPE>((batch_id + 1) * batch_size),
                                             this->sp_local_receiver->proc_row_width) -
                                    1;

            auto dst_start_index =
                    this->sp_local_receiver->proc_col_width * grid->rank_in_col;
            auto dst_end_index =
                    std::min(static_cast<INDEX_TYPE>(this->sp_local_receiver->proc_col_width *
                                                     (grid->rank_in_col + 1)),
                             this->sp_local_receiver->gCols) -
                    1;

            if (local) {
                if (col_major) {
                    calc_embedding(source_start_index, source_end_index, dst_start_index,
                                   dst_end_index, csr_block, csr_block_output, lr, batch_id,
                                   batch_size, block_size, fetch_from_temp_cache);
                } else {
                    calc_embedding_row_major(source_start_index, source_end_index,
                                             dst_start_index, dst_end_index, csr_block,
                                             csr_block_output, lr, batch_id, batch_size,
                                             block_size, fetch_from_temp_cache);
                }
            } else {
                for (int r = start_process; r < end_process; r++) {

                    if (r != grid->rank_in_col) {

                        int computing_rank = (grid->rank_in_col >= r)
                                             ? (grid->rank_in_col - r) % grid->col_world_size
                                             : (grid->col_world_size - r + grid->rank_in_col) % grid->col_world_size;

                        dst_start_index = this->sp_local_receiver->proc_row_width * computing_rank;
                        dst_end_index =
                                std::min(static_cast<INDEX_TYPE>(
                                                 this->sp_local_receiver->proc_row_width * (computing_rank + 1)),
                                         this->sp_local_receiver->gCols) -
                                1;

                        if (col_major) {
                            calc_embedding(source_start_index, source_end_index,
                                           dst_start_index, dst_end_index, csr_block,
                                           csr_block_output, lr, batch_id, batch_size,
                                           block_size, fetch_from_temp_cache);
                        } else {
                            calc_embedding_row_major(source_start_index, source_end_index,
                                                     dst_start_index, dst_end_index, csr_block,
                                                     csr_block_output, lr, batch_id, batch_size,
                                                     block_size, fetch_from_temp_cache);
                        }
                    }
                }
            }
        }

        inline void calc_embedding(INDEX_TYPE source_start_index,
                                   INDEX_TYPE source_end_index,
                                   INDEX_TYPE dst_start_index, INDEX_TYPE dst_end_index,
                                   CSRLocal<VALUE_TYPE> *csr_block, CSRLocal<VALUE_TYPE> *csr_block_output,
                                   VALUE_TYPE lr, int batch_id, int batch_size,
                                   int block_size, bool temp_cache) {
            if (csr_block->handler != nullptr) {
                CSRHandle *csr_handle = csr_block->handler.get();

#pragma omp parallel for schedule(static)
                for (INDEX_TYPE i = dst_start_index; i <= dst_end_index; i++) {

                    INDEX_TYPE local_dst = i - (grid)->rank_in_col *
                                               (this->sp_local_receiver)->proc_row_width;
                    int target_rank = (int) (i / (this->sp_local_receiver)->proc_row_width);
                    bool fetch_from_cache =
                            target_rank == (grid)->rank_in_col ? false : true;


                    bool matched = false;
                    std::array<VALUE_TYPE, embedding_dim> array_ptr;
                    bool col_inserted = false;
                    for (INDEX_TYPE j = static_cast<INDEX_TYPE>(csr_handle->rowStart[i]);
                         j < static_cast<INDEX_TYPE>(csr_handle->rowStart[i + 1]); j++) {
                        if (csr_handle->col_idx[j] >= source_start_index and
                            csr_handle->col_idx[j] <= source_end_index) {
                            VALUE_TYPE forceDiff[embedding_dim];
                            auto source_id = csr_handle->col_idx[j];
                            auto index = source_id - batch_id * batch_size;

                            if (!matched) {
                                if (fetch_from_cache) {
                                    unordered_map <INDEX_TYPE, CacheEntry<VALUE_TYPE, embedding_dim>>
                                            &arrayMap =
                                            (temp_cache)
                                            ? (*this->dense_local_b->tempCachePtr)[target_rank]
                                            : (*this->dense_local_b->cachePtr)[target_rank];
                                    array_ptr = arrayMap[i].value;
                                }
                                matched = true;
                            }
                            VALUE_TYPE val = 0;
                            for (int d = 0; d < embedding_dim; d++) {
                                if (!fetch_from_cache) {
                                    val += (this->dense_local_a)->nCoordinates[i * embedding_dim + d] *
                                           (this->dense_local_b)->nCoordinates[local_dst * embedding_dim + d] * lr;
                                } else {
                                    val += (this->dense_local_a)->nCoordinates[i * embedding_dim + d] * array_ptr[d] *
                                           lr;
                                }
                            }
                            CSRHandle *csr_handle_output = csr_block_output->handler.get();
                            csr_handle_output->values[j] = val;
                        }
                    }
                }
            }
        }

        inline void calc_embedding_row_major(INDEX_TYPE source_start_index,
                                             INDEX_TYPE source_end_index, INDEX_TYPE dst_start_index,
                                             INDEX_TYPE dst_end_index, CSRLocal<VALUE_TYPE> *csr_block,
                                             CSRLocal<VALUE_TYPE> *csr_block_output, VALUE_TYPE lr, int batch_id,
                                             int batch_size, int block_size, bool temp_cache) {
            if (csr_block->handler != nullptr) {
                CSRHandle *csr_handle = csr_block->handler.get();

#pragma omp parallel for schedule(static) // enable for full batch training or // batch size larger than 1000000
                for (INDEX_TYPE i = source_start_index; i <= source_end_index; i++) {

                    INDEX_TYPE index = i - batch_id * batch_size;

                    for (INDEX_TYPE j = static_cast<INDEX_TYPE>(csr_handle->rowStart[i]);
                         j < static_cast<INDEX_TYPE>(csr_handle->rowStart[i + 1]); j++) {
                        auto dst_id = csr_handle->col_idx[j];

                        if (dst_id >= dst_start_index and dst_id < dst_end_index) {
                            INDEX_TYPE local_dst =
                                    dst_id - (grid)->rank_in_col *
                                             (this->sp_local_receiver)->proc_col_width;
                            int target_rank =
                                    (int) (dst_id / (this->sp_local_receiver)->proc_col_width);
                            bool fetch_from_cache =
                                    target_rank == (grid)->rank_in_col ? false : true;

                            VALUE_TYPE forceDiff[embedding_dim];
                            std::array<VALUE_TYPE, embedding_dim> array_ptr;

                            if (fetch_from_cache) {
                                unordered_map <INDEX_TYPE, CacheEntry<VALUE_TYPE, embedding_dim>>
                                        &arrayMap =
                                        (temp_cache)
                                        ? (*this->dense_local_b->tempCachePtr)[target_rank]
                                        : (*this->dense_local_b->cachePtr)[target_rank];
                                array_ptr = arrayMap[dst_id].value;
                            }
                            auto t = start_clock();
                            VALUE_TYPE val = 0;
                            for (int d = 0; d < embedding_dim; d++) {
                                if (!fetch_from_cache) {
                                    val += (this->dense_local_a)->nCoordinates[i * embedding_dim + d] *
                                           (this->dense_local_b)->nCoordinates[local_dst * embedding_dim + d] * lr;
                                } else {
                                    val += (this->dense_local_a)->nCoordinates[i * embedding_dim + d] * array_ptr[d] *
                                           lr;
                                }
                            }
                            CSRHandle *csr_handle_output = csr_block_output->handler.get();
                            csr_handle_output->values[j] = val;
                            auto time = stop_clock_get_elapsed(t);
                            timing_info[index] += time;

                        }
                    }
                }
            }
        }
    };
} // namespace distblas::algo

