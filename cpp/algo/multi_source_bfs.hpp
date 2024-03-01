#pragma once

#include "../core/sparse_mat.hpp"
#include "../core/sparse_mat_tile.hpp"
#include "../algo/spgemm_with_tiling.hpp"

using namespace distblas::core;

namespace distblas::algo {

template <typename INDEX_TYPE, typename VALUE_TYPE, size_t embedding_dim>
class MultiSourceBFS {

private:
  distblas::core::SpMat<VALUE_TYPE> *sp_local_receiver;
  distblas::core::SpMat<VALUE_TYPE> *sp_local_sender;
  distblas::core::SpMat<VALUE_TYPE> *sp_local_native;
  distblas::core::SpMat<VALUE_TYPE> *sparse_local;

  Process3DGrid *grid;

  // cache size controlling hyper parameter
  double alpha = 0;

  // hyper parameter controls the  computation and communication overlapping
  double beta = 1.0;

  // hyper parameter controls the switching the sync vs async communication
  bool sync = true;

  // hyper parameter controls the col major or row major  data access
  bool col_major = false;

  double tile_width_fraction;

  bool hash_spgemm = false;

public:
  MultiSourceBFS(distblas::core::SpMat<VALUE_TYPE> *sp_local_native,
                      distblas::core::SpMat<VALUE_TYPE> *sp_local_receiver,
                      distblas::core::SpMat<VALUE_TYPE> *sp_local_sender,
                      distblas::core::SpMat<VALUE_TYPE> *sparse_local,
                      Process3DGrid *grid, double alpha, double beta,
                      bool col_major, bool sync_comm,
                      double tile_width_fraction, bool hash_spgemm)
      : sp_local_native(sp_local_native), sp_local_receiver(sp_local_receiver),
        sp_local_sender(sp_local_sender), sparse_local(sparse_local),
        grid(grid), alpha(alpha), beta(beta), col_major(col_major),
        sync(sync_comm), tile_width_fraction(tile_width_fraction) {
    this->hash_spgemm = hash_spgemm;
  }

  json execute(int iterations, int batch_size, VALUE_TYPE lr) {
    json jobj;
    distblas::core::SpMat<VALUE_TYPE> *sparse_input = nullptr;
    unique_ptr<DenseMat<INDEX_TYPE,VALUE_TYPE,embedding_dim>> state_holder= make_unique<DenseMat<INDEX_TYPE,VALUE_TYPE,embedding_dim>>(grid,sp_local_receiver->proc_row_width);

    for (int i = 0; i < iterations; i++) {
      size_t total_memory = 0;
      double output_nnz = 0;
      if (i == 0) {
        sparse_input = sparse_local;
      }
      auto rows =  sp_local_receiver->proc_row_width;
      auto cols = static_cast<INDEX_TYPE>(embedding_dim);
      auto sparse_out = make_shared<distblas::core::SpMat<VALUE_TYPE>>(grid,rows,cols,hash_spgemm);
      unique_ptr<distblas::algo::SpGEMMAlgoWithTiling<INDEX_TYPE, VALUE_TYPE,embedding_dim>>
          spgemm_algo = unique_ptr<distblas::algo::SpGEMMAlgoWithTiling<
              INDEX_TYPE, VALUE_TYPE, embedding_dim>>(
              new distblas::algo::SpGEMMAlgoWithTiling<INDEX_TYPE, VALUE_TYPE,
                                                       embedding_dim>(
                  sp_local_native, sp_local_receiver, sp_local_sender,
                  sparse_input, sparse_out.get(), grid, alpha, beta, col_major,
                  sync, tile_width_fraction, hash_spgemm,state_holder.get()));

      auto t = start_clock();
      spgemm_algo.get()->algo_spgemm(1, batch_size, lr);
      this->update_state_holder(sparse_input,state_holder.get());
      stop_clock_and_add(t, "Total Time");
      total_memory += get_memory_usage();


      output_nnz =static_cast<double>((sparse_out->csr_local_data)->handler->rowStart[(sparse_out->csr_local_data)->handler->rowStart.size() - 1]);
      double totalSum = std::accumulate((*(state_holder->nnz_count)).begin(), (*(state_holder->nnz_count)).end(), 0);
      (*(sparse_input->csr_local_data)) =(*(sparse_out->csr_local_data));
      add_perf_stats(totalSum,"Output NNZ");
      if (output_nnz>0) {
        add_perf_stats(output_nnz, "BFS Frontier");
      }
      add_perf_stats(total_memory, "Memory usage");

      jobj[i]=json_perf_statistics();
      reset_performance_timers();
    }
   return jobj;
  }

  void update_state_holder( distblas::core::SpMat<VALUE_TYPE> *sparse_input, DenseMat<INDEX_TYPE,VALUE_TYPE,embedding_dim> *dense_mat){
    CSRHandle *handle = sparse_input->csr_local_data->handler.get();
    #pragma omp parallel for
    for(auto i=0;i<handle->rowStart.size()-1;i++){
      auto bfs_frontier=(*(dense_mat->nnz_count))[i];
      for(auto j=handle->rowStart[i];j<handle->rowStart[i+1];j++){
        auto d = handle->col_idx[j];
        (*(dense_mat->state_metadata))[i][d]=1;
        (*(dense_mat->nnz_count))[i]++;
      }
    }
  }
};

} // namespace distblas::algo

