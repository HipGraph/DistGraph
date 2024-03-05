#pragma once

#include "../core/sparse_mat.hpp"
#include "../core/sparse_mat_tile.hpp"
#include "../algo/spgemm_with_tiling.hpp"

using namespace distblas::core;

namespace distblas::algo {

template <typename INDEX_TYPE, typename VALUE_TYPE, size_t embedding_dim>
class Baseline {

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
  Baseline(distblas::core::SpMat<VALUE_TYPE> *sp_local_native,
                 distblas::core::SpMat<VALUE_TYPE> *sp_local_receiver,
                 distblas::core::SpMat<VALUE_TYPE> *sp_local_sender,
                 distblas::core::SpMat<VALUE_TYPE> *sparse_local,
                 Process3DGrid *grid, double alpha, double beta, bool col_major,
                 bool sync_comm, double tile_width_fraction, bool hash_spgemm)
      : sp_local_native(sp_local_native), sp_local_receiver(sp_local_receiver),
        sp_local_sender(sp_local_sender), sparse_local(sparse_local),
        grid(grid), alpha(alpha), beta(beta), col_major(col_major),
        sync(sync_comm), tile_width_fraction(tile_width_fraction) {
    this->hash_spgemm = hash_spgemm;
  }

  json execute(int iterations, int batch_size, VALUE_TYPE lr) {
    json jobj;

    for (int i = 0; i < iterations; i++) {
      size_t total_memory = 0;
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
      spgemm_algo.get()->algo_spgemm(1, batch_size, lr,enable_remote);
      auto size_r = sparse_out->csr_local_data->handler->rowStart.size();
      double output_nnz = sparse_out->csr_local_data->handler->rowStart[size_r-1];
      auto density =   (output_nnz/(sp_local_receiver->proc_row_width*embedding_dim))*100;
      stop_clock_and_add(t, "Total Time");
      total_memory += get_memory_usage();
      auto sparsity = 1 - density;
      add_perf_stats(sparsity, "Sparsity");

      add_perf_stats(total_memory, "Memory usage");

      jobj[i]=json_perf_statistics();
      reset_performance_timers();
    }
    return jobj;
  }


};
}