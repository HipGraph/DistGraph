#pragma once

#include "../../core/sparse_mat.hpp"
#include "../../core/sparse_mat_tile.hpp"
#include "../spgemm/spgemm_with_tiling.hpp"

using namespace distblas::core;

namespace distblas::algo {

template <typename INDEX_TYPE, typename VALUE_TYPE, size_t embedding_dim>
class BaselineSpMM {

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
  BaselineSpMM(distblas::core::SpMat<VALUE_TYPE> *sp_local_native,
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
    int batches=0;
    if (sp_local_receiver->proc_row_width % batch_size == 0) {
      batches =
          static_cast<int>(sp_local_receiver->proc_row_width / batch_size);
    } else {
      batches =
          static_cast<int>(sp_local_receiver->proc_row_width / batch_size) + 1;
    }

    for (int i = 0; i < iterations; i++) {
      auto t = start_clock();
      size_t total_memory = 0;
      auto dense_mat = unique_ptr<DenseMat<INDEX_TYPE, VALUE_TYPE, embedding_dim>>(
          new DenseMat<INDEX_TYPE, VALUE_TYPE, embedding_dim>(grid, sp_local_receiver->proc_row_width));
      auto dense_mat_output = unique_ptr<DenseMat<INDEX_TYPE, VALUE_TYPE, embedding_dim>>(
          new DenseMat<INDEX_TYPE, VALUE_TYPE, embedding_dim>(grid, sp_local_receiver->proc_row_width));

      auto embedding_algo =
              make_unique<distblas::algo::SpMMAlgo<INDEX_TYPE, VALUE_TYPE, embedding_dim>>(
                      sp_local_native, sp_local_receiver,
                      sp_local_sender, dense_mat.get(),
                      dense_mat_output.get(), grid, alpha, beta, col_major, sync);

      cout << " rank " << grid->rank_in_col << " spmm algo started  " << endl;
      embedding_algo.get()->algo_spmm(1, batch_size, lr);

      stop_clock_and_add(t, "Total Time");
      double totalLocalSpGEMM = std::accumulate((embedding_algo->timing_info).begin(), (embedding_algo->timing_info).end(), 0.0)/16;
      add_perf_stats(totalLocalSpGEMM,"Local SpMM");
      jobj[i]=json_perf_statistics();
      reset_performance_timers();
    }
    return jobj;
  }


};
}