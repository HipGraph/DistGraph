#pragma once

#include "../../core/sparse_mat.hpp"
#include "../../core/sparse_mat_tile.hpp"
#include "../spgemm/spgemm_with_tiling.hpp"
#include "baseline_spmm.hpp"
#include "../dist_graph_blas.hpp"

using namespace distblas::core;
using namespace  distblas::net;

namespace distblas::algo {

template <typename INDEX_TYPE, typename VALUE_TYPE>
class SpMM: public distblas::algo::DistGraphBLAS<INDEX_TYPE,VALUE_TYPE> {

private:

  int rows=0;
  int embedding_dim=0;

  SpMMAlgo<INDEX_TYPE,VALUE_TYPE>* spMMAlgo;

  DenseMat<INDEX_TYPE,VALUE_TYPE>* input_dense_mat;
  DenseMat<INDEX_TYPE,VALUE_TYPE>* output;

public:
  SpMM(Process3DGrid* grid, distblas::core::SpMat<INDEX_TYPE,VALUE_TYPE> *sparse_mat,
       DenseMat<INDEX_TYPE,VALUE_TYPE>* input_dense_mat,
       DenseMat<INDEX_TYPE,VALUE_TYPE>* output,
       double alpha, double beta)
      :DistGraphBLAS<INDEX_TYPE,VALUE_TYPE>(grid,sparse_mat,alpha,beta)  {
      cout << "SpMM parent completed" << endl;
      this->input_dense_mat=input_dense_mat;
      this->output = output;
      cout << " rank " << this->grid->rank_in_col << " SpMMAlgo constructor calling" << endl;
      if (this->sp_local_receiver->csr_local_data.get()) {
          cout << " rank "<< this->grid->rank_in_col  <<"csr_local_data initialized properly "<<endl;
      }
      distblas::core::CSRHandle *handle = this->sp_local_receiver->csr_local_data.get()->handler.get();
      cout << " rank "<< this->grid->rank_in_col  << "access handler passed "<<handle->values.size()<< endl;
      auto embedding_algo =
              make_unique<distblas::algo::SpMMAlgo<INDEX_TYPE, VALUE_TYPE>>(
                      this->sparse_local, this->sp_local_receiver,
                      this->sp_local_sender,input_dense_mat,
                      output, this->grid, this->alpha, this->beta, false);
      cout << " rank " << this->grid->rank_in_col << " spmm algo initialization completed" << endl;
      spMMAlgo = embedding_algo.get();
      cout << " rank " << this->grid->rank_in_col << " spmm algo assign completed" << endl;
  }

  json execute() {
      cout << " rank " << this->grid->rank_in_col << " spmm algo started  about to execute  spMMAlgo" << endl;
    json jobj;

    auto t = start_clock();
    size_t total_memory = 0;

    spMMAlgo->execute(1, this->batch_size, 1.0);

    stop_clock_and_add(t, "Total Time");
    double totalLocalSpGEMM = std::accumulate((spMMAlgo->timing_info).begin(), (spMMAlgo->timing_info).end(), 0.0)/16;
    add_perf_stats(totalLocalSpGEMM,"Local SpMM");
    jobj =json_perf_statistics();
    reset_performance_timers();

    return jobj;
  }
};
}