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

      this->sparse_local = sparse_mat;
      this->grid=grid;
      this->alpha=alpha;
      this->beta=beta;
      auto localBRows = divide_and_round_up(sparse_mat->gCols,grid->col_world_size);
      auto localARows = divide_and_round_up(sparse_mat->gRows,grid->col_world_size);
      this->batch_size = localARows;

      this->sparse_local->batch_size = this->batch_size;
      this->sparse_local->proc_row_width = localARows;
      this->sparse_local->proc_col_width = localBRows;

      vector<Tuple<VALUE_TYPE>> copiedVector(sparse_mat->coords);
      auto shared_sparseMat_sender = make_shared<distblas::core::SpMat<INDEX_TYPE,VALUE_TYPE>>(grid,
                                                                                               copiedVector, sparse_mat->gRows,
                                                                                               sparse_mat->gCols, sparse_mat->gNNz, this->batch_size,
                                                                                               localARows, localBRows, false, true);
      this->sp_local_sender =  shared_sparseMat_sender.get();


      auto shared_sparseMat_receiver = make_shared<distblas::core::SpMat<INDEX_TYPE,VALUE_TYPE>>(grid,
                                                                                                 copiedVector, sparse_mat->gRows,
                                                                                                 sparse_mat->gCols, sparse_mat->gNNz, this->batch_size,
                                                                                                 localARows, localBRows, true, false);

      this->sp_local_receiver = shared_sparseMat_receiver.get();

      auto partitioner = unique_ptr<GlobalAdjacency1DPartitioner>(new GlobalAdjacency1DPartitioner(grid));

      cout << " rank " << grid->rank_in_col << " partitioning data started  " << endl;

      partitioner.get()->partition_data<INDEX_TYPE,VALUE_TYPE>(this->sp_local_sender);
      partitioner.get()->partition_data<INDEX_TYPE,VALUE_TYPE>(this->sp_local_receiver);
      partitioner.get()->partition_data<INDEX_TYPE,VALUE_TYPE>(this->sparse_local);

      cout <<  " rank " << grid->rank_in_col << " partitioning data completed  " << endl;

      this->sparse_local->initialize_CSR_blocks(true);
      this->sp_local_sender->initialize_CSR_blocks(true);
      this->sp_local_receiver->initialize_CSR_blocks(true);
      cout <<  " rank " << grid->rank_in_col << " initialize_CSR_blocks completed " << endl;

      distblas::core::CSRHandle *handle = this->sp_local_receiver->csr_local_data.get()->handler.get();
      cout << " rank "<< this->grid->rank_in_col  << "values "<<handle->values.size()<<" rowStart "<<handle->rowStart.size()<<" col "<<handle->col_idx.size()<< endl;


      cout << "SpMM parent completed" << endl;
      this->input_dense_mat=input_dense_mat;
      this->output = output;
      cout << " rank " << this->grid->rank_in_col << " SpMMAlgo constructor calling" << endl;

      distblas::core::CSRHandle *handle_2 = this->sp_local_receiver->csr_local_data.get()->handler.get();
      cout << " rank "<< this->grid->rank_in_col  << " outside values "<<handle_2->values.size()<<" outside rowStart "<<handle_2->rowStart.size()<<" outside col "<<handle_2->col_idx.size()<< endl;
      auto embedding_algo =
              make_unique<distblas::algo::SpMMAlgo<INDEX_TYPE, VALUE_TYPE>>(
                      this->sparse_local, this->sp_local_receiver,
                      this->sp_local_sender,input_dense_mat,
                      output, this->grid, this->alpha, this->beta, false);
      cout << " rank " << this->grid->rank_in_col << " spmm algo initialization completed" << endl;

      distblas::core::CSRHandle *handle_3 = this->sp_local_receiver->csr_local_data.get()->handler.get();
      cout << " rank "<< this->grid->rank_in_col  << " embedding_algo values "<<handle_3->values.size()<<" embedding_algo rowStart "<<handle_3->rowStart.size()<<" embedding_algo col "<<handle_3->col_idx.size()<< endl;

      spMMAlgo = embedding_algo.get();

      cout << " rank " << this->grid->rank_in_col << " spmm algo assign completed" << endl;
      spMMAlgo->execute(1, this->batch_size, 1.0);
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