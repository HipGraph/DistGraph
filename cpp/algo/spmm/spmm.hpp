#pragma once

#include "../../core/sparse_mat.hpp"
#include "../../core/sparse_mat_tile.hpp"
#include "../spgemm/spgemm_with_tiling.hpp"
#include "baseline_spmm.hpp"
#include "../../net/data_comm.hpp"
#include "../../partition/partitioner.hpp"

using namespace distblas::core;
using namespace  distblas::net;
using namespace  distblas::partition;

namespace distblas::algo {

template <typename INDEX_TYPE, typename VALUE_TYPE>
class SpMM{

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
       double alpha, double beta){

      auto localBRows = divide_and_round_up(sparse_mat->gCols,grid->col_world_size);
      auto localARows = divide_and_round_up(sparse_mat->gRows,grid->col_world_size);

      sparse_mat->batch_size = localARows;
      sparse_mat->proc_row_width = localARows;
      sparse_mat->proc_col_width = localBRows;

      vector<Tuple<VALUE_TYPE>> copiedVector(sparse_mat->coords);
      auto shared_sparseMat_sender = make_shared<distblas::core::SpMat<INDEX_TYPE,VALUE_TYPE>>(grid,
                                                                                               copiedVector, sparse_mat->gRows,
                                                                                               sparse_mat->gCols, sparse_mat->gNNz, sparse_mat->batch_size,
                                                                                               localARows, localBRows, false, true);


      auto shared_sparseMat_receiver = make_shared<distblas::core::SpMat<INDEX_TYPE,VALUE_TYPE>>(grid,
                                                                                                 copiedVector, sparse_mat->gRows,
                                                                                                 sparse_mat->gCols, sparse_mat->gNNz, sparse_mat->batch_size,
                                                                                                 localARows, localBRows, true, false);

      auto partitioner = unique_ptr<GlobalAdjacency1DPartitioner>(new GlobalAdjacency1DPartitioner(grid));

      partitioner.get()->partition_data<INDEX_TYPE,VALUE_TYPE>(shared_sparseMat_sender.get());
      partitioner.get()->partition_data<INDEX_TYPE,VALUE_TYPE>(shared_sparseMat_receiver.get());
      partitioner.get()->partition_data<INDEX_TYPE,VALUE_TYPE>(sparse_mat);

      shared_sparseMat_sender->initialize_CSR_blocks(true);
      shared_sparseMat_receiver->initialize_CSR_blocks(true);
      sparse_mat->initialize_CSR_blocks(true);

      auto embedding_algo =
              make_unique<distblas::algo::SpMMAlgo<INDEX_TYPE, VALUE_TYPE>>(
                      sparse_mat.get(), shared_sparseMat_receiver.get(),
                      shared_sparseMat_sender.get(),input_dense_mat,
                      output, grid, alpha, beta, false);
      spMMAlgo = embedding_algo.get();
      spMMAlgo->execute(1, sparse_mat->batch_size, 1.0);
  }
};
}