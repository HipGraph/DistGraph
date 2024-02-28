#pragma once

#include "../core/sparse_mat.hpp"
#include "../core/sparse_mat_tile.hpp"
#include "../algo/spgemm_with_tiling.hpp"

using namespace distblas::core;

namespace distblas::algo {

template <typename INDEX_TYPE, typename VALUE_TYPE, size_t embedding_dim>
class SpGEMMSpMMIterative {

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
  SpGEMMSpMMIterative(distblas::core::SpMat<VALUE_TYPE> *sp_local_native,
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

  void execute(int iterations, int batch_size, VALUE_TYPE lr) {

    distblas::core::SpMat<VALUE_TYPE> *sparse_input = nullptr;
    unique_ptr<DenseMat<INDEX_TYPE,VALUE_TYPE,embedding_dim>> state_holder= make_unique<DenseMat<INDEX_TYPE,VALUE_TYPE,embedding_dim>>(grid,sp_local_receiver->proc_row_width);

    for (int i = 0; i < iterations; i++) {
      double output_sparsity = 0;
      if (i == 0) {
        sparse_input = sparse_local;
      }
      auto rows =  sp_local_receiver->proc_row_width;
      auto cols = static_cast<INDEX_TYPE>(embedding_dim);
      auto sparse_out = make_shared<distblas::core::SpMat<VALUE_TYPE>>(grid,rows,cols,hash_spgemm);
      unique_ptr<distblas::algo::SpGEMMAlgoWithTiling<INDEX_TYPE, VALUE_TYPE,
                                                      embedding_dim>>
          spgemm_algo = unique_ptr<distblas::algo::SpGEMMAlgoWithTiling<
              INDEX_TYPE, VALUE_TYPE, embedding_dim>>(
              new distblas::algo::SpGEMMAlgoWithTiling<INDEX_TYPE, VALUE_TYPE,
                                                       embedding_dim>(
                  sp_local_native, sp_local_receiver, sp_local_sender,
                  sparse_input, sparse_out.get(), grid, alpha, beta, col_major,
                  sync, tile_width_fraction, hash_spgemm));
      spgemm_algo.get()->algo_spgemm(1, batch_size, lr);

      output_sparsity =(sparse_out->csr_local_data)->handler->rowStart[(sparse_out->csr_local_data)->handler->rowStart.size() - 1];
      output_sparsity =100 * (output_sparsity /(((sparse_out->csr_local_data)->handler->rowStart.size() - 1) *embedding_dim));
      cout << " rank " << grid->rank_in_col << " output sparsity "<< output_sparsity << endl;
      (*(sparse_input->csr_local_data)) = (*(sparse_out->csr_local_data));
    }
  }
};

} // namespace distblas::algo

