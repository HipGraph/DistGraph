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
    int batches=0;
    if (sp_local_receiver->proc_row_width % batch_size == 0) {
      batches =
          static_cast<int>(sp_local_receiver->proc_row_width / batch_size);
    } else {
      batches =
          static_cast<int>(sp_local_receiver->proc_row_width / batch_size) + 1;
    }

    auto main_comm =
        unique_ptr<TileDataComm<INDEX_TYPE, VALUE_TYPE, embedding_dim>>(
            new TileDataComm<INDEX_TYPE, VALUE_TYPE, embedding_dim>(
                sp_local_receiver, sp_local_sender, sparse_local, grid, alpha,
                batches, tile_width_fraction, hash_spgemm));
    main_comm.get()->onboard_data(false);

    for (int i = 0; i < iterations; i++) {
      size_t total_memory = 0;
      double bfs_frontier = 0;
      if (i == 0) {
        sparse_input = sparse_local;
      }
      auto rows =  sp_local_receiver->proc_row_width;
      auto cols = static_cast<INDEX_TYPE>(embedding_dim);
      auto sparse_out = make_shared<distblas::core::SpMat<VALUE_TYPE>>(grid,rows,cols,hash_spgemm);
      bfs_frontier =static_cast<double>((sparse_input->csr_local_data)->handler->rowStart[(sparse_input->csr_local_data)->handler->rowStart.size() - 1]);

      auto density =   (bfs_frontier/(sp_local_receiver->proc_row_width*embedding_dim))*100;
      int enable_mode = density>1.0?1:0;
      int global_mode;
      MPI_Allreduce(&enable_mode, &global_mode, 1, MPI_INT, MPI_SUM,grid->col_world);
      bool enable_remote = global_mode>0?true:false;

      cout<<grid->rank_in_col<<" iteration "<<i<<" enable remote "<<enable_remote<<endl;
      unique_ptr<distblas::algo::SpGEMMAlgoWithTiling<INDEX_TYPE, VALUE_TYPE,embedding_dim>>
          spgemm_algo = unique_ptr<distblas::algo::SpGEMMAlgoWithTiling<
              INDEX_TYPE, VALUE_TYPE, embedding_dim>>(
              new distblas::algo::SpGEMMAlgoWithTiling<INDEX_TYPE, VALUE_TYPE,
                                                       embedding_dim>(
                  sp_local_native, sp_local_receiver, sp_local_sender,
                  sparse_input, sparse_out.get(), grid, alpha, beta, col_major,
                  sync, tile_width_fraction, hash_spgemm,state_holder.get()));


      auto t = start_clock();
      spgemm_algo.get()->algo_spgemm(1, batch_size, lr,false);
      this->update_state_holder(sparse_input,state_holder.get());
      stop_clock_and_add(t, "Total Time");
      total_memory += get_memory_usage();

      double totalSum = std::accumulate((*(state_holder->nnz_count)).begin(), (*(state_holder->nnz_count)).end(), 0);
      (*(sparse_input->csr_local_data)) =(*(sparse_out->csr_local_data));
      main_comm->update_local_input(sparse_input);
      add_perf_stats(totalSum,"Output NNZ");
      if (bfs_frontier>0) {
        add_perf_stats(bfs_frontier, "BFS Frontier");
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

