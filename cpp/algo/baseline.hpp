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
                 bool sync_comm, double tile_width_fraction,bool hash_spgemm)
      : sp_local_native(sp_local_native), sp_local_receiver(sp_local_receiver),
        sp_local_sender(sp_local_sender), sparse_local(sparse_local),
        grid(grid), alpha(alpha), beta(beta), col_major(col_major),
        sync(sync_comm), tile_width_fraction(tile_width_fraction) {
    this->hash_spgemm = hash_spgemm;
  }

  json execute(int iterations, int batch_size, VALUE_TYPE lr, bool test_remote=false) {
    json jobj;

//    double fraction_array[] = {0.25,0.5,0.75,1};
//    int len =1;
//    if (test_remote){
//      len = 4;
//      iterations = iterations+1;
//    }
//    int count_i=0;
//    for(int w=0;w<len;w++ ){
//      if (test_remote) {
//        tile_width_fraction = fraction_array[w];
//      }
//
//      for(int h=0;h<len;h++){
//        if (test_remote){
//          batch_size = sp_local_receiver->proc_row_width * fraction_array[h];
//          sp_local_receiver->batch_size = batch_size;
//          sp_local_sender->batch_size = batch_size;
//          sparse_local->batch_size = batch_size;
//          sp_local_native->batch_size=batch_size;
//        }
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
          bool enabled =false;
          size_t total_memory = 0;
          auto rows =  sp_local_receiver->proc_row_width;
          auto cols = static_cast<INDEX_TYPE>(embedding_dim);
          auto sparse_out = make_shared<distblas::core::SpMat<VALUE_TYPE>>(grid,rows,cols,hash_spgemm);
          auto main_comm =
              unique_ptr<TileDataComm<INDEX_TYPE, VALUE_TYPE, embedding_dim>>(
                  new TileDataComm<INDEX_TYPE, VALUE_TYPE, embedding_dim>(
                      sp_local_receiver, sp_local_sender, sparse_local, grid, alpha,
                      batches, tile_width_fraction, hash_spgemm));
//          if (test_remote) {
//            if (i % iterations == 0) {
//              main_comm.get()->onboard_data(false);
//            } else if (test_remote) {
//              main_comm.get()->onboard_data(true);
//              enabled = true;
//            }
//          }else {
            main_comm.get()->onboard_data(false);
//          }

          unique_ptr<distblas::algo::SpGEMMAlgoWithTiling<INDEX_TYPE, VALUE_TYPE,embedding_dim>>
              spgemm_algo = unique_ptr<distblas::algo::SpGEMMAlgoWithTiling<
                  INDEX_TYPE, VALUE_TYPE, embedding_dim>>(
                  new distblas::algo::SpGEMMAlgoWithTiling<INDEX_TYPE, VALUE_TYPE,
                                                           embedding_dim>(
                      sp_local_native, sp_local_receiver, sp_local_sender,
                      sparse_local, sparse_out.get(), grid, alpha, beta, col_major,
                      sync, tile_width_fraction, hash_spgemm,main_comm.get()));

//          if (test_remote) {
//            if (i % iterations == 0) {
//              spgemm_algo.get()->algo_spgemm(1, batch_size, lr, false);
//            } else {
//              spgemm_algo.get()->algo_spgemm(1, batch_size, lr, true);
//            }
//          } else {
            spgemm_algo.get()->algo_spgemm(1, batch_size, lr, false);
//          }
          stop_clock_and_add(t, "Total Time");
          auto size_r = sparse_out->csr_local_data->handler->rowStart.size();
          double output_nnz = sparse_out->csr_local_data->handler->rowStart[size_r-1];
          double density =   (output_nnz/static_cast<double >((sp_local_receiver->proc_row_width*embedding_dim)))*100;

          double totalLocalSpGEMM = std::accumulate((spgemm_algo->timing_info).begin(), (spgemm_algo->timing_info).end(), 0.0)/16;
          add_perf_stats(totalLocalSpGEMM,"Local SpGEMM");
          total_memory += get_memory_usage();
          auto sparsity = 100 - density;
          add_perf_stats(output_nnz, "Output NNZ");
          if (sparsity>0) {
            add_perf_stats(sparsity, "Sparsity");
          }

          add_perf_stats(total_memory, "Memory usage");
          json out = json_perf_statistics();
//          out["tile_width_fraction"] = fraction_array[w];
//          out["tile_height_fraction"] = fraction_array[h];
          out["remote_enabled"] =  enabled;
          jobj[i]=out;
//          count_i++;
          reset_performance_timers();
        }
//      }
//    }

    return jobj;
  }


};
}