#pragma once
#include "../core/sparse_mat_tile.hpp"
#include "../net/tile_based_data_comm.hpp"


using namespace std;
using namespace distblas::core;
using namespace distblas::net;
using namespace Eigen;

using namespace distblas::core;

namespace distblas::algo {
template <typename INDEX_TYPE, typename VALUE_TYPE, size_t embedding_dim>
class SpGEMMAlgoWithTiling {

private:
  distblas::core::SpMat<VALUE_TYPE> *sparse_local_output;
  distblas::core::SpMat<VALUE_TYPE> *sparse_local;
  distblas::core::SpMat<VALUE_TYPE> *sp_local_receiver;
  distblas::core::SpMat<VALUE_TYPE> *sp_local_sender;
  distblas::core::SpMat<VALUE_TYPE> *sp_local_native;
  Process3DGrid *grid;

  std::unordered_map<int, unique_ptr<DataComm<INDEX_TYPE, VALUE_TYPE, embedding_dim>>> data_comm_cache;

  //record temp local output
  unique_ptr<vector<unordered_map<INDEX_TYPE,VALUE_TYPE>>> output_ptr;

  //cache size controlling hyper parameter
  double alpha = 0;

  //hyper parameter controls the  computation and communication overlapping
  double beta = 1.0;

  //hyper parameter controls the switching the sync vs async commiunication
  bool sync = true;

  //hyper parameter controls the col major or row major  data access
  bool col_major = false;

public:
  SpGEMMAlgoWithTiling(distblas::core::SpMat<VALUE_TYPE> *sp_local_native,
             distblas::core::SpMat<VALUE_TYPE> *sp_local_receiver,
             distblas::core::SpMat<VALUE_TYPE> *sp_local_sender,
             distblas::core::SpMat<VALUE_TYPE> *sparse_local,
             distblas::core::SpMat<VALUE_TYPE> *sparse_local_output,
             Process3DGrid *grid, double alpha, double beta, bool col_major, bool sync_comm)
      : sp_local_native(sp_local_native), sp_local_receiver(sp_local_receiver),
        sp_local_sender(sp_local_sender), sparse_local(sparse_local), grid(grid),
        alpha(alpha), beta(beta),col_major(col_major),sync(sync_comm),
        sparse_local_output(sparse_local_output) {}



  void algo_spgemm(int iterations, int batch_size, VALUE_TYPE lr) {
    auto t = start_clock();

    int batches = 0;
    int last_batch_size = batch_size;

    if (sp_local_receiver->proc_row_width % batch_size == 0) {
      batches =
          static_cast<int>(sp_local_receiver->proc_row_width / batch_size);
    } else {
      batches =
          static_cast<int>(sp_local_receiver->proc_row_width / batch_size) + 1;
      last_batch_size =
          sp_local_receiver->proc_row_width - batch_size * (batches - 1);
    }

    cout << " rank " << grid->rank_in_col << " total batches " << batches<< endl;

    // This communicator is being used for negative updates and in alpha > 0 to
    // fetch initial embeddings
    auto main_comm = unique_ptr<TileDataComm<INDEX_TYPE, VALUE_TYPE, embedding_dim>>(
        new TileDataComm<INDEX_TYPE, VALUE_TYPE, embedding_dim>(
            sp_local_receiver, sp_local_sender, sparse_local, grid,  alpha,batches,0.5));
    main_comm.get()->onboard_data();


    stop_clock_and_add(t, "Total Time");
  }
};
} // namespace distblas::algo
