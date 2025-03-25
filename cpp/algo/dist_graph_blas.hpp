#pragma once
#include "../net/process_3D_grid.hpp"
#include "../core/common.h"
#include <vector>
#include <memory>
#include "../partition/partitioner.hpp"


using namespace distblas::net;

namespace distblas::algo {

    template <typename INDEX_TYPE, typename VALUE_TYPE>
    class DistGraphBLAS {

    private:

    public:

        //cache size controlling hyper parameter
        double alpha = 0;

        //hyper parameter controls the  computation and communication overlapping
        double beta = 1.0;

        //hyper parameter controls the switching the sync vs async commiunication
        bool sync = true;

        //hyper parameter controls the col major or row major  data access
        bool col_major = false;

        Process3DGrid *grid;

        distblas::core::SpMat<INDEX_TYPE,VALUE_TYPE> *sparse_local;
        distblas::core::SpMat<INDEX_TYPE,VALUE_TYPE> *sp_local_receiver;
        distblas::core::SpMat<INDEX_TYPE,VALUE_TYPE> *sp_local_sender;

        INDEX_TYPE batch_size=0;

        DistGraphBLAS(Process3DGrid* grid,
                    distblas::core::SpMat<INDEX_TYPE,VALUE_TYPE>* sparse_mat,double alpha, double beta){

            sparse_local = sparse_mat;

            auto localBRows = divide_and_round_up(sparse_mat->gCols,grid->col_world_size);
            auto localARows = divide_and_round_up(sparse_mat->gRows,grid->col_world_size);
            batch_size = localARows;

            sparse_mat->batch_size = batch_size;
            sparse_mat->proc_row_width = localARows;
            sparse_mat->proc_col_width = localBRows;

            vector<Tuple<VALUE_TYPE>> copiedVector(sparse_mat->coords);
            auto shared_sparseMat_sender = make_shared<distblas::core::SpMat<INDEX_TYPE,VALUE_TYPE>>(grid,
                                                                                                     copiedVector, sparse_mat->gRows,
                                                                                                     sparse_mat->gCols, sparse_mat->gNNz, batch_size,
                                                                                                     localARows, localBRows, false, true);
            sp_local_sender =  shared_sparseMat_sender.get();


            auto shared_sparseMat_receiver = make_shared<distblas::core::SpMat<INDEX_TYPE,VALUE_TYPE>>(grid,
                                                                                                       copiedVector, sparse_mat->gRows,
                                                                                                       sparse_mat->gCols, sparse_mat->gNNz, batch_size,
                                                                                                       localARows, localBRows, true, false);

            sp_local_receiver = shared_sparseMat_receiver.get();

            auto partitioner = unique_ptr<GlobalAdjacency1DPartitioner>(
                    new GlobalAdjacency1DPartitioner(grid));

            cout << " rank " << rank << " partitioning data started  " << endl;

            partitioner.get()->partition_data<INDEX_TYPE,VALUE_TYPE>(sp_local_sender);
            partitioner.get()->partition_data<INDEX_TYPE,VALUE_TYPE>(sp_local_receiver);
            partitioner.get()->partition_data<INDEX_TYPE,VALUE_TYPE>(sparse_local);

            cout << " rank " << rank << " partitioning data completed  " << endl;

            sparse_local->initialize_CSR_blocks(true);
            sp_local_sender->initialize_CSR_blocks(true);
            sp_local_receiver->initialize_CSR_blocks(true);

        }

        virtual json execute();
    };
}
