#pragma once

#include "../../core/sparse_mat.hpp"
#include "../../core/sparse_mat_tile.hpp"
#include "../spgemm/spgemm_with_tiling.hpp"
#include "../fusedMM/fusedmm.hpp"
#include "gat_layer.hpp"

using namespace distblas::core;

namespace distblas::algo {

    template<typename INDEX_TYPE, typename VALUE_TYPE, size_t features_per_head>
    class GAT {

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

        vector <GATLayer<INDEX_TYPE,VALUE_TYPE,features_per_head>> gat_layers;


        vector<unique_ptr<DenseMat<INDEX_TYPE,VALUE_TYPE,features_per_head>>> buffers;


    public:
        GAT(distblas::core::SpMat<VALUE_TYPE> *sp_local_native,
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

        void addLayer(GATLayer<INDEX_TYPE,VALUE_TYPE,features_per_head> layer) {
            gat_layers.emplace_back(std::move(layer));
        }

        void computeGAT(int i, int j){
            auto  dense_output = make_unique<DenseMat<INDEX_TYPE,VALUE_TYPE,features_per_head>>(grid,buffers[i]->rows,gat_layers[i].weights[j]->cols,true);
            buffers[i]->multiply(gat_layers[i].weights[j].get(),dense_output.get());
        }

        json execute() {
            auto t = start_clock();
            buffers.resize(gat_layers.size()+1);
            cout<<" first buffer initialization completed "<<endl;
            buffers[0]= make_unique<DenseMat<INDEX_TYPE, VALUE_TYPE, features_per_head>>(grid,sparse_local->proc_row_width,gat_layers[0].input_features);
            cout<<" first buffer initialization completed "<<endl;
            for(int i=0;i<gat_layers.size();++i){
                buffers[i+1]= make_unique<DenseMat<INDEX_TYPE, VALUE_TYPE, features_per_head>>(grid,sparse_local->proc_row_width,gat_layers[i].num_heads*features_per_head,true);
                for(int j=0;j<gat_layers[i].num_heads;++j){
                    gat_layers[i].weights[j] = make_unique<DenseMat<INDEX_TYPE, VALUE_TYPE, features_per_head>>(grid,buffers[i]->cols);
                    cout<<" gat layer initialization completed "<<i<<endl;
                }
            }
            for(int i=0;i<gat_layers.size();++i){
                for(int j=0;j<gat_layers[i].weights.size();++j){
                    cout<<" computed gat  "<<i<<" "<<j<<endl;
                     computeGAT(i,j);
                }
            }

            stop_clock_and_add(t, "Total Time");
            return json_perf_statistics();
        }
    };
}