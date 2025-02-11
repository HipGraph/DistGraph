#pragma once

#include "../../core/sparse_mat.hpp"
#include "../../core/sparse_mat_tile.hpp"
#include "../../algo/spgemm_with_tiling.hpp"
#include "../fusedmm.hpp"
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

        vector <GATLayer<INDEX_TYPE, VALUE_TYPE, features_per_head>> gat_layers;

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

        void addLayer(GATLayer <INDEX_TYPE, VALUE_TYPE, features_per_head> layer) {
            gat_layers.push_back(layer);
        }

        json execute() {
            json jobj;
            auto t = start_clock();

            for(int i=0;i<gat_layers.size();++i){
                for(int j=0;j<gat_layers[i].weights.size();++j){
                    cout<<" layer "<<i<<" weight "<<j<<endl;
                }
            }

            stop_clock_and_add(t, "Total Time");
            jobj[i] = json_perf_statistics();
            reset_performance_timers();
            return jobj;
        }
    };
}