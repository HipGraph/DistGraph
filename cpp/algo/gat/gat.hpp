#pragma once

#include "../../core/sparse_mat.hpp"
#include "../../core/sparse_mat_tile.hpp"
#include "../spgemm/spgemm_with_tiling.hpp"
#include "../fusedMM/fusedmm.hpp"
#include "gat_layer.hpp"
#include "../sddmm/sddmm.hpp"
#include "../spmm/spmm.hpp"

using namespace distblas::core;

namespace distblas::algo {

    template<typename INDEX_TYPE, typename VALUE_TYPE, size_t features_per_head>
    class GAT {

    private:
        distblas::core::SpMat<VALUE_TYPE> *sp_local_receiver;
        distblas::core::SpMat<VALUE_TYPE> *sp_local_sender;
        distblas::core::SpMat<VALUE_TYPE> *sp_local_native;

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


        void applyLeakyRelu(distblas::core::SpMat<VALUE_TYPE>* sp_mat, double alpha){
            auto csr_local = std::move(sp_mat->csr_local_data);
            auto handler = std::move(csr_local->handler);

            #pragma omp parallel for
            for(int i=0;i<handler->row_idx.size()-1;++i){
                for(int j=handler->row_idx[i];j<handler->row_idx[i+1];++j){
                    if (handler->values[j]<0){
                        handler->values[j]=alpha*handler->values[j];
                    }
                }
            }
        }

        void computeGAT(int i, int j){
            auto  dense_output = make_unique<DenseMat<INDEX_TYPE,VALUE_TYPE,features_per_head>>(grid,buffers[i]->rows,gat_layers[i].weights[j]->cols,true);
            buffers[i]->multiply(gat_layers[i].weights[j].get(),dense_output.get());
            cout<<" dense computing layer  "<<i<<"  head "<<j<<" completed "<<endl;


            auto sparse_output = make_unique<distblas::core::SpMat<VALUE_TYPE>>(*sp_local_native);

            auto sddmm_algo = make_unique<distblas::algo::SDDMM<INDEX_TYPE, VALUE_TYPE, features_per_head>>(
                    sp_local_native, sp_local_receiver,
                    sp_local_sender,dense_output.get(),dense_output.get(),sparse_output.get(),
                    grid,
                    alpha, beta,col_major,sync);

            sddmm_algo->execute(1,sp_local_native->proc_row_width,1.0);
            cout<<" sddmm computing layer  "<<i<<"  head "<<j<<" completed "<<endl;

            applyLeakyRelu(sparse_output.get(),0.001);

            cout<<" applying  leaky relu  "<<i<<"  head "<<j<<" completed "<<endl;
            auto dense_mat_output = make_unique<DenseMat<INDEX_TYPE, VALUE_TYPE, features_per_head>>(grid, sparse_output->proc_row_width);

            auto spmm = make_unique<distblas::algo::SpMMAlgo<INDEX_TYPE, VALUE_TYPE, features_per_head>>(
                    sparse_output.get(), sp_local_receiver,
                    sp_local_sender,dense_output.get(),dense_mat_output.get(),
                            grid,
                            alpha, beta,col_major,sync);
            spmm->execute(1,sp_local_native->proc_row_width,1.0);

            cout<<" applying  spmm "<<i<<"  head "<<j<<" completed "<<endl;
//
//            assginNextInput(i,j,dense_mat_output.get());
        }




    public:
        GAT(distblas::core::SpMat<VALUE_TYPE> *sp_local_native,
            distblas::core::SpMat<VALUE_TYPE> *sp_local_receiver,
            distblas::core::SpMat<VALUE_TYPE> *sp_local_sender,
            Process3DGrid *grid, double alpha, double beta, bool col_major,
            bool sync_comm, double tile_width_fraction, bool hash_spgemm)
                : sp_local_native(sp_local_native), sp_local_receiver(sp_local_receiver),
                  sp_local_sender(sp_local_sender),grid(grid), alpha(alpha), beta(beta), col_major(col_major),
                  sync(sync_comm), tile_width_fraction(tile_width_fraction) {
            this->hash_spgemm = hash_spgemm;
        }

        void addLayer(GATLayer<INDEX_TYPE,VALUE_TYPE,features_per_head> layer) {
            gat_layers.emplace_back(std::move(layer));
        }

        json execute() {
            auto t = start_clock();
            buffers.resize(gat_layers.size()+1);
            cout<<"  buffer resizing  completed "<<endl;
            buffers[0]= make_unique<DenseMat<INDEX_TYPE, VALUE_TYPE, features_per_head>>(grid,sp_local_native->proc_row_width,
                    gat_layers[0].input_features);
            cout<<" first buffer initialization completed "<<endl;
            for(int i=0;i<gat_layers.size();++i){
                buffers[i+1]= make_unique<DenseMat<INDEX_TYPE, VALUE_TYPE, features_per_head>>(grid,sp_local_native->proc_row_width,
                        gat_layers[i].num_heads*features_per_head,true);

                for(int j=0;j<gat_layers[i].num_heads;++j){
                    gat_layers[i].weights[j] = make_unique<DenseMat<INDEX_TYPE, VALUE_TYPE, features_per_head>>(grid,buffers[i]->cols);
                    cout<<" gat layer initialization completed "<<i<<endl;
                }
            }
            for(int i=0;i<gat_layers.size();++i){
                for(int j=0;j<gat_layers[i].weights.size();++j){
                    cout<<" computing layer  "<<i<<"  head "<<j<<endl;
                     computeGAT(i,j);
                }
            }

            stop_clock_and_add(t, "Total Time");
            return json_perf_statistics();
        }
    };
}