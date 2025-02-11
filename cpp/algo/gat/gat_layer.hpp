#pragma once
#include "../../core/dense_mat.hpp"
#include "../../net/process_3D_grid.hpp"

using namespace distblas::core;

using namespace distblas::core;
using namespace std;
using namespace distblas::net;

namespace distblas::algo {

    template <typename INDEX_TYPE, typename VALUE_TYPE, int features_per_head>
    class GATLayer {

    private:
        int input_features;
        int num_heads;
       // for multi head weight matrices
    public:
        vector<unique_ptr<DenseMat<INDEX_TYPE,VALUE_TYPE,features_per_head>>> weights;
        GATLayer(Process3DGrid *grid,int input_features, int num_heads) {
            this->input_features = input_features;
            this->num_heads = num_heads;
            weights.resize(num_heads);

            for(int i=0;i<num_heads;i++){
                weights[i] = make_unique<DenseMat<INDEX_TYPE, VALUE_TYPE, features_per_head>>(
                        grid,num_heads* features_per_head);
            }
        }
    };
}