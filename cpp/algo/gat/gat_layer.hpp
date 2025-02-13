#pragma once
#include "../../core/dense_mat.hpp"
#include "../../net/process_3D_grid.hpp"

using namespace distblas::core;

using namespace distblas::core;
using namespace std;
using namespace distblas::net;

namespace distblas::algo {

    template<typename INDEX_TYPE, typename VALUE_TYPE, size_t features_per_head>
    class GATLayer {

    private:

    public:
        int input_features;
        int num_heads;
        vector<unique_ptr<DenseMat<INDEX_TYPE,VALUE_TYPE,features_per_head>>> weights;
        GATLayer(Process3DGrid* grid,int input_features,  int num_heads) {
            this->input_features = input_features;
            this->num_heads = num_heads;
//            this->features_per_head = features_per_head;
        }
    };
}