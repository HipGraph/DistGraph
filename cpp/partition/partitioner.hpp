#pragma once
#include <algorithm>
#include <parallel/algorithm>
#include <numeric>
#include <mpi.h>
#include "../core/sparse_mat.hpp"
#include "../net/process_3D_grid.hpp"
#include "../core/common.h"
#include <iostream>



using namespace std;
using namespace distblas::core;
using namespace distblas::net;

namespace distblas::partition  {

  class Partitioner {

  public:

    virtual int block_owner(int row_block, int col_block) = 0;
  };

  class GlobalAdjacency1DPartitioner : public Partitioner {

  public:
    int rows_per_block;
    int cols_per_block;
    Process3DGrid *process_3D_grid;


    GlobalAdjacency1DPartitioner(int gRows, int gCols, Process3DGrid *process_3D_grid);

    ~GlobalAdjacency1DPartitioner();

    int block_owner(int row_block, int col_block);

    int get_owner_Process(int row, int column, bool transpose);

    template <typename T>
    void partition_data(distblas::core::SpMat<T> *sp_mat, bool transpose) {

      int world_size = process_3D_grid->world_size;
      int my_rank = process_3D_grid->global_rank;

      vector<int> sendcounts(world_size, 0);
      vector<int> recvcounts(world_size, 0);

      vector<int> offsets, bufindices;

      vector<Tuple<T>> coords = sp_mat->coords;

//      cout<<" rank "<<my_rank<<"  coords size  "<<coords.size()<<endl;

      Tuple<T>* sendbuf = new Tuple<T>[coords.size()];

#pragma omp parallel for
      for(int i = 0; i < coords.size(); i++) {
        int owner = get_owner_Process(coords[i].row, coords[i].col, transpose);
#pragma omp atomic update
        sendcounts[owner]++;
      }
      prefix_sum(sendcounts, offsets);
      bufindices = offsets;


//      for(int i=0;i<sendcounts.size();i++){
//        cout<<" rank "<<my_rank<<"  size "<<sendcounts[i]<<endl;
//      }

#pragma omp parallel for
      for(int i = 0; i < coords.size(); i++) {
        int owner = get_owner_Process(coords[i].row, coords[i].col, transpose);

        int idx;
#pragma omp atomic capture
        idx = bufindices[owner]++;

        sendbuf[idx].row = transpose ? coords[i].col : coords[i].row;
        sendbuf[idx].col = transpose ? coords[i].row : coords[i].col;
        sendbuf[idx].value = coords[i].value;
      }


      // Broadcast the number of nonzeros that each processor is going to receive
      MPI_Alltoall(sendcounts.data(), 1, MPI_INT, recvcounts.data(), 1,
                   MPI_INT, process_3D_grid->global);

      vector<int> recvoffsets;
      prefix_sum(recvcounts, recvoffsets);

      // Use the sizing information to execute an AlltoAll
      int total_received_coords =
          std::accumulate(recvcounts.begin(), recvcounts.end(), 0);

//      cout<<" rank "<<my_rank<<" total_coords "<<total_received_coords<<endl;


      (sp_mat->coords).resize(total_received_coords);

      MPI_Alltoallv(sendbuf, sendcounts.data(), offsets.data(),
                    SPTUPLE, (sp_mat->coords).data(), recvcounts.data(), recvoffsets.data(),
                    SPTUPLE, process_3D_grid->global);

      // TODO: Parallelize the sort routine?
      //std::sort((result->coords).begin(), (result->coords).end(), column_major);
      __gnu_parallel::sort((sp_mat->coords).begin(), (sp_mat->coords).end(), column_major<T>);
      delete[] sendbuf;
    }

  };

}
