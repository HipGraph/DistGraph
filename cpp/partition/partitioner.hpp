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


      Tuple<T>* sendbuf = new Tuple<T>[coords.size()];

#pragma omp parallel for
      for(int i = 0; i < coords.size(); i++) {
        int owner = get_owner_Process(coords[i].row,
                                      coords[i].col,
                                      transpose);
#pragma omp atomic update
        sendcounts[owner]++;
      }
      prefix_sum(sendcounts, offsets);
      bufindices = offsets;


      cout<<" rank "<< my_rank << " sendcount completed "<<endl;
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

      cout<<" rank "<< my_rank << " sendbyuf completed "<<endl;


      // Broadcast the number of nonzeros that each processor is going to receive
      MPI_Alltoall(sendcounts.data(), 1, MPI_INT, recvcounts.data(), 1,
                   MPI_INT, process_3D_grid->global);

      cout<<" rank "<< my_rank << " broadcasting completed "<<endl;

      vector<int> recvoffsets;
      prefix_sum(recvcounts, recvoffsets);

      // Use the sizing information to execute an AlltoAll
      int total_received_coords =
          std::accumulate(recvcounts.begin(), recvcounts.end(), 0);

      (sp_mat->coords).resize(total_received_coords);

      cout<<" rank "<< my_rank << " MPI_Alltoallv starting "<<endl;

      MPI_Alltoallv(sendbuf, sendcounts.data(), offsets.data(),
                    SPTUPLE, (sp_mat->coords).data(), recvcounts.data(), recvoffsets.data(),
                    SPTUPLE, process_3D_grid->global);


      cout<<" rank "<< my_rank << " MPI_Alltoallv completed "<<endl;


      for (typename vector<Tuple<T>>::iterator it = (sp_mat->coords).begin(); it != (sp_mat->coords).end(); ++it) {
         T val =  (*it).value ;
      }
      cout<<" rank "<< my_rank << " Loop traversing completed "<<endl;

      // TODO: Parallelize the sort routine?
//      std::sort((sp_mat->coords).begin(), (sp_mat->coords).end(), column_major<T>);
      __gnu_parallel::sort((sp_mat->coords).begin(), (sp_mat->coords).end(), column_major<T>);
      cout<<" rank "<< my_rank << " delete sorting completeed "<<endl;
      delete[] sendbuf;
      cout<<" rank "<< my_rank << " delete sendbuf completeed "<<endl;
    }

  };

}
