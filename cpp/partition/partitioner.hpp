#pragma once
#include "../core/common.h"
#include "../core/sparse_mat.hpp"
#include "../net/process_3D_grid.hpp"
#include <algorithm>
#include <iostream>
#include <mpi.h>
#include <numeric>
#include <parallel/algorithm>

using namespace std;
using namespace distblas::core;
using namespace distblas::net;

namespace distblas::partition {

class Partitioner {

public:
  virtual int block_owner(int row_block, int col_block) = 0;
};

class GlobalAdjacency1DPartitioner : public Partitioner {

public:
  Process3DGrid *process_3D_grid;

  GlobalAdjacency1DPartitioner(Process3DGrid *process_3D_grid);

  ~GlobalAdjacency1DPartitioner();

  int block_owner(int row_block, int col_block);

  int get_owner_Process(uint64_t row, uint64_t column, uint64_t  proc_row_width, uint64_t  proc_col_width, uint64_t gCols,bool transpose);

  template <typename T>
  void partition_data(distblas::core::SpMat<T> *sp_mat, bool transpose) {

    int world_size = process_3D_grid->world_size;
    int my_rank = process_3D_grid->global_rank;

    int considered_row_width;
    if (rank == world_size - 1) {
      considered_row_width = shared_sparseMat.get()->gRows -
                             sp_mat->proc_row_width * (grid.get()->world_size - 1);
    }


    Tuple<T> *sendbuf = new Tuple<T>[sp_mat->coords.size()];

    if (world_size > 1) {
      vector<int> sendcounts(world_size, 0);
      vector<int> recvcounts(world_size, 0);

      vector<int> offsets, bufindices;

      vector<Tuple<T>> coords = sp_mat->coords;



#pragma omp parallel for
      for (int i = 0; i < coords.size(); i++) {
        int owner = get_owner_Process(coords[i].row, coords[i].col,
                                      considered_row_width,
                                      sp_mat->proc_col_width,
                                      sp_mat->gCols,transpose);
#pragma omp atomic update
        sendcounts[owner]++;
      }
      prefix_sum(sendcounts, offsets);
      bufindices = offsets;

#pragma omp parallel for
      for (int i = 0; i < coords.size(); i++) {
        int owner = get_owner_Process(coords[i].row, coords[i].col,
                                      considered_row_width,
                                      sp_mat->proc_col_width,
                                      sp_mat->gCols,transpose);

        int idx;
#pragma omp atomic capture
        idx = bufindices[owner]++;

        //        sendbuf[idx].row = transpose ? coords[i].col : coords[i].row;
        //        sendbuf[idx].col = transpose ? coords[i].row : coords[i].col;
        sendbuf[idx].row = coords[i].row;
        sendbuf[idx].col = coords[i].col;
        sendbuf[idx].value = coords[i].value;
      }

      // Broadcast the number of nonzeros that each processor is going to
      // receive
      MPI_Alltoall(sendcounts.data(), 1, MPI_INT, recvcounts.data(), 1, MPI_INT,
                   process_3D_grid->global);

      vector<int> recvoffsets;
      prefix_sum(recvcounts, recvoffsets);

      // Use the sizing information to execute an AlltoAll
      int total_received_coords =
          std::accumulate(recvcounts.begin(), recvcounts.end(), 0);

      (sp_mat->coords).resize(total_received_coords);

      MPI_Alltoallv(sendbuf, sendcounts.data(), offsets.data(), SPTUPLE,
                    (sp_mat->coords).data(), recvcounts.data(),
                    recvoffsets.data(), SPTUPLE, process_3D_grid->global);

      // TODO: Parallelize the sort routine?
      //      std::sort((sp_mat->coords).begin(), (sp_mat->coords).end(),
      //      column_major<T>);
    }
    __gnu_parallel::sort((sp_mat->coords).begin(), (sp_mat->coords).end(),
                         column_major<T>);

    delete[] sendbuf;
  }
};

} // namespace distblas::partition
