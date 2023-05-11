#pragma once
#include <mpi.h>
#include <string>
#include <vector>
#include "../core/sparse_mat.hpp"

using namespace std;
using namespace distblas::core;
namespace distblas::io {
/**
 * This class implements IO operations of DistBlas library.
 */
class ParallelIO {
private:

public:

  ParallelIO();
  ~ParallelIO();

  /**
   * Interface for parallel reading of Matrix Market formatted files
   * @param file_path
   */
  template <typename T>
  void ParallelIO::parallel_read_MM(string file_path, const SpMat<T> &spmat_ptr) {
    MPI_Comm WORLD;
    MPI_Comm_dup(MPI_COMM_WORLD, &WORLD);

    int proc_rank, num_procs;
    MPI_Comm_rank(WORLD, &proc_rank);
    MPI_Comm_size(WORLD, &num_procs);

    shared_ptr<CommGrid> simpleGrid;
    simpleGrid.reset(new CommGrid(WORLD, num_procs, 1));

    unique_ptr<PSpMat_s32p64_Int> G =
        unique_ptr<PSpMat_s32p64_Int>(new PSpMat_s32p64_Int(simpleGrid));

    uint64_t nnz;

    G.get()->ParallelReadMM(file_path, true, maximum<T>());

    nnz = G.get()->getnnz();
    if (proc_rank == 0) {
      cout << "File reader read " << nnz << " nonzeros." << endl;
    }
    SpTuples<int64_t, T> tups(G.get()->seq());
    tuple<int64_t, int64_t, T> *values = tups.tuples;

    vector<Tuple<int>> coords;
    coords.resize(tups.getnnz());

#pragma omp parallel for
    for (int i = 0; i < tups.getnnz(); i++) {
      coords[i].row = get<0>(values[i]);
      coords[i].col = get<1>(values[i]);
      coords[i].value = get<2>(values[i]);
    }

    int rowIncrement = G->getnrow() / num_procs;

 //TODO: parallalize this
    for(int i = 0; i < coords.size(); i++) {
      coords[i].r += rowIncrement * proc_rank;
    }

    spmat_ptr->coords = coords;
    spmat_ptr->gRows = G->gRows;
    spmat_ptr->gCols = G->gCols;
    spmat_ptr->gNNz = G->gNNz;

  }
};
} // namespace distblas::io
