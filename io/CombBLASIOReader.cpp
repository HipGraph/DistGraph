/**
 * This implementation contains the CombBLAS based parallel IO Implementation.
 */
#include "CombBLAS/CombBLAS.h"

using namespace combblas;

namespace distblas::io {

typedef SpParMat<int64_t, int, SpDCCols<int32_t, int>> PSpMat_s32p64_Int;

template <typename T> ParallelIO<T>::ParallelIO() {}

template <typename T>
void ParallelIO<T>::parallel_read_MM(string file_path,
                                     vector<Tuple<T>> &coords) {
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

  coords.resize(tups.getnnz());

  for (int i = 0; i < tups.getnnz(); i++) {
    coords[i].row = get<0>(values[i]);
    coords[i].col = get<1>(values[i]);
    coords[i].value = get<2>(values[i]);
  }
}

template <typename T> ParallelIO<T>::~ParallelIO() {}
}