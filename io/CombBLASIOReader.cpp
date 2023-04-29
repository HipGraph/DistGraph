/**
 * This implementation contains the CombBLAS based parallel IO Implementation.
 */
#include "../include/DistBLAS/ParallelIO.hpp"
#include "CombBLAS/CombBLAS.h"

using namespace distblas::io;
using namespace combblas;

typedef SpParMat<int64_t, int, SpDCCols<int32_t, int>> PSpMat_s32p64_Int;

ParallelIO::ParallelIO() {}

void ParallelIO::parallel_read_MM(string file_path) {
  MPI_Comm WORLD;
  MPI_Comm_dup(MPI_COMM_WORLD, &WORLD);

  int proc_rank, num_procs;
  MPI_Comm_rank(WORLD, &proc_rank);
  MPI_Comm_size(WORLD, &num_procs);

  shared_ptr<CommGrid> simpleGrid;
  simpleGrid.reset(new CommGrid(WORLD, num_procs, 1));

  PSpMat_s32p64_Int *G;
  uint64_t nnz;

  G = new PSpMat_s32p64_Int(simpleGrid);

  G->ParallelReadMM(file_path, true, maximum<double>());

  nnz = G->getnnz();
  if (proc_rank == 0) {
    cout << "File reader read " << nnz << " nonzeros." << endl;
  }
  SpTuples<int64_t, int> tups(G->seq());
}

ParallelIO::~ParallelIO() {}
