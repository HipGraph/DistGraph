#include "../cpp/io/parrallel_IO.hpp"
#include "../cpp/core/sparse_mat.hpp"
#include "../cpp/partition/partitioner.hpp"
#include "../cpp/core/common.h"
#include <iostream>
#include <memory>
#include <string>
#include <fstream>
#include <cstring>
#include "../cpp/core/csr_local.hpp"
#include "../cpp/core/dense_mat.hpp"

using namespace std;
using namespace distblas::io;
using namespace distblas::partition;

int main(int argc, char **argv) {
  string file_path = argv[1];

  cout << " file_path " << file_path << endl;

  MPI_Init(&argc, &argv);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  initialize_mpi_datatypes<int>();

  auto reader = unique_ptr<ParallelIO>(new ParallelIO());

  auto shared_sparseMat = shared_ptr<distblas::core::SpMat<int>>(new distblas::core::SpMat<int>());

  cout<<" rank "<< rank << " reading data from file path:  "<<file_path<<endl;
  reader.get()->parallel_read_MM<int>(file_path, shared_sparseMat.get());
  cout<<" rank "<< rank << " reading data from file path:  "<<file_path<<" completed "<<endl;

  auto grid = unique_ptr<Process3DGrid>(new Process3DGrid(2, 1, 1, 1));

  auto  partitioner = unique_ptr<GlobalAdjacency1DPartitioner>
      (new GlobalAdjacency1DPartitioner(shared_sparseMat.get()->gRows,
                                        shared_sparseMat.get()->gCols,
                                        grid.get()));

  cout<<" rank "<< rank << " partitioning data started  "<<endl;

  partitioner.get()->partition_data(shared_sparseMat.get(), false);




  int localBRows = divide_and_round_up(shared_sparseMat.get()->gCols,grid.get()->world_size);
  int localARows = divide_and_round_up(shared_sparseMat.get()->gRows,grid.get()->world_size);


  shared_sparseMat.get()->divide_block_cols(localBRows,grid.get()->world_size, true);
  shared_sparseMat.get()->sort_by_rows();
  shared_sparseMat.get()->divide_block_rows(localARows,localBRows,
                                            grid.get()->world_size, true);

  cout<<" rank "<< rank << " partitioning data completed  "<<endl;


  cout<<" rank "<< rank << " initialization of CSR started  "<<endl;
  shared_sparseMat.get()->initialize_CSR_blocks(localARows,localBRows,-1, false);
  cout<<" rank "<< rank << " initialization of CSR completed  "<<endl;

//  shared_sparseMat.get()->print_blocks_and_cols();

  cout<<" rank "<< rank << " creation of dense matrices started  "<<endl;
  auto dense_mat = unique_ptr<DenseMat>(new DenseMat(4,4,0.0,1.0));
  dense_mat.get()->print_matrix();
  cout<<" rank "<< rank << " creation of dense matrices completed  "<<endl;


  cout<<" rank "<< rank << " processing completed  "<<endl;

  int col_rank;
  MPI_Comm_rank(shared_sparseMat.get()->col_world, &col_rank);

  cout<<" rank "<< rank << " col rank  "<<col_rank<<endl;

  MPI_Finalize();
  return 0;
}