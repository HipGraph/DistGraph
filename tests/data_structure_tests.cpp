#include "../cpp/core/common.h"
#include "../cpp/core/csr_local.hpp"
#include "../cpp/core/dense_mat.hpp"
#include "../cpp/core/sparse_mat.hpp"
#include "../cpp/io/parrallel_IO.hpp"
#include "../cpp/partition/partitioner.hpp"
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

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

  auto shared_sparseMat =
      shared_ptr<distblas::core::SpMat<int>>(new distblas::core::SpMat<int>());
  auto shared_sparseMat_Trans =
      shared_ptr<distblas::core::SpMat<int>>(new distblas::core::SpMat<int>());

  cout << " rank " << rank << " reading data from file path:  " << file_path
       << endl;
  reader.get()->parallel_read_MM<int>(file_path, shared_sparseMat.get());
  cout << " rank " << rank << " reading data from file path:  " << file_path
       << " completed " << endl;

  vector<Tuple<int>> copiedVector(shared_sparseMat.get()->coords);
  shared_sparseMat_Trans.get()->coords= copiedVector;

  auto grid = unique_ptr<Process3DGrid>(new Process3DGrid(2, 1, 1, 1));

  auto partitioner =
      unique_ptr<GlobalAdjacency1DPartitioner>(new GlobalAdjacency1DPartitioner(
          shared_sparseMat.get()->gRows, shared_sparseMat.get()->gCols,
          grid.get()));

  cout << " rank " << rank << " partitioning data started  " << endl;

  partitioner.get()->partition_data(shared_sparseMat.get(), false);
  partitioner.get()->partition_data(shared_sparseMat_Trans.get(), true);

  int localBRows = divide_and_round_up(shared_sparseMat.get()->gCols,
                                       grid.get()->world_size);
  int localARows = divide_and_round_up(shared_sparseMat.get()->gRows,
                                       grid.get()->world_size);


  shared_sparseMat.get()->divide_block_cols(localARows,localBRows, grid.get()->world_size,
                                            true,false);
  shared_sparseMat.get()->sort_by_rows();
  shared_sparseMat.get()->divide_block_rows(
      15000, localBRows,  true, false);

  shared_sparseMat_Trans.get()->divide_block_cols(15000,localBRows,
                                                  2, true,true);
  shared_sparseMat_Trans.get()->sort_by_rows();
  shared_sparseMat_Trans.get()->divide_block_rows(
      localARows, 15000, true, true);

//  shared_sparseMat.get()->print_coords(false);
//  shared_sparseMat_Trans.get()->print_coords(true);
  cout << " rank " << rank << " partitioning data completed  " << endl;

  cout << " rank " << rank << " initialization of CSR started  " << endl;
  shared_sparseMat.get()->initialize_CSR_blocks(15000, localBRows, -1,
                                                false);
  shared_sparseMat_Trans.get()->initialize_CSR_blocks(localARows, 15000,
                                                      -1, true);
  cout << " rank " << rank << " initialization of CSR completed  " << endl;

  shared_sparseMat.get()->print_blocks_and_cols(false);
  shared_sparseMat_Trans.get()->print_blocks_and_cols(true);

  vector<vector<int>> id_list;
  shared_sparseMat_Trans.get()->fill_col_ids(0,id_list);

  if (rank=0) {
    for (int i = 0; i < id_list.size(); i++) {
      cout << " rank " << i << endl;
      for (int k = 0; k < id_list[i].size(); k++) {
        cout << id_list[i][k] << " " << endl;
      }
    }
  }

  //  cout<<" rank "<< rank << " creation of dense matrices started  "<<endl;
  //  auto dense_mat = unique_ptr<DenseMat>(new DenseMat(4,4,0.0,1.0));
  //  dense_mat.get()->print_matrix();
  //  cout<<" rank "<< rank << " creation of dense matrices completed  "<<endl;

  cout << " rank " << rank << " processing completed  " << endl;

  int col_rank;
  MPI_Comm_rank(grid.get()->col_world, &col_rank);

  cout << " rank " << rank << " col rank  " << col_rank << endl;

  MPI_Finalize();
  return 0;
}