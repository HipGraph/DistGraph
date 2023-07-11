#include "../cpp/core/common.h"
#include "../cpp/core/csr_local.hpp"
#include "../cpp/core/dense_mat.hpp"
#include "../cpp/core/sparse_mat.hpp"
#include "../cpp/io/parrallel_IO.hpp"
#include "../cpp/net/data_comm.hpp"
#include "../cpp/partition/partitioner.hpp"
#include "../cpp/embedding/algo.hpp"
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

using namespace std;
using namespace distblas::io;
using namespace distblas::partition;
using namespace distblas::net;

int main(int argc, char **argv) {
  string file_path = argv[1];

  cout << " file_path " << file_path << endl;

  int batch_size = 300;

  MPI_Init(&argc, &argv);
  int rank;
  int world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  initialize_mpi_datatypes<int, double, 2>();

  auto reader = unique_ptr<ParallelIO>(new ParallelIO());
  auto grid = unique_ptr<Process3DGrid>(new Process3DGrid(1, 1, 1, 1));

  auto shared_sparseMat =
      shared_ptr<distblas::core::SpMat<int>>(new distblas::core::SpMat<int>());
  //  auto shared_sparseMat_Trans =
  //      shared_ptr<distblas::core::SpMat<int>>(new
  //      distblas::core::SpMat<int>());

  cout << " rank " << rank << " reading data from file path:  " << file_path
       << endl;
  reader.get()->parallel_read_MM<int>(file_path, shared_sparseMat.get());
  cout << " rank " << rank << " reading data from file path:  " << file_path
       << " completed " << endl;
  //  reader.get()->parallel_read_MM<int>(file_path,
  //  shared_sparseMat_Trans.get());
  //  shared_sparseMat.get()->print_coords(false);
  //  shared_sparseMat.get()->print_coords(false);

  int localBRows = divide_and_round_up(shared_sparseMat.get()->gCols,
                                       grid.get()->world_size);
  int localARows = divide_and_round_up(shared_sparseMat.get()->gRows,
                                       grid.get()->world_size);

  cout << " rank " << rank << " localBRows  " << localBRows
       << " localARows "<<localARows << endl;

  shared_sparseMat.get()->block_row_width = batch_size;
  shared_sparseMat.get()->block_col_width = batch_size;
  shared_sparseMat.get()->proc_row_width = localARows;
  shared_sparseMat.get()->proc_col_width = localBRows;

  cout << " rank " << rank << " gROWs  " << shared_sparseMat.get()->gRows
       <<  "gCols" << shared_sparseMat.get()->gCols << endl;

  vector<Tuple<int>> copiedVector(shared_sparseMat.get()->coords);
  auto shared_sparseMat_Trans = make_shared<distblas::core::SpMat<int>>(
      copiedVector, shared_sparseMat.get()->gRows,
      shared_sparseMat.get()->gCols, shared_sparseMat.get()->gNNz, localARows,
      batch_size, localARows, localBRows,false);

  vector<Tuple<int>> copiedVectorTwo(shared_sparseMat.get()->coords);
  auto shared_sparseMat_combined = make_shared<distblas::core::SpMat<int>>(
      copiedVectorTwo, shared_sparseMat.get()->gRows,
      shared_sparseMat.get()->gCols, shared_sparseMat.get()->gNNz, localARows,
      batch_size, localARows, localBRows,true);

  auto partitioner =
      unique_ptr<GlobalAdjacency1DPartitioner>(new GlobalAdjacency1DPartitioner(
          shared_sparseMat.get()->gRows, shared_sparseMat.get()->gCols,
          grid.get()));

  cout << " rank " << rank << " partitioning data started  " << endl;

  partitioner.get()->partition_data(shared_sparseMat_Trans.get(), true);
  partitioner.get()->partition_data(shared_sparseMat.get(), false);
  partitioner.get()->partition_data(shared_sparseMat_combined.get(), false);
  //

  shared_sparseMat.get()->divide_block_cols(
      300, localBRows, grid.get()->world_size, true, false);
  shared_sparseMat.get()->sort_by_rows();
  shared_sparseMat.get()->divide_block_rows(300, localBRows, true, false);
//
//
  shared_sparseMat_Trans.get()->divide_block_cols(300, localBRows, 1, true,
                                                  true);
  shared_sparseMat_Trans.get()->sort_by_rows();
  shared_sparseMat_Trans.get()->divide_block_rows(localARows,localBRows, true,
                                                  true);
//
//
//  shared_sparseMat_combined.get()->divide_block_cols(
//      localBRows, localBRows, grid.get()->world_size, true, false);
//  shared_sparseMat_combined.get()->sort_by_rows();
//  shared_sparseMat_combined.get()->divide_block_rows(300, localBRows, true, false);




//  cout << " rank " << rank << " partitioning data completed  " << endl;
//
//  cout << " rank " << rank << " initialization of CSR started  " << endl;
//  shared_sparseMat.get()->initialize_CSR_blocks(300, 300, localARows,
//                                                localBRows, -1, false);
//  cout << " rank " << rank << " initialization of  CSR completed  " << endl;
//  cout << " rank " << rank << " initialization of transpose CSR started  "
//       << endl;
//  shared_sparseMat_Trans.get()->initialize_CSR_blocks(
//      localARows, 300, localARows, localBRows, -1, true);
//  cout << " rank " << rank << " initialization of transpose CSR completed  "
//       << endl;
//
//
//  shared_sparseMat_combined.get()->initialize_CSR_blocks(300, localBRows, localARows,
//                                                localBRows, -1, false);
  //  shared_sparseMat.get()->print_blocks_and_cols(false);
  //  shared_sparseMat_Trans.get()->print_blocks_and_cols(true);

  cout << " rank " << rank << " creation of dense matrices started  " << endl;
//  auto dense_mat = shared_ptr<DenseMat<double, 2>>(
//      new DenseMat<double, 2>(localARows, 0, 1.0, grid.get()->world_size));
  //    dense_mat.get()->print_matrix();
  cout << " rank " << rank << " creation of dense matrices completed  " << endl;

//  auto communicator =
//      unique_ptr<DataComm<int, double, 2>>(new DataComm<int, double, 2>(
//          shared_sparseMat.get(), shared_sparseMat_Trans.get(), dense_mat.get(),
//          grid.get()));

  cout << " rank " << rank << " async started  " << endl;

//  unique_ptr<distblas::embedding::EmbeddingAlgo<int,double,2>> embedding_algo =
//      unique_ptr<distblas::embedding::EmbeddingAlgo<int,double,2>>(new distblas::embedding::EmbeddingAlgo<int,double,2>(shared_sparseMat_combined.get(),
//                                                                                                                            dense_mat.get(),
//                                                                                                                            communicator.get(),
//                                                                                                                            grid.get(),5,-5));
//
//  embedding_algo.get()->algo_force2_vec_ns(1200,300,5,0.02);
  cout << " rank " << rank << " async completed  " << endl;

//  dense_mat.get()->print_cache();

  cout << " rank " << rank << " processing completed  " << endl;

  MPI_Finalize();
  return 0;
}
