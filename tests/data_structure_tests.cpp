#include "../cpp/core/common.h"
#include "../cpp/core/csr_local.hpp"
#include "../cpp/core/dense_mat.hpp"
#include "../cpp/core/sparse_mat.hpp"
#include "../cpp/embedding/algo.hpp"
#include "../cpp/io/parrallel_IO.hpp"
#include "../cpp/net/data_comm.hpp"
#include "../cpp/partition/partitioner.hpp"
#include <chrono>
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

  // Initialize MPI DataTypes
  initialize_mpi_datatypes<int, double, 2>();

  // Creating reader
  auto reader = unique_ptr<ParallelIO>(new ParallelIO());

  // Creating ProcessorGrid
  auto grid = unique_ptr<Process3DGrid>(new Process3DGrid(1, 1, 1, 1));

  auto shared_sparseMat =
      shared_ptr<distblas::core::SpMat<int>>(new distblas::core::SpMat<int>());

  cout << " rank " << rank << " reading data from file path:  " << file_path
       << endl;

  auto start_io = std::chrono::high_resolution_clock::now();
  reader.get()->parallel_read_MM<int>(file_path, shared_sparseMat.get());
  auto end_io = std::chrono::high_resolution_clock::now();

  cout << " rank " << rank << " reading data from file path:  " << file_path
       << " completed " << endl;

  int localBRows = divide_and_round_up(shared_sparseMat.get()->gCols,
                                       grid.get()->world_size);
  int localARows = divide_and_round_up(shared_sparseMat.get()->gRows,
                                       grid.get()->world_size);

  cout << " rank " << rank << " localBRows  " << localBRows << " localARows "
       << localARows << endl;

  shared_sparseMat.get()->block_row_width = batch_size;
  shared_sparseMat.get()->block_col_width = batch_size;
  shared_sparseMat.get()->proc_row_width = localARows;
  shared_sparseMat.get()->proc_col_width = localBRows;

  cout << " rank " << rank << " gROWs  " << shared_sparseMat.get()->gRows
       << "gCols" << shared_sparseMat.get()->gCols << endl;

  vector<Tuple<int>> copiedVector(shared_sparseMat.get()->coords);
  auto shared_sparseMat_Trans = make_shared<distblas::core::SpMat<int>>(
      copiedVector, shared_sparseMat.get()->gRows,
      shared_sparseMat.get()->gCols, shared_sparseMat.get()->gNNz, localARows,
      batch_size, localARows, localBRows, false);

  vector<Tuple<int>> copiedVectorTwo(shared_sparseMat.get()->coords);
  auto shared_sparseMat_combined = make_shared<distblas::core::SpMat<int>>(
      copiedVectorTwo, shared_sparseMat.get()->gRows,
      shared_sparseMat.get()->gCols, shared_sparseMat.get()->gNNz, localARows,
      batch_size, localARows, localBRows, true);

  auto partitioner =
      unique_ptr<GlobalAdjacency1DPartitioner>(new GlobalAdjacency1DPartitioner(
          shared_sparseMat.get()->gRows, shared_sparseMat.get()->gCols,
          grid.get()));

  cout << " rank " << rank << " partitioning data started  " << endl;

  partitioner.get()->partition_data(shared_sparseMat_Trans.get(), true);
  partitioner.get()->partition_data(shared_sparseMat.get(), false);
  partitioner.get()->partition_data(shared_sparseMat_combined.get(), false);

  cout << " rank " << rank << " partitioning data completed  " << endl;

  auto ini_csr_start =
      std::chrono::high_resolution_clock::now();
  shared_sparseMat.get()->initialize_CSR_blocks(300, 300, true, false);
  auto ini_csr_end1 =
      std::chrono::high_resolution_clock::now();

  shared_sparseMat_Trans.get()->initialize_CSR_blocks(localARows, 300, true,
                                                      true);
  auto ini_csr_end2 =
      std::chrono::high_resolution_clock::now();
  shared_sparseMat_combined.get()->initialize_CSR_blocks(300, localBRows, true,
                                                         false);

  auto ini_csr_end =
      std::chrono::high_resolution_clock::now();

  auto ini_csr_duration = std::chrono::duration_cast<std::chrono::microseconds>(
                            ini_csr_end - ini_csr_start)
                            .count();
  auto ini_csr_duration1 = std::chrono::duration_cast<std::chrono::microseconds>(
                              ini_csr_end1 - ini_csr_start)
                              .count();
  auto ini_csr_duration2 = std::chrono::duration_cast<std::chrono::microseconds>(
                              ini_csr_end2 - ini_csr_end1)
                              .count();

  cout << " rank " << rank << " CSR block initialization completed  " << endl;
  auto dense_mat = shared_ptr<DenseMat<double, 2>>(
      new DenseMat<double, 2>(localARows, 0, 1.0, grid.get()->world_size));

  //    dense_mat.get()->print_matrix();
  cout << " rank " << rank << " creation of dense matrices completed  " << endl;

  auto communicator =
      unique_ptr<DataComm<int, double, 2>>(new DataComm<int, double, 2>(
          shared_sparseMat.get(), shared_sparseMat_Trans.get(), dense_mat.get(),
          grid.get()));

    cout << " rank " << rank << " async started  " << endl;

  unique_ptr<distblas::embedding::EmbeddingAlgo<int, double, 2>>
      embedding_algo =
          unique_ptr<distblas::embedding::EmbeddingAlgo<int, double, 2>>(
              new distblas::embedding::EmbeddingAlgo<int, double, 2>(
                  shared_sparseMat_combined.get(), dense_mat.get(),
                  communicator.get(), grid.get(), 5, -5));

  auto end_init = std::chrono::high_resolution_clock::now();

  embedding_algo.get()->algo_force2_vec_ns(1200, 300, 5, 0.02);

  auto end_train = std::chrono::high_resolution_clock::now();
  //  cout << " rank " << rank << " async completed  " << endl;

//  dense_mat.get()->print_matrix_rowptr();

  auto io_duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end_io - start_io)
          .count();
  auto init_duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end_init - end_io)
          .count();
  auto train_duration = std::chrono::duration_cast<std::chrono::microseconds>(
                            end_train - end_init)
                            .count();

  cout << " io: " << (io_duration / 1000)
       << " initialization: " << (init_duration / 1000)
       << " training: " << (train_duration / 1000)
      << " ini CSR duration: "<<(ini_csr_duration/1000)
       <<" ini_csr_duration1 "<<(ini_csr_duration1/1000)
       <<" ini_csr_duration2 "<<(ini_csr_duration2/1000)
       <<endl;

  MPI_Finalize();
  return 0;
}
