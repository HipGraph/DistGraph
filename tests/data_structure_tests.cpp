#include "../cpp/core/common.h"
#include "../cpp/core/csr_local.hpp"
#include "../cpp/core/dense_mat.hpp"
#include "../cpp/core/sparse_mat.hpp"
#include "../cpp/io/parrallel_IO.hpp"
#include "../cpp/net/data_comm.hpp"
#include "../cpp/partition/partitioner.hpp"
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

  int batch_size = 15000;

  MPI_Init(&argc, &argv);
  int rank;
  int world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  initialize_mpi_datatypes<int, double, 10>();

  auto reader = unique_ptr<ParallelIO>(new ParallelIO());
  auto grid = unique_ptr<Process3DGrid>(new Process3DGrid(2, 1, 1, 1));

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

  shared_sparseMat.get()->block_row_width = batch_size;
  shared_sparseMat.get()->block_col_width = batch_size;
  shared_sparseMat.get()->proc_row_width = localARows;
  shared_sparseMat.get()->proc_col_width = localBRows;

  vector<Tuple<int>> copiedVector(shared_sparseMat.get()->coords);
  auto shared_sparseMat_Trans = make_shared<distblas::core::SpMat<int>>(
      copiedVector, shared_sparseMat.get()->gRows,
      shared_sparseMat.get()->gCols, shared_sparseMat.get()->gNNz, localARows,
      batch_size, localARows, localBRows);

  auto partitioner =
      unique_ptr<GlobalAdjacency1DPartitioner>(new GlobalAdjacency1DPartitioner(
          shared_sparseMat.get()->gRows, shared_sparseMat.get()->gCols,
          grid.get()));

  cout << " rank " << rank << " partitioning data started  " << endl;

  partitioner.get()->partition_data(shared_sparseMat_Trans.get(), true);
  partitioner.get()->partition_data(shared_sparseMat.get(), false);
  //

  shared_sparseMat.get()->divide_block_cols(
      15000, localBRows, grid.get()->world_size, true, false);
  shared_sparseMat.get()->sort_by_rows();
  shared_sparseMat.get()->divide_block_rows(15000, localBRows, true, false);

  shared_sparseMat_Trans.get()->divide_block_cols(15000, localBRows, 2, true,
                                                  true);
  shared_sparseMat_Trans.get()->sort_by_rows();
  shared_sparseMat_Trans.get()->divide_block_rows(localARows, 15000, true,
                                                  true);

  cout << " rank " << rank << " partitioning data completed  " << endl;

  cout << " rank " << rank << " initialization of CSR started  " << endl;
  shared_sparseMat.get()->initialize_CSR_blocks(15000, localBRows, localARows,
                                                localBRows, -1, false);
  cout << " rank " << rank << " initialization of  CSR completed  " << endl;
  cout << " rank " << rank << " initialization of transpose CSR started  "
       << endl;
  shared_sparseMat_Trans.get()->initialize_CSR_blocks(
      localARows, 15000, localARows, localBRows, -1, true);
  cout << " rank " << rank << " initialization of transpose CSR completed  "
       << endl;

  //  shared_sparseMat.get()->print_blocks_and_cols(false);
  //  shared_sparseMat_Trans.get()->print_blocks_and_cols(true);

  vector<uint64_t> id_list_trans;
  vector<uint64_t> id_list;
  shared_sparseMat_Trans.get()->fill_col_ids(0, 0, id_list_trans, true, true);
  shared_sparseMat.get()->fill_col_ids(0, 0, id_list, false, true);

  cout << " rank " << rank << " creation of dense matrices started  " << endl;
  auto dense_mat = shared_ptr<DenseMat<double, 10>>(
      new DenseMat<double, 10>(localARows, 0, 1.0, grid.get()->world_size));
  //    dense_mat.get()->print_matrix();
  cout << " rank " << rank << " creation of dense matrices completed  " << endl;

  auto communicator =
      unique_ptr<DataComm<int, double, 10>>(new DataComm<int, double, 10>(
          shared_sparseMat.get(), shared_sparseMat_Trans.get(), dense_mat.get(),
          grid.get()));

  cout << " rank " << rank << " async started  " << endl;
  MPI_Request request;
  std::vector<DataTuple<double, 10>> *results_init =
      new vector<DataTuple<double, 10>>();
  communicator.get()->async_transfer(0, true, true, results_init, request);
  communicator.get()->populate_cache(results_init,request);

  for (int i = 0; i < (localARows / batch_size); i++) {
    MPI_Request request;
    unique_ptr<std::vector<DataTuple<double, 10>>> results_postive_ptr =
        unique_ptr<std::vector<DataTuple<double, 10>>>(new vector<DataTuple<double, 10>>());
    communicator.get()->async_transfer(i, true, true, results_postive_ptr.get(), request);
    communicator.get()->populate_cache(results_postive_ptr.get(),request);

    MPI_Request request_two;
    unique_ptr<std::vector<DataTuple<double, 10>>> results_negative_ptr =
        unique_ptr<std::vector<DataTuple<double, 10>>>(new vector<DataTuple<double, 10>>());
    vector<uint64_t> random_number_vec =
        generate_random_numbers(0, 60000, i, 10);
//    communicator.get()->async_transfer(random_number_vec, true,
//                                       results_negative_ptr.get(), request_two);
//    communicator.get()->populate_cache(results_negative_ptr.get(), request_two);

  }
//  delete results_init;
  cout << " rank " << rank << " async completed  " << endl;

//  dense_mat.get()->print_cache();

  cout << " rank " << rank << " processing completed  " << endl;

  MPI_Finalize();
  return 0;
}
