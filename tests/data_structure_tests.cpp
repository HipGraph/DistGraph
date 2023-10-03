#include "../cpp/algo/algo.hpp"
#include "../cpp/algo/spmm.hpp"
#include "../cpp/core/common.h"
#include "../cpp/core/csr_local.hpp"
#include "../cpp/core/dense_mat.hpp"
#include "../cpp/core/json.hpp"
#include "../cpp/core/sparse_mat.hpp"
#include "../cpp/io/parrallel_IO.hpp"
#include "../cpp/net/data_comm.hpp"
#include "../cpp/partition/partitioner.hpp"
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

using json = nlohmann::json;

using namespace std;
using namespace distblas::io;
using namespace distblas::partition;
using namespace distblas::net;

int main(int argc, char **argv) {

  const int dimension = 128;

  string input_file = "";
  string output_file = "embedding.txt";
  string data_set_name = "";

  int batch_size = 16384;
  double alpha = 0;
  double beta = 0.25;
  int iterations = 2;
  int ns = 5;
  double lr = 0.02;

  bool spmm = false;

  for (int p = 0; p < argc; p++) {
    if (strcmp(argv[p], "-input") == 0) {
      input_file = argv[p + 1];
    } else if (strcmp(argv[p], "-output") == 0) {
      output_file = argv[p + 1];
    } else if (strcmp(argv[p], "-batch") == 0) {
      batch_size = atoi(argv[p + 1]);
    } else if (strcmp(argv[p], "-iter") == 0) {
      iterations = atoi(argv[p + 1]);
    } else if (strcmp(argv[p], "-alpha") == 0) {
      alpha = atof(argv[p + 1]);
    } else if (strcmp(argv[p], "-lr") == 0) {
      lr = atof(argv[p + 1]);
    } else if (strcmp(argv[p], "-nsamples") == 0) {
      ns = atoi(argv[p + 1]);
    } else if (strcmp(argv[p], "-beta") == 0) {
      beta = atof(argv[p + 1]);
    } else if (strcmp(argv[p], "-dataset") == 0) {
      data_set_name = argv[p + 1];
    }
  }

  MPI_Init(&argc, &argv);
  int rank;
  int world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
//  batch_size = batch_size/world_size;

  // Initialize MPI DataTypes
  initialize_mpi_datatypes<int, double, dimension>();

  // Creating reader
  auto reader = unique_ptr<ParallelIO>(new ParallelIO());

  // Creating ProcessorGrid
  auto grid = unique_ptr<Process3DGrid>(new Process3DGrid(world_size, 1, 1, 1));

  auto shared_sparseMat =
      shared_ptr<distblas::core::SpMat<int>>(new distblas::core::SpMat<int>());

  cout << " rank " << rank << " reading data from file path:  " << input_file
       << endl;

  auto start_io = std::chrono::high_resolution_clock::now();
  reader.get()->parallel_read_MM<int>(input_file, shared_sparseMat.get(), true);
  auto end_io = std::chrono::high_resolution_clock::now();

  cout << " rank " << rank << " reading data from file path:  " << input_file
       << " completed " << endl;

  auto localBRows = divide_and_round_up(shared_sparseMat.get()->gCols,
                                        grid.get()->world_size);
  auto localARows = divide_and_round_up(shared_sparseMat.get()->gRows,
                                        grid.get()->world_size);

  // To enable full batch size
//  batch_size = localARows;

  cout << " rank " << rank << " localBRows  " << localBRows << " localARows "
       << localARows << endl;

  shared_sparseMat.get()->batch_size = batch_size;
  shared_sparseMat.get()->proc_row_width = localARows;
  shared_sparseMat.get()->proc_col_width = localBRows;

  cout << " rank " << rank << " gROWs  " << shared_sparseMat.get()->gRows
       << "gCols" << shared_sparseMat.get()->gCols << endl;

  vector<Tuple<int>> copiedVector(shared_sparseMat.get()->coords);
  auto shared_sparseMat_sender = make_shared<distblas::core::SpMat<int>>(
      copiedVector, shared_sparseMat.get()->gRows,
      shared_sparseMat.get()->gCols, shared_sparseMat.get()->gNNz, batch_size,
      localARows, localBRows, false, true);

  auto shared_sparseMat_receiver = make_shared<distblas::core::SpMat<int>>(
      copiedVector, shared_sparseMat.get()->gRows,
      shared_sparseMat.get()->gCols, shared_sparseMat.get()->gNNz, batch_size,
      localARows, localBRows, true, false);

  auto partitioner = unique_ptr<GlobalAdjacency1DPartitioner>(
      new GlobalAdjacency1DPartitioner(grid.get()));

  cout << " rank " << rank << " partitioning data started  " << endl;

  partitioner.get()->partition_data(shared_sparseMat_sender.get());
  partitioner.get()->partition_data(shared_sparseMat_receiver.get());
  partitioner.get()->partition_data(shared_sparseMat.get());

  cout << " rank " << rank << " partitioning data completed  " << endl;

  shared_sparseMat.get()->initialize_CSR_blocks();
  shared_sparseMat_sender.get()->initialize_CSR_blocks();
  shared_sparseMat_receiver.get()->initialize_CSR_blocks();

  cout << " rank " << rank << " CSR block initialization completed  " << endl;
  auto dense_mat = shared_ptr<DenseMat<int, double, dimension>>(
      new DenseMat<int, double, dimension>(grid.get(), localARows));

  if (spmm) {
    auto dense_mat_output = shared_ptr<DenseMat<int, double, dimension>>(
        new DenseMat<int, double, dimension>(grid.get(), localARows));

    unique_ptr<distblas::algo::SpMMAlgo<int, double, dimension>>

        embedding_algo =
            unique_ptr<distblas::algo::SpMMAlgo<int, double, dimension>>(
                new distblas::algo::SpMMAlgo<int, double, dimension>(
                    shared_sparseMat.get(), shared_sparseMat_receiver.get(),
                    shared_sparseMat_sender.get(), dense_mat.get(),
                    dense_mat_output.get(), grid.get(), alpha, beta, 5, -5));

    MPI_Barrier(MPI_COMM_WORLD);
    cout << " rank " << rank << "  algo started  " << endl;
    embedding_algo.get()->algo_spmm(iterations, batch_size, lr);

  } else {

    unique_ptr<distblas::algo::EmbeddingAlgo<int, double, dimension>>

        embedding_algo =
            unique_ptr<distblas::algo::EmbeddingAlgo<int, double, dimension>>(
                new distblas::algo::EmbeddingAlgo<int, double, dimension>(
                    shared_sparseMat.get(), shared_sparseMat_receiver.get(),
                    shared_sparseMat_sender.get(), dense_mat.get(), grid.get(),
                    alpha, beta, 5, -5));

    MPI_Barrier(MPI_COMM_WORLD);
    cout << " rank " << rank << "  algo started  " << endl;
    embedding_algo.get()->algo_force2_vec_ns(iterations, batch_size, ns, lr);
  }
  cout << " rank " << rank << " async completed  " << endl;
  //
  //
  //  cout << " rank " << rank << " training completed  " << endl;
  ofstream fout;
  fout.open("perf_output", std::ios_base::app);

  json j_obj;
  j_obj["alpha"] = alpha;
  j_obj["beta"] = beta;
  j_obj["algo"] = "Embedding";
  j_obj["p"] = world_size;
  j_obj["data_set"] = data_set_name;
  j_obj["perf_stats"] = json_perf_statistics();
  if (rank == 0) {
    fout << j_obj.dump(4) << "," << endl;
  }

  fout.close();
  //
  //  reader->parallel_write("embedding.txt",
  //  dense_mat.get()->nCoordinates,localARows, dimension);
  //    dense_mat.get()->print_matrix_rowptr(0);

  MPI_Finalize();
  return 0;
}
