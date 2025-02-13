#include "algo/embedding/algo.hpp"
#include "algo/spmm/spmm.hpp"
#include "core/common.h"
#include "core/csr_local.hpp"
#include "core/dense_mat.hpp"
#include "core/json.hpp"
#include "core/sparse_mat.hpp"
#include "io/parrallel_IO.hpp"
#include "net/data_comm.hpp"
#include "partition/partitioner.hpp"
#include "core/json.hpp"
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "algo/spgemm/spgemm.hpp"
#include "net/tile_based_data_comm.hpp"
#include "algo/spgemm/spgemm_with_tiling.hpp"
#include "algo/embedding/sparse_embedding.hpp"
#include "algo/bfs/multi_source_bfs.hpp"
#include "algo/baseline.hpp"
#include "algo/spmm/baseline_spmm.hpp"
#include "algo/fusedMM/baseline_fused_mm.hpp"
#include "algo/gat/gat.hpp"
#include "algo/gat/gat_layer.hpp"

using json = nlohmann::json;

using namespace std;
using namespace distblas::io;
using namespace distblas::partition;
using namespace distblas::net;
using namespace  distblas::core;



int main(int argc, char **argv) {

  const int dimension = 128;


  string input_file = "";
  string output_file = "embedding.txt";
  string data_set_name = "";

  int batch_size = 16384;
  double alpha = 0;
  double beta = 0.25;
  int iterations = 30;
  int ns = 5;
  double lr = 0.02;

  bool spmm = false;

  bool spgemm = false;

  bool col_major = false;
  bool sync_comm  = false;

  bool fix_batch_training = false;

  double density=0.5;

  bool save_results = false;

  string sparse_data_file ="";

  double output_sparsity=0;

   double tile_width_fraction=1;
   double tile_height_fraction=1;
   bool enable_remote=false;

   bool sparse_embedding=false;

   bool msbfs=false;

   bool fusedMM=false;

   bool gat=false;

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
    } else if (strcmp(argv[p], "-col_major") == 0) {
      int val = atoi(argv[p + 1]);
      col_major = (val != 0) ? true : false;
    } else if (strcmp(argv[p], "-sync_comm") == 0) {
      int val = atoi(argv[p + 1]);
      sync_comm = (val != 0) ? true : false;
    } else if (strcmp(argv[p], "-fix_batch_training") == 0) {
      int full_batch_tra = atoi(argv[p + 1]);
      fix_batch_training = full_batch_tra == 1 ? true : false;
    }else if (strcmp(argv[p], "-spmm") == 0) {
      int enable_spmm = atoi(argv[p + 1]);
      spmm = enable_spmm == 1 ? true : false;
    }else if (strcmp(argv[p], "-spgemm") == 0) {
      int enable_spgemm = atoi(argv[p + 1]);
      spgemm = enable_spgemm == 1 ? true : false;
    }else if (strcmp(argv[p], "-sparse_embedding") == 0) {
      int res = atof(argv[p + 1]);
      sparse_embedding = res == 1 ? true : false;
    }else if (strcmp(argv[p], "-msbfs") == 0) {
      int res = atof(argv[p + 1]);
      msbfs = res == 1 ? true : false;
    }else if (strcmp(argv[p], "-density") == 0) {
      density = atof(argv[p + 1]);
    }else if (strcmp(argv[p], "-save_results") == 0) {
      int save_res = atoi(argv[p + 1]);
      save_results = save_res == 1 ? true : false;
    }else if (strcmp(argv[p], "-input_sparse_file") == 0) {
      sparse_data_file = argv[p + 1];
    } else if (strcmp(argv[p], "-tile_width_fraction") == 0) {
      tile_width_fraction = atof(argv[p + 1]);
    }else if (strcmp(argv[p], "-tile_height_fraction") == 0) {
      tile_height_fraction = atof(argv[p + 1]);
    }else if (strcmp(argv[p], "-enable_remote") == 0) {
      int res = atof(argv[p + 1]);
      enable_remote = res == 1 ? true : false;
    }else if (strcmp(argv[p], "-fusedMM") == 0) {
        int res = atof(argv[p + 1]);
        fusedMM = res == 1 ? true : false;
    }else if (strcmp(argv[p], "-gat") == 0) {
        int res = atof(argv[p + 1]);
        gat = res == 1 ? true : false;
    }
  }

//  }

  MPI_Init(&argc, &argv);
  int rank;
  int world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  if (fix_batch_training) {
    batch_size = batch_size / world_size;
  }


  // Initialize MPI DataTypes
  if (!(spgemm or sparse_embedding)) {
    initialize_mpi_datatypes<VALUE_TYPE, dimension>();
  }else{
    initialize_mpi_datatypes<VALUE_TYPE, sp_tuple_max_dim>();
  }


//  // Creating reader
  auto reader = unique_ptr<ParallelIO>(new ParallelIO());

  // Creating ProcessorGrid
  auto grid = unique_ptr<Process3DGrid>(new Process3DGrid(world_size, 1, 1, 1));

  auto shared_sparseMat =
      shared_ptr<distblas::core::SpMat<VALUE_TYPE>>(new distblas::core::SpMat<VALUE_TYPE>(grid.get()));

  cout << " rank " << rank << " reading data from file path:  " << input_file<< endl;

  auto start_io = std::chrono::high_resolution_clock::now();

  reader.get()->parallel_read_MM<int64_t,int,VALUE_TYPE>(input_file, shared_sparseMat.get(),true);

  cout << " rank " << rank << " gROWs  " << shared_sparseMat.get()->gRows<< "gCols" << shared_sparseMat.get()->gCols << endl;
  cout << " rank " << rank << " reading data from file path:  " << input_file<< " completed " << endl;



  auto localBRows = divide_and_round_up(shared_sparseMat.get()->gCols,grid.get()->col_world_size);
  auto localARows = divide_and_round_up(shared_sparseMat.get()->gRows,grid.get()->col_world_size);

  // To enable full batch size
    if (spmm or spgemm or fusedMM) {
      batch_size = localARows;
    }

    if (spgemm and tile_height_fraction<1){
      batch_size = localARows*tile_height_fraction;
    }

  shared_sparseMat.get()->batch_size = batch_size;
  shared_sparseMat.get()->proc_row_width = localARows;
  shared_sparseMat.get()->proc_col_width = localBRows;

  vector<Tuple<VALUE_TYPE>> copiedVector(shared_sparseMat.get()->coords);
  auto shared_sparseMat_sender = make_shared<distblas::core::SpMat<VALUE_TYPE>>(grid.get(),
                                                                                copiedVector, shared_sparseMat.get()->gRows,
                                                                                shared_sparseMat.get()->gCols, shared_sparseMat.get()->gNNz, batch_size,
                                                                                localARows, localBRows, false, true);

  auto shared_sparseMat_receiver = make_shared<distblas::core::SpMat<VALUE_TYPE>>(grid.get(),
                                                                                  copiedVector, shared_sparseMat.get()->gRows,
                                                                                  shared_sparseMat.get()->gCols, shared_sparseMat.get()->gNNz, batch_size,
                                                                                  localARows, localBRows, true, false);



  cout << " rank " << rank << " localBRows  " << localBRows << " localARows "<< localARows << endl;

  vector<Tuple<VALUE_TYPE>> sparse_coo;
  auto sparse_input = make_shared<distblas::core::SpMat<VALUE_TYPE>>(grid.get());
  if (spgemm & save_results) {
    int local_cols = divide_and_round_up(static_cast<int>(dimension),grid->col_world_size);
    reader->build_sparse_random_matrix(localARows, shared_sparseMat.get()->gRows,
                                       local_cols,static_cast<int>(dimension), density, 0,sparse_coo,
                                       output_file+"/sparse_local.txt",grid.get(),false);
    cout<<" rank "<<grid->rank_in_col<<" nnz "<<sparse_coo.size()<<endl;
  } else if (spgemm) {
    reader.get()->parallel_read_MM<int64_t,VALUE_TYPE,VALUE_TYPE>(sparse_data_file, sparse_input.get(),false,true);
    sparse_input.get()->batch_size = batch_size;
    sparse_input.get()->proc_row_width = localARows;
    sparse_input.get()->proc_col_width = static_cast<int>(dimension);
  }

  if (!save_results) {
    auto end_io = std::chrono::high_resolution_clock::now();

    auto partitioner = unique_ptr<GlobalAdjacency1DPartitioner>(
        new GlobalAdjacency1DPartitioner(grid.get()));

    cout << " rank " << rank << " partitioning data started  " << endl;

    partitioner.get()->partition_data<VALUE_TYPE>(
        shared_sparseMat_sender.get());
    partitioner.get()->partition_data<VALUE_TYPE>(
        shared_sparseMat_receiver.get());
    partitioner.get()->partition_data<VALUE_TYPE>(shared_sparseMat.get());

    cout << " rank " << rank << " partitioning data completed  " << endl;

    shared_sparseMat.get()->initialize_CSR_blocks(true);
    shared_sparseMat_sender.get()->initialize_CSR_blocks(true);
    shared_sparseMat_receiver.get()->initialize_CSR_blocks(true);
  }
  if (spgemm and !save_results){
    cout << " rank " << rank << " input gROWs  " << sparse_input.get()->gRows<< "input gCols" << sparse_input.get()->gCols << endl;
    cout << " rank " << rank << " input partitioning started   " << endl;
//    partitioner.get()->partition_data<VALUE_TYPE>(sparse_input.get());
    cout << " rank " << rank << " input partitioning data completed  " << endl;
    sparse_input->initialize_CSR_blocks(true);
    cout << " rank " << rank << " input csr  completed  " << endl;
  }

  cout << " rank " << rank << " CSR block initialization completed  " << endl;

//  dense_local->print_cache(i);
//  dense_mat.get()->print_matrix_rowptr(-1);
 json perf_stats;
  if (spmm) {
        unique_ptr<distblas::algo::BaselineSpMM<INDEX_TYPE, VALUE_TYPE, dimension>> spgemm_algo = unique_ptr<distblas::algo::BaselineSpMM<INDEX_TYPE, VALUE_TYPE, dimension>>(
            new distblas::algo::BaselineSpMM<INDEX_TYPE, VALUE_TYPE, dimension>(
                shared_sparseMat.get(), shared_sparseMat_receiver.get(),
                shared_sparseMat_sender.get(), sparse_input.get(),
                grid.get(),
                alpha, beta,col_major,sync_comm, tile_width_fraction,false));


        MPI_Barrier(MPI_COMM_WORLD);
        cout << " rank " << rank << " SpMM algo started  " << endl;
        perf_stats =  spgemm_algo.get()->execute(iterations, batch_size,lr);
        cout << " rank " << rank << " SpMM algo completed  " << endl;

  }else if(fusedMM){
      unique_ptr<distblas::algo::BaselineFusedMM<INDEX_TYPE, VALUE_TYPE, dimension>> fused_algo = unique_ptr<distblas::algo::BaselineFusedMM<INDEX_TYPE, VALUE_TYPE, dimension>>(
              new distblas::algo::BaselineFusedMM<INDEX_TYPE, VALUE_TYPE, dimension>(
                      shared_sparseMat.get(), shared_sparseMat_receiver.get(),
                      shared_sparseMat_sender.get(), sparse_input.get(),
                      grid.get(),
                      alpha, beta,col_major,sync_comm, tile_width_fraction,false));


      MPI_Barrier(MPI_COMM_WORLD);
      cout << " rank " << rank << " FusedMM algo started  " << endl;
      perf_stats =  fused_algo.get()->execute(iterations, batch_size,lr);
      cout << " rank " << rank << " FusedMM algo completed  " << endl;
  }else if(gat){
      unique_ptr<distblas::algo::GAT<INDEX_TYPE, VALUE_TYPE, 256>> gat = make_unique<
               distblas::algo::GAT<INDEX_TYPE, VALUE_TYPE, 256>>(
                      shared_sparseMat.get(), shared_sparseMat_receiver.get(),
                      shared_sparseMat_sender.get(), sparse_input.get(),
                      grid.get(),
                      alpha, beta,col_major,sync_comm, tile_width_fraction,false);
      gat->addLayer(distblas::algo::GATLayer<INDEX_TYPE,VALUE_TYPE,256>(grid.get(),1024,4));
      gat->addLayer(distblas::algo::GATLayer<INDEX_TYPE,VALUE_TYPE,256>(grid.get(),1024,6));
      gat->addLayer(distblas::algo::GATLayer<INDEX_TYPE,VALUE_TYPE,256>(grid.get(),1024,2));
      gat->addLayer(distblas::algo::GATLayer<INDEX_TYPE,VALUE_TYPE,256>(grid.get(),1024,7));

      MPI_Barrier(MPI_COMM_WORLD);
      cout << " rank " << rank << " gat algo started  " << endl;
      perf_stats = gat->execute();
      cout << " rank " << rank << " gat algo completed  " << endl;
  }else if(spgemm and !save_results){
//    bool has_spgemm =dimension>spa_threshold?true:false;
    bool has_spgemm =true;
//    auto sparse_out = make_shared<distblas::core::SpMat<VALUE_TYPE>>(grid.get(),localARows,dimension,has_spgemm);

//    unique_ptr<distblas::algo::SpGEMMAlgo<INDEX_TYPE, VALUE_TYPE, dimension>> spgemm_algo = unique_ptr<distblas::algo::SpGEMMAlgo<INDEX_TYPE, VALUE_TYPE, dimension>>(
//                new distblas::algo::SpGEMMAlgo<INDEX_TYPE, VALUE_TYPE, dimension>(
//                    shared_sparseMat.get(), shared_sparseMat_receiver.get(),
//                    shared_sparseMat_sender.get(), sparse_input.get(),sparse_out.get(),
//                    grid.get(),
//                    alpha, beta,col_major,sync_comm));

//    unique_ptr<distblas::algo::SpGEMMAlgoWithTiling<INDEX_TYPE, VALUE_TYPE, dimension>> spgemm_algo = unique_ptr<distblas::algo::SpGEMMAlgoWithTiling<INDEX_TYPE, VALUE_TYPE, dimension>>(
//        new distblas::algo::SpGEMMAlgoWithTiling<INDEX_TYPE, VALUE_TYPE, dimension>(
//            shared_sparseMat.get(), shared_sparseMat_receiver.get(),
//            shared_sparseMat_sender.get(), sparse_input.get(),sparse_out.get(),
//            grid.get(),
//            alpha, beta,col_major,sync_comm, tile_width_fraction,has_spgemm));

        unique_ptr<distblas::algo::Baseline<INDEX_TYPE, VALUE_TYPE, dimension>> spgemm_algo = unique_ptr<distblas::algo::Baseline<INDEX_TYPE, VALUE_TYPE, dimension>>(
            new distblas::algo::Baseline<INDEX_TYPE, VALUE_TYPE, dimension>(
                shared_sparseMat.get(), shared_sparseMat_receiver.get(),
                shared_sparseMat_sender.get(), sparse_input.get(),
                grid.get(),
                alpha, beta,col_major,sync_comm, tile_width_fraction,has_spgemm));


    MPI_Barrier(MPI_COMM_WORLD);
    cout << " rank " << rank << " spgemm baseline algo started  " << endl;
    perf_stats =  spgemm_algo.get()->execute(iterations, batch_size,lr,enable_remote);
    cout << " rank " << rank << " spgemm baseline algo completed  " << endl;
//    output_sparsity = (sparse_out->csr_local_data)->handler->rowStart[(sparse_out->csr_local_data)->handler->rowStart.size()-1];
//    output_sparsity = 100*(output_sparsity/(((sparse_out->csr_local_data)->handler->rowStart.size()-1)*dimension));
//    reader->parallel_write_csr<double>(output_file+"/sparse_embedding.txt",(sparse_out->csr_local_data)->handler.get(),grid.get(), localARows,shared_sparseMat.get()->gRows,dimension);

  }else if (msbfs and !save_results){
    bool has_spgemm =dimension>spa_threshold?true:false;
            unique_ptr<distblas::algo::MultiSourceBFS<INDEX_TYPE, VALUE_TYPE, dimension>> spgemm_algo = unique_ptr<distblas::algo::MultiSourceBFS<INDEX_TYPE, VALUE_TYPE, dimension>>(
                new distblas::algo::MultiSourceBFS<INDEX_TYPE, VALUE_TYPE, dimension>(
                    shared_sparseMat.get(), shared_sparseMat_receiver.get(),
                    shared_sparseMat_sender.get(), sparse_input.get(),
                    grid.get(),
                    alpha, beta,col_major,sync_comm, tile_width_fraction,has_spgemm));
    MPI_Barrier(MPI_COMM_WORLD);
    cout << " rank " << rank << " msbfs algo started  " << endl;
    perf_stats =  spgemm_algo.get()->execute(iterations, batch_size,lr);
    cout << " rank " << rank << " msbfs algo completed  " << endl;

  } else if (sparse_embedding and !save_results){
    bool has_spgemm =dimension>spa_threshold?true:false;
    auto sparse_out = make_shared<distblas::core::SpMat<VALUE_TYPE>>(grid.get(),localARows,dimension,has_spgemm,true);
    unique_ptr<distblas::algo::SparseEmbedding<INDEX_TYPE, VALUE_TYPE, dimension>> spgemm_algo = unique_ptr<distblas::algo::SparseEmbedding<INDEX_TYPE, VALUE_TYPE, dimension>>(
            new distblas::algo::SparseEmbedding<INDEX_TYPE, VALUE_TYPE, dimension>(
                shared_sparseMat.get(), shared_sparseMat_receiver.get(),
                shared_sparseMat_sender.get(), sparse_out.get(),
                grid.get(),
                alpha, beta,col_major,sync_comm, tile_width_fraction,has_spgemm));
    spgemm_algo.get()->algo_sparse_embedding(iterations, batch_size,ns,lr,density,enable_remote);
    perf_stats = json_perf_statistics();
    reader->parallel_write(output_file+"/embedding.txt",sparse_out.get()->dense_collector.get(),
                           localARows, dimension, grid.get(),shared_sparseMat.get());
  } else if (!save_results) {
    auto dense_mat = shared_ptr<DenseMat<INDEX_TYPE, VALUE_TYPE, dimension>>(
        new DenseMat<INDEX_TYPE, VALUE_TYPE, dimension>(grid.get(), localARows));

    unique_ptr<distblas::algo::EmbeddingAlgo<INDEX_TYPE, VALUE_TYPE, dimension>>

        embedding_algo =
            unique_ptr<distblas::algo::EmbeddingAlgo<INDEX_TYPE, VALUE_TYPE, dimension>>(
                new distblas::algo::EmbeddingAlgo<INDEX_TYPE, VALUE_TYPE, dimension>(
                    shared_sparseMat.get(), shared_sparseMat_receiver.get(),
                    shared_sparseMat_sender.get(), dense_mat.get(), grid.get(),
                    alpha, beta, 5, -5,col_major,sync_comm));
    MPI_Barrier(MPI_COMM_WORLD);
    cout << " rank " << rank << " embedding algo started  " << endl;
    embedding_algo.get()->algo_force2_vec_ns(iterations, batch_size, ns, lr);
    perf_stats = json_perf_statistics();
    reader->parallel_write(output_file+"/embedding.txt",dense_mat.get()->nCoordinates,localARows, dimension, grid.get(),shared_sparseMat.get());
  }
  cout << " rank " << rank << " algo completed  " << endl;
//
  if (!save_results) {
    ofstream fout;
    fout.open("perf_output", std::ios_base::app);
    ////
    json j_obj;
    j_obj["alpha"] = alpha;
    j_obj["beta"] = beta;
    j_obj["algo"] = "Embedding";
    j_obj["p"] = world_size;
    //  j_obj["sparsity"] = density;
    j_obj["data_set"] = data_set_name;
    j_obj["d"] = dimension;
    j_obj["batch_size"] = batch_size;
    j_obj["tile_width_fraction"] = tile_width_fraction;
    //  if (spgemm){
    //    j_obj["output_nnz"] = output_sparsity;
    //  }
    j_obj["perf_stats"] = perf_stats;
    if (rank == 0) {
      fout << j_obj.dump(4) << "," << endl;
    }
    //
    fout.close();
  }
// reader->parallel_write(output_file+"/embedding.txt",dense_mat.get()->nCoordinates,localARows, dimension, grid.get(),shared_sparseMat.get());
 if(spgemm & save_results) {
   int local_cols = divide_and_round_up(static_cast<int>(dimension),grid->col_world_size);
   reader->parallel_write(output_file+"/sparse_local.txt",sparse_coo,grid.get(), local_cols,shared_sparseMat.get()->gRows,static_cast<int>(dimension),true);
 }


  MPI_Finalize();
  return 0;
}
