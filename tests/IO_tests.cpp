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

  string output_path =  "output.txt"+ to_string(rank);
  char stats[500];
  strcpy(stats, output_path.c_str());
  ofstream fout(stats, std::ios_base::app);

  auto reader = unique_ptr<ParallelIO>(new ParallelIO());

  auto shared_sparseMat = shared_ptr<distblas::core::SpMat<int>>(new distblas::core::SpMat<int>());

  reader.get()->parallel_read_MM<int>(file_path, shared_sparseMat.get());

  for(int i=0; i<shared_sparseMat.get()->coords.size();i++){
    fout<<shared_sparseMat.get()->coords[i].row << " "
         << shared_sparseMat.get()->coords[i].col<<" "
         << shared_sparseMat.get()->coords[i].value
         <<endl;
  }

  auto grid = unique_ptr<Process3DGrid>(new Process3DGrid(4, 1, 1, 1));

  auto  partitioner = unique_ptr<GlobalAdjacency1DPartitioner>
      (new GlobalAdjacency1DPartitioner(shared_sparseMat.get()->gRows,
                                        shared_sparseMat.get()->gCols,
                                        grid.get()));

  cout<<" rank "<< rank << " partitioning data started "<<endl;

  partitioner.get()->partition_data(shared_sparseMat.get(), false);

  cout<<" rank "<< rank << " partitioning data stopped "<<endl;


//  string output_path1 =  "output_partitioned.txt"+ to_string(rank);
//  char stats1[500];
//  strcpy(stats1, output_path1.c_str());
//  ofstream fout1(stats1, std::ios_base::app);

//
//  for(int i=0; i<shared_sparseMat.get()->coords.size();i++){
//    fout1<<shared_sparseMat.get()->coords[i].row << " "
//         << shared_sparseMat.get()->coords[i].col<<" "
//         << shared_sparseMat.get()->coords[i].value
//         <<endl;
//  }



  int localBRows = divide_and_round_up(shared_sparseMat.get()->gCols,grid.get()->world_size);
  int localARows = divide_and_round_up(shared_sparseMat.get()->gRows,grid.get()->world_size);
  cout<<" rank "<< rank << " divide_block_cols "<<endl;
  shared_sparseMat.get()->divide_block_cols(localBRows,grid.get()->world_size, false);


  cout<<" rank "<< rank << " localARows "<<localARows<<endl;
  cout<<" rank "<< rank << " localBRows "<<localBRows<<endl;

  shared_sparseMat.get()->initialize_CSR_blocks(localARows,localBRows,-1, false);



  cout<< " gRows "<<shared_sparseMat.get()->gRows
       <<" gCols"<<shared_sparseMat.get()->gCols<<" NNz "<< shared_sparseMat.get()->gNNz<<endl;

  cout<<" rank "<<rank<< " size "<<shared_sparseMat.get()->coords.size()<<endl;

  MPI_Finalize();
  return 0;
}