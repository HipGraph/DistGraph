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

  MPI_Init(&argc, &argv);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  vector<int> random_number_vec = generate_random_numbers(0,60000,1234,10);
  string output_path = "random_number_generators_"+ to_string(rank)+".txt"
  char stats[500];
  strcpy(stats, output_path.c_str());
  ofstream fout(stats, std::ios_base::app);

  for(int i=0;i< random_number_vec.size();i++){
    fout<<random_number_vec[i]<<" ";
  }
  fout<<endl;

  MPI_Finalize();
  return 0;
}
