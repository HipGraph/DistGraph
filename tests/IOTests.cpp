#include "../include/DistBLAS/ParallelIO.hpp"
#include <iostream>
#include <memory>

using namespace std;
using namespace distblas::io;

int main(int argc, char **argv) {
  string file_path = argv[1];

  cout << " file_path " << file_path << endl;

  MPI_Init(&argc, &argv);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  auto reader = unique_ptr<ParallelIO>(new ParallelIO());

  vector<Tuple> tuples = reader.get()->parallel_read_MM(file_path);

  cout<<" rank "<<rank<< " size "<<tuples.size()<<endl;

  MPI_Finalize();
  return 0;
}