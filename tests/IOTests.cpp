#include "../include/DistBLAS/ParallelIO.hpp"
#include <iostream>
#include <memory>
#include <string>
#include <fstream>
#include <cstring>

using namespace std;
using namespace distblas::io;

int main(int argc, char **argv) {
  string file_path = argv[1];

  cout << " file_path " << file_path << endl;


  MPI_Init(&argc, &argv);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);



  string output_path =  "output.txt"+ to_string(rank);
  char stats[500];
  strcpy(stats, output_path.c_str());
  ofstream fout(stats, std::ios_base::app);



  auto reader = unique_ptr<ParallelIO>(new ParallelIO());

  vector<Tuple<int>> tuples;

  reader.get()->parallel_read_MM<int>(file_path, tuples);

  for(int i=0; i<tuples.size();i++){
    fout<<tuples[i].row << " "<< tuples[i].col<<" "<< tuples[i].value <<endl;
  }


  cout<<" rank "<<rank<< " size "<<tuples.size()<<endl;

  MPI_Finalize();
  return 0;
}