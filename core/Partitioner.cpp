#include "../include/DistBLAS/Partitioner.hpp"
using namespace distblas::core;
using namespace std;

Partitioner::Partitioner(MPI_Comm comm) {
 this->comm = comm;
}

Partitioner::~Partitioner() {

}

template <typename T>
SpMat<T>* redistribute_data(SpMat<T> &spmat, bool  inplace) {

  int num_procs, proc_rank;
  MPI_Comm_size(this->comm, &num_procs);
  MPI_Comm_rank(this->comm, &proc_rank);

  vector<int> sendcounts(num_procs, 0);
  vector<int> recvcounts(num_procs, 0);

  vector<int> offsets, bufindices;


  unique_ptr<Tuple<T>[]> sendbuf_ptr(new Tuple<T>[spmat.coords.size()]);

//#pragma omp parallel for
//  for(int i = 0; i < coords.size(); i++) {
//    int owner = dist->getOwner(coords[i].r, coords[i].c, transpose);
//#pragma omp atomic update
//    sendcounts[owner]++;
//  }
//  prefix_sum(sendcounts, offsets);
//  bufindices = offsets;
//
//#pragma omp parallel for
//  for(int i = 0; i < coords.size(); i++) {
//    int owner = dist->getOwner(coords[i].r, coords[i].c, transpose);
//
//    int idx;
//#pragma omp atomic capture
//    idx = bufindices[owner]++;
//
//    sendbuf[idx].r = transpose ? coords[i].c : coords[i].r;
//    sendbuf[idx].c = transpose ? coords[i].r : coords[i].c;
//    sendbuf[idx].value = coords[i].value;
//  }
//
//
//  // Broadcast the number of nonzeros that each processor is going to receive
//  MPI_Alltoall(sendcounts.data(), 1, MPI_INT, recvcounts.data(), 1,
//               MPI_INT, dist->world);
//
//  vector<int> recvoffsets;
//  prefix_sum(recvcounts, recvoffsets);
//
//  // Use the sizing information to execute an AlltoAll
//  int total_received_coords =
//      std::accumulate(recvcounts.begin(), recvcounts.end(), 0);
//
//  SpmatLocal* result;
//
//  if(in_place) {
//    result = this;
//  }
//  else {
//    result = new SpmatLocal();
//  }
//
//  result->M = transpose ? this->N : this->M;
//  result->N = transpose ? this->M : this->N;
//  result->dist_nnz = this->dist_nnz;
//
//  result->initialized = true;
//  (result->coords).resize(total_received_coords);
//
//  MPI_Alltoallv(sendbuf, sendcounts.data(), offsets.data(),
//                SPCOORD, (result->coords).data(), recvcounts.data(), recvoffsets.data(),
//                SPCOORD, dist->world
//  );
//
//  // TODO: Parallelize the sort routine?
//  //std::sort((result->coords).begin(), (result->coords).end(), column_major);
//  __gnu_parallel::sort((result->coords).begin(), (result->coords).end(), column_major);
//  delete[] sendbuf;

  return spmat;

}

