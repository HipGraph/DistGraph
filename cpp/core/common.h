#ifndef COMMON_HEADER
#define COMMON_HEADER

#include <cstdint> // int64_t
#include <vector>
#include <mpi.h>

using namespace std;

namespace distblas::core {

template <typename T> struct Tuple {
  int64_t row;
  int64_t col;
  T value;
};

template <typename T> struct CSR {
  int64_t row;
  int64_t col;
  T value;
};

template <typename T>
bool column_major(Tuple<T> &a, Tuple<T> &b) {
  if(a.col == b.col) {
    return a.row < b.row;
  }
  else {
    return a.col< b.col;
  }
}

template <typename T>
bool row_major(Tuple<T> &a, Tuple<T> &b) {
  if(a.row == b.row) {
    return a.col < b.col;
  }
  else {
    return a.row < b.row;
  }
}

extern MPI_Datatype SPTUPLE;

template <typename T>
void initialize_mpi_datatypes() {
  const int nitems = 3;
  int blocklengths[3] = {1, 1, 1};
  MPI_Datatype types[3];
  if (std::is_same<T, int>::value) {
    types[3] = {MPI_UINT64_T, MPI_UINT64_T, MPI_INT};
  } else {
    //TODO:Need to support all datatypes
    types[3] = {MPI_UINT64_T, MPI_UINT64_T, MPI_DOUBLE};
  }

  MPI_Aint offsets[3];
  offsets[0] = offsetof(Tuple<T>, row);
  offsets[1] = offsetof(Tuple<T>, col);
  offsets[2] = offsetof(Tuple<T>, value);
  MPI_Type_create_struct(nitems, blocklengths, offsets, types, &SPTUPLE);
  MPI_Type_commit(&SPTUPLE);
}

int divide_and_round_up(int num, int denom);

void prefix_sum(vector<int> &values, vector<int> &offsets);

} // namespace distblas::core

#endif