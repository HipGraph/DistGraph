#ifndef COMMON_HEADER
#define COMMON_HEADER

#include <cstddef>
#include <cstdint> // int64_t
#include <mpi.h>
#include <vector>
#include <mkl_spblas.h>
#include <iostream>

using namespace std;

namespace distblas::core {

int divide_and_round_up(int num, int denom);

void prefix_sum(vector<int> &values, vector<int> &offsets);

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


struct CSRHandle {
  vector<double> values;
  vector<MKL_INT> col_idx;
  vector<MKL_INT> rowStart;
  vector<MKL_INT> row_idx;
  sparse_matrix_t mkl_handle;
};

//TODO: removed reference type due to binding issue
template <typename T>
bool column_major(Tuple<T> a, Tuple<T> b) {
  if (a.col == b.col) {
    return a.row < b.row;
  } else {
    return a.col < b.col;
  }
}

template <typename T> bool row_major(Tuple<T> &a, Tuple<T> &b) {
  if (a.row == b.row) {
    return a.col < b.col;
  } else {
    return a.row < b.row;
  }
}

extern MPI_Datatype SPTUPLE;

template <typename T>
void initialize_mpi_datatypes() {
  const int nitems = 3;
  int blocklengths[3] = {1, 1, 1};
  MPI_Datatype*  types = new MPI_Datatype[3];
  types[0] = MPI_UINT64_T;
  types[1] = MPI_UINT64_T;
  MPI_Aint offsets[3];
  if (std::is_same<T, int>::value) {
    types[2] = MPI_INT;
    offsets[0] = offsetof(Tuple<int>, row);
    offsets[1] = offsetof(Tuple<int>, col);
    offsets[2] = offsetof(Tuple<int>, value);
  } else {
    // TODO:Need to support all datatypes
    types[2] = MPI_DOUBLE;
    offsets[0] = offsetof(Tuple<double>, row);
    offsets[1] = offsetof(Tuple<double>, col);
    offsets[2] = offsetof(Tuple<double>, value);
  }

  MPI_Type_create_struct(nitems, blocklengths, offsets, types, &SPTUPLE);
  MPI_Type_commit(&SPTUPLE);
  delete[] types;
}

}; // namespace distblas::core

#endif