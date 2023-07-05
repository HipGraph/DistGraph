#ifndef COMMON_HEADER
#define COMMON_HEADER

#include <Eigen/Dense>
#include <cstddef>
#include <cstdint> // int64_t
#include <iostream>
#include <mkl_spblas.h>
#include <mpi.h>
#include <random>
#include <vector>

using namespace std;

namespace distblas::core {

int divide_and_round_up(int num, int denom);

vector<int> generate_random_numbers(int lower_bound, int upper_bound, int seed,
                                    int ns);

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

template <typename T> struct DataTuple {
  uint64_t col;
//  Eigen::Matrix<T, Eigen::Dynamic, 1> value;
//  DataTuple(int size) : col(0), value(size){}
  vector<T> value;
  DataTuple() : col(0), value(std::vector<T>(10)) {}
};


struct CSRHandle {
  vector<double> values;
  vector<MKL_INT> col_idx;
  vector<MKL_INT> rowStart;
  vector<MKL_INT> row_idx;
  sparse_matrix_t mkl_handle;
};

// TODO: removed reference type due to binding issue
template <typename T> bool column_major(Tuple<T> a, Tuple<T> b) {
  if (a.col == b.col) {
    return a.row < b.row;
  } else {
    return a.col < b.col;
  }
}

template <typename T> bool row_major(Tuple<T> a, Tuple<T> b) {
  if (a.row == b.row) {
    return a.col < b.col;
  } else {
    return a.row < b.row;
  }
}

extern MPI_Datatype SPTUPLE;

extern MPI_Datatype DENSETUPLE;

template <typename T> void initialize_mpi_datatype_SPTUPLE() {
  const int nitems = 3;
  int blocklengths[3] = {1, 1, 1};
  MPI_Datatype *types = new MPI_Datatype[3];
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

template <typename T>
void initialize_mpi_datatype_DENSETUPLE(int size) {
  MPI_Datatype vectorType;
  MPI_Type_contiguous(size, MPI_DOUBLE, &vectorType);
  MPI_Type_commit(&vectorType);

  const int nitems = 2;
  int blocklengths[2] = {1, 1};
  MPI_Datatype *types = new MPI_Datatype[2];
  types[0] = MPI_UINT64_T;
  types[1] = vectorType;

  MPI_Aint offsets[2];

  DataTuple<T> dummyTuple; // Dummy struct to get displacements

  MPI_Get_address(&dummyTuple.col, &offsets[0]);
  MPI_Get_address(&dummyTuple.value, &offsets[1]);
  offsets[1] -= offsets[0];
  offsets[0] = 0;
//  if (std::is_same<T, int>::value) {
//    types[1] = MPI_INT;
//    offsets[0] = offsetof(DataTuple<int>, col);
//    offsets[1] = offsetof(DataTuple<int>, value);
//  } else {
//    // TODO:Need to support all datatypes
//    types[1] = MPI_DOUBLE;
//    offsets[0] = offsetof(DataTuple<double>, col);
//    offsets[1] = offsetof(DataTuple<double>, value);
//
//  }
  cout << "offsets[0]  " << offsets[0] << " offsets[1] " << offsets[1]
       << endl;

  MPI_Type_create_struct(nitems, blocklengths, offsets, types, &DENSETUPLE);
  MPI_Type_commit(&DENSETUPLE);
  delete[] types;
}

template <typename SPT, typename DENT>
void initialize_mpi_datatypes(int size) {
  initialize_mpi_datatype_SPTUPLE<SPT>();
//  initialize_mpi_datatype_DENSETUPLE<DENT>(size);
}

}; // namespace distblas::core

#endif