#ifndef COMMON_HEADER
#define COMMON_HEADER

#include "mpi_type_creator.hpp"
#include <Eigen/Dense>
#include <cstddef>
#include <cstdint> // int64_t
#include <iostream>
#include <mkl_spblas.h>
#include <mpi.h>
#include <random>
#include <vector>
#include <chrono>
#include <iostream>
#include <fstream>
#include "json.hpp"
#include <unordered_map>
#include <unistd.h>
#include <unordered_set>
#include <algorithm>

using namespace std;
using namespace std::chrono;
using json = nlohmann::json;

typedef chrono::time_point<std::chrono::steady_clock> my_timer_t;

namespace distblas::core {

int divide_and_round_up(uint64_t num, int denom);

vector<uint64_t> generate_random_numbers(int lower_bound, int upper_bound, int seed,
                                    int ns);

void prefix_sum(vector<int> &values, vector<int> &offsets);

size_t get_memory_usage();

void reset_performance_timers();

void stop_clock_and_add(my_timer_t &start, string counter_name);

void add_memory(size_t mem, string counter_name);

void add_datatransfers(uint64_t count, string counter_name);

void print_performance_statistics();

my_timer_t start_clock();

double stop_clock_get_elapsed(my_timer_t &start);

json json_perf_statistics();

int get_proc_length(double beta, int world_size);

int get_end_proc(int  starting_index, double beta, int world_size);

std::unordered_set<uint64_t> random_select(const std::unordered_set<uint64_t>& originalSet, int count);

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

template <typename T, size_t size> struct DataTuple {
  uint64_t col;
  std::array<T, size> value;
};

template <typename T, size_t size> struct CacheEntry {
  std::array<T, size> value;
  int inserted_batch_id;
  int inserted_itr;
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


extern vector<string> perf_counter_keys;

extern map<string, int> call_count;
extern map<string, double> total_time;

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

template <typename T,size_t embedding_dim>
void initialize_mpi_datatype_DENSETUPLE() {
  DataTuple<T,embedding_dim> p;
  DENSETUPLE = CreateCustomMpiType(p, p.col, p.value);
}

template <typename SPT, typename DENT, size_t embedding_dim>
void initialize_mpi_datatypes() {
  initialize_mpi_datatype_SPTUPLE<SPT>();
  initialize_mpi_datatype_DENSETUPLE<DENT,embedding_dim>();
}

template <typename DENT, size_t MAXBOUND>
DENT  scale(DENT v){
  if(v > MAXBOUND) return MAXBOUND;
  else if(v < -MAXBOUND) return -MAXBOUND;
  else return v;
}





}; // namespace distblas::core

#endif