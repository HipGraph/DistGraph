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

const int row_max = 22;
const int sp_tuple_max_dim= 512;

const int hash_scale = 107;

const int spa_threshold=1024;

using INDEX_TYPE = uint64_t;

using VALUE_TYPE = double;

const VALUE_TYPE MAX_BOUND = 5;
const VALUE_TYPE MIN_BOUND = -5;

typedef chrono::time_point<std::chrono::steady_clock> my_timer_t;

namespace distblas::core {

int divide_and_round_up(INDEX_TYPE num, int denom);

vector<INDEX_TYPE> generate_random_numbers(int lower_bound, int upper_bound, int seed,
                                    int ns);

void prefix_sum(vector<int> &values, vector<int> &offsets);

size_t get_memory_usage();

void reset_performance_timers();

void stop_clock_and_add(my_timer_t &start, string counter_name);

void add_perf_stats(size_t mem, string counter_name);

void add_perf_stats(INDEX_TYPE count, string counter_name);

void add_perf_stats(INDEX_TYPE count, string counter_name);

void print_performance_statistics();

my_timer_t start_clock();

double stop_clock_get_elapsed(my_timer_t &start);

json json_perf_statistics();

int get_proc_length(double beta, int world_size);

int get_end_proc(int  starting_index, double beta, int world_size);

std::unordered_set<INDEX_TYPE> random_select(const std::unordered_set<INDEX_TYPE>& originalSet, int count);




template <typename VALUE_TYPE> struct Tuple {
  int64_t row;
  int64_t col;
  VALUE_TYPE value;
};

template <typename INDEX_TYPE> struct TileTuple {
  int batch_id;
  int tile_id;
  INDEX_TYPE count=0;
  INDEX_TYPE send_merge_count=0;
  int mode=0;
};

template <typename VALUE_TYPE> struct CSR {
  int64_t row;
  int64_t col;
  VALUE_TYPE value;
};

template <typename VALUE_TYPE, size_t size> struct SpTuple {
  std::array<INDEX_TYPE, row_max> rows;
  std::array<INDEX_TYPE, size> cols;
  std::array<VALUE_TYPE, size> values;
};

template <typename VALUE_TYPE, size_t size> struct DataTuple {
  INDEX_TYPE col;
  std::array<VALUE_TYPE, size> value;
};

template <typename VALUE_TYPE, size_t size> struct CacheEntry {
  std::array<VALUE_TYPE, size> value;
  int inserted_batch_id;
  int inserted_itr;
};

template <typename VALUE_TYPE> struct SparseCacheEntry {
  vector<VALUE_TYPE> values = vector<VALUE_TYPE>();
  vector<INDEX_TYPE> cols = vector<INDEX_TYPE>();
  int inserted_batch_id;
  int inserted_itr;
  bool force_delete;
};



struct CSRHandle {
  vector<double> values;
  vector<MKL_INT> col_idx;
  vector<MKL_INT> rowStart;
  vector<MKL_INT> row_idx;
  sparse_matrix_t mkl_handle;

  CSRHandle& operator=(const CSRHandle& other) {
    if (this != &other) {
      this->values = other.values;
      this->col_idx = other.col_idx;
      this->rowStart = other.rowStart;
      this->row_idx = other.row_idx;
    }
    return *this;
  }
};

template <typename T>
bool CompareTuple(const Tuple<T>& obj1, const Tuple<T>& obj2) {
  // Customize the comparison logic based on your requirements
  return obj1.col == obj2.col;
}

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

extern MPI_Datatype SPARSETUPLE;

extern MPI_Datatype TILETUPLE;

extern vector<string> perf_counter_keys;

extern map<string, int> call_count;
extern map<string, double> total_time;

template <typename VALUE_TYPE> void initialize_mpi_datatype_SPTUPLE() {
  const int nitems = 3;
  int blocklengths[3] = {1, 1, 1};
  MPI_Datatype *types = new MPI_Datatype[3];
  types[0] = MPI_UINT64_T;
  types[1] = MPI_UINT64_T;
  MPI_Aint offsets[3];
  if (std::is_same<VALUE_TYPE, int>::value) {
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

template <typename VALUE_TYPE,size_t embedding_dim>
void initialize_mpi_datatype_DENSETUPLE() {
  DataTuple<VALUE_TYPE,embedding_dim> p;
  DENSETUPLE = CreateCustomMpiType(p, p.col, p.value);
}

template <typename VALUE_TYPE,size_t embedding_dim>
void initialize_mpi_datatype_SPARSETUPLE() {
  SpTuple<VALUE_TYPE,embedding_dim> p;
  SPARSETUPLE = CreateCustomMpiType(p,p.rows, p.cols, p.values);
}

template <typename INDEX_TYPE>
void initialize_mpi_datatype_TILETUPLE() {
   TileTuple<INDEX_TYPE> p;
   TILETUPLE = CreateCustomMpiType(p,p.batch_id, p.tile_id, p.count,p.send_merge_count,p.mode);
}

template <typename VALUE_TYPE, size_t embedding_dim>
void initialize_mpi_datatypes() {
  initialize_mpi_datatype_SPTUPLE<VALUE_TYPE>();
  initialize_mpi_datatype_DENSETUPLE<VALUE_TYPE,embedding_dim>();
  initialize_mpi_datatype_SPARSETUPLE<VALUE_TYPE,embedding_dim>();
  initialize_mpi_datatype_TILETUPLE<INDEX_TYPE>();
}

template <typename VALUE_TYPE, size_t MAXBOUND>
VALUE_TYPE  scale(VALUE_TYPE v){
  if(v > MAXBOUND) return MAXBOUND;
  else if(v < -MAXBOUND) return -MAXBOUND;
  else return v;
}



}; // namespace distblas::core

#endif