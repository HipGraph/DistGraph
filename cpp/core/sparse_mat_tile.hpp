#pragma once
#include "../net/process_3D_grid.hpp"
#include <algorithm>
#include <memory>
#include <set>


using namespace distblas::net;

namespace distblas::core {

template <typename INDEX_TYPE, typename VALUE_TYPE> class SparseTile {

private:
public:
  int id;
  INDEX_TYPE row_starting_index;
  INDEX_TYPE row_end_index;
  INDEX_TYPE col_start_index;
  INDEX_TYPE col_end_index;

  Process3DGrid *grid;

  unordered_map<INDEX_TYPE, unordered_map<int, bool>> id_to_proc_mapping;
  unordered_set<INDEX_TYPE> col_id_set;
  unordered_set<INDEX_TYPE> row_id_set;
  int mode = 0; // mode=0 local and mode=1 remote

  int64_t total_transferrable_datacount=-1;
  int64_t total_receivable_datacount=-1;

  static double tile_width_fraction;

  SparseTile(Process3DGrid *grid, int id, INDEX_TYPE row_starting_index,
             INDEX_TYPE row_end_index, INDEX_TYPE col_start_index,
             INDEX_TYPE col_end_index)
      : grid(grid), id(id), row_starting_index(row_starting_index),
        row_end_index(row_end_index), col_start_index(col_start_index),
        col_end_index(col_end_index){}

  SparseTile(Process3DGrid *grid) : grid(grid) {}

  void insert(INDEX_TYPE col_index) { col_id_set.insert(col_index); }
  void insert_row_index(INDEX_TYPE row_index) { row_id_set.insert(row_index); }

  static int get_tile_id(int batch_id, INDEX_TYPE col_index,
                         INDEX_TYPE proc_col_width, int rank) {
    int tiles_per_process_row = static_cast<int>(1/(tile_width_fraction));
     auto tile_width =
        (proc_col_width % tiles_per_process_row) == 0
            ? (proc_col_width/tiles_per_process_row)
            : ((proc_col_width/tiles_per_process_row) + 1);
    return static_cast<int>(col_index/tile_width);
  }

  static INDEX_TYPE get_tile_width(INDEX_TYPE proc_col_width) {
    int tiles_per_process_row = static_cast<int>(1 / (tile_width_fraction));
    return (proc_col_width % tiles_per_process_row) == 0
               ? (proc_col_width/tiles_per_process_row)
               : ((proc_col_width/tile_width_fraction) + 1);
  }

  static int get_tiles_per_process_row() {
    return static_cast<int>(1/(tile_width_fraction));
  }

};

template <typename INDEX_TYPE, typename VALUE_TYPE>
double SparseTile<INDEX_TYPE, VALUE_TYPE>::tile_width_fraction = 0.5;
} // namespace distblas::core
