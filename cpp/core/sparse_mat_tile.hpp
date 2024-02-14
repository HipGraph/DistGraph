#pragma once
#include <algorithm>
#include <memory>
#include "../net/process_3D_grid.hpp"
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

  Process3DGrid* grid;

  unordered_map<INDEX_TYPE, unordered_map<int,bool>> &id_to_proc_mapping;
  unordered_set<INDEX_TYPE> col_id_set;
  int mode = 0; // mode=0 local and mode=1 remote

  SparseTile(Process3DGrid *grid, int id, INDEX_TYPE row_starting_index, INDEX_TYPE row_end_index,
             INDEX_TYPE col_start_index, INDEX_TYPE col_end_index)
      : grid(grid),id(id), row_starting_index(row_starting_index),
        row_end_index(row_end_index), col_start_index(col_start_index),
        col_end_index(col_end_index) {
  }

  void insert(INDEX_TYPE col_index){
    col_id_set.insert(col_index);
  }

  static int get_tile_id(int batch_id, INDEX_TYPE col_index, INDEX_TYPE proc_col_width, int rank, double tile_width_fraction){
    int tiles_per_process_row = static_cast<int>(1/(tile_width_fraction));
    int tile_width = (proc_col_width%tiles_per_process_row)==0?static_cast<int>(proc_col_width*tile_width_fraction):(static_cast<int>(proc_col_width*tile_width_fraction)+1);
    auto local_i = col_index- (proc_col_width * rank);
    int  tile_col_id = static_cast<int>(local_i/tile_width);
    return batch_id * tiles_per_process_row + tile_col_id;
  }
};

} // namespace distblas::core
