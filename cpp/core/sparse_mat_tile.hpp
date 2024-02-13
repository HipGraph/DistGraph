#pragma once
#include <algorithm>
#include <memory>
#include "../net/process_3D_grid.hpp"

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

  unordered_map<INDEX_TYPE, unordered_map<int,bool>> &id_to_proc_mapping
  vector<unordered_set<INDEX_TYPE>> proc_to_id_mapping;
  int mode = 0; // mode=0 local and mode=1 remote

  SparseTile(Process3DGrid *grid, int id, INDEX_TYPE row_starting_index, INDEX_TYPE row_end_index,
             INDEX_TYPE col_start_index, INDEX_TYPE col_end_index)
      : grid(grid),id(id), row_starting_index(row_starting_index),
        row_end_index(row_end_index), col_start_index(col_start_index),
        col_end_index(col_end_index) {
    col_indices = make_unique<std::vector<INDEX_TYPE>>();
    id_to_proc_mapping =  ;
    proc_to_id_mapping =  vector<unordered_set<INDEX_TYPE>>(grid->world_size);
  }
};

} // namespace distblas::core
