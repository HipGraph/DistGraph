#include "../core/sparse_mat.hpp"
#include "../net/process_3D_grid.hpp"
#include "partitioner.hpp"

using namespace  distblas::partition;

GlobalAdjacency1DPartitioner::GlobalAdjacency1DPartitioner(int gRows, int gCols, Process3DGrid *process_3D_grid){
  this->proc_row_width = divide_and_round_up(gRows, process_3D_grid->world_size);
  this->proc_col_width = gCols;
  this->process_3D_grid =  process_3D_grid;

}

GlobalAdjacency1DPartitioner::~GlobalAdjacency1DPartitioner(){

}

int GlobalAdjacency1DPartitioner::block_owner(int row_block, int col_block) {
  int rowRank = row_block;
  return process_3D_grid->get_global_rank(rowRank, 0, 0);

}

int GlobalAdjacency1DPartitioner::get_owner_Process(int row, int column, bool transpose) {
  if(!transpose) {
    return block_owner(row / proc_row_width,  column/proc_col_width);
  }
  else {
    return block_owner(column / proc_row_width,  row/proc_col_width);
  }
}
