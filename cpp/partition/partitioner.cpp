#pragma once
#include "../core/sparse_mat.hpp"
#include "../net/process_3D_grid.hpp"
#include "partitioner.hpp"

using namespace  distblas::partition;

GlobalAdjacency1DPartitioner::GlobalAdjacency1DPartitioner(int gRows, int gCols, Process3DGrid *process_3D_grid){
  this->rows_per_block = divide_and_round_up(gRows, process_3D_grid->world_size);
  this->cols_per_block = gCols;
  this->process_3D_grid =  process_3D_grid;

}

GlobalAdjacency1DPartitioner::~GlobalAdjacency1DPartitioner(){

}

int GlobalAdjacency1DPartitioner::block_owner(int row_block, int col_block) {
  int rowRank = row_block;
  return process_3D_grid->get_global_rank(rowRank, 1, 0);

}

int GlobalAdjacency1DPartitioner::get_owner_Process(int row, int column, bool transpose) {
  if(! transpose) {
    return blockOwner(row / rows_per_block,  column/cols_per_block);
  }
  else {
    return blockOwner(column / rows_per_block,  row/cols_per_block);
  }
}
