#pragma once
#include <iostream>
#include <mpi.h>
#include <vector>
#include "common.h"
#include "distributed_mat.hpp"
#include "csr_local.hpp"
#include <parallel/algorithm>
#include <algorithm>


using namespace std;

namespace distblas::core {

/**
 * This class represents the Sparse Matrix
 */
template <typename T>
class SpMat: public DistributedMat {

public:
  int gRows, gCols, gNNz;
  vector<Tuple<T>> coords;


  vector<uint64_t> block_col_starts;
  vector<CSRLocal<T>*> csr_blocks;
  vector<uint64_t> block_row_starts;

  /**
   * Constructor for Sparse Matrix representation of  Adj matrix
   * @param coords  (src, dst, value) Tuple vector loaded as input
   * @param gRows   total number of Rows in Distributed global Adj matrix
   * @param gCols   total number of Cols in Distributed global Adj matrix
   * @param gNNz     total number of NNz in Distributed global Adj matrix
   */
  SpMat(vector<Tuple<T>> &coords, int gRows, int gCols, int gNNz) {
    this->gRows = gRows;
    this->gCols = gCols;
    this->gNNz = gNNz;
    this->coords = coords;
  }

  SpMat() {

  }


  void divide_block_cols(int block_width, int target_divisions, bool mod_ind) {
    block_col_starts.clear();
    // Locate block starts within the local sparse matrix (i.e. divide a long
    // block row into subtiles)
    int current_start = 0;

    for(uint64_t i = 0; i < coords.size(); i++) {
      while(coords[i].col >= current_start) {
        block_col_starts.push_back(i);
        current_start += block_width;
      }

      // This modding step helps indexing.
      if(mod_ind) {
        coords[i].col %= block_width;
      }
    }

    assert(block_col_starts.size() <= target_divisions + 1);

    while(block_col_starts.size() < target_divisions + 1) {
      block_col_starts.push_back(coords.size());
    }

  }


  void sort_by_rows() {
    for(int i = 0; i < block_col_starts.size() - 1; i++) {
      __gnu_parallel::sort(coords.begin() + block_col_starts[i],
                           coords.begin() + block_col_starts[i+1],
                           row_major<T>);
    }
  }

  void divide_block_rows(int block_width_row, int block_width_col,
                         int target_divisions, bool mod_ind) {
    block_row_starts.clear();
    // Locate block starts within the local sparse matrix (i.e. divide a long
    // block row into subtiles)
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    for (uint64_t i = 0;i< block_col_starts.size() - 1;i++) {

      cout<<" rank "<< rank <<" block start "<< block_col_starts[i]
           << " block end "<< block_col_starts[i+1] << endl;
      int current_start = block_width_col*rank;
      for (uint64_t j = block_col_starts[i]; j < block_col_starts[i+1]; j++) {
        while (coords[j].row >= current_start) {
          block_row_starts.push_back(j);
          cout<<"rank"<<rank<<"adding block_row "<<j<< " candidate numbers "
               <<block_col_starts[i+1] << " : "<<block_col_starts[i] << endl;

          int current_step =   std::min(static_cast<int>(block_width_row),
                                      static_cast<int>(coords[block_col_starts[i+1]].row));
          current_start += block_width_row;
          cout<<"rank "<<rank<<" next checking start "<<current_start<<endl;
        }

        // This modding step helps indexing.
        if (mod_ind) {
          //        coords[i].col %= block_width;
          coords[j].row %= block_width_row;
        }
      }
    }

//    assert(block_row_starts.size() <= target_divisions + 1);

//    while(block_row_starts.size() < target_divisions + 1) {
      block_row_starts.push_back(coords.size());
//    }

  }

  void print_blocks_and_cols() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

      int current_col_block = 0;
      for(int j=0;j<block_row_starts.size()-1;j++){

            if (block_row_starts[j]> block_col_starts[current_col_block+1]){
              ++current_col_block;
            }

            string output_path =  "blocks_rank"+ to_string(rank)+"_col_"+to_string(current_col_block)+"_row_"+to_string(j)+".txt";
            char stats[500];
            strcpy(stats, output_path.c_str());
            ofstream fout(stats, std::ios_base::app);
            int num_coords = block_row_starts[j+1] -block_row_starts[j];

            for (int i = 0; i < num_coords; i++) {

              fout <<(coords.data()+block_row_starts[j]+i)->row<<" "<<(coords.data()+block_row_starts[j]+i)->col<< endl;
            }

    }
  }


  void initialize_CSR_blocks(int block_rows, int block_cols, int max_nnz, bool transpose) {

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    csr_blocks =  vector<CSRLocal<T>*>(block_col_starts.size()-1,nullptr);
    if(max_nnz == -1) {
      cout<<" rank "<< rank <<" block size "<<block_col_starts.size()<<" full code size "<<coords.size() <<endl;


      for (int i=0; i < block_col_starts.size() - 1; i++){
        int num_coords = block_col_starts[i + 1] - block_col_starts[i];
        cout<<" rank "<< rank <<" i "<<i<<"  coords "<<num_coords<<
            " block_col_starts[i] " <<block_col_starts[i] <<endl;
      }

      for(int i = 0; i < block_col_starts.size() - 1; i++) {
        int num_coords = block_col_starts[i + 1] - block_col_starts[i];

        if(num_coords > 0) {
          cout<<" rank "<< rank <<" i "<<i<<"  coords "<<num_coords<<
              " block_col_starts[i] " <<block_col_starts[i] <<endl;

          cout<<" rank "<< rank <<" i "<<i<<"  blockrows "<<block_rows<<" block_cols "<<block_cols
               <<" num_coords "<<num_coords<<endl;

          CSRLocal<T>* block
              = new CSRLocal<T>(block_rows, block_cols,
                                num_coords, coords.data() + block_col_starts[i],
                                num_coords, transpose);

          cout<<" rank "<< rank <<" i "<<i<<"  coords "<<num_coords<<" csr creation completed "<<endl;
          csr_blocks[i]=block;
        }
//        else {
//          csr_blocks.push_back(nullptr);
//        }
      }
    }
    else {
      int num_coords = block_col_starts[1] - block_col_starts[0];
      CSRLocal<T>* block = new CSRLocal<T>(block_rows, block_cols,
                                           max_nnz, coords.data(),
                                           num_coords, transpose);
      block_col_starts.push_back(block);
    }
  }


  ~SpMat() {
    for(int i = 0; i < csr_blocks.size(); i++) {
      if(csr_blocks[i] != nullptr) {
        delete csr_blocks[i];
      }
    }
  }

};

}
