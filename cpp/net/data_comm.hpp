#pragma once
#include "../core/dense_mat.hpp"
#include "../core/sparse_mat.hpp"
#include <iostream>
#include <mpi.h>
#include <vector>

using namespace distblas::core;

    namespace distblas::net {

  /**
   * This class represents the data transfer related operations across processes
   * based on internal data connectivity patterns.
   */
  template <typename SPT, typename  DENT>
  class DataComm {

  private:
    distblas::core::SpMat<SPT> *sp_local;
    distblas::core::SpMat<SPT> *sp_local_trans;
    distblas::core::DenseMat *dense_local;

  public:
    DataComm(distblas::core::SpMat<SPT> *sp_local, distblas::core::SpMat<SPT> *sp_local_trans,
             DenseMat *dense_local) {
      this->sp_local = sp_local;
      this->sp_local_trans = sp_local_trans;
      this->dense_local = dense_local;

    }
    void invoke(int batch_id, bool fetch_all) {

      int no_of_nodes_per_proc_list =
          (this->sp_local->proc_col_width / this->sp_local->block_col_width);
      int no_of_nodes_per_proc_list_trans =
          (this->sp_local_trans->proc_row_width /
           this->sp_local_trans->block_row_width);

      int no_of_lists =
          (this->sp_local->proc_col_width / this->sp_local->block_col_width);

      int no_of_lists_trans = (this->sp_local_trans->proc_row_width /
                               this->sp_local_trans->block_row_width);
    }
  };
}
