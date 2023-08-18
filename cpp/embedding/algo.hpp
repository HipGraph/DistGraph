#pragma once
#include "../core/common.h"
#include "../core/csr_local.hpp"
#include "../core/dense_mat.hpp"
#include "../core/sparse_mat.hpp"
#include "../net/data_comm.hpp"
#include "../net/process_3D_grid.hpp"
#include <Eigen/Dense>
#include <chrono>
#include <math.h>
#include <memory>
#include <mpi.h>
#include <unordered_map>

using namespace std;
using namespace distblas::core;
using namespace distblas::net;
using namespace Eigen;

namespace distblas::embedding {
template <typename SPT, typename DENT, size_t embedding_dim>

class EmbeddingAlgo {

private:
  DenseMat<DENT, embedding_dim> *dense_local;
  distblas::core::SpMat<SPT> *sp_local;
  distblas::core::SpMat<SPT> *sp_local_metadata;
  distblas::core::SpMat<SPT> *sp_local_trans;
  Process3DGrid *grid;
  DataComm<SPT, DENT, embedding_dim> *data_comm;
  DENT MAX_BOUND, MIN_BOUND;
  std::unordered_map<int, unique_ptr<DataComm<SPT, DENT, embedding_dim>>>
      data_comm_cache;

public:
  EmbeddingAlgo(distblas::core::SpMat<SPT> *sp_local,
                distblas::core::SpMat<SPT> *sp_local_metadata,
                distblas::core::SpMat<SPT> *sp_local_trans,
                DenseMat<DENT, embedding_dim> *dense_local,
                DataComm<SPT, DENT, embedding_dim> *data_comm,
                Process3DGrid *grid, DENT MAX_BOUND, DENT MIN_BOUND) {
    this->data_comm = data_comm;
    this->grid = grid;
    this->dense_local = dense_local;
    this->sp_local = sp_local;
    this->sp_local_metadata = sp_local_metadata;
    this->sp_local_trans = sp_local_trans;
    this->MAX_BOUND = MAX_BOUND;
    this->MIN_BOUND = MIN_BOUND;
  }

  DENT scale(DENT v) {
    if (v > MAX_BOUND)
      return MAX_BOUND;
    else if (v < -MAX_BOUND)
      return -MAX_BOUND;
    else
      return v;
  }

  void algo_force2_vec_ns(int iterations, int batch_size, int ns, DENT lr) {

    int batches = 0;
    int last_batch_size = batch_size;
    if (sp_local->proc_row_width % batch_size == 0) {
      batches = static_cast<int>(sp_local->proc_row_width / batch_size);
    } else {
      batches = static_cast<int>(sp_local->proc_row_width / batch_size) +
                1; // TODO:Error prone
      last_batch_size = sp_local->proc_row_width - batch_size * (batches - 1);
    }

    cout << " rank " << this->grid->global_rank << " total batches " << batches
         << endl;

    for (int i = 0; i < batches; i++) {
      auto communicator = unique_ptr<DataComm<SPT, DENT, embedding_dim>>(
          new DataComm<SPT, DENT, embedding_dim>(
              sp_local_metadata, sp_local_trans, dense_local, grid));
      data_comm_cache.insert(std::make_pair(i, std::move(communicator)));
    }

    DENT *prevCoordinates = static_cast<DENT *>(
        ::operator new(sizeof(DENT[batch_size * embedding_dim])));

    for (int i = 0; i < 1; i++) {

      for (int j = 0; j < batches; j++) {
        cout<<" rank "<<grid->global_rank<<" processing iteration "<<i<<"batch " << j <<endl;
        int seed = j + i;

        for (int i = 0; i < batch_size; i += 1) {
          int IDIM = i * embedding_dim;
          for (int d = 0; d < embedding_dim; d++) {
            prevCoordinates[IDIM + d] = 0;
          }
        }

        // negative samples generation
        vector<uint64_t> random_number_vec =
            generate_random_numbers(0, (this->sp_local)->gRows, seed, ns);

        CSRLinkedList<SPT> *batch_list = (this->sp_local)->get_batch_list(0);

        auto head = batch_list->getHeadNode();
        CSRLocal<SPT> *csr_block_local = (head.get())->data.get();
        CSRLocal<SPT> *csr_block_remote = nullptr;

        if (this->grid->world_size > 1) {
          auto remote = (head.get())->next;
          csr_block_remote = (remote.get())->data.get();
        }

        cout<<" rank "<<grid->global_rank<<" processing iteration "<<i<<"batch " << j<<" csr processing completed" <<endl;

        int working_rank = 0;
        bool fetch_remote =
            (working_rank == ((this->grid)->global_rank)) ? false : true;

        int considering_batch_size = batch_size;

        if (j == batches - 1) {
          considering_batch_size = last_batch_size;
        }

        MPI_Request request_batch_update;
        unique_ptr<std::vector<DataTuple<DENT, embedding_dim>>> update_ptr =
            unique_ptr<std::vector<DataTuple<DENT, embedding_dim>>>(
                new vector<DataTuple<DENT, embedding_dim>>());
        if (this->grid->world_size > 1) {

          if (i == 0) {
            data_comm_cache[j].get()->async_transfer(j, false, update_ptr.get(),
                                                     request_batch_update);
          } else if (i > 0) {
            data_comm_cache[j].get()->async_re_transfer(update_ptr.get(),
                                                        request_batch_update);
          }
        }
        cout<<" rank "<<grid->global_rank<<" processing iteration "<<i<<"batch " << j<<" async_transfer completed" <<endl;
//        this->calc_t_dist_grad_rowptr(csr_block_local, prevCoordinates, lr, j,
//                                      batch_size, considering_batch_size);
        cout<<" rank "<<grid->global_rank<<" processing iteration "<<i<<"batch " << j<<" calc_t_dist_grad_rowptr completed" <<endl;
        if (this->grid->world_size > 1) {
//          data_comm_cache[j].get()->populate_cache(update_ptr.get(),
//                                                   request_batch_update);

          cout<<" rank "<<grid->global_rank<<" processing iteration "<<i<<"batch " << j<<" populate_cache completed" <<endl;
//          this->calc_t_dist_grad_rowptr(csr_block_remote, prevCoordinates, lr,
//                                        j, batch_size, considering_batch_size);

//          cout<<" rank "<<grid->global_rank<<" processing iteration "<<i<<"batch " << j<<" calc_t_dist_grad_rowptr completed" <<endl;
//          MPI_Request request;
//          unique_ptr<std::vector<DataTuple<DENT, embedding_dim>>>
//              results_negative_ptr =
//                  unique_ptr<std::vector<DataTuple<DENT, embedding_dim>>>(
//                      new vector<DataTuple<DENT, embedding_dim>>());
//          this->data_comm->async_transfer(random_number_vec, false,
//                                          results_negative_ptr.get(), request);
//          this->data_comm->populate_cache(results_negative_ptr.get(), request);

        }

//        this->calc_t_dist_replus_rowptr(prevCoordinates, random_number_vec, lr,
//                                        j, batch_size, considering_batch_size);

//        cout<<" rank "<<grid->global_rank<<" processing iteration "<<i<<"batch " << j<<"  calc_t_dist_replus_rowptr completed" <<endl;

//        this->update_data_matrix_rowptr(prevCoordinates, j, batch_size);

//        cout<<" rank "<<grid->global_rank<<" processing iteration "<<i<<"batch " << j<<"  update_data_matrix_rowptr completed" <<endl;
      }
    }
  }

  inline void calc_t_dist_grad_rowptr(CSRLocal<SPT> *csr_block,
                                      DENT *prevCoordinates, DENT lr,
                                      int batch_id, int batch_size,
                                      int block_size) {

    auto row_base_index = batch_id * batch_size;

    if (csr_block->handler != nullptr) {
      CSRHandle *csr_handle = csr_block->handler.get();

      //#pragma omp parallel for schedule(static)
      for (uint64_t i = row_base_index; i < row_base_index + block_size; i++) {
        uint64_t row_id = i;
        int ind = i - row_base_index;

        DENT forceDiff[embedding_dim];
#pragma forceinline
#pragma omp simd
        for (uint64_t j = static_cast<uint64_t>(csr_handle->rowStart[i]);
             j < static_cast<uint64_t>(csr_handle->rowStart[i + 1]); j++) {

          uint64_t global_col_id = static_cast<uint64_t>(csr_handle->values[j]);

          uint64_t local_col =
              global_col_id -
              (this->grid)->global_rank * (this->sp_local)->proc_row_width;
          int target_rank =
              (int)(global_col_id / (this->sp_local)->proc_row_width);
          bool fetch_from_cache =
              target_rank == (this->grid)->global_rank ? false : true;

          if (fetch_from_cache) {

            std::array<DENT, embedding_dim> colvec =
                (this->dense_local)
                    ->fetch_data_vector_from_cache(target_rank, global_col_id);
            DENT attrc = 0;
            for (int d = 0; d < embedding_dim; d++) {
              forceDiff[d] = (this->dense_local)
                                 ->nCoordinates[row_id * embedding_dim + d] -
                             colvec[d];
              attrc += forceDiff[d] * forceDiff[d];
            }
            DENT d1 = -2.0 / (1.0 + attrc);
            for (int d = 0; d < embedding_dim; d++) {
              forceDiff[d] = scale(forceDiff[d] * d1);
              prevCoordinates[ind * embedding_dim + d] += (lr)*forceDiff[d];
            }

          } else {

            DENT attrc = 0;
            for (int d = 0; d < embedding_dim; d++) {
              forceDiff[d] = (this->dense_local)
                                 ->nCoordinates[row_id * embedding_dim + d] -
                             (this->dense_local)
                                 ->nCoordinates[local_col * embedding_dim + d];

              attrc += forceDiff[d] * forceDiff[d];
            }
            DENT d1 = -2.0 / (1.0 + attrc);
            for (int d = 0; d < embedding_dim; d++) {
              forceDiff[d] = scale(forceDiff[d] * d1);
              prevCoordinates[ind * embedding_dim + d] += (lr)*forceDiff[d];
            }
          }
        }
      }
    }
  }

  inline void calc_t_dist_replus_rowptr(DENT *prevCoordinates,
                                        vector<uint64_t> &col_ids, DENT lr,
                                        int batch_id, int batch_size,
                                        int block_size) {

    int row_base_index = batch_id * batch_size;

    //#pragma omp parallel for schedule(static)
    for (int i = 0; i < block_size; i++) {
      uint64_t row_id = static_cast<uint64_t>(i + row_base_index);
      DENT forceDiff[embedding_dim];
#pragma forceinline
#pragma omp simd
      for (int j = 0; j < col_ids.size(); j++) {
        uint64_t global_col_id = col_ids[j];
        uint64_t local_col_id =
            global_col_id -
            static_cast<uint64_t>(
                ((this->grid)->global_rank * (this->sp_local)->proc_row_width));
        bool fetch_from_cache = false;

        int owner_rank =
            static_cast<int>(global_col_id / (this->sp_local)->proc_row_width);
        if (owner_rank != (this->grid)->global_rank) {
          fetch_from_cache = true;
        }
        if (fetch_from_cache) {
          DENT repuls = 0;
          std::array<DENT, embedding_dim> colvec =
              (this->dense_local)
                  ->fetch_data_vector_from_cache(owner_rank, global_col_id);
          for (int d = 0; d < embedding_dim; d++) {
            forceDiff[d] =
                (this->dense_local)->nCoordinates[row_id * embedding_dim + d] -
                colvec[d];
            repuls += forceDiff[d] * forceDiff[d];
          }
          DENT d1 = 2.0 / ((repuls + 0.000001) * (1.0 + repuls));
          for (int d = 0; d < embedding_dim; d++) {
            forceDiff[d] = scale(forceDiff[d] * d1);
            prevCoordinates[i * embedding_dim + d] += (lr)*forceDiff[d];
          }
        } else {
          DENT repuls = 0;
          for (int d = 0; d < embedding_dim; d++) {
            forceDiff[d] =
                (this->dense_local)->nCoordinates[row_id * embedding_dim + d] -
                (this->dense_local)
                    ->nCoordinates[local_col_id * embedding_dim + d];
            repuls += forceDiff[d] * forceDiff[d];
          }
          DENT d1 = 2.0 / ((repuls + 0.000001) * (1.0 + repuls));
          for (int d = 0; d < embedding_dim; d++) {
            forceDiff[d] = scale(forceDiff[d] * d1);
            prevCoordinates[i * embedding_dim + d] += (lr)*forceDiff[d];
          }
        }
      }
    }
  }

  inline void update_data_matrix_rowptr(DENT *prevCoordinates, int batch_id,
                                        int batch_size) {

    int row_base_index = batch_id * batch_size;
    int end_row = std::min((batch_id + 1) * batch_size,
                           ((this->sp_local)->proc_row_width));

    for (int i = 0; i < (end_row - row_base_index); i++) {
#pragma omp simd
      for (int d = 0; d < embedding_dim; d++) {
        (this->dense_local)
            ->nCoordinates[(row_base_index + i) * embedding_dim + d] +=
            prevCoordinates[i * embedding_dim + d];
      }
    }
  }

  // Eigen based implementation

  //  inline void calc_t_dist_grad_attrac(Matrix<DENT, Dynamic, embedding_dim>
  //  &values, DENT lr,
  //                          CSRLocal<SPT> *csr_block, int batch_id,
  //                          int col_batch_id, int batch_size) {
  //
  //    int row_base_index = batch_id * batch_size;
  //    //    cout<<"executing  calc_t_dist_grad_attrac"<<endl;
  //    DENT *prevCoordinates;
  //    prevCoordinates = static_cast<DENT *>(
  //        ::operator new(sizeof(DENT[values.rows() * embedding_dim])));
  //    if (csr_block->handler != nullptr) {
  //      //      cout<<"inside   csr_block handler"<<endl;
  //      CSRHandle *csr_handle = csr_block->handler.get();
  //
  //#pragma omp parallel for schedule(static)
  //      for (int i = 0; i < values.rows(); i++) {
  //        uint64_t row_id = static_cast<uint64_t>(i + row_base_index);
  //        Eigen::Matrix<DENT, 1, embedding_dim> row_vec =
  //            (this->dense_local)->fetch_local_eigen_vector(row_id);
  //        DENT forceDiff[embedding_dim];
  //#pragma forceinline
  //#pragma omp simd
  //        for (uint64_t j = static_cast<uint64_t>(csr_handle->rowStart[i]);
  //             j < static_cast<uint64_t>(csr_handle->rowStart[i + 1]); j++) {
  //          uint64_t global_col_id =
  //          static_cast<uint64_t>(csr_handle->values[j]); uint64_t local_col =
  //              global_col_id -
  //              (this->grid)->global_rank * (this->sp_local)->proc_row_width;
  //          Eigen::Matrix<DENT, 1, embedding_dim> col_vec;
  //
  //          int target_rank =
  //              (int)(global_col_id / (this->sp_local)->proc_row_width);
  //          bool fetch_from_cache =
  //              target_rank == (this->grid)->global_rank ? false : true;
  //          //          cout<<"("<<i<<","<<global_col_id<<")"<<endl;
  //          if (fetch_from_cache) {
  //            Eigen::Matrix<DENT, embedding_dim, 1> col_vec_trans =
  //                (this->dense_local)
  //                    ->fetch_data_vector_from_cache(target_rank,
  //                    global_col_id);
  //            col_vec = col_vec_trans.transpose();
  //          } else {
  //            col_vec =
  //            (this->dense_local)->fetch_local_eigen_vector(local_col);
  //          }
  //
  //          Eigen::Matrix<DENT, 1, embedding_dim> t =
  //              (row_vec.array() - col_vec.array());
  //          DENT d1 = -2.0 / (1.0 + t.array().square().sum());
  //          Eigen::Matrix<DENT, 1, embedding_dim> clamped_vector =
  //              (t.array() * d1)
  //                  .cwiseMax(this->MIN_BOUND)
  //                  .cwiseMin(this->MAX_BOUND) *
  //              lr;
  //          values.row(i) = values.row(i).array() + clamped_vector.array();
  //        }
  //      }
  //    }
  //  }
  //
  //  inline void calc_t_dist_grad_repulsive(Matrix<DENT, Dynamic,
  //  embedding_dim> &values,
  //                             vector<uint64_t> &col_ids, DENT lr, int
  //                             batch_id, int batch_size) {
  //
  //    int row_base_index = batch_id * batch_size;
  //
  //#pragma omp parallel for schedule(static)
  //    for (int i = 0; i < values.rows(); i++) {
  //      uint64_t row_id = static_cast<uint64_t>(i + row_base_index);
  //      for (int j = 0; j < col_ids.size(); j++) {
  //        uint64_t global_col_id = col_ids[j];
  //        uint64_t local_col_id =
  //            global_col_id -
  //            static_cast<uint64_t>(
  //                ((this->grid)->global_rank *
  //                (this->sp_local)->proc_row_width));
  //        bool fetch_from_cache = false;
  //
  //        int owner_rank =
  //            static_cast<int>(global_col_id /
  //            (this->sp_local)->proc_row_width);
  //        if (owner_rank != (this->grid)->global_rank) {
  //          fetch_from_cache = true;
  //        }
  //        Eigen::Matrix<DENT, 1, embedding_dim> col_vec;
  //
  //        if (fetch_from_cache) {
  //          Eigen::Matrix<DENT, embedding_dim, 1> col_vec_trans =
  //              (this->dense_local)
  //                  ->fetch_data_vector_from_cache(owner_rank, global_col_id);
  //          col_vec = col_vec_trans.transpose();
  //        } else {
  //
  //          col_vec =
  //          (this->dense_local)->fetch_local_eigen_vector(local_col_id);
  //        }
  //        Eigen::Matrix<DENT, 1, embedding_dim> row_vec =
  //            (this->dense_local)->fetch_local_eigen_vector(row_id);
  //
  //        Eigen::Matrix<DENT, 1, embedding_dim> t =
  //            row_vec.array() - col_vec.array();
  //
  //        DENT d1 = 2.0 / ((t.array().square().sum() + 0.000001) *
  //                         (1.0 + t.array().square().sum()));
  //        Eigen::Matrix<DENT, 1, embedding_dim> clamped_vector =
  //            (t.array() * d1)
  //                .cwiseMax(this->MIN_BOUND)
  //                .cwiseMin(this->MAX_BOUND) *
  //            lr;
  //        values.row(i) = values.row(i).array() + clamped_vector.array();
  //      }
  //    }
  //  }
  //
  //  inline void update_data_matrix(Matrix<DENT, Dynamic, embedding_dim>
  //  &values,
  //                                 int batch_id, int batch_size) {
  //
  //    int row_base_index = batch_id * batch_size;
  //    int end_row =
  //        std::min((batch_id + 1) * batch_size,
  //        (this->sp_local)->proc_row_width);
  //    ((this->dense_local)->matrixPtr.get())
  //        ->block(row_base_index, 0, end_row - row_base_index, embedding_dim)
  //        += values;
  //  }
};
} // namespace distblas::embedding
