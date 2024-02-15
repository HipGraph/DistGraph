#pragma once
#include "../core/sparse_mat_tile.hpp"
#include "data_comm.hpp"
#include <math.h>
#include <memory>

using namespace distblas::core;
using namespace std;

namespace distblas::net {

template <typename INDEX_TYPE, typename VALUE_TYPE, size_t embedding_dim>
class TileDataComm : public DataComm<INDEX_TYPE, VALUE_TYPE, embedding_dim> {

private:
  shared_ptr<vector<vector<vector<SparseTile<INDEX_TYPE, VALUE_TYPE>>>>>
      receiver_proc_tile_map;
  shared_ptr<vector<vector<vector<SparseTile<INDEX_TYPE, VALUE_TYPE>>>>>
      sender_proc_tile_map;

   int total_batches;

   double tile_width_fraction;
   int tiles_per_process_row;
public:
  TileDataComm(distblas::core::SpMat<VALUE_TYPE> *sp_local_receiver,
               distblas::core::SpMat<VALUE_TYPE> *sp_local_sender,
               distblas::core::SpMat<VALUE_TYPE> *sparse_local,
               Process3DGrid *grid, double alpha, int total_batches,
               double tile_width_fraction)
      : DataComm<INDEX_TYPE, VALUE_TYPE, embedding_dim>(
            sp_local_receiver, sp_local_sender, sparse_local, grid, -1, alpha) {
     tiles_per_process_row = static_cast<int>(1 / (tile_width_fraction));
     this->total_batches = total_batches;
     this->tile_width_fraction = tile_width_fraction;
     SparseTile<INDEX_TYPE,VALUE_TYPE>::tile_width_fraction=tile_width_fraction;
    receiver_proc_tile_map =
        make_shared<vector<vector<vector<SparseTile<INDEX_TYPE, VALUE_TYPE>>>>>(
            total_batches, vector<vector<SparseTile<INDEX_TYPE, VALUE_TYPE>>>(
                               grid->col_world_size,
                               vector<SparseTile<INDEX_TYPE, VALUE_TYPE>>(
                                   tiles_per_process_row, SparseTile<INDEX_TYPE,VALUE_TYPE>(grid))));
    sender_proc_tile_map =
        make_shared<vector<vector<vector<SparseTile<INDEX_TYPE, VALUE_TYPE>>>>>(
            total_batches, vector<vector<SparseTile<INDEX_TYPE, VALUE_TYPE>>>(
                               grid->col_world_size,
                               vector<SparseTile<INDEX_TYPE, VALUE_TYPE>>(
                                   tiles_per_process_row, SparseTile<INDEX_TYPE,VALUE_TYPE>(grid))));

    if (alpha == 0) {
      #pragma omp parallel for
      for (int i = 0; i < total_batches; i++) {
        INDEX_TYPE row_starting_index_receiver = i * sp_local_receiver->batch_size;
        auto row_end_index_receiver = std::min(
            std::min((row_starting_index_receiver + sp_local_receiver->batch_size),sp_local_receiver->proc_row_width),sp_local_receiver->gRows);

        for (int j = 0; j < grid->col_world_size; j++) {
          INDEX_TYPE row_starting_index_sender = i * sp_local_receiver->batch_size +
                                      sp_local_receiver->proc_row_width * j;
          INDEX_TYPE row_end_index_sender = std::min(
              std::min((row_starting_index_sender + sp_local_receiver->batch_size),
                       static_cast<INDEX_TYPE>((j + 1) * sp_local_receiver->proc_row_width)),sp_local_receiver->gRows);
          for (int k = 0; k < tiles_per_process_row; k++) {
            auto tile_width =
                SparseTile<INDEX_TYPE, VALUE_TYPE>::get_tile_width(
                    sp_local_receiver->proc_col_width);
            INDEX_TYPE col_starting_index_receiver = k * tile_width + sp_local_receiver->proc_col_width * j;

            INDEX_TYPE col_end_index_receiver = std::min(std::min((col_starting_index_receiver + tile_width),
             static_cast<INDEX_TYPE>((j + 1) * sp_local_receiver->proc_col_width)), sp_local_receiver->gCols);


            INDEX_TYPE col_starting_index_sender = k * tile_width;
            auto col_end_index_sender = std::min(std::min((col_starting_index_sender + tile_width),
                                                          sp_local_receiver->proc_col_width),sp_local_receiver->gCols);

            (*receiver_proc_tile_map)[i][j][k].id = k;
            (*receiver_proc_tile_map)[i][j][k].row_starting_index =row_starting_index_receiver;
            (*receiver_proc_tile_map)[i][j][k].row_end_index = row_end_index_receiver;
            (*receiver_proc_tile_map)[i][j][k].col_start_index =col_starting_index_receiver;
            (*receiver_proc_tile_map)[i][j][k].col_end_index = col_end_index_receiver;

            (*sender_proc_tile_map)[i][j][k].id = k;
            (*sender_proc_tile_map)[i][j][k].row_starting_index =row_starting_index_sender;
            (*sender_proc_tile_map)[i][j][k].row_end_index = row_end_index_sender;
            (*sender_proc_tile_map)[i][j][k].col_start_index =col_starting_index_sender;
            (*sender_proc_tile_map)[i][j][k].col_end_index = col_end_index_sender;

          }
        }
      }
    }
  }

  ~TileDataComm() {}



  void onboard_data()  {
    if (this->alpha == 0) {
      for(int i = 0; i < total_batches; i++){
        this->sp_local_receiver->find_col_ids_with_tiling(
            i, 0, this->grid->col_world_size,
            receiver_proc_tile_map.get(), this->receive_indices_to_proc_map, 0);
        // calculating sending data cols
        this->sp_local_sender->find_col_ids_with_tiling(
            i, 0, this->grid->col_world_size,
            sender_proc_tile_map.get(), this->send_indices_to_proc_map, 0);
      }
      // This represents the case for pulling
      this->sparse_local->get_transferrable_datacount(sender_proc_tile_map.get(),total_batches,true);
      this->sparse_local->get_transferrable_datacount(receiver_proc_tile_map.get(),total_batches,false);

      int tiles_per_process = SparseTile<INDEX_TYPE,VALUE_TYPE>::get_tiles_per_process_row();
      auto itr = total_batches * this->grid->col_world_size * tiles_per_process;

      auto per_process_messages = total_batches*tiles_per_process;

      unique_ptr<vector<TileTuple<INDEX_TYPE>>> send_tile_meta = make_unique<vector<TileTuple<INDEX_TYPE>>>(itr);
      unique_ptr<vector<TileTuple<INDEX_TYPE>>> receive_tile_meta = make_unique<vector<TileTuple<INDEX_TYPE>>>(itr);

      #pragma omp parallel for
      for(auto in=0;in<itr;in++){
        auto i = in / (this->grid->col_world_size * tiles_per_process);
        auto j = (in / tiles_per_process) % this->grid->col_world_size;
        auto k = in % tiles_per_process;

        auto offset = j*per_process_messages;
        auto index = offset+ i*tiles_per_process +k;
        TileTuple<INDEX_TYPE> t;
        t.batch_id = i;
        t.tile_id = k;
        t.count = (*sender_proc_tile_map)[i][j][k].total_transferrable_datacount;
        (*send_tile_meta)[index]=t;
      }

      MPI_Alltoall((*send_tile_meta).data(),per_process_messages , TILETUPLE,
                   (*receive_tile_meta).data(), per_process_messages, TILETUPLE, this->grid->col_world);

      #pragma omp parallel for
      for(auto in=0;in<itr;in++){
        auto i = in / (this->grid->col_world_size * tiles_per_process);
        auto j = (in / tiles_per_process) % this->grid->col_world_size;
        auto k = in % tiles_per_process;
        auto offset = j*per_process_messages;
        auto index = offset+ i*tiles_per_process +k;
        TileTuple<INDEX_TYPE> t = (*receive_tile_meta)[index];
        if (t.batch_id==i and t.tile_id==k){
          (*receiver_proc_tile_map)[i][j][k].total_receivable_datacount=t.count;
        }
      }

      for(auto in=0;in<itr;in++){
        auto i = in / (this->grid->col_world_size * tiles_per_process);
        auto j = (in / tiles_per_process) % this->grid->col_world_size;
        auto k = in % tiles_per_process;
        auto offset = j*per_process_messages;
        cout<<" rank "<<this->grid->rank_in_col<<" batch_id "<<i<<" process "<<j<<" tile id  "
             <<k<<" receivables "<<(*receiver_proc_tile_map)[i][j][k].total_receivable_datacount<<" "<<" transferrable "
             <<" receivables "<<(*receiver_proc_tile_map)[i][j][k].total_transferrable_datacount<<" col "
             <<(*receiver_proc_tile_map)[i][j][k].col_id_set.size()<<" row "<<(*receiver_proc_tile_map)[i][j][k].row_id_set.size()<<endl;
      }
    }
  }
};

} // namespace distblas::net
