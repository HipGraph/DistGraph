#pragma once
#include "../core/sparse_mat_tile.hpp"
#include "../net/tile_based_data_comm.hpp"


using namespace std;
using namespace distblas::core;
using namespace distblas::net;
using namespace Eigen;

using namespace distblas::core;

namespace distblas::algo {
template <typename INDEX_TYPE, typename VALUE_TYPE, size_t embedding_dim>
class SpGEMMAlgoWithTiling {

private:
  distblas::core::SpMat<VALUE_TYPE> *sparse_local_output;
  distblas::core::SpMat<VALUE_TYPE> *sparse_local;
  distblas::core::SpMat<VALUE_TYPE> *sp_local_receiver;
  distblas::core::SpMat<VALUE_TYPE> *sp_local_sender;
  distblas::core::SpMat<VALUE_TYPE> *sp_local_native;
  Process3DGrid *grid;

  std::unordered_map<int, unique_ptr<DataComm<INDEX_TYPE, VALUE_TYPE, embedding_dim>>> data_comm_cache;

  //record temp local output
  unique_ptr<vector<unordered_map<INDEX_TYPE,VALUE_TYPE>>> output_ptr;

  //cache size controlling hyper parameter
  double alpha = 0;

  //hyper parameter controls the  computation and communication overlapping
  double beta = 1.0;

  //hyper parameter controls the switching the sync vs async commiunication
  bool sync = true;

  //hyper parameter controls the col major or row major  data access
  bool col_major = false;

  double tile_width_fraction;

public:
  SpGEMMAlgoWithTiling(distblas::core::SpMat<VALUE_TYPE> *sp_local_native,
             distblas::core::SpMat<VALUE_TYPE> *sp_local_receiver,
             distblas::core::SpMat<VALUE_TYPE> *sp_local_sender,
             distblas::core::SpMat<VALUE_TYPE> *sparse_local,
             distblas::core::SpMat<VALUE_TYPE> *sparse_local_output,
             Process3DGrid *grid, double alpha, double beta, bool col_major, bool sync_comm, double tile_width_fraction)
      : sp_local_native(sp_local_native), sp_local_receiver(sp_local_receiver),
        sp_local_sender(sp_local_sender), sparse_local(sparse_local), grid(grid),
        alpha(alpha), beta(beta),col_major(col_major),sync(sync_comm),
        sparse_local_output(sparse_local_output), tile_width_fraction(tile_width_fraction) {}



  void algo_spgemm(int iterations, int batch_size, VALUE_TYPE lr) {
    auto t = start_clock();
    size_t total_memory = 0;
    int batches = 0;
    int last_batch_size = batch_size;

    if (sp_local_receiver->proc_row_width % batch_size == 0) {
      batches =
          static_cast<int>(sp_local_receiver->proc_row_width / batch_size);
    } else {
      batches =
          static_cast<int>(sp_local_receiver->proc_row_width / batch_size) + 1;
      last_batch_size =
          sp_local_receiver->proc_row_width - batch_size * (batches - 1);
    }

    cout << " rank " << grid->rank_in_col << " total batches " << batches<< endl;

    // This communicator is being used for negative updates and in alpha > 0 to
    // fetch initial embeddings
    auto main_comm = unique_ptr<TileDataComm<INDEX_TYPE, VALUE_TYPE, embedding_dim>>(
        new TileDataComm<INDEX_TYPE, VALUE_TYPE, embedding_dim>(
            sp_local_receiver, sp_local_sender, sparse_local, grid,  alpha,batches,tile_width_fraction));

    // Buffer used for receive MPI operations data
    unique_ptr<std::vector<SpTuple<VALUE_TYPE,sp_tuple_max_dim>>> update_ptr = unique_ptr<std::vector<SpTuple<VALUE_TYPE,sp_tuple_max_dim>>>(new vector<SpTuple<VALUE_TYPE,sp_tuple_max_dim>>());

    //Buffer used for send MPI operations data
    unique_ptr<vector<SpTuple<VALUE_TYPE,sp_tuple_max_dim>>> sendbuf_ptr = unique_ptr<vector<SpTuple<VALUE_TYPE,sp_tuple_max_dim>>>(new vector<SpTuple<VALUE_TYPE,sp_tuple_max_dim>>());


    main_comm.get()->onboard_data();

    cout << " rank " << grid->rank_in_col << " on board data completed " << endl;

    int total_tiles = SparseTile<INDEX_TYPE,VALUE_TYPE>::get_tiles_per_process_row();

    CSRLocal<VALUE_TYPE> *csr_block =
        (col_major) ? (this->sp_local_receiver)->csr_local_data.get()
                    : (this->sp_local_native)->csr_local_data.get();

    int considering_batch_size = batch_size;

    for (int i = 0; i < iterations; i++) {

      for (int j = 0; j < batches; j++) {
        cout<<" rank "<<grid->rank_in_col<<" batch "<<j<<endl;
        if (j == batches - 1) {
          considering_batch_size = last_batch_size;
        }


        // One process computations without MPI operations
        if (this->grid->col_world_size == 1) {
          // local computations for 1 process
          this->calc_t_dist_grad_rowptr(csr_block,  lr, i,j,
                                        batch_size, considering_batch_size,
                                        0,  0, 0,false,main_comm.get(),this->sparse_local_output);

        } else {
          if( (this->sparse_local_output)->hash_spgemm) {
            this->execute_pull_model_computations(
                sendbuf_ptr.get(), update_ptr.get(), i, j,
                main_comm.get(), csr_block, batch_size,
                considering_batch_size, lr, 1, true, 0, true, true, this->sparse_local_output);

            (this->sparse_local_output)->initialize_hashtables();

            //compute remote computations
            this->calc_t_dist_grad_rowptr((this->sp_local_sender)->csr_local_data.get(),  lr, i,j,
                                          batch_size, considering_batch_size,
                                          2,  0, this->grid->col_world_size,true,main_comm.get(),nullptr);
          }

          this->execute_pull_model_computations(
              sendbuf_ptr.get(), update_ptr.get(), i, j,
              main_comm.get(), csr_block, batch_size,
              considering_batch_size, lr,  1,
              true, 0, true,false, this->sparse_local_output);
//          this->calc_t_dist_grad_rowptr((this->sp_local_sender)->csr_local_data.get(),  lr, i,j,
//                                        batch_size, considering_batch_size,
//                                        2,  0, this->grid->col_world_size,false,main_comm.get(),this->sparse_local_output);

        }
        total_memory += get_memory_usage();
      }
      (this->sparse_local)->purge_cache();
    }
    (this->sparse_local_output)->initialize_CSR_blocks();
    total_memory = total_memory / (iterations * batches);
    add_memory(total_memory, "Memory usage");
    stop_clock_and_add(t, "Total Time");
  }


  inline void execute_pull_model_computations(
      std::vector<SpTuple<VALUE_TYPE,sp_tuple_max_dim>> *sendbuf,
      std::vector<SpTuple<VALUE_TYPE,sp_tuple_max_dim>> *receivebuf, int iteration,
      int batch, TileDataComm<INDEX_TYPE, VALUE_TYPE, embedding_dim> *main_comm,
      CSRLocal<VALUE_TYPE> *csr_block, int batch_size, int considering_batch_size,
      double lr,  int comm_initial_start, bool local_execution,
      int first_execution_proc, bool communication, bool symbolic, DistributedMat* output) {

    int proc_length = get_proc_length(beta, this->grid->col_world_size);
    int prev_start = comm_initial_start;

     auto tiles_per_process= SparseTile<INDEX_TYPE,VALUE_TYPE>::get_tiles_per_process_row();

    for (int k = prev_start; k < this->grid->col_world_size; k += proc_length) {
      int end_process = get_end_proc(k, beta, this->grid->col_world_size);

      MPI_Request req;

      if (communication and (symbolic or !output->hash_spgemm)) {

        main_comm->transfer_sparse_data(sendbuf, receivebuf,  iteration,
                                        batch, k, end_process,0,tiles_per_process);

      }
      if (k == comm_initial_start) {
        // local computation
        this->calc_t_dist_grad_rowptr(
            csr_block,  lr, iteration,batch, batch_size,
            considering_batch_size, 0,
            first_execution_proc, prev_start,symbolic, main_comm,output);
      } else if (k > comm_initial_start) {
        int prev_end_process = get_end_proc(prev_start, beta, grid->col_world_size);

        this->calc_t_dist_grad_rowptr(csr_block,  lr, iteration,batch,
                                      batch_size, considering_batch_size, 1,
                                      prev_start, prev_end_process,symbolic, main_comm, output);
      }
      prev_start = k;
    }
    int prev_end_process = get_end_proc(prev_start, beta, grid->col_world_size);

    // updating last remote fetched data vectors
    this->calc_t_dist_grad_rowptr(csr_block,  lr, iteration,batch,
                                  batch_size, considering_batch_size,
                                  1,prev_start, prev_end_process,symbolic, main_comm, output);
    // dense_local->invalidate_cache(i, j, true);
  }



  inline void calc_t_dist_grad_rowptr(CSRLocal<VALUE_TYPE> *csr_block,
                                      VALUE_TYPE lr, int itr, int batch_id, int batch_size, int block_size,
                                      int mode, int start_process,int end_process,
                                      bool symbolic,TileDataComm<INDEX_TYPE,VALUE_TYPE, embedding_dim> *main_com, DistributedMat* output) {
    if (mode==0) {//local computation
      auto source_start_index = batch_id * batch_size;
      auto source_end_index = std::min(std::min(static_cast<INDEX_TYPE>((batch_id + 1) * batch_size),
                                       this->sp_local_receiver->proc_row_width),this->sp_local_receiver->gRows);
      auto dst_start_index =
          this->sp_local_receiver->proc_col_width * this->grid->rank_in_col;
      auto dst_end_index =
          std::min(static_cast<INDEX_TYPE>(this->sp_local_receiver->proc_col_width *
                                           (this->grid->rank_in_col + 1)),
                   this->sp_local_receiver->gCols);
      calc_embedding_row_major(source_start_index, source_end_index,
                               dst_start_index, dst_end_index, csr_block,
                               lr, batch_id, batch_size,
                               block_size,symbolic,mode,output);
    } else if (mode==1) {//remote pull
      for (int r = start_process; r < end_process; r++) {
        if (r != grid->rank_in_col) {
            int computing_rank =(grid->rank_in_col >= r)? (grid->rank_in_col - r) % grid->col_world_size: (grid->col_world_size - r + grid->rank_in_col) %grid->col_world_size;
            for (int tile = 0;tile <(*main_com->receiver_proc_tile_map)[batch_id][computing_rank].size();tile++) {
              if ((*main_com->receiver_proc_tile_map)[batch_id][computing_rank][tile].mode ==0) {
                auto source_start_index =  (*main_com->receiver_proc_tile_map)[batch_id][computing_rank][tile].row_starting_index;
                auto source_end_index =  (*main_com->receiver_proc_tile_map)[batch_id][computing_rank][tile].row_end_index;
                auto dst_start_index = (*main_com->receiver_proc_tile_map)[batch_id][computing_rank][tile].col_start_index;
                auto dst_end_index = (*main_com->receiver_proc_tile_map)[batch_id][computing_rank][tile].col_end_index;

                calc_embedding_row_major(source_start_index, source_end_index,
                                         dst_start_index, dst_end_index,
                                         csr_block, lr, batch_id, batch_size,
                                         block_size, symbolic,mode,output);
                if (itr==0 and !symbolic){
                  add_tiles(1,"Locally Computed Tiles");
                }
              }
            }
        }
      }
    }else { //execute remote computations
      for (int r = start_process; r < end_process; r++) {
        if (r != grid->rank_in_col) {
          int computing_rank =(grid->rank_in_col >= r)? (grid->rank_in_col - r) % grid->col_world_size: (grid->col_world_size - r + grid->rank_in_col) %grid->col_world_size;
          for (int tile = 0;tile <(*main_com->sender_proc_tile_map)[batch_id][computing_rank].size();tile++) {
            SparseTile<INDEX_TYPE,VALUE_TYPE>& sp_tile = (*main_com->sender_proc_tile_map)[batch_id][computing_rank][tile];
            if (sp_tile.mode ==0) {
              auto source_start_index =  sp_tile.row_starting_index;
              auto source_end_index =  sp_tile.row_end_index;
              auto dst_start_index = sp_tile.col_start_index;
              auto dst_end_index = sp_tile.col_end_index;
              sp_tile.initialize_output_DS_if(0);
//              calc_embedding_row_major(source_start_index, source_end_index,
//                                       dst_start_index, dst_end_index,
//                                       csr_block, lr, batch_id, batch_size,
//                                       block_size, symbolic,mode,sp_tile);
//              if (itr==0 and !symbolic){
                add_tiles(1,"Remote Computed Tiles");
//              }
            }
          }
        }
      }

    }
  }

  inline void calc_embedding_row_major(INDEX_TYPE source_start_index,
                                       INDEX_TYPE source_end_index, INDEX_TYPE dst_start_index,
                                       INDEX_TYPE dst_end_index, CSRLocal<VALUE_TYPE> *csr_block,VALUE_TYPE lr, int batch_id,
                                       int batch_size, int block_size, bool symbolic, int mode, DistributedMat *output) {
    if (csr_block->handler != nullptr) {
      CSRHandle *csr_handle = csr_block->handler.get();


      #pragma omp parallel for schedule(static) // enable for full batch training or // batch size larger than 1000000
      for (INDEX_TYPE i = source_start_index; i < source_end_index; i++) {

        INDEX_TYPE index = i - source_start_index;
        int max_reach=0;

        for (INDEX_TYPE j = static_cast<INDEX_TYPE>(csr_handle->rowStart[i]);j < static_cast<INDEX_TYPE>(csr_handle->rowStart[i + 1]); j++) {
          auto dst_id = csr_handle->col_idx[j];
          if (dst_id >= dst_start_index and dst_id < dst_end_index) {
            INDEX_TYPE local_dst =
                (mode==0 or mode ==1)? dst_id - (this->grid)->rank_in_col *
                             (this->sp_local_receiver)->proc_col_width:dst_id;
            int target_rank =
                (int)(dst_id/(this->sp_local_receiver)->proc_col_width);
            bool fetch_from_cache =
                (target_rank == (this->grid)->rank_in_col or mode==2) ? false : true;

            vector<INDEX_TYPE> remote_cols;
            vector<VALUE_TYPE> remote_values;

            if (fetch_from_cache) {
              unordered_map<INDEX_TYPE, SparseCacheEntry<VALUE_TYPE>>
                  &arrayMap = (*this->sparse_local->tempCachePtr)[target_rank];
              remote_cols = arrayMap[dst_id].cols;
              remote_values =arrayMap[dst_id].values;

            }

            CSRHandle *handle = ((this->sparse_local)->csr_local_data)->handler.get();

            if (!fetch_from_cache) {
              int count = handle->rowStart[local_dst+1]- handle->rowStart[local_dst];
              if (symbolic) {
                INDEX_TYPE val =(*(output->sparse_data_counter))[index] +count;
                (*(output->sparse_data_counter))[index] =std::min(val, static_cast<INDEX_TYPE>(embedding_dim));
              }else if (output->hash_spgemm) {
                INDEX_TYPE ht_size = (*(output->sparse_data_collector))[index].size();
                for (auto k = handle->rowStart[local_dst]; k < handle->rowStart[local_dst + 1]; k++) {
                  auto  d = (handle->col_idx[k]);
                  INDEX_TYPE hash = (d*hash_scale) & (ht_size-1);
                  auto value =  lr *handle->values[k];
                  int max_count=10;
                  int count=0;
                  while(count<max_count){
                    if ((*(output->sparse_data_collector))[index][hash].col==d){
                      (*(output->sparse_data_collector))[index][hash].value = (*(output->sparse_data_collector))[index][hash].value + value;
                      break;
                    }else if ((*(output->sparse_data_collector))[index][hash].col==-1){
                      (*(output->sparse_data_collector))[index][hash].col = d;
                      (*(output->sparse_data_collector))[index][hash].value =   value;
                      break;
                    }else {
                      hash = (hash+100) & (ht_size-1);
                      count++;
                    }
                  }
                }
              }else {
                for (auto k = handle->rowStart[local_dst]; k < handle->rowStart[local_dst + 1]; k++) {
                  auto d = (handle->col_idx[k]);
                  (*(output->dense_collector))[index][d] += lr*(handle->values[k]);
                }
              }
            }else{
              int count = remote_cols.size();
              if (symbolic){
                INDEX_TYPE val  = (*(output->sparse_data_counter))[index]+ count;
                (*(output->sparse_data_counter))[index] = std::min(val,static_cast<INDEX_TYPE>(embedding_dim));
              }else if (output->hash_spgemm) {
                INDEX_TYPE ht_size = (*(output->sparse_data_collector))[index].size();
                for (int m = 0; m < remote_cols.size(); m++) {
                  auto d = remote_cols[m];
                  auto value =  lr *remote_values[m];
                  INDEX_TYPE hash = (d*hash_scale) & (ht_size-1);
                  int max_count=10;
                  int count=0;
                  while (count<max_count) {
                    if ((*(output->sparse_data_collector))[index][hash].col == d) {
                      (*(output->sparse_data_collector))[index][hash].value = (*(output->sparse_data_collector))[index][hash].value + value;
                      break;
                    } else if ((*(output->sparse_data_collector))[index][hash].col ==-1) {
                      (*(output->sparse_data_collector))[index][hash].col = d;
                      (*(output->sparse_data_collector))[index][hash].value = value;
                      break;
                    } else {
                      hash =(hash + 100) &(ht_size -1);
                      count++;
                    }
                  }
                }
              }else{
                for (int m = 0; m < remote_cols.size(); m++) {
                  auto d = remote_cols[m];
                  (*(output->dense_collector))[index][d] += lr*remote_values[m];
                }

              }
            }
          }
        }
      }
    }
  }



};
} // namespace distblas::algo
