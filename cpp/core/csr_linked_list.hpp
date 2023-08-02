#pragma once
#include "common.h"
#include "csr_local.hpp"
#include <memory>
#include <random>

using namespace std;

namespace distblas::core {

template <typename T> struct CSRLocalNode {
  shared_ptr<CSRLocal<T>> data = nullptr;
  shared_ptr<CSRLocalNode<T>> next = nullptr;
  uint64_t id;
};

template <typename T> class CSRLinkedList {

private:
  shared_ptr<CSRLocalNode<T>> head;
//  shared_ptr<CSRLocalNode<T>> tail;
  int total_nodes;

public:
  vector<CSRLocal<T> *> direct_ref;
  CSRLinkedList(int num_of_nodes) {
    head = nullptr;
    direct_ref = vector<CSRLocal<T>*>(num_of_nodes);
    total_nodes = num_of_nodes;
  }

  ~CSRLinkedList() {}

  void insert(MKL_INT rows, MKL_INT cols, MKL_INT max_nnz, Tuple<T> *coords,
              int num_coords, bool transpose, uint64_t id) {

    auto newNode = shared_ptr<CSRLocalNode<T>>(new CSRLocalNode<T>());
    newNode.get()->id = id;
//    shared_ptr<CSRLocal<T>> data_ptr = make_shared<CSRLocal<T>>(rows, cols, max_nnz, coords,
//                                                                num_coords, transpose);
//    newNode.get()->data = data_ptr;
//    int index = static_cast<int>(id);
//    //    cout<<" index "<<index<<endl;
//    direct_ref[index] = data_ptr.get();
//
//    if (this->head == nullptr) {
//      head = newNode;
//
//    }
//
//    else {
//      shared_ptr<CSRLocalNode<T>> temp = head;
//      while (temp.get()->next != nullptr) {
//        temp = temp.get()->next;
//      }
//      temp.get()->next = newNode;
//
//    }
  }

  shared_ptr<CSRLocalNode<T>> getHeadNode() { return head; }
};

} // namespace distblas::core
