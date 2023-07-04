#pragma once
#include "common.h"
#include "csr_local.hpp"
#include <memory>
#include <random>


using namespace std;

namespace distblas::core {

template <typename T>
struct CSRLocalNode {
  shared_ptr<CSRLocal<T>> data = nullptr;
  shared_ptr<CSRLocalNode<T>> next = nullptr;
  uint64_t id;
};

template <typename T>
class CSRLinkedList {

private:
  shared_ptr<CSRLocalNode<T>> head;



public:

  CSRLinkedList() {
    head = nullptr;

  }

  ~CSRLinkedList() {

  }

  void insert(MKL_INT rows, MKL_INT cols, MKL_INT max_nnz, Tuple<T> *coords,
              int num_coords, bool transpose, uint64_t id) {

    auto newNode = shared_ptr<CSRLocalNode<T>>(new CSRLocalNode<T>());
    newNode.get()->id=id;
    newNode.get()->data = make_shared<CSRLocal<T>>( rows,  cols,  max_nnz,  coords,
                                                   num_coords,  transpose);
    if (this->head == nullptr) {
      head = newNode;
    }else {
      shared_ptr<CSRLocalNode<T>> temp = head;
      while(temp.get()->next != nullptr) {
        temp = temp.get()->next;
      }
      temp.get()->next = newNode;
    }
  }

  shared_ptr<CSRLocalNode<T>>  getHeadNode() {
    return head;
  }



};

} // namespace distblas::core
