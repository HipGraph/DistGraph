#pragma once
#include "common.h"
#include "csr_local.hpp"
#include <memory>

using namespace std;

namespace distblas::core {

template <typename T>
struct CSRLocalNode {
  shared_ptr<CSRLocal<T>> data;
  shared_ptr<CSRLocalNode<T>> next;
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
//   CSRLocalNode<T>* temp = head;
//   while(temp != nullptr) {
//     CSRLocalNode<T>* nextTemp = temp->next;
//     delete temp;
//     temp = nextTemp;
//   }
  }

  void insert(CSRLocal<T>* dataPoint) {

    auto newNode = shared_ptr<CSRLocalNode<T>>(new CSRLocalNode<T>());
    newNode.get()->data = shared_ptr<CSRLocal<T>>(dataPoint);
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
