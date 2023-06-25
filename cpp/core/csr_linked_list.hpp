#pragma once
#include "common.h"
#include "csr_local.hpp"
#include <memory>

using namespace std;

namespace distblas::core {

template <typename T>
struct CSRLocalNode {
  unique_ptr<CSRLocal<T>> data;
  unique_ptr<CSRLocalNode<T>> next;
};

template <typename T>
class CSRLinkedList {

private:
  unique_ptr<CSRLocalNode<T>> head;

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

    auto newNode = unique_ptr<CSRLocalNode<T>>(new CSRLocalNode<T>());
    newNode.get()->data = unique_ptr<CSRLocal<T>>(dataPoint);
    if (this->head == nullptr) {
      head = newNode;
    }else {
      unique_ptr<CSRLocalNode<T>> temp = head;
      while(temp.get()->next != nullptr) {
        temp = temp.get()->next;
      }
      temp.get()->next = newNode;
    }
  }

  unique_ptr<CSRLocalNode<T>>  getHeadNode() {
    return head;
  }

};

} // namespace distblas::core
