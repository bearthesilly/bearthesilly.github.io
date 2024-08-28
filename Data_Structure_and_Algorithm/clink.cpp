#include <iostream>
#include <stdlib.h>
#include <time.h>
using namespace std;

struct Node{
    Node(int data = 0): data_(data), next_(nullptr){}
    int data_;
    Node* next_;
};

class Clink{
public:
    Clink(){
        // 初始化的时候, 指向头节点; new Node()在开辟的时候, 同时也会调用构造函数进行初始化
        head_ = new Node();
    }
    ~Clink(){ // 一定不是简简单单的释放头指针就完了! 理解为什么需要p head_两个指针完成操作!
        Node *p = head_;
        while (p != nullptr){
            head_ = head_->next_;
            delete p;
            p = head_;
        }
        head_ = nullptr;
    }

    void InsertTail(int val){ // 链表尾插法
        // 先找到当前链表的末尾节点, 然后生成新节点; 如何找到尾节点呢? 判断地址域是不是空指针!
        Node *p = head_;
        while (p->next_ != nullptr){
            p = p->next_;
        }
        Node *node = new Node(val);
        p->next_ = node;        
    }

    void InsertHead(int val){ // 链表头插法; 注意修改的顺序!!
        Node *node = new Node(val);
        node->next_ = head_->next_;
        head_->next_ = node;
    }

    void Remove(int val){ // 删除节点; 理解为什么p q要两个结构体指针来操作!
        Node *p = head_->next_;
        Node *q = head_;
        while (p != nullptr){
            if (p->data_ == val){
                q->next_ = p->next_;
                delete p; // 释放p对应的node
                return;
            }
            else{
                q = p;
                p = p->next_;
            }
        }
    }

    bool Find(int val){
        Node *p = head_->next_;
        while (p != nullptr){
            if (p->data_ == val){
                return true;
            }
            else{
                p = p->next_;
            }
        }
        return false;
    }

    void RemoveAll(int val){
        Node *p = head_->next_;
        Node *q = head_;
        while (p != nullptr){
            if (p->data_ == val){
                q->next_ = p->next_;
                delete p;
                p = q->next_;
            }
            else{
                q = p;
                p = p->next_;
            }
        }
    }

    void Show(){
        // 注意这里指针的设计! 这样可以防止尾节点的数据忘记被打印! 
        Node *p = head_->next_;
        while (p != nullptr){
            cout << p->data_ << " ";
            p = p->next_;
        }
        cout << endl;
    }
private:
    Node *head_;
    friend void ReverseLink(Clink &link);
};

void ReverseLink(Clink &link){
    Node *p = link.head_->next_;
    if (p == nullptr){return;}
    link.head_->next_ = nullptr;
    while (p != nullptr){
        Node *q = p->next_;
        p->next_ = link.head_->next_;
        link.head_->next_ = p;
        p = q;
    }
}

int main(){
    Clink link;
    srand(time(0));
    for (int i = 0; i < 10; i++){
        int val = rand()%100;
        link.InsertTail(val);
        cout << val << " ";
    }
    cout << endl;
    link.InsertTail(200);
    link.Show();
    link.Remove(200);
    link.Show();
    link.InsertHead(233);
    link.InsertHead(233);
    link.InsertTail(233);
    link.Show();
    link.RemoveAll(233);
    link.Show();
    ReverseLink(link);
    link.Show();
    return 0;
}