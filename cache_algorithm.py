from __future__ import print_function
import random

class MyNode(object):
    """docstring for MyNode"""
    def __init__(self):
        self.empty = True

    def print(self):
        if self.empty:            
            print(self.empty)
        else:
            print(self.empty, self.key)

class PLRU(object):
    """docstring for LRU"""
    def __init__(self, size, p):
        #   super().__init__()
        # print(size, p)
        self.hit = 0
        self.update = 0
        self.size = size
        self.ssd = {}
        self.head = MyNode()
        self.head.next = self.head
        self.head.prev = self.head
        self.p = p
        self.listSize = 1
        self.latesthit = 0
        # Adjust the size
        self.change_size(size)

    def copy(self, ssd, size=None, p=None):
        self.hit = ssd.hit
        self.latesthit = 0
        self.update = ssd.update
        if size == None:
            self.size = ssd.size
        else:
            self.size = size
        if p == None:
            self.p = ssd.p
        else:
            self.p = p
        # print("size=", self.size)
        self.change_size(self.size)
        node = self.head
        copynode = ssd.head
        for i in range(0, min(self.size, ssd.size)):
            if copynode.empty==True:    #空闲节点不复制
                break       
            node.empty = copynode.empty
            node.key = copynode.key
            self.ssd[node.key] = node
            node = node.next
            copynode = copynode.next
        self.listSize = self.size
        # print("hit=", self.hit, "update=", self.update)
        # print(self.size)
        # print(self.ssd)
        # print("p=", self.p)
        # print(ssd.ssd)
        # node = self.head
        # for i in range(self.size):
        #     print(node)
        #     node.print()
        #     node = node.next

    def __len__(self):
        return len(self.ssd)

    def clear(self):
        for node in self.dli():
            node.empty = True
        self.ssd.clear()

    def is_hit(self, key):
        if key in self.ssd:
            self.hit += 1
            self.latesthit += 1
            return True
        return False

    def get_hit(self):
        return self.hit

    def add_hit(self):
        self.hit += 1
        self.latesthit += 1

    def get_update(self):
        return self.update

    def add_update(self):
        self.update += 1

    def get_size(self):
        return self.size

    def get_p(self):
        return self.p

    #para : block to update
    #function : if hit, move block to head; else, p probabity to update new block
    #return : {evictedBlock, updatedBlock} if updated; None if not
    def update_cache(self, key, roll=None):
        # if roll==-1:
        #     print(self.size, key)
        # First, see if any value is stored under 'key' in the cache already.
        # If so we are going to replace that value with the new one.
        if key in self.ssd:
            # if roll==-1:
            #     print("hit?")
            # Lookup the node
            node = self.ssd[key]
            # Update the list ordering.
            self.mtf(node, roll)
            self.head = node
            return (None, -1) 
        # Ok, no value is currently stored under 'key' in the cache. We need
        # to choose a node to place the new item in. There are two cases. If
        # the cache is full some item will have to be pushed out of the
        # cache. We want to choose the node with the least recently used
        # item. This is the node at the tail of the list. If the cache is not
        # full we want to choose a node that is empty. Because of the way the
        # list is managed, the empty nodes are always together at the tail
        # end of the list. Thus, in either case, by chooseing the node at the
        # tail of the list our conditions are satisfied.

        # test p
        if roll==None:            
            random.seed()
            roll = random.random()
        if roll>=self.p:            #不选择加载到ssd中
            # print(roll, self.p, roll>=self.p)
            return (None, -1)

        # if roll==-1:
        #     print("test p")
        # Since the list is circular, the tail node directly preceeds the
        # 'head' node.
        self.update += 1
        node = self.head.prev
        if roll == -1:
            self.head.print()
            node.print()
        oldKey = None
        # print(node.empty)
        # If the node already contains something we need to remove the old
        # key from the dictionary.
        if not node.empty:
            oldKey = node.key
            del self.ssd[node.key]

        # Place the new key and value in the node
        node.empty = False      #非空节点
        node.key = key
        
        # Add the node to the dictionary under the new key.
        self.ssd[key] = node

        # We need to move the node to the head of the list. The node is the
        # tail node, so it directly preceeds the head node due to the list
        # being circular. Therefore, the ordering is already correct, we just
        # need to adjust the 'head' variable.
        self.head = node
        return (oldKey, key)

    def delete_cache(self, key):

        # Lookup the node, then remove it from the hash ssd.
        if key not in self.ssd:
            return
        node = self.ssd[key]
        del self.ssd[key]

        node.empty = True

        

        # Because this node is now empty we want to reuse it before any
        # non-empty node. To do that we want to move it to the tail of the
        # list. We move it so that it directly preceeds the 'head' node. This
        # makes it the tail node. The 'head' is then adjusted. This
        # adjustment ensures correctness even for the case where the 'node'
        # is the 'head' node.
        self.mtf(node)
        self.head = node.next

    

    # This method adjusts the ordering of the doubly linked list so that
    # 'node' directly precedes the 'head' node. Because of the order of
    # operations, if 'node' already directly precedes the 'head' node or if
    # 'node' is the 'head' node the order of the list will be unchanged.
    def mtf(self, node, roll=None):
        if roll == -1:
            self.head.print()
            node.print()
            node.next.print()
            node.prev.print()
        node.prev.next = node.next
        node.next.prev = node.prev

        if roll == -1:
            print(node.prev.next.key)
            print(node.next.prev.key)

        node.prev = self.head.prev
        node.next = self.head.prev.next

        if roll == -1:
            print(self.head == node)
            print(self.head.prev.next.key)
            print(node.prev.key)
            print(node.next.key)

        node.next.prev = node
        node.prev.next = node
        if roll == -1:
            print(node.next.prev.key)
            print(node.prev.next.key)

        if roll == -1:
            self.head.print()
            node.print()
            node.next.print()
            node.prev.print()
    # This method returns an iterator that iterates over the non-empty nodes
    # in the doubly linked list in order from the most recently to the least
    # recently used.
    def dli(self):
        node = self.head
        for i in range(len(self.ssd)):
            yield node
            node = node.next

    def change_p(self, p):
        self.p = p

    def change_size(self, size):
        # print(size, self.listSize)
        self.size = size      
        l = []
        freenode = self.listSize - size #需要释放的个数，为负代表需要添加的个数
        if size > self.listSize:
            self.add_tail_node(size - self.listSize)
        elif size < self.listSize:
            l= self.remove_tail_node(self.listSize - size)
        return (l, freenode)

    # Increases the size of the cache by inserting n empty nodes at the tail
    # of the list.
    def add_tail_node(self, n):
        for i in range(n):
            node = MyNode()
            node.next = self.head
            node.prev = self.head.prev

            self.head.prev.next = node
            self.head.prev = node
        self.listSize += n

    # Decreases the size of the list by removing n nodes from the tail of the
    # list.
#修改
    def remove_tail_node(self, n):
        # print("tag", self.listSize, n)
        assert self.listSize > n
        l = []
        for i in range(n):
            node = self.head.prev
            if not node.empty:           #非空节点，即ssd中有数据的地方
                del self.ssd[node.key]  
                l.append(node.key)
            # Splice the tail node out of the list
            self.head.prev = node.prev
            node.prev.next = self.head

        self.listSize -= n
        return l

    def get_top_n(self, number):
        node = self.head
        # print("number", number, "size", len(self.ssd))
        l = []
        for i in range(0, min(number, len(self.ssd))):
            # print("i", i)
            # print("node", node.empty)
            # print(node.key)
            l.append(node.key)
            node = node.next
        # print("debug", len(l), l==None)
        return l

    def get_tail_n(self, number):
        node = self.head.prev
        while node.empty:
            node = node.prev
        
        l = []
        for i in range(0, min(number, len(self.ssd))):
            l.append(node.key)
            node = node.prev
        # print("debug", len(l), l==None)
        return l

    def print_sample(self):
        print("print LRU ssd")
        if len(self.ssd) <= 100:            
            node = self.head
            for i in range(len(self.ssd)):
                print(node.key, end=",")
                node = node.next        
            print()
        print("p", self.p, "s", self.size)
        print("hit", self.hit)
        print("write", self.update)

    def update_cache_k(self, throt, potentialDict):

        node = potentialDict.head
        # print("potential dict")
        # print(len(potentialDict.ssd))
        # potentialDict.print_sample()
        throt = min(throt, len(potentialDict.ssd))
        for i in range(1, throt):
            node = node.next
        for i in range(0, throt):
            self.update_cache(node.key)
            # print(node.key)
            node = node.prev

    def is_full(self):
        if(len(self.ssd) >= self.size):
            return True
        return False

    def get_parameters(self):
        size = self.size
        p = self.p
        update = self.update
        hit = self.hit
        return (size, p, update, hit)