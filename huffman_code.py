##ref: https://blog.csdn.net/Jack_ffffff/article/details/112340859

import math
#节点类
class Node:
    def __init__(self, value):
        self.value = value
        self.prev = None
        self.left = None
        self.right = None
    def is_left(self):
        return self.prev.left == self

#每一个节点赋值，调用Node类默认方法初始化
def set_value(values):
    nodes = [Node(value) for value in values]
    return nodes

#创建哈夫曼树
def create_huffman_tree(nodes):
    nodes_huf = nodes[:]
    while len(nodes_huf) != 1:
        nodes_huf.sort(key=lambda object: object.value)
        #自底向上构造树
        node_left = nodes_huf.pop(0)
        node_right = nodes_huf.pop(0)
        node_prev = Node(node_left.value + node_right.value)
        node_prev.left = node_left
        node_prev.right = node_right
        node_left.prev = node_prev
        node_right.prev = node_prev
        nodes_huf.append(node_prev)
    nodes_huf[0].prev = None
    return nodes_huf[0]  #概率之和1

#哈夫曼编码
def huffman_encoding(nodes, node_huf):
    codes = [''] * len(nodes)
    for i in range(len(nodes)):
        node_tmp = nodes[i]
        while node_tmp != node_huf:
            if node_tmp.is_left():
                codes[i] = '1' + codes[i]
            else:
                codes[i] = '0' + codes[i]
            #从底向上查找
            node_tmp = node_tmp.prev
    return codes

#哈夫曼解码
def huffman_decoding(code, node_huf):
    node_tmp = node_huf
    for i in range(len(code)):
        if node_tmp.left != 0:
            if code[i] == '1':  #容易写成数字1
                node_tmp = node_tmp.left
            else:
                node_tmp = node_tmp.right
    return node_tmp.value

def count_value(list):
    n = len(list)
    set1 = set(list)
    dict={}
    for i in set1:
        dict.update({i:list.count(i)/n})
    return dict

def build_table(l):
    dict = count_value(l)
    val = list(dict.keys())
    fre = [float(i) for i in list(dict.values())]
    nodes = set_value(fre)
    nodes_huf = create_huffman_tree(nodes)
    codes = huffman_encoding(nodes, nodes_huf)
    # print('{0} → {1}.'.format(val, codes))
    res = {}
    for i in range(len(val)):
        res.update({val[i]:codes[i]})
    return res


def decode(str, dict):
    new_d = {v:k for k,v in dict.items()}
    res = list()
    temp = ""
    for i in str:
        temp = temp + i
        if temp in new_d.keys():
            res.append(new_d.get(temp))
            temp = ""
    return res