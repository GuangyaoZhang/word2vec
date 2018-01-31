
import numpy as np
from numba import jit
import math
import random
from heapq import *
import struct

from multiprocessing import Process
from multiprocessing import Manager
import cProfile




class Word2vec():
    vocab_size = 0
    def __init__(self):
        self.vocab_size = 0
        self.layer1size = 200

        self.dic = dict()
        self.file = open('./data/text8','rb')

        self.readDic()
        Word.vocab_size = self.vocab_size

        self.window = 8
        self.sample = 1e-3

        self.initNet()
        sorted_list = sorted(list(self.dic.items()),key = lambda t:t[1].codelen,reverse=True)
        i = [(i[0],i[1].codelen) for i in sorted_list]
        print(i)
        # self.train()
        # # self.multitrain()
        # wri = open('text8-vector.bin', 'wb')
        # self.sorted_list = sorted(self.dic.items(), key=lambda t: t[1].weight, reverse=True)
        # wri.write(b'%ld %ld\n' % (self.vocab_size, self.layer1size))
        # for i in self.sorted_list:
        #     if (i[1].weight):
        #         wri.write(bytes(i[0] + ' ', encoding="utf8"))
        #         for j in i[1].vec:
        #             s = struct.pack('f', j)
        #             wri.write(s)
        #         wri.write(b'\n')

    def multitrain(self):
        m = Manager()
        d = m.dict()
        d.update(self.dic)
        mul_li = list()
        total = 4
        self.max_sentence_len = 1000

        self.neu1 = np.zeros(self.layer1size)
        self.neu1e = np.zeros(self.layer1size)
        self.max_sentence_len = 1000
        self.sentence = list()
        self.sentences = list()

        self.alpha_first = 0.025
        self.alpha = 0.025
        self.word_position = 0
        self.word_count = 0
        time = 0
        for word in self.content_include_enter:
            if word == '\n' or len(self.sentence) > self.max_sentence_len:
                self.sentences.append(self.sentence)
                self.sentence = list()
            elif self.dic[word].weight:
                ran = (math.sqrt(self.dic[word].weight / (self.sample * self.train_words)) + 1) * (
                self.sample * self.train_words) / self.dic[word].weight
                rand = random.random()
                if ran < rand:
                    pass

                else:
                    self.sentence.append(word)
                    print(time)
                    time += 1
        print(self.sentences)
        print(len(self.sentences))



        for i in range(total):
            mul_li.append(Process(target=self.train_thread,args=(i,total)))
        for i in range(total):
            mul_li[i].start()
        for i in range(total):
            mul_li[i].join()


    def readDic(self):
        file = self.file.read()
        print(file)
        print(type(file))
        print(len(file))
        self.content = str(file).split()
        self.content_include_enter = str(file).split(' ')
        # self.train_words = len(self.content)#??????????????????????
        i = 0
        for word in self.content:
            if word in self.dic:
                self.dic[word].weightup()
            else:
                self.dic.setdefault(word, Word(word,1))
            i+=1
            if(i%1000000 == 0):
                print(i)

        self.reduceDic()

        self.vocab_size = 0

        self.content_size = 0

        for i in self.content:
            if(self.dic[i].weight):
                self.content_size+=1
        for i in self.dic.items():
            if(i[1].weight):
                self.vocab_size+=1
        Word.vocab_size = self.vocab_size
        self.train_words = self.content_size
        print(self.content_size)
        print(self.vocab_size)



    def reduceDic(self):
        for i in self.dic.items():
            if(i[1].weight<5):
                i[1].weight = 0


    def initNet(self):
        self.createBinaryTree()


        # 测试：通过dict.items修改dict数值
        # a = {2: [10], 3: [20]}
        # for i in a.items():
        #     i[1][0] = 1000
        # print(a)
        # 运行结果：
        # {2: [1000], 3: [1000]}


    def createBinaryTree(self):

        # vocab_list = list(self.dic.items())
        vocab_list = []
        for i in self.dic.items():
            if i[1].weight:
                vocab_list.append(i)

        self.vocab = sorted([i[1] for i in vocab_list],key = lambda t:t.weight)

        heapify(self.vocab)
    # 这里有一个有趣的现象，初始数组的不同将会导致堆化后的数组排序不同
    #     具体来说
    #     vocab_list = list(self.dic.items())
    #     self.vocab = sorted([i[1] for i in vocab_list], key=lambda t: t.weight)
    #     print([[i.name, i.weight] for i in self.vocab])
    #     heapify(self.vocab)
    #     print([[i.name, i.weight] for i in self.vocab])
    #     和
    #     vocab_list = list(self.dic.items())
    #     self.vocab = sorted([i[1] for i in vocab_list], key=lambda t: t.weight,reverse= Ture)
    #     print([[i.name, i.weight] for i in self.vocab])
    #     heapify(self.vocab)
    #     print([[i.name, i.weight] for i in self.vocab])
    #     执行结果不同
# 利用递归实现，内存溢出?
        while len(self.vocab) > 1:
            r = heappop(self.vocab)
            l = heappop(self.vocab)
            n = Word(None, r.weight + l.weight)
            n.setChildren(l, r)
            heappush(self.vocab, n)
# 单层复制
        def oneLayerDeepCopy(self,li):
            y = []
            for i in li:
                y.append(i)
            return y

        def codeIt(self,node):
            i = 0
            if not(node.left and node.right):
                    self.dic[node.name] = node
            else:
                node.left.codelen = node.codelen + 1
                node.right.codelen = node.codelen + 1
                print(node.codelen)
                node.left.code = node.code+'1'
                node.right.code = node.code+'0'
                node.left.vecs = oneLayerDeepCopy(self,node.vecs)
                node.right.vecs = oneLayerDeepCopy(self,node.vecs)
                if(i%10000 ==0):
                    print(i+1)
                i+=1
                node.left.vecs.append(node)
                node.right.vecs.append(node)
                codeIt(self,node.left)
                codeIt(self,node.right)


#python 对象名是指针
        self.vocab[0].codelen = 0
        self.vocab[0].code = ''

        codeIt(self,self.vocab[0])

        for i in self.dic.items():
            i[1].vec = (np.random.random(self.layer1size)-0.5)/self.layer1size

        # 测试复制深度是否足够浅
        # print(self.dic['the'].vecs[0].vec)
        # self.dic['of'].vecs[0].vec = 0
        # print(self.dic['the'].vecs[0].vec)

        # #  Again
        # while len(self.vocab) > 1:
        #         l = heappop(self.vocab)
        #         r = heappop(self.vocab)
        #         n = Word(None, r.weight + l.weight)
        #         l.setparent()
        #         heappush(self.vocab, n)


    def train(self):
        self.neu1 = np.zeros(self.layer1size)
        self.neu1e = np.zeros(self.layer1size)
        self.max_sentence_len = 1000
        self.sentence = list()
        self.sentences = list()




        self.alpha_first = 0.025
        self.alpha = 0.025
        self.word_position = 0
        self.word_count = 0

        # for word in self.content_include_enter:
        #     if self.dic[word].weight:
        #         ran = (math.sqrt(self.dic[word].weight/(self.sample*self.train_words)) + 1) * (self.sample*self.train_words)/self.dic[word].weight
        #         rand = random.random()
        #         if ran < rand:
        #             pass
        #         else:
        #             self.sentence.append(word)
        #
        #     if word == '\n' or len(self.sentence)>self.max_sentence_len:
        #         self.sentences.append(self.sentence)
        #         self.sentence = list()
        # print(self.sentences)
        # print(len(self.sentences))
        # word_ac = 0
        # for sen in self.sentences:
        #     word_ac+=len(sen)
        # print(word_ac)
        # for sentence in self.sentences:
        #     self.word_position = 0
        #
        #     self.word_a = 0
        #     if (self.word_a > 10000):
        #         self.word_a = 0
        #         print("Alpha: %f  Progress: %.2f%%", self.alpha, self.word_count / (self.train_words + 1) * 100)
        #         self.alpha = self.alpha_first * (1 - self.word_count / (self.train_words + 1))
        #         if (self.alpha < self.alpha_first * 0.0001):
        #             self.alpha = self.alpha_first * 0.0001
        #
        #     for word in sentence:
        #         self.neu1 = np.zeros(self.layer1size)
        #         self.neu1e = np.zeros(self.layer1size)
        #         if (self.word_count % 10000 == 0):
        #             print("Alpha: %f  Progress: %f" % (self.alpha, self.word_count / (word_ac + 1) * 100))
        #             self.alpha = self.alpha_first * (1 - self.word_count / (word_ac + 1))
        #             if (self.alpha < self.alpha_first * 0.0001):
        #                 self.alpha = self.alpha_first * 0.0001
        #         b = random.randint(1,self.window+1)
        #         for i in range(b+1):
        #             if(i ==0):
        #                 continue
        #             if(self.word_position-i>=0):
        #                 self.neu1+=self.dic[sentence[self.word_position-i]].vec
        #             if(self.word_position+i<len(self.sentence)):
        #                  self.neu1+= self.dic[sentence[self.word_position + i]].vec
        #         self.word_position = self.word_position+1
        #         self.word_count = self.word_count + 1
        #         self.word_a+=1
        #
        #         def sigmoid(x):
        #             return 1 / (1 + math.exp(-x))
        #
        #
        #         #学习 for zip
        #         for vec_node,code in zip(self.dic[word].vecs,self.dic[word].code):
        #             # print(self.neu1)
        #             # print(vec_node.vec)
        #             # print(np.dot(self.neu1.T,vec_node.vec))
        #             q = sigmoid(np.dot(self.neu1.T,vec_node.vec))
        #             g = self.alpha*(1-int(code)-q)
        #             self.neu1e+=g*vec_node.vec
        #             vec_node.vec+=g*self.neu1
        #         for i in range(b):
        #             if(i ==0):
        #                 continue
        #             if(self.word_position-i>=0):
        #                 self.dic[sentence[self.word_position-i]].vec+=self.neu1e
        #             if(self.word_position+i<len(self.sentence)):
        #                 self.dic[sentence[self.word_position + i]].vec+=self.neu1e

        li = []
        for word in self.content_include_enter:
            if self.dic[word].weight:
                    ran = (math.sqrt(self.dic[word].weight/(self.sample*self.train_words)) + 1) * (self.sample*self.train_words)/self.dic[word].weight
                    rand = random.random()
                    if ran < rand:
                        pass
                    else:
                        li.append(word)

        word_ac = len(li)
        print(word_ac)

        self.word_a=0
        self.content = li
        for word in self.content:
            self.neu1 = np.zeros(self.layer1size)
            self.neu1e = np.zeros(self.layer1size)
            if (self.word_count % 10000 == 0):
                print("Alpha: %f  Progress: %f" % (self.alpha, self.word_count / (word_ac + 1) * 100))
                self.alpha = self.alpha_first * (1 - self.word_count / (word_ac + 1))
                if (self.alpha < self.alpha_first * 0.0001):
                    self.alpha = self.alpha_first * 0.0001
            b = random.randint(1,self.window+1)
            i = 0
            for i in range(b+1):
                if(i ==0):
                    continue
                if(self.word_position-i>=0):
                    self.neu1+=self.dic[self.content[self.word_position-i]].vec
                if(self.word_position+i<len(self.content)):
                     self.neu1+= self.dic[self.content[self.word_position + i]].vec
            self.word_position = self.word_position+1
            self.word_count = self.word_count + 1
            self.word_a+=1

            def sigmoid(x):
                return 1 / (1 + math.exp(-x))


            #学习 for zip
            for vec_node,code in zip(self.dic[word].vecs,self.dic[word].code):
                # print(self.neu1)
                # print(vec_node.vec)
                # print(np.dot(self.neu1.T,vec_node.vec))
                q = sigmoid(np.dot(self.neu1,vec_node.vec))
                g = self.alpha*(1-int(code)-q)
                self.neu1e+=g*vec_node.vec
                vec_node.vec+=g*self.neu1
            for i in range(b):
                if(i ==0):
                    continue
                if(self.word_position-i>=0):
                    self.dic[self.content[self.word_position - i]].vec+=self.neu1e
                if(self.word_position+i<len(self.sentence)):
                    self.dic[self.content[self.word_position + i]].vec+=self.neu1e


    def train_thread(self,id,total):
        self.neu1 = np.zeros(self.layer1size)
        self.neu1e = np.zeros(self.layer1size)
        self.max_sentence_len = 1000


        self.alpha_first = 0.025
        self.alpha = 0.025
        self.word_position = 0
        self.word_count = 0


        time = 0
        # self.content_include_enter = self.content_include_enter[len(self.content_include_enter)]



        self.sentences = self.sentences[id*(len(self.sentences)//total):(id+1)*(len(self.sentences)//total)]

        for sentence in self.sentences:
            self.word_position = 0
            self.word_a = 0
            if (self.word_a > 10000):
                self.word_a = 0
                print("Alpha: %f  Progress: %.2f%%", self.alpha, self.word_count / (self.train_words + 1) * 100)
                self.alpha = self.alpha_first * (1 - self.word_count / (self.train_words + 1))
                if (self.alpha < self.alpha_first * 0.0001):
                    self.alpha = self.alpha_first * 0.0001
            print(self.word_count)
            for word in sentence:
                self.neu1 = np.zeros(self.layer1size)
                self.neu1e = np.zeros(self.layer1size)

                b = random.randint(1,self.window+1)
                for i in range(b):
                    if(i ==0):
                        continue
                    if(self.word_position-i>=0):
                        self.neu1 +=self.dic[sentence[self.word_position-i]].vec
                    if(self.word_position+i<len(self.sentence)):
                         self.neu1 += self.dic[sentence[self.word_position + i]].vec
                self.word_position = self.word_position+1
                self.word_count = self.word_count + 1
                print(self.word_count)
                print(self.word_position)
                self.word_a+=1

                def sigmoid(x):
                    return 1 / (1 + math.exp(-x))


                #学习 for zip
                for vec_node,code in zip(self.dic[word].vecs,self.dic[word].code):
                    # print(self.neu1)
                    # print(vec_node.vec)
                    # print(np.dot(self.neu1.T,vec_node.vec))
                    q = sigmoid(np.dot(self.neu1,vec_node.vec))
                    g = self.alpha*(1-int(code)-q)
                    self.neu1e+=g*vec_node.vec
                    vec_node.vec+=g*self.neu1
                for i in range(b):
                    if(i ==0):
                        continue
                    if(self.word_position-i>=0):
                        self.dic[sentence[self.word_position-i]].vec+=self.neu1e
                    if(self.word_position+i<len(self.sentence)):
                        self.dic[sentence[self.word_position + i]].vec+=self.neu1e
class Word():

    layer1size = 200

    def __init__(self,name,weight):
        self.name = name
        self.weight = weight
        self.vec = np.zeros(Word.layer1size)
        self.vecs = list()

        self.left = None
        self.rigth = None
        self.codelen = 1000
    def setParent(self,pr):
        self.parent = pr

    def setChildren(self, ln, rn):
        self.left = ln
        self.right = rn

    def __lt__(self, other):
        return self.weight<other.weight

    def weightup(self,num = 1):
        self.weight+=num

def start():
    word2vec = Word2vec()
start()
