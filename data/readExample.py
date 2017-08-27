#coding: utf-8

file=open(r'C:\\Users\\lenovo\\workspace1\\Qbs\\src\\共享单车\\共享单车keywords.txt','r',encoding='utf-8')
data=file.read()
keywords=data.split(' ')
print(len(keywords),keywords[1])

file1=open(r'C:\\Users\\lenovo\\workspace1\\Qbs\\src\\共享单车\\共享单车keySentence.txt','r',encoding='utf-8')
data1=file1.read()
keySentence=data1.split('\n')
print(len(keySentence),keySentence[1])
'''
Created on 2017年7月11日

@author: lenovo
'''
