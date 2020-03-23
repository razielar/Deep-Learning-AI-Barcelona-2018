#!/usr/bin/env python
# coding: utf-8

# Enumerate tutorial
# February 4th 2020

import itertools

f=['UUU', 'UUC']
i=['AUU', 'AUC', 'AUA']

pool=[f,i]

for i in itertools.product(*pool): #unpack the list
    print(''.join(i))


