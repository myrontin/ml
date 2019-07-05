# -*- coding: utf-8 -*-
"""
Created on Tue May 15 20:37:25 2018

@author: Myron
"""
#####################################################################################################################
#This program attempts to solve a problem in numerical computation. Generate four lists of positive integers.       #
#List A contains 10 elements. List B has 50, list C 250, and list D 1000.                                           #
#Use a linear congruential method to generate the four lists. Each integer should be between 1 and 10,000.          #
#Divide each list into two sublists ( each sub-list need not contain precisely half of the original list elements)  # 
#so that the differences of the sums of the numbers in the two sublists is minimized.                               #
#####################################################################################################################

import numpy as np
import random
import heapq

## generate number
np.random.seed(48)
x1= np.random.randint(low=0,high=10000,size=10)
np.random.seed(52)
x2= np.random.randint(low=0,high=10000,size=50)
np.random.seed(83)
x3= np.random.randint(low=0,high=10000,size=250)
np.random.seed(102)
x4= np.random.randint(low=0,high=10000,size=1000)

## create chromosoime
#def genChromo(n):
#    m1=np.ones(n//2,dtype=int)
#    m2=np.zeros(n//2,dtype=int)
#    chromo = np.concatenate((m1,m2))
#    return chromo
#
#chromo = genChromo(10)
#Population =[] 
#for x in range (0,10):
#    np.random.shuffle(chromo)
#    Population.append(list(chromo))

# population initialization
def genPopulation(offspring):
    n = offspring.size
    np.random.shuffle(offspring)
    m1 = np.copy(offspring[0:n//2])
    m2 = np.copy(offspring[n//2:n])
    pop = [m1,m2,0]
    return pop

# fitness function
def fitnessPt(p1, p2):
    pt = np.absolute(np.sum(p1)-np.sum(p2))
    return pt

# generatie random position for crossover
def randomPos(size):
    ranPos=random.sample(range(0,size),size//2)
    return ranPos

# crossover 
def crossover(chromo):
    child = []
    child.append(np.copy(chromo[0]))
    child.append(np.copy(chromo[1]))
    n=child[0].size
    ranPos = randomPos(n)
    for x in range(0,len(ranPos)):
        temp = np.copy(child[0][ranPos[x]])
        child[0][ranPos[x]]= np.copy(child[1][ranPos[x]])
        child[1][ranPos[x]] = temp
    child = [child[0],child[1],0]
    return child
    
# selection (GA)
def nextGen(population, fitnessPt):
    newPopulation = []
    n=len(fitnessPt)
    # 80% from min
    smallest = heapq.nsmallest(int(n*0.4),fitnessPt)
    # 20% random
    ranPos = random.sample(range(0,len(fitnessPt)),int(n*0.1))
    for x in range(0, len(smallest)):
        newPopulation.append(list(population[fitnessPt.index(smallest[x])]))     
    for x in range(0,len(ranPos)):
         newPopulation.append(list(population[ranPos[x]])) 
    for x in range(0,len(newPopulation)):
        newPopulation[x][2]=newPopulation[x][2]+1
    return newPopulation

# init population  
population =[]    
for x in range(0,5*len(x4)):
    population.append(genPopulation(x4))

# GA
run = 100
minFitnesspt =99999999
for i in range(0,run):
    if minFitnesspt ==0:
        break
    for x in range(0,len(population)):
        population.append(crossover(population[x]))
        
    fitnessPoint = []
    
    for x in range(0,len(population)):
       fitnessPoint.append(fitnessPt(population[x][1],population[x][0]))  
       
    population = nextGen(population, fitnessPoint)  
    newfitnessPoint=[]
    
    for x in range(0,len(population)):
       newfitnessPoint.append(fitnessPt(population[x][1],population[x][0]))
    print("Current Generation: ",i," Min Fitness Point: ", min(newfitnessPoint)) 
    minFitnesspt = min(newfitnessPoint)

print(population[newfitnessPoint.index(minFitnesspt)])
























