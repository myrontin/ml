# -*- coding: utf-8 -*-
"""
Created on Sat May 19 17:50:42 2018

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

# selection (PSO)
def nextGenPSO(population, fitnessPt):
    newPopulation = []
    n=len(fitnessPt)
    # 80% from min
    smallest = heapq.nsmallest(int(n*0.8),fitnessPt)
    # 20% random
    ranPos = random.sample(range(0,len(fitnessPt)),int(n*0.2))
    for x in range(0, len(smallest)):
        newPopulation.append(list(population[fitnessPt.index(smallest[x])]))     
    for x in range(0,len(ranPos)):
         newPopulation.append(list(population[ranPos[x]])) 
    for x in range(0,len(newPopulation)):
        newPopulation[x][2]=newPopulation[x][2]+1
    return newPopulation

## init population  
population =[]    
for x in range(0,10*len(x4)):
    population.append(genPopulation(x4))

# PSO
run = 100
minFitnesspt =99999999
for i in range(0,run):
    if minFitnesspt ==0:
        break
    fitnessPoint = []
    for x in range(0,len(population)):
       fitnessPoint.append(fitnessPt(population[x][1],population[x][0]))  
    population = nextGenPSO(population, fitnessPoint)  
    newfitnessPoint=[]
    for x in range(0,len(population)):
       newfitnessPoint.append(fitnessPt(population[x][1],population[x][0]))
    print("Current Generation: ",i," Max Fitness Point: ", max(newfitnessPoint)) 
    minFitnesspt = min(newfitnessPoint)

print(population[newfitnessPoint.index(minFitnesspt)])