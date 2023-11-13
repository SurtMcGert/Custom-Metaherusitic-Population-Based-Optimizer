import numpy as np

def batAlgorithm(objective, dimensions, lowerBound, upperBound, epochs, populationSize, alpha=0.9, Amin = 0, gamma=0.9, fmin = 0, fmax = 100):
    #Initialize parameters for generation t=0
    #Position
    x = np.random.uniform(lowerBound, upperBound, size=(populationSize, dimensions))
    #Velocity
    v = np.random.uniform(lowerBound, upperBound, size=(populationSize, dimensions))
    #Loudness
    A = np.random.uniform(1, 2, size=(populationSize,1))
    #Frequency
    f = np.random.uniform(fmin, fmax, size=(populationSize,1))
    #Pulse rate
    r = np.random.uniform(0, 1, size=(populationSize,1))
    #Initial pulse rates
    r0 = r
    #Generation
    t = 0
    #Limit to which can be considered "among the best solutions"
    randomSolutionIndexLimit = 0.1
    
    #Sort bats based on fitness values
    fitnessValues = np.apply_along_axis(objective, 1, x)
    #List of best fitness values
    bestFitnessArray = np.sort(fitnessValues)
    #List of indexes of the bats based on best fitness values
    xbestSorted = np.argsort(fitnessValues)
    #Position of best bat
    xbest = xbestSorted[0]
    #Best bat's fitness value
    bestFitness = bestFitnessArray[0]
    
    #Main loop
    while t < epochs:
        
        #Iterate over each bat
        for bat in range(populationSize):
            #Get random beta and rand values
            beta = np.random.rand()
            rand = np.random.rand()
            #Update frequency, velocity, and position of each bat
            f[bat] = fmin + (fmax-fmin) * beta
            v[bat] = v[bat] + (x[bat]-x[xbestSorted[0]])*f[bat]
            x[bat] = x[bat] + v[bat]
            
            #Select new solution for bat based on one of the best solutions
            if (rand > r[bat]):
                maxBestLimit = randomSolutionIndexLimit * populationSize
                randomSolution = np.random.choice(xbestSorted[0:maxBestLimit])
                xnew = x[randomSolution] + np.random.uniform(-1, 1, size = dimensions)*np.mean(A)
                newFitness = objective(xnew)
               
            #Accept solution based on loudness
            if (rand < A[bat] and newFitness < bestFitness):
                #Store new position
                x[bat] = xnew

                #Update loudness
                A[bat] = alpha * A[bat]
                #If loudness falls below minimum loudness (Amin) set back the loudness to Amin
                if A[bat] < Amin:
                    A[bat] = Amin
                
                #Update pulse rate
                r[bat] = r0[bat] * (1 - exp(-gamma*t))
                
        #Sort bats based on fitness values
        fitnessValues = np.apply_along_axis(objective, 1, x)
        #List of best fitness values
        bestFitnessArray = np.sort(fitnessValues)
        #List of indexes of the bats based on best fitness values
        xbestSorted = np.argsort(fitnessValues)
        #Position of best bat
        xbest = xbestSorted[0]
        #Best bat's fitness value
        bestFitness = bestFitnessArray[0]
        
        #Increase generation count
        t += 1
        
    return xbest, bestFitness