import numpy as np
import time


def BFO(Position, CostFunction, VarMin, VarMax, MaxIt):
    nPop, nVar = Position.shape
    # BFO parameters
    VarSize = (1, nVar)
    solution = {'Position': None, 'Cost': None}
    pop = [solution.copy() for _ in range(nPop)]

    BestSol = {'Position': None, 'Cost': np.inf}

    # Initialize population
    for i in range(nPop):
        pop[i]['Position'] = Position[i]
        pop[i]['Cost'] = CostFunction(pop[i]['Position'])
        if pop[i]['Cost'] < BestSol['Cost']:
            BestSol = pop[i].copy()

    BestCosts = np.zeros(MaxIt)

    U = 0.7  # Initialize U outside the loop
    J = 1  # Initialize J outside the loop
    ct = time.time()
    for it in range(1, MaxIt + 1):
        P = abs(1 - it / (np.sqrt(1 + it ** 2))) + np.random.rand() / np.sqrt(it)
        U = (1 - it / MaxIt) * np.cos(it * np.arccos(U))
        J = (J - (J / it) / MaxIt) * U
        R = 2 * np.random.rand()
        SD0 = 1
        SD = SD0 - (P / np.pi) * (np.arctan(it * P / np.pi))

        Costs = np.array([pop[i]['Cost'] for i in range(nPop)])
        WorstCost = np.max(Costs)

        alpha = 1 - (it / MaxIt) ** 2
        beta = 1 - alpha

        newpop = []

        # Search oysters
        for i in range(nPop):
            newsol = {'Position': None, 'Cost': None}
            A = np.arange(nPop)
            A = np.delete(A, i)
            j = np.random.choice(A)

            if np.random.rand() < P:
                newsol['Position'] = pop[i]['Position'] + (
                        np.random.rand(*VarSize) * (pop[j]['Position'] - J * pop[i]['Position']))
            else:
                newsol['Position'] = pop[i]['Position'] + (
                        np.random.rand(*VarSize) * (BestSol['Position'] - J * pop[i]['Position']))

            newsol['Position'] = np.maximum(newsol['Position'], VarMin[i])
            newsol['Position'] = np.minimum(newsol['Position'], VarMax[i])
            newsol['Cost'] = CostFunction(newsol['Position'][0])

            if newsol['Cost'] < pop[i]['Cost']:
                pop[i] = newsol.copy()
                if pop[i]['Cost'] < BestSol['Cost']:
                    BestSol = pop[i].copy()

        # Means
        MMM = np.zeros(VarSize)
        for i in range(nPop):
            MMM += pop[i]['Position']
        MMM /= nPop

        # Escape
        for i in range(nPop):
            newsol = {'Position': None, 'Cost': None}
            if np.random.rand() < P:
                newsol['Position'] = pop[i]['Position'] + (np.random.rand(*VarSize) * (BestSol['Position'] - J * MMM))
            else:
                newsol['Position'] = np.random.uniform(VarMin[i], VarMax[i], size=VarSize)

            newsol['Position'] = np.maximum(newsol['Position'], VarMin[i])
            newsol['Position'] = np.minimum(newsol['Position'], VarMax[i])
            newsol['Cost'] = CostFunction(newsol['Position'][0])

            if newsol['Cost'] < pop[i]['Cost']:
                pop[i] = newsol.copy()
                if pop[i]['Cost'] < BestSol['Cost']:
                    BestSol = pop[i].copy()

        # Matting
        for i in range(nPop):
            A = np.arange(nPop)
            A = np.delete(A, i)
            j1 = np.random.choice(A)
            A = np.delete(A, np.where(A == j1))
            j2 = np.random.choice(A)
            A = np.delete(A, np.where(A == j2))
            j3 = np.random.choice(A)

            newsol = {'Position': None, 'Cost': None}
            newsol['Position'] = pop[j1]['Position'] + (
                    np.random.rand(*VarSize) * (pop[j2]['Position'] - pop[j3]['Position']))
            newsol['Position'] = np.maximum(newsol['Position'], VarMin[i])
            newsol['Position'] = np.minimum(newsol['Position'], VarMax[i])
            newsol['Cost'] = CostFunction(newsol['Position'][0])

            if newsol['Cost'] < pop[i]['Cost']:
                pop[i] = newsol.copy()
                if pop[i]['Cost'] < BestSol['Cost']:
                    BestSol = pop[i].copy()

        BestCosts[it - 1] = BestSol['Cost']

        # Sort population by cost
        SortOrder = np.argsort([pop[i]['Cost'] for i in range(nPop)])
        pop = [pop[i] for i in SortOrder]

        # Select top N/4 solutions
        N = int(np.ceil(nPop / 4))
        pop_plus = pop[:N]

    Best_position = pop[SortOrder[0]]['Position'][0]
    Best_Fit = pop[SortOrder[0]]['Cost']

    ct = time.time() - ct
    return Best_Fit, BestCosts, Best_position, ct
