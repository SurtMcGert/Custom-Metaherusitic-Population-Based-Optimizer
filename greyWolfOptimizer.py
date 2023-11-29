import numpy as np

def grey_wolf_optimization(objectiveFunction, num_dimensions, num_wolves, num_iterations, bound_lower, bound_upper):
    # 
    # Initialization
    #
    
    # Wolves are randomly placed in the decision space
    position_wolves = np.random.uniform(bound_lower, bound_upper, size=(num_wolves, num_dimensions))
    alpha_position = position_wolves[0, :].copy()
    beta_position = position_wolves[1, :].copy()
    delta_position = position_wolves[2, :].copy()

    #
    # Main loop
    #
    for iteration in range(num_iterations):
        # Encirclement behaviour - as time goes on, the algorithm gets more exploitative
        a = 2 - 2 * (iteration / num_iterations)  
        # Influence position update during attacking phase, as time goes on, stronger (linearly decreases from -1 to -2)
        a2 = -1 + iteration * ((-1) / num_iterations)

        # Update positions of wolves
        for i in range(num_wolves):
            # Random dimensions (stochasticity)
            r1, r2 = np.random.rand(num_dimensions), np.random.rand(num_dimensions)
            
            # More randomness, used to modulate absolute differences
            C1, C2 = 2 * r1, 2 * r2
            
            # Scaling coefficients
            # a is defined above, and decreases over time
            A1, A2 = 2 * a * r1 - a, 2 * a * r2 - a
            
            # Absolute differences
            D_alpha, D_beta, D_delta = np.abs(C1 * alpha_position - position_wolves[i, :]), \
                                      np.abs(C2 * beta_position - position_wolves[i, :]), \
                                      np.abs(C2 * delta_position - position_wolves[i, :])

            # Position is updated based on the top three wolves' positions
            position_wolves[i, :] = (alpha_position - A1 * D_alpha) + (beta_position - A2 * D_beta) + (delta_position - a2 * D_delta)

            # Boundary check
            # np.clip sets any points outside the boundaries to the value of the closest boundary itself
            position_wolves[i, :] = np.clip(position_wolves[i, :], bound_lower, bound_upper)

        # Update fitness values
        # np.apply_along_axis applies the objective formula to every wolf in the 2D decision space, and returns a single, 1D list
        fitnessValues = np.apply_along_axis(objectiveFunction, 1, position_wolves)

        # Update positions
        index_alpha = np.argsort(fitnessValues)[0]
        index_beta = np.argsort(fitnessValues)[1]
        index_delta = np.argsort(fitnessValues)[2]
        
        alpha_position = position_wolves[index_alpha, :].copy()
        beta_position = position_wolves[index_beta, :].copy()
        delta_position = position_wolves[index_delta, :].copy()

        # Display best fitness value every 10 iterations
        if iteration % 10 == 0:
            print(f"Iteration {iteration}, Best Fitness = {np.min(fitnessValues)}")

    # Return the best solution, along with its fitness value
    return alpha_position, np.min(fitnessValues)
       