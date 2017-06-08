from numpy import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def compute_error(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) **2
    return totalError / float(len(points))

def step_gradient(b_current, m_current, points, learning_rate):
    #gradient descent
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return [new_b, new_m]

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m

    #plot
    for i in range(0, len(points)):
        plt.scatter(points[i][0], points[i][1], c ='b')

    for i in range(num_iterations):
        x1,x2 = 20,80
        y1,y2 = m * x1 + b,m * x2 + b
        plt.plot([x1,x2],[y1,y2],c='r')
        b, m = step_gradient(b, m, array(points), learning_rate)
    plt.show()
    return [b,m]

def run():
    points = genfromtxt('data.csv', delimiter=',')
    #hyperparameters
    learning_rate = 0.0001
    #y = mx + b
    initial_b = 0
    initial_m = 0
    num_iterations = 1000
    print ("Initial Error = {0}".format(compute_error(initial_b, initial_m, points)))
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print ("Final Error = {0}".format(compute_error(b, m, points)))
    print ("After {0} iterations b = {1}, m = {2}".format(num_iterations, b, m))
    #plot
    for i in range(0, len(points)):
        plt.scatter(points[i][0], points[i][1], c ='b')
    x1,x2 = 20,80
    y1,y2 = m * x1 + b,m * x2 + b
    plt.plot([x1,x2],[y1,y2],c='r')
    plt.show()

if __name__ == '__main__':
    run()
