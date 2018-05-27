import random
import math
import array
import utils

class SPSA_minimization:
    def __init__(self,f,theta0,max_iter,constraints=None,options={}):
        '''
        Args:
            f(function): the function to minimize
            theta0(dict): the starting point of the minimization
            max_iter(int): the number of iterations of the algorithm
            constraints(function,Optional) : a function which maps the current point to the
                closest point of the search domain .
            options(dict,optional): Optional settings of the spsa algorithm parameters,
                Default values taken from the reference articles are used if not present in options

        '''
        #Store the arguments
        self.f = f
        self.theta0 = theta0
        self.max_iter = max_iter
        self.constraints = constraints
        self.options = options
        #some attributes to provide an history of evaluations
        self.previous_gradient = {}
        self.rprop_previous_g = {}
        self.rprop_previous_delta = {}

        self.history_eval = array.array('d',range(1000))
        self.history_theta = [theta0 for k in range(1000)]
        self.history_count = 0

        self.best_eval = array.array('d',range(1000))
        self.best_theta = [theta0 for k in range(1000)]
        self.best_count =  0
        #These constants are used throughout the spsa algorithm
        self.a = options.get('a',1.1)
        self.c = options.get('c',0.1)

        self.alpha = options.get('alpha',0.70) #theoretical alpha = 0.601,must be <= 1
        self.gamma = options.get('gamma',0.12) #theoretical gamma = 0.101,must be <= 1/6
        self.A = options.get('A',max_iter/10.0)

    def run(self):
        '''
            Return a point which is (hopefully) a minimizer of the goal function f,
            starting from point theta0
            Returns:
                the point (as a dict) which is (hopefully) a minimize of 'f'.
        '''
        k = 0
        theta = self.theta0
        while True:
            if self.constraints is not None:
                theta = self.constraints(theta)
            print("theta = " + utils.pretty(theta))
            c_k = self.c/((k+1)**self.gamma)
            a_k = self.a/((k+1 + self.A) ** self.alpha)
            gradient = self.approximate_gradient(theta,c_k)
            #For steepest descent we update via a constant small step in the gradient direction
            mu = -0.01/max(1.0,utils.norm2(gradient))
            theta = utils.linear_combinaison(1.0,theta,mu,gradient)

            ## For RPROP, we update with information about the sign of the gradients
            theta = utils.linear_combinaison(1.0,theta,-0.01,self.rprop(theta,gradient))
            #We then move to the point which gives the best average of goal
            (avg_goal,avg_theta) = self.average_best_evals(30)
            theta = utils.linear_combinaison(0.8,theta,0.2,avg_theta)

            k = k +1
            if k >= self.max_iter:
                break;

            if (k % 100 == 0) or (k <= 1000):
                (avg_goal,avg_theta) = self.average_evaluations(30)
                print("iter = " + str(k))
                print("mean goal (all) = " + str(avg_goal))
                print("mean theta (all) = " + utils.pretty(avg_theta))

                (avg_goal,avg_theta) = self.average_best_evals(30)
                print('mean goal (best) = ' + str(avg_goal))
                print('mean theta (best) = ' + utils.pretty(avg_theta))
            print('-----------------------------------------------------------')
        return theta

    def evaluate_goal(self,theta):
        '''
            Return the evaluation of the goal function f at point theta.
        '''
        v = self.f(**theta)
        #store the value in history
        self.history_eval[self.history_count %1000] = v
        self.history_theta[self.history_count%1000] = theta
        self.history_count += 1

        return v

    def approximate_gradient(self,theta,c):
        '''
            Return an approximation of the gradient of f at point theta.
            On repeated calls, the esperance of the series of returned values
            converges almost surely to the true gradient of f at theta.
        '''
        if self.history_count > 0:
            current_goal, _ = self.average_evaluations(30)
        else:
            current_goal = 100000000000000000.0
        bernouilli = self.create_bernouilli(theta)

        count = 0
        while True:
            state = random.getstate()
            theta1 = utils.linear_combinaison(1.0,theta,c,bernouilli)
            f1 = self.evaluate_goal(theta1)
            random.setstate(state)
            theta2 = utils.linear_combinaison(1.0,theta,-c,bernouilli)
            f2 = self.evaluate_goal(theta2)

            if f1 != f2:
                break;
            count = count +1
            if count >= 100:
                break;
        #Update the gradient
        gradient = {}
        for (name,value) in theta.items():
            gradient[name] = (f1-f2)/(2.0*c*bernouilli[name])

        if(f1 > current_goal) and (f2 > current_goal):
            print('function seems not decreasing')
            gradient = utils.linear_combinaison(0.1,gradient)
        gradient = utils.linear_combinaison(0.1,gradient,0.9,self.previous_gradient)
        self.previous_gradient = gradient
        #Store the best the two evals f1 and f2 (or both)
        if(f1 <= current_goal):
            self.best_eval[self.best_count %1000] = f1
            self.best_theta[self.best_count%1000] = theta1
            self.best_count +=1
        if(f2 <= current_goal):
            self.best_eval[self.best_count%1000] = f2
            self.best_theta[self.best_count%1000] = theta2
            self.best_count += 1
        #Return the estimation of the new gradient
        return gradient

    def create_bernouilli(self,m):
        '''
            Create a random direction to estimate the stochastic gradient.
        '''
        bernouilli = {}
        for (name,value) in m.items():
            bernouilli[name] = 1 if random.randint(0,1) else -1

        g = utils.norm2(self.previous_gradient)
        d = utils.norm2(bernouilli)
        if g > 0.00001:
            bernouilli = utils.linear_combinaison(0.55,bernouilli,0.25*d/g,self.previous_gradient)

        for (name,value) in m.items():
            if bernouilli[name] == 0.0:
                bernouilli[name] = 0.2
            if abs(bernouilli[name]) < 0.2:
                bernouilli[name] = 0.2*utils.sign_of(bernouilli[name])
        return bernouilli
    def average_evaluations(self,n):
        '''
            Return the average of the n last evaluation of the goal function.
        '''
        assert(self.history_count >0)
        if n <= 0 : n = 1
        if n > 1000: n = 1000
        if n > self.history_count: n = self.history_count

        sum_eval = 0.0
        sum_theta = utils.linear_combinaison(0.0,self.theta0)
        for i in range(n):
            j = ((self.history_count -1)%1000) -i
            if j < 0 : j += 1000
            if j >= 1000: j -= 1000

            sum_eval += self.history_eval[j]
            sum_theta = utils.sum(sum_theta,self.history_theta[j])
        #return the average
        alpha = 1.0 /(1.0*n)
        return (alpha*sum_eval,utils.linear_combinaison(alpha,sum_theta))

    def average_best_evals(self,n):
        '''
            Return the average of the n last best evaluation of the goal function.
        '''
        if n <= 0: n = 1
        if n > 1000 : n = 1000
        if n > self.best_count: n = self.best_count

        sum_eval = 0.0
        sum_theta = utils.linear_combinaison(0.0,self.theta0)
        for i in range(n):
            j = ((self.best_count -1) % 1000) -i
            if j < 0 : j+= 1000
            if j >= 1000 : j -= 1000
            sum_eval += self.best_eval[j]
            sum_theta = utils.sum(sum_theta,self.best_theta[j])
        #return the average
        alpha = 1.0 /(1.0*n)
        return (alpha*sum_eval,utils.linear_combinaison(alpha,sum_theta))

    def rprop(self,theta,gradient):
        #get the previous g of the RPROP algorithm
        if self.rprop_previous_g != {}:
            previous_g = self.rprop_previous_g
        else:
            previous_g = gradient
        #get the previous delta of the RPROP algorithm
        if self.rprop_previous_delta != {}:
            delta = self.rprop_previous_delta
        else:
            delta = gradient
            delta = utils.copy_and_fill(delta,0.5)

        p = utils.hadamard_product(previous_g,gradient)
        print('gradient = ' + utils.pretty(gradient))
        print('old_g = ' + utils.pretty(previous_g))
        print('p = ' + utils.pretty(p))

        g = {}
        eta = {}
        for (name,value) in p.items():
            if p[name] >0 : eta[name] = 1.1 #building speed
            if p[name] <0 : eta[name] = 0.5 #we have passed a local minima :slow down
            if p[name] == 0: eta[name] = 1.0

            delta[name] = eta[name] *delta[name]
            delta[name] = min(50.0,delta[name])
            g[name] = gradient[name]
        print('g = ' + utils.pretty(g))
        print('eta =' + utils.pretty(eta))
        print('delta = ' + utils.pretty(delta))
        #store the current g and delta for the next call of the RPROP algorithm
        self.rprop_previous_g = g
        self.rprop_previous_delta = delta
        #calculate the update for the current RPROP
        s = utils.hadamard_product(delta,utils.sign(g))
        print('sign(g) = ' + utils.pretty(utils.sign(g)))
        print(' s = ' + utils.pretty(s))
        return s

if __name__ == "__main__":
    def g(**args):
        x = args["x"]
        return x*x
    print(SPSA_minimization(g,{"x":3.0},1000).run())
