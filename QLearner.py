"""
Template for implementing QLearner  (c) 2015 Tucker Balch
"""

import numpy as np
import random as rand

class QLearner(object):

    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):

        self.verbose = verbose
        self.num_actions = num_actions
        self.s = 0
        self.a = 0
        self.num_states = num_states
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.T = np.zeros((num_states,num_actions,num_states));
        self.T_c = np.zeros((num_states,num_actions,num_states));
        self.T_c.fill(0.0001);
        self.R = np.zeros((self.num_states,self.num_actions));
        self.Q = np.zeros((num_states,num_actions));

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """

        random_action = rand.randint(0, self.num_actions - 1);

        if (rand.uniform(0.0,1.0) <= self.rar):
            action = random_action;
        else:
            action = np.argmax(self.Q[s, :]);

        self.s = s;
        self.a = action;
        if self.verbose: print "s =", s,"a =",action
        return action

    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """
        random_action = rand.randint(0, self.num_actions-1);

        self.Q[self.s, self.a] = self.Q[self.s, self.a] * (1 - self.alpha) + self.alpha * (r + self.gamma * self.Q[s_prime, np.argmax(self.Q[s_prime, :])])

        if (rand.uniform(0.0,1.0) <= self.rar):
            action = random_action;
        else:
            action = np.argmax(self.Q[s_prime, :]);

        self.rar = self.rar*self.radr;

        self.T_c[self.s,self.a,s_prime] += 1;

        Tc_allstates = np.sum(self.T_c[self.s,self.a,:]);
        self.T[self.s, self.a, s_prime] = self.T_c[self.s, self.a, s_prime] / Tc_allstates;
        self.R[self.s,self.a] = (1-self.alpha)*self.R[self.s,self.a] + self.alpha * r;

        for d in range(0,self.dyna):
            s_dyna = rand.randint(0,self.num_states-1);
            a_dyna = rand.randint(0,self.num_actions-1);
            s_prime_dyna = np.argmax(self.T[s_dyna,a_dyna,:]);

            r_dyna = self.R[s_dyna,a_dyna];
            self.Q[s_dyna, a_dyna] = self.Q[s_dyna, a_dyna] * (1 - self.alpha) + self.alpha * (r_dyna + self.gamma * self.Q[s_prime_dyna, np.argmax(self.Q[s_prime_dyna, :])]);

        self.s = s_prime;
        self.a = action;

        if self.verbose: print "s =", s_prime,"a =",action,"r =",r
        return action

    def author(self):
        return 'pdesai75'

if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
