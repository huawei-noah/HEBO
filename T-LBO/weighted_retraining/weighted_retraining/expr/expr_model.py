""" Contains code for the arithmetic expression model (Grammar VAE) """

from keras import backend as K

import weighted_retraining.weighted_retraining.expr.eq_grammar as G

masks_K = K.variable(G.masks)
ind_of_ind_K = K.variable(G.ind_of_ind)

MAX_LEN = 15
DIM = G.D

