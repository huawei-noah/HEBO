

#def ts(model, X):
#    return model.gp.posterior_samples_f(X, size=1)

def ucb(model, X):
    return model.ucb(X)
