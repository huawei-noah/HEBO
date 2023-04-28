import mlflow
import os
import psutil
import numpy as np
import logging
import time
import networkx as nx
import functools
import asyncio

# Taken from https://codereview.stackexchange.com/questions/188539/python-code-to-retry-function
def retry(retry_count=5, delay=5):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return_e = None
            for _ in range(retry_count):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    return_e = e
                    logging.exception("Exception")
                    logging.warn("Retrying")
                    time.sleep(delay)
            raise return_e
        return wrapper
    return decorator

def usage_psutil():
    process = psutil.Process(os.getpid())
    return time.process_time(), process.memory_info()[0]

def correctConnection(trueTree, sampleTree):
    trueEdges = trueTree.edges()
    sampleEdges = sampleTree.edges()
    intersection = list(set(trueEdges).intersection(sampleEdges))
    if len(sampleEdges) == 0:
        return 0
    return float(len(intersection)) / len(sampleEdges)
    
def correctSeparation(trueTree, sampleTree):
    n = trueTree.number_of_nodes()
    num = 0
    den = 0
    for i in range(n):
        for j in range(n):
            if not sampleTree.has_edge(i,j):
                den += 1
                if not trueTree.has_edge(i,j):
                    num += 1
    if den == 0:
        return 0
    return float(num) / den

def make_edges_set(g):
    return set([frozenset(x) for x in g.edges()])

def f1_score(true_graph, graph):
    true_edges = (make_edges_set(true_graph))
    edges = (make_edges_set(graph))

    true_positives = true_edges.intersection(edges)
    # false_positives = edges - true_edges

    relevant_elements = true_edges

    if len(edges) == 0:
        return None, None, None

    # false_positives.union(true_positives)= edges
    precision = len(true_positives)/len(edges)
    recall = len(true_positives)/len(relevant_elements)

    if len(true_positives) == 0:
        return None, None, None

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1, precision, recall

# https://stackoverflow.com/questions/12122021/python-implementation-of-a-graph-similarity-grading-algorithm
# Note that we are comparing with graphs with the same number of nodes, so this is not a problem.
def eigenvector_similarity(graph1, graph2):
    def select_k(spectrum, minimum_energy = 0.9):
        running_total = 0.0
        total = sum(spectrum)
        if total == 0.0:
            return len(spectrum)
        for i in range(len(spectrum)):
            running_total += spectrum[i]
            if running_total / total >= minimum_energy:
                return i + 1
        return len(spectrum)

    laplacian1 = nx.spectrum.laplacian_spectrum(graph1)
    laplacian2 = nx.spectrum.laplacian_spectrum(graph2)

    k1 = select_k(laplacian1)
    k2 = select_k(laplacian2)
    k = min(k1, k2)

    return sum((laplacian1[:k] - laplacian2[:k])**2)

class MlflowLogger(object):
    def __init__(self):

        self.true_graph = None
        self.t_graph = 0

        # Optimal
        self.y_opt = None
        self.y_best = np.inf
        
        self.cum_instant_regret = 0
        self.cum_best_regret = 0
                
        self.t_y = 0
        
        # =====
        self.t_cost = 0
        self.cum_acq_cost = 0

        self.cost_metrics = {}

        # =====
        # HACK 
        self.Y_hist = []

        # =====
        self.t_ba = 0
        self.ba_metrics = {}
    def update_truth(self, y_opt, true_graph):
        self.y_opt = y_opt
        self.true_graph = true_graph

    def log_init_y(self, y):

        self.y_best = y
        
        _metrics_y = {
            'y': y,
            'y_best':self.y_best,
            'N': 0 }

        if self.y_opt != None:
            # Instant regret and best regret
            # ==============================
            # We flip the formula due to the way it is optimized
            instant_regret = y - self.y_opt
            
            # We flip the formula due to the way it is optimized
            best_regret = self.y_best - self.y_opt
            
            # cum_regrets
            #================================
            self.cum_instant_regret = instant_regret
            self.cum_best_regret = best_regret
            
            # avg cum regrets
            # ===============
            # Essential as we do not want to div by 0
            
            avg_cum_instant_regret = self.cum_instant_regret
            avg_cum_best_regret = self.cum_best_regret
            
            #logging.info("Regret {} {}".format(best_regret, cum_best_regret))
            _metrics_y = {
                **_metrics_y,
                'instant_regret': instant_regret,
                'best_regret': best_regret,
                'cum_instant_regret': self.cum_instant_regret,
                'cum_best_regret': self.cum_best_regret,
                'avg_cum_instant_regret': avg_cum_instant_regret,
                'avg_cum_best_regret': avg_cum_best_regret}
        #print("y", self.t_y)
        _metrics_y = {
                **_metrics_y,
                **self.cost_metrics,
                **self.ba_metrics}
        self.cost_metrics = {}
        self.ba_metrics = {}
        self.log_metrics_retry(_metrics_y, step=0)

    def log_y(self, y):

        self.t_y += 1
        # Current y and best y
        # ====================
        
        if y < self.y_best:
            self.y_best = y
        
        _metrics_y = {
            'y': y,
            'y_best':self.y_best,
            'N': self.t_y }

        if self.y_opt != None:
            # Instant regret and best regret
            # ==============================
            # We flip the formula due to the way it is optimized
            instant_regret = y - self.y_opt
            
            # We flip the formula due to the way it is optimized
            best_regret = self.y_best - self.y_opt
            
            # cum_regrets
            #================================
            self.cum_instant_regret += instant_regret
            self.cum_best_regret += best_regret
            
            # avg cum regrets
            # ===============
            # Essential as we do not want to div by 0
            
            avg_cum_instant_regret = self.cum_instant_regret/(self.t_y+1)
            avg_cum_best_regret = self.cum_best_regret/(self.t_y+1)
            
            #logging.info("Regret {} {}".format(best_regret, cum_best_regret))
            _metrics_y = {
                **_metrics_y,
                'instant_regret': instant_regret,
                'best_regret': best_regret,
                'cum_instant_regret': self.cum_instant_regret,
                'cum_best_regret': self.cum_best_regret,
                'avg_cum_instant_regret': avg_cum_instant_regret,
                'avg_cum_best_regret': avg_cum_best_regret}
        #print("y", self.t_y)
        _metrics_y = {
                **_metrics_y,
                **self.cost_metrics,
                **self.ba_metrics}
        self.cost_metrics = {}
        self.ba_metrics = {}
        self.log_metrics_retry(_metrics_y, step=self.t_y)

    def log_battack(self, attack_success, predicted_lbl):
        self.t_ba += 1
        self.ba_metrics = {
            "attack_success" : attack_success,
            "predicted_lbl" : predicted_lbl}

    def log_cost(self, total_cost):
        self.t_cost += 1
        self.cum_acq_cost += total_cost
        
        avg_cum_acq_cost = self.cum_acq_cost/self.t_cost
       
        # Why log this to a file?
        # Also log the cpu and ram
        process_time, mem_used = usage_psutil()
        self.cost_metrics = {
            "acq_cost" : total_cost,
            "cum_acq_cost" : self.cum_acq_cost,
            "avg_cum_acq_cost" : avg_cum_acq_cost,
            "process_time" : process_time,
            "mem_used": mem_used}
        #print("cost", self.t_cost)
        #self.log_metrics_retry(_metrics, step=self.t_cost)
    def log_cost_ba(self):
        process_time, mem_used = usage_psutil()
        self.cost_metrics = {
            "process_time" : process_time,
            "mem_used": mem_used}
    def log_graph_metrics(self, learnt_graph):

        if self.true_graph == None:
            return

        correct_connections = correctConnection(self.true_graph, learnt_graph)
        correct_separations = correctSeparation(self.true_graph, learnt_graph)

        f1, precision, recall = f1_score(self.true_graph, learnt_graph)

        _graph_metrics = {
            "correct_conn" : correct_connections,
            "correct_sep" : correct_separations,
            "eigen_sim" : eigenvector_similarity(self.true_graph, learnt_graph),
            "no_conn_comp" : nx.number_connected_components(learnt_graph)}
        if f1 != None:
            _graph_metrics["f1"] = f1
            _graph_metrics["precision"] = precision
            _graph_metrics["recall"] = recall
        else:
            logging.info("F1 unavaliable")

        self.log_metrics_retry(_graph_metrics, step=self.t_graph)
        
        self.t_graph += 1
    @retry(retry_count=5, delay=10)
    def log_metrics_retry(self, metrics, step):
        mlflow.log_metrics(metrics, step=step)

    @retry(retry_count=5, delay=10)
    def log_artifacts(self, local_dir, artifact_path=None):
        mlflow.log_artifacts(local_dir, artifact_path)

    @retry(retry_count=5, delay=10)
    def set_tags(self, tags):
        mlflow.set_tags(tags)

    @retry(retry_count=5, delay=10)
    def set_tag(self, key, value):
        mlflow.set_tag(key, value)

    @retry(retry_count=5, delay=10)
    def log_metric(self, key, value, step=None):
        mlflow.log_metric(key, value, step)
    @retry(retry_count=5, delay=10)
    def log_param(self, key, value):
        mlflow.log_param(key, value)
