import argparse

class GeneralArgumentParser(object):
    """"Argument parser for the main running file main.py."""
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='General argument parser')
        self.general_parameters()
    
    def parse_args(self):
        return self.parser.parse_args()

    def general_parameters(self):
        self.parser.add_argument('--experiment', type=int, default=0,
                            help='Experiment number')
        self.parser.add_argument('--num-exps', type=int, default=1,
                                 help='number of simultaneous experiments to run in the parameter sweep (default 1)')                                                                                              
        self.parser.add_argument('--smoketest', default=False, action='store_true',
                                 help='Perform a smoke test (default False)')