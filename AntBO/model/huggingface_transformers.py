from pathlib import Path
import sys
import os
ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent)
sys.path.insert(0, ROOT_PROJECT)

from transformers import AutoTokenizer, \
                        AutoModelForMaskedLM, \
                        TrainingArguments, \
                        Trainer, AutoModel

from model.base import BaseModel
import torch
from torch.nn import Module
from dataloader.base import SequenceDataLoader
import math
import glob
import pandas as pd
import numpy as np
import os

class ProteinBERT(BaseModel, Module):
    def __init__(self, config, mode, use_cuda):
        BaseModel.__init__(self)
        Module.__init__(self)
        '''
        config: dictionary of model configuration
                path: path to bert model
                modelname: name of model default, prot_bert_bfd
                device: cuda or cpu
                epochs: number of epochs to finetune
                data:
                    path: Path to CDR3 data
                    test_size: fraction to use for testing
                    modelname: name of transformer tokenizer model default, prot_bert_bfd
                    antigens: list of antigens to finetune, if None use all antigens
                    batch_size : batch size of dataloader
                    nm_workers: Number of workers to use for dataloader
                    seed: seed for dataloader
        '''
        self.config = config
        self.use_cuda = use_cuda
        self.mode = mode
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.use_cuda else "cpu")
        self.n_gpu = torch.cuda.device_count()
        # self.device = 'cuda:2'
        # Reproducibility of Experiments
        torch.manual_seed(self.config['seed'])
        np.random.seed(self.config['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if not self.config['modelname']:
            self.config['modelname'] = 'prot_bert_bfd'
            self.config['data']['modelname'] = self.config['modelname']

        self.configure_BERT()

    def configure_BERT(self):
        self.tokenizer = AutoTokenizer.from_pretrained(f"{self.config['path']}/{self.config['modelname']}")
        if self.mode == 'train':
            self.train_args =TrainingArguments(
                                output_dir = f"{self.config['path']}/OutputFinetuneBERT{self.config['modelname']}",
                                num_train_epochs = self.config['epochs'],
                                per_device_train_batch_size = self.config['batch_size'],
                                per_device_eval_batch_size = self.config['batch_size'],
                                warmup_steps = self.config['warmup'],
                                weight_decay = self.config['weight_decay'],
                                logging_dir = f"{self.config['path']}/OutputFinetuneBERT{self.config['modelname']}/logs",
                                logging_steps =self.config['logsteps'],
                                do_train = True,
                                do_eval=True,
                                evaluation_strategy = "epoch",
                                gradient_accumulation_steps = 64,
                                run_name = f"finetune{self.config['modelname']}",
                                no_cuda = not self.use_cuda,
            )
            self.model = AutoModelForMaskedLM.from_pretrained(f"{self.config['path']}/{self.config['modelname']}")
        else:
            self.model = AutoModel.from_pretrained(f"{self.config['path']}/{self.config['modelname']}")

        self.model.to(self.device)

    def generate(self, out_len, nm_seq, input_seq="A E T C Z"):
        from transformers import AutoModel, pipeline
        self.model.eval()
        input_ids = torch.tensor(self.tokenizer.encode(input_seq, add_special_tokens=False))
        generate_seq = model.generate(
                        input_ids = input_ids,
                        max_length = out_len,
                        temperature = 1.0,
                        top_k = 0,
                        repetition_penalty = 1.0,
                        do_sample = True,
                        num_return_sequences = nm_seq
        )
        output_seq = ["".join(tokenizer.decode(seq)) for seq in generate_seq]
        with open(f"{self.path}/generated_sequences.txt", 'w') as f:
            for seq in output_seq:
                f.write(f"{seq}\n")

    def embedding(self, x, itern=0):
        if itern == 0:
            from transformers import pipeline
            self.pipeline = pipeline('feature-extraction', model=self.model, tokenizer=self.tokenizer, device=self.device)
        x = [" ".join(x_i) for x_i in x]
        seq = [re.sub(r"[UZOB],", "X", seq) for seq in x]
        representation = self.pipeline(seq)
        return representation

    def fit(self):
        dataset = SequenceDataLoader(self.config['data'], self.tokenizer)
        trainset, testset = dataset.StandardDataLoader()
        self.trainer = Trainer(
                        model = self.model,
                        args = self.train_args,
                        train_dataset = trainset,
                        eval_dataset = testset,
                    )
        self.trainer.train()
        eval_results = self.trainer.evaluate()
        print(f"Perplexity - {math.exp(eval_results['eval_loss']):.2f}")

import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser(add_help=True,
                                     description='Finetuning BERT on CDR3 sequence')
    parser.add_argument('--antigens', type=list, default=['1ADQ_A', '1FBI_X', '1HOD_C', '1NSN_S', '1OB1_C', '1WEJ_F', '2YPV_A', '3RAJ_A', '3VRL_C'], help='List of Antigen to perform BO')
    parser.add_argument('--use_cuda', type=bool, default=True, help='GPU Flag')
    parser.add_argument('--device_ids', type=list, default=['2', '3'], help='Cuda device to use')
    parser.add_argument('--mode', type=str, default='train', help='Use BERT in one of the three modes: train, generate or embedding')

    config = {'path': "/nfs/aiml/asif/ProtBERT",
              'modelname': 'prot_bert_bfd',
              'epochs': 10,
              'batch_size':320,
              'data': {'path': "/nfs/aiml/asif/CDRdata",
                        'modelname': None, 'test_size': 0.2,
                       'antibody': 'Murine', 'antigens': None,
                       'seed':42, 'return_energy': False,
                },
              'mode': 'train',
              'seed' : 42,
              'warmup': 1000,
              'weight_decay': 0.01,
              'logsteps': 200,

        }
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(id for id in args.device_ids)

    #config['data']['antigens'] = args.antigens
    # antigens = [antigen.strip().split()[1] for antigen in open(f"/nfs/aiml/asif/CDRdata/antigens.txt", 'r') if
    #             antigen != '\n']

    config['data']['antigens'] = args.antigens

    prot_bert = ProteinBERT(config, args.mode, use_cuda=True)
    if args.mode == 'train':
        prot_bert.fit()
    elif args.mode == 'generate':
        args.prot_bert.generate()
    elif args.mode == 'embedding':
        args.embedding()
    else:
        assert 0, f"{args.config['mode']} Not Implemented"