import os
from argparse import Namespace

import weighted_retraining.weighted_retraining.expr.eq_grammar as G
from utils.utils_save import ROOT_PROJECT
from weighted_retraining.weighted_retraining import utils
from weighted_retraining.weighted_retraining.expr.equation_vae import EquationGrammarModelTorch
from weighted_retraining.weighted_retraining.expr.expr_data_pt import WeightedExprDataset
from weighted_retraining.weighted_retraining.expr.expr_dataset import get_filepath
from weighted_retraining.weighted_retraining.expr.expr_model_pt import EquationVaeTorch

MAX_LEN = 15


def instentiate_expr_datamodule() -> WeightedExprDataset:
    """ Create a WeightedExprDataset """

    ignore_percentile = 65
    dataset_path = os.path.join(ROOT_PROJECT, 'weighted_retraining/data/expr')
    data_seed = 0
    good_percentile = 5
    weight_type = 'rank'
    rank_weight_k = 1
    weight_quantile = None
    val_frac = .1
    property_key = 'scores'
    second_key = 'expr'
    batch_size = 128
    predict_target = False
    metric_loss = None

    hparams = Namespace()
    hparams.ignore_percentile = ignore_percentile
    hparams.data_seed = data_seed
    hparams.good_percentile = good_percentile
    hparams.weight_type = weight_type
    hparams.dataset_path = dataset_path
    hparams.rank_weight_k = rank_weight_k
    hparams.weight_quantile = weight_quantile
    hparams.val_frac = val_frac
    hparams.property_key = property_key
    hparams.second_key = second_key
    hparams.batch_size = batch_size
    hparams.predict_target = predict_target
    hparams.metric_loss = metric_loss

    hparams.dataset_path = get_filepath(hparams.ignore_percentile, hparams.dataset_path, hparams.data_seed,
                                        good_percentile=hparams.good_percentile)

    datamodule = WeightedExprDataset(hparams, utils.DataWeighter(hparams), add_channel=False)
    datamodule.setup()
    return datamodule


def instentiate_expr_model() -> EquationGrammarModelTorch:
    """ Instantiate an EquationGrammarModelTorch """
    data_info = G.gram.split('\n')

    latent_dim = 25
    beta_start = 1
    beta_final = 1
    beta_step = 1
    beta_step_freq = 1
    beta_warmup = 1

    hparams = Namespace()
    hparams.latent_dim = latent_dim
    hparams.beta_final = beta_final
    hparams.beta_start = beta_start
    hparams.beta_step = beta_step
    hparams.beta_step_freq = beta_step_freq
    hparams.beta_warmup = beta_warmup

    model = EquationGrammarModelTorch(
        EquationVaeTorch(hparams, len(data_info), MAX_LEN),
    )

    return model
