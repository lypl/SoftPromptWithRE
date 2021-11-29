import ast
import json
from pickle import load
import random
from re import template
import statistics
from abc import ABC
import os
from collections import defaultdict
from copy import deepcopy
from typing import List, Dict

import torch
import wandb
import logging
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from NLImodelsWrapper import NLIRelationWrapper
from utils import InputExample, top_k_accuracy, find_optimal_threshold, apply_threshold, get_new_token

logger = logging.getLogger(__name__)

class NLIConfig(ABC):
    """Abstract class for a PET configuration that can be saved to and loaded from a json file."""

    def __repr__(self):
        return repr(self.__dict__)

    def save(self, path: str):
        """Save this config to a file."""
        with open(path, 'w', encoding='utf8') as fh:
            json.dump(self.__dict__, fh)

    @classmethod
    def load(cls, path: str):
        """Load a config from a file."""
        cfg = cls.__new__(cls)
        with open(path, 'r', encoding='utf8') as fh:
            cfg.__dict__ = json.load(fh)
        return cfg


class NLITrainConfig(NLIConfig):
    """Configuration for training a model."""

    def __init__(self, device, save_optiprompt_dir: str, prompt_type, train_batch_size: int = 8, eval_batch_size: int = 8, eval_step: int = 10,
                 max_steps: int = -1, num_train_epoch: int = 3, gradient_accumulation_steps: int = 1, check_step: int = 10,
                 learning_rate: float = 3e-3,  warmup_proportion: float = 0.1, seed: int = 42):
        """
        Create a new training config.

        """
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.max_steps = max_steps
        self.num_train_epoch = num_train_epoch
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.warmup_proportion = warmup_proportion
        self.eval_step = eval_step
        self.save_optiprompt_dir = save_optiprompt_dir
        self.check_step = check_step
        self.seed = seed
        self.device = device
        self.prompt_type = prompt_type
        # self.max_num_relvec = max_num_relvec
        # self.relvec_construct_mode = relvec_construct_mode
        # self.template_type = template_type
        self.marker_learning_rate = marker_learning_rate
        self.marker_warmup_proportion = marker_warmup_proportion
        self.marker_save_model_dir = marker_save_model_dir
        self.marker_weight_decay = marker_weight_decay
        self.marker_adam_epsilon = marker_adam_epsilon
        self.marker_num_train_epoch = marker_num_train_epoch
        self.marker_train_batch_size = marker_train_batch_size
        self.marker_gradient_accumulation_steps = marker_gradient_accumulation_steps
        self.marker_max_grad_norm = marker_max_grad_norm
        
        



class NLIEvalConfig(NLIConfig):
    """Configuration for evaluating a model."""

    def __init__(self, device: str, topk: int, n_gpu: int = 1, per_gpu_eval_batch_size: int = 8, metrics: List[str] = None):
        """
        Create a new evaluation config.

        :param device: the device to use ('cpu' or 'gpu')
        :param n_gpu: the number of gpus to use
        :param per_gpu_eval_batch_size: the number of evaluation examples per batch and gpu
        :param metrics: the evaluation metrics to use (default: accuracy only)
        :param decoding_strategy: the decoding strategy for PET with multiple masks ('default', 'ltr', or 'parallel')
        :param priming: whether to use priming
        """
        self.device = device
        # self.n_gpu = n_gpu
        self.per_gpu_eval_batch_size = per_gpu_eval_batch_size
        self.metrics = metrics
        self.topk = topk
        self.use_threshold = None


def NLIforward(wrapper: NLIRelationWrapper, eval_data: List[InputExample], config: NLIEvalConfig, experiment_info: Dict = None) -> Dict:
    """
    Evaluate a NLImodel.

    :param model: the model to evaluate
    :param eval_data: the examples for evaluation
    :param config: the evaluation config
    :param priming_data: an optional list of priming data to use
    :return: a dictionary containing the model's logits, predictions and (if any metrics are given) scores
    """

    metrics = config.metrics if config.metrics else ['precision']
    device = torch.device(config.device if config.device else "cuda" if torch.cuda.is_available() else "cpu")

    wrapper.model.to(device)
    # outputs = wrapper.predict(eval_data, config.device, config.per_gpu_eval_batch_size, config.topk)
    outputs = wrapper.eval(eval_data, config.device, config.per_gpu_eval_batch_size)
    
    # outputs:num_example * num_relation
    result = defaultdict(str)
    result['predictions'] = outputs
    result['experiment_info'] = experiment_info

    # readable_predictions
    all_topics = wrapper.predict(outputs, config.topk)
    readable_results = []
    wrong_readable_results = []
    wrong_label_cnt = {}
    for i, res in enumerate(all_topics):
        candidate_label_id = [] # List[tuple]
        prob = outputs[i]
        # print("prob's dim:")
        # print(prob.shape)
        _i = 0
        for x in prob:
            if x > 0.0:
                candidate_label_id.append((_i, x))
            _i += 1
        to_write = {
            "example_info": {
                "subj": eval_data[i].subj,
                "obj": eval_data[i].obj,
                "context": eval_data[i].context,
                "label": eval_data[i].label
            },
            "top-k_res": res,
            "candidate_label_id": candidate_label_id

        }
        readable_results.append(to_write)
        # to_write["top-k_res"]: List[(label, confidence), (), ()...]
        if "NA" not in to_write["example_info"]["label"] and "no_relation" not in to_write["example_info"]["label"] and to_write["example_info"]["label"] not in to_write["top-k_res"][0]:
            wrong_readable_results.append(to_write)
            if to_write["example_info"]["label"] not in wrong_label_cnt.keys():
                wrong_label_cnt[to_write["example_info"]["label"]] = 1
            else:
                wrong_label_cnt[to_write["example_info"]["label"]] += 1


    result['readable_predictions'] = readable_results
    result['wrong_readable_predictions'] = wrong_readable_results
    result['wrong_label_cnt'] = wrong_label_cnt

    labels = []
    for data in eval_data:
        labels.append(wrapper.rel2id[data.label])
    # if config.use_threshold is None:
    #     optimal_threshold, _ = find_optimal_threshold(labels, outputs)
    #     output_ = apply_threshold(outputs, threshold=optimal_threshold)
    # else:
    #     output_ = outputs.argmax(-1)
    output_ = outputs.argmax(-1)
    pre, rec, f1, _ = precision_recall_fscore_support(labels, output_, average="micro", labels=list(range(1, len(labels))))
    scores = {}
    for metric in metrics:
        if metric == "precision":
            scores[metric] = pre
        elif metric == "recall":
            scores[metric] = rec
        elif metric == "f1-score":
            scores[metric] = f1
        elif metric == "top-1":
            scores[metric] = top_k_accuracy(outputs, labels, k=1)
        elif metric == "top-3":
            scores[metric] = top_k_accuracy(outputs, labels, k=3)
        elif metric == "top-5":
            scores[metric] = top_k_accuracy(outputs, labels, k=5)
        elif metric == "top-curve":
            scores[metric] = [top_k_accuracy(outputs, labels, k=i) for i in range(1, len(labels) + 1)]
        else:
            raise ValueError(f"Metric '{metric}' not implemented")

    result['scores'] = scores
    
    return result


def prompt_tuning(wrapper: NLIRelationWrapper, train_data: List[InputExample], dev_data: List[InputExample], test_data: List[InputExample], 
                  config: NLITrainConfig, EvalConfig: NLIEvalConfig) -> Dict:
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    device = torch.device(config.device if config.device else "cuda" if torch.cuda.is_available() else "cpu")
    wrapper.model.to(device) # !!!
    wandb.watch(wrapper.model)

    # 用prompt重写所有relation的template

    # 这里不需要，因为首次需要用manT的embedding走一次并存起来，所以这里写在wrapper.tuning_train()里面
    # wrapper.save_optiprompt()

    # tuning relvec
    wrapper.tuning_train(train_data, dev_data, device, config.learning_rate, config.eval_step, 
                         config.prompt_type, config.warmup_proportion, 
                         config.save_optiprompt_dir, config.max_steps, config.eval_batch_size, 
                         config.num_train_epoch, 
                         config.train_batch_size, 
                         config.check_step, config.gradient_accumulation_steps, topk=EvalConfig.topk)
    
    # 该wrappertuning后的model_parameters 已经保存了吗？⭐
    # res_wrapper = load_optiprompt(config.save_optiprompt_dir, )
    # result = NLIforward(res_wrapper, test_data, EvalConfig)
    # return result

def marker_tuning(wrapper: NLIRelationWrapper, train_data: List[InputExample], dev_data: List[InputExample], test_data: List[InputExample], 
                  config: NLITrainConfig, EvalConfig: NLIEvalConfig) -> Dict:
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    device = torch.device(config.device if config.device else "cuda" if torch.cuda.is_available() else "cpu")
    wrapper.model.to(device) # !!!
    wandb.watch(wrapper.model)
    # dev_data是用来model selection的，test_data最终测试结果，尝试方法是否work时暂时不需要做test_data的部分，之后补上
    wrapper.marker_tuning_train(train_data, dev_data, device, config.marker_learning_rate,  
                         config.marker_warmup_proportion, 
                         config.marker_save_model_dir, config.eval_batch_size, 
                         config.marker_weight_decay,
                         config.marker_adam_epsilon,
                         config.marker_num_train_epoch, 
                         config.marker_train_batch_size, 
                         config.check_step, config.marker_gradient_accumulation_steps, 
                         config.marker_max_grad_norm, topk=EvalConfig.topk)
    
    # 该wrappertuning后的model_parameters 已经保存了吗？⭐
    """
    train_data: List[InputExample], dev_data: List[InputExample], device, learning_rate: float, 
                     warmup_proportion: float, save_marker_token_data_dir: str, eval_batch_size: int, weight_decay: float,
                     adam_epsilon, num_train_epochs: int, train_batch_size: int, check_step: int, gradient_accumulation_steps: int = 1, 
                     max_grad_norm, topk: int = 1
    """

