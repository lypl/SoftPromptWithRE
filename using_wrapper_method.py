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
from utils import InputExample, top_k_accuracy, apply_threshold, get_new_token

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
                 learning_rate: float = 3e-3,  warmup_proportion: float = 0.1, seed: int = 42,
                 marker_learning_rate=3e-5, marker_warmup_proportion=0.1, marker_save_model_dir=None,
                 marker_weight_decay=1e-2, marker_adam_epsilon=1e-8, marker_num_train_epoch=5, marker_train_batch_size=32,
                 marker_gradient_accumulation_steps=2, marker_max_grad_norm=1.0):
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

def get_wrong_examples(eval_data, outputs, all_topics):
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
        # to_write["top-k_res"]: List[(label, confidence), (), ()...]
        if "NA" not in to_write["example_info"]["label"] and "no_relation" not in to_write["example_info"]["label"] and to_write["example_info"]["label"] not in to_write["top-k_res"][0]:
            wrong_readable_results.append(to_write)
    return wrong_readable_results

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
    # wrapper.model.load_state_dict() 因为是用来测试初始化的结果，暂且不需要从某个checkpoint load
    wrapper.model.to(device)
    # outputs = wrapper.predict(eval_data, config.device, config.per_gpu_eval_batch_size, config.topk)
    micro_f1, f1_by_relation, outputs, all_topics = wrapper.evaluate_RE(eval_data, device, config.per_gpu_eval_batch_size, config.topk)
    
    # outputs:num_example * num_relation
    result = defaultdict(str)
    result['predictions'] = outputs
    result['experiment_info'] = experiment_info

    wrong_readable_results = get_wrong_examples(test_data, outputs, all_topics)

    result['wrong_readable_predictions'] = wrong_readable_results
    result['micro_f1'] = micro_f1
    result['f1_by_relation'] = f1_by_relation

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
                         config.check_step, config.marker_max_grad_norm, 
                         config.marker_gradient_accumulation_steps, 
                         topk=EvalConfig.topk)
    
    wrapper.model.load_state_dict(torch.load(os.path.join(config.marker_save_model_dir, "parameter.pkl")))
    test_micro_f1, test_f1_by_relation, outputs, topics = wrapper.evaluate_RE(test_data, device, EvalConfig.per_gpu_eval_batch_size, EvalConfig.topk) # evaluate()这里直接用RE的结果来查看tuning的结果
    wrong_examples = get_wrong_examples(test_data, outputs, topics)
    result = {
        "micro_f1": test_micro_f1,
        "f1_by_relation": test_f1_by_relation,
        "wrong_examples": wrong_examples,
    }
    return result

   
