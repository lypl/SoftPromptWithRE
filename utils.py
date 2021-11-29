""" This code include some data structs and useful function 
    utils.py不应该调用同目录下别的包。否则形成循环调用，报错：import error: cannot import name 'relation'
"""

import copy
import json
import time
import random
from logging import Logger
import os
import pickle
from collections import defaultdict
from typing import Dict, List, Optional, Union

import torch
import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support

from torch.nn import functional as F
from torch.utils.data import Dataset


class relation(object):
    """A set of information about a relation, 在NLI这条实验路径上则仅加载其template_file; 用于prompt tuning时加载data_file、template等
    """

    def __init__(self, label: str, ID=None, name =None, meta=None, templates=None):
        """
        Create new relation.

        :param label: the relation 
        :param ID: idx for TACRED
        :param name: idx for TREX-1p/2p
        :param templates: the template list recording to this relation
        :param meta: an optional dictionary to store arbitrary meta information
        """
        self.ID: int = ID # "rel2id" in dataset (TACRED:[0, 40] ) 
        self.name: str = name # for TREX-1p:P19/P20/P413...
        self.label: str = label # relation's str e.g. "place of birth" / "per:alternate_names"
        # self.template_file: str = template_file
        self.meta = meta if meta else {}
        self.templates: List[str] = templates
        # self.prompts = prompts # {prompt_type: List[str]}

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serialize this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self, fp):
        """Serialize this instance to a JSON string."""
        return json.dump(self.to_dict(), indent=4, sort_keys=True,fp=fp) + "\n"

    @staticmethod
    def load_relations(path: str) -> List['relation']:
        """Load a set of relations from a file"""
        with open(path, 'rb') as fh:
            return pickle.load(fh)

    @staticmethod
    def save_relations(relations: List['relation'], path: str) -> None:
        """Save a set of relations to a file"""
        with open(path, 'wb') as fh:
            pickle.dump(relations, fh)
    

    def __repr__(self):
        return str(self.to_json_string())


class InputExample(object):
    """A raw input example consisting of data use for NLI"""

    def __init__(self, idx, subj, obj, ori_context, context, label, train_label=None, pair_type=None, logits: List[float] = None, meta: Optional[Dict] = None):
        """
        Create a new InputExample.

        :param subj: first entity
        :param obj: second entity *order-sensitive
        :param context: a premise with subj and obj
        :param label: the relation between subj and obj
        :param logits: an optional list of per-class logits **logits就是不必担心值在0和1之间、各种情况相加等于一，但是又可以被容易地转换成那种格式的表示概率的值
        :param meta: an optional dictionary to store arbitrary meta information 数据集中除必要信息外的附加信息
        :param idx: an numeric index -- must have
        """
        # coarse attribution
        self.subj: str = subj
        self.obj: str = obj
        self.ori_context: List = ori_context
        self.context: str = context
        self.pair_type: List[str] = pair_type # type of subj and obj
        self.label: str = label # relation
        self.logits = logits  # 方便之后输出所有example的信息和结果
        self.idx = idx
        self.meta = meta if meta else {}

        # attributions to construct InputFeature
       
        # for giving tokenizers
        self.raw_texts_to_tokenizer: List[str] = None # Dict{template_file_name(template_construct_method): List[contxt+[sep]+hypothesis]} 取消，NLI路径上不需要自动观察哪个relation的template好，这个可以将所有结果获得之后再手动观察；仅需要观察哪种template构造方法获得的所有relation的template好，所以实验设置命令行参数指示从哪个template文件（方法）得到所有raw_texts即可
        self.train_label: List[int]  = train_label # NLImodel 's classifier的分类结果（3种）
        

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serialize this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serialize this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @staticmethod
    def load_examples(path: str) -> List['InputExample']:
        """Load a set of input examples from a file"""
        with open(path, 'rb') as fh:
            return pickle.load(fh)

    @staticmethod
    def save_examples(examples: List['InputExample'], path: str) -> None:
        """Save a set of input examples to a file"""
        with open(path, 'wb') as fh:
            pickle.dump(examples, fh)


class InputFeatures(object):
    """A set of numeric features obtained from an :class:`InputExample`"""

    def __init__(self, corresponce_to_InputExample_idx: int, input_ids: torch.tensor, attention_mask: torch.tensor, label: torch.tensor, token_type_ids: torch.tensor = None,  logits=None, train_label = None,
                 meta: Optional[Dict] = None):
        """
        Create new InputFeatures.

        :param input_ids: the input ids corresponding to the original text or text sequence
        :param attention_mask: an attention mask, with 0 = no attention, 1 = attention
        :param token_type_ids: segment ids as used by BERT
        :param label: the label
        :param mlm_labels: an optional sequence of labels used for auxiliary language modeling
        :param logits: an optional sequence of per-class logits
        :param meta: an optional dictionary to store arbitrary meta information
        :param corresponce_to_InputExample_idx: an optional numeric index
        """
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label
        self.train_label = train_label
        # self.mlm_labels = mlm_labels
        self.logits = logits
        self.corresponce_to_InputExample_idx = corresponce_to_InputExample_idx
        self.meta = meta if meta else {}

    def __repr__(self):
        return str(self.to_json_string())

    def pretty_print(self, tokenizer):
        return f'input_ids         = {tokenizer.convert_ids_to_tokens(self.input_ids)}\n' + \
               f'attention_mask    = {self.attention_mask}\n' + \
               f'token_type_ids    = {self.token_type_ids}\n' + \
               f'logits            = {self.logits}\n' + \
               f'label             = {self.label}'
               #    f'mlm_labels        = {self.mlm_labels}\n' + \

    def to_dict(self):
        """Serialize this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serialize this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"



class DictDataset(Dataset):
    """A dataset of tensors that uses a dictionary for key-value mappings"""

    def __init__(self, **tensors):
        tensors.values()

        assert all(next(iter(tensors.values())).size(0) == tensor.size(0) for tensor in tensors.values())
        self.tensors = tensors

    def __getitem__(self, index):
        return {key: tensor[index] for key, tensor in self.tensors.items()}

    def __len__(self):
        return next(iter(self.tensors.values())).size(0)


def np_softmax(x, dim=-1):
    e = np.exp(x)
    return e / np.sum(e, axis=dim, keepdims=True)


def np_sigmoid(x, dim=-1):
    return 1 / (1 + np.exp(-x))


def save_result_for_a_experiment(content, data_dir, exprement_name, for_tuning=False):
    data_dir = os.path.join(data_dir, exprement_name)
    if for_tuning:
        path = os.path.join(os.getcwd(), "wrong_samples", "wrong_examples.txt")
        with open(path, 'a', encoding='UTF-8') as f:
            f.write("before tuning and after initialize results: ")
            f.write("\n\n")
            f.write(json.dumps(content['scores'], sort_keys=True, indent=4, separators=(',', ': ')))
            f.write("\n\n")
        return
    
    preds_data_path = os.path.join(data_dir, "preds.txt")
    metrics_result_data_path = os.path.join(data_dir, "results.txt")
    np.savetxt(preds_data_path, content['predictions'], fmt='%f', delimiter=', ')
    with open(metrics_result_data_path, 'a', encoding='UTF-8') as f:
        # f.write(json.dumps(content['readable_predictions'], sort_keys=True, indent=4, separators=(',', ': ')))
        f.write("\n\n")
        f.write(json.dumps(content['scores'], sort_keys=True, indent=4, separators=(',', ': ')))

    now = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())
    info_str = ""
    for k, v in content["experiment_info"].items():
        info_str += "_"
        if k == "epoch_num":
            info_str += "epoch_"
        info_str += str(v)
        
    file_name = "wrong_examples_" + now + info_str + ".txt"
           
    wrong_examples_path = os.path.join(os.getcwd(), "wrong_samples", file_name) # hard code 
    with open(wrong_examples_path, 'a', encoding='UTF-8') as f:
        f.write(json.dumps(content['wrong_readable_predictions'], sort_keys=True, indent=4, separators=(',', ': ')))
        f.write("\n\n")
        f.write(json.dumps(content['wrong_label_cnt'], sort_keys=True, indent=4, separators=(',', ': ')))
        f.write("\n\n")
        f.write(json.dumps(content['scores'], sort_keys=True, indent=4, separators=(',', ': ')))


def f1_score_(labels, preds, n_labels=42):
    return f1_score(labels, preds, labels=list(range(1, n_labels)), average="micro")


def precision_recall_fscore_(labels, preds, n_labels=42):
    p, r, f, _ = precision_recall_fscore_support(labels, preds, labels=list(range(1, n_labels)), average="micro")
    return p, r, f


def apply_threshold(output, threshold=0.0, ignore_negative_prediction=True):
    """Applies a threshold to determine whether is a relation or not"""
    output_ = output.copy()
    if ignore_negative_prediction:
        output_[:, 0] = 0.0
    activations = (output_ >= threshold).sum(-1).astype(np.int)
    output_[activations == 0, 0] = 1.00

    return output_.argmax(-1)


def find_optimal_threshold(labels, output, granularity=1000, metric=f1_score_):
    thresholds = np.linspace(0, 1, granularity)
    values = []
    for t in thresholds:
        preds = apply_threshold(output, threshold=t)
        values.append(metric(labels, preds))

    best_metric_id = np.argmax(values)
    best_threshold = thresholds[best_metric_id]

    return best_threshold, values[best_metric_id]

def top_k_accuracy(output, labels, k=1):
    # relation为NA的example是没有被统计进去的，所有如果所有example都是NA的关系，则会报除零错l_sum = 0
    preds = np.argsort(output)[:, ::-1][:, :k] # 概率最高的前k个relation的id
    ret = 0
    l_sum = 0
    for l, p in zip(labels, preds):
        if l != 13:
            # NA的id
            l_sum += 1
        if l in p:
            ret += 1
    if l_sum == 0:
        print("all examples' relation are 'NA'. The accuracy is: ")
        print(ret / labels.shape[0])
        return ret / labels.shape[0]
    return ret / l_sum
    # return sum(l in p and l > 0 for l, p in zip(labels, preds)) / (labels > 0).sum()

def load_vocab(vocab_filename):
    with open(vocab_filename, "r") as f:
        lines = f.readlines()
    vocab = [x.strip() for x in lines]
    return vocab

def get_new_token(vid, max_num_relvec):
    assert(vid > 0 and vid <= max_num_relvec)
    return '[V%d]'%(vid)

def get_marked_sentence(ori_context, subj_st, subj_ed, obj_st, obj_ed, subj_type, obj_type, marker_name):
        marker_types = ["entity_marker", "entity_marker_punct", "typed_marker", "typed_marker_punct"]
        ctx = []
        new_tokens = []
        if marker_name == marker_types[0]:
            for i, token in enumerate(ori_context):
                if i == subj_st:
                    ctx.append("[E1]")
                    new_tokens.append("[E1]")
                if i == subj_ed:
                    ctx.append("[/E1]")
                    new_tokens.append("[/E1]")
                if i == obj_st:
                    ctx.append("[E2]")
                    new_tokens.append("[E2]")
                if i == obj_ed:
                    ctx.append("[/E2]")
                    new_tokens.append("[/E2]")
                ctx.append(token)
        elif marker_name == marker_types[1]:
            for i, token in enumerate(ori_context):
                if i == subj_st:
                    ctx.append("@")
                if i == subj_ed:
                    ctx.append("@")
                if i == obj_st:
                    ctx.append("#")
                if i == obj_ed:
                    ctx.append("#")
                ctx.append(token)
        elif marker_name == marker_types[2]:
            for i, token in enumerate(ori_context):
                if i == subj_st:
                    res = "[E1:{}]".format(subj_type)
                    ctx.append(res)
                    if res not in new_tokens:
                        new_tokens.append(res)
                if i == subj_ed:
                    res = "[/E1:{}]".format(subj_type)
                    ctx.append(res)
                    if res not in new_tokens:
                        new_tokens.append(res)
                if i == obj_st:
                    res = "[E2:{}]".format(obj_type)
                    ctx.append(res)
                    if res not in new_tokens:
                        new_tokens.append(res)
                if i == obj_ed:
                    res = "[/E2:{}]".format(obj_type)
                    ctx.append(res)
                    if res not in new_tokens:
                        new_tokens.append(res)
                ctx.append(token)
        elif marker_name == marker_types[3]:
            # 注意实体类型都是小写
            for i, token in enumerate(ori_context):
                if i == subj_st:
                    res = "@ * {}".format(subj_type)
                    ctx.append(res)
                    if res not in new_tokens:
                        new_tokens.append(res)
                if i == subj_ed:
                    res = "* @"
                    ctx.append(res)
                if i == obj_st:
                    res = "# ^ {}".format(obj_type)
                    ctx.append(res)
                    if res not in new_tokens:
                        new_tokens.append(res)
                if i == obj_ed:
                    res = "^ #"
                    ctx.append(res)
                ctx.append(token)
        return ctx, new_tokens
        
