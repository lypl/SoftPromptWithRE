""" This code define config class and wrap models' function for NLI """

import json
import random
import jsonpickle
import os
import wandb
import sys
from collections import defaultdict
from typing import List, Dict, Optional

from numpy.core.fromnumeric import shape
from numpy.lib.shape_base import expand_dims

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from tqdm import trange, tqdm

from load_relation import get_relations
from transformers import AdamW, get_linear_schedule_with_warmup, PreTrainedTokenizer, AutoConfig, AutoTokenizer, \
                         AutoModelForSequenceClassification, RobertaForSequenceClassification, AlbertForSequenceClassification, BartForSequenceClassification, \
                         DebertaForSequenceClassification
from transformers import __version__ as transformers_version

import logging

from utils import relation, InputExample, InputFeatures, DictDataset, np_softmax, load_vocab, top_k_accuracy, get_new_token
from load_relation import  get_relations


NLIWRAPPER = "sequence_classifier"
WRAPPER_TYPE = [NLIWRAPPER]
# 一种wrapper_type(如NLImodel 属于序列分类sequence_classifier)指导 使用哪种processor和哪种transformers中提供的model_architecture(因为这两者都需要model信息，所以需要用wrapper_type统一)


MODEL_CLASSES = {
    'roberta': {
        'config': AutoConfig,
        'tokenizer': AutoTokenizer,
        NLIWRAPPER: AutoModelForSequenceClassification, # for construct model_architecture in transformers
        # 'base_model': RobertaForSequenceClassification.roberta
    },
    'albert': {
        'config': AutoConfig,
        'tokenizer': AutoTokenizer, 
        NLIWRAPPER: AutoModelForSequenceClassification,
        # 'base_model': AlbertForSequenceClassification.albert
    },
    'bart': {
        'config': AutoConfig,
        'tokenizer': AutoTokenizer,
        NLIWRAPPER: AutoModelForSequenceClassification,
        # 'base_model': BartForSequenceClassification.bart
    },
    'deberta': {
        'config': AutoConfig,
        'tokenizer': AutoTokenizer, 
        NLIWRAPPER: AutoModelForSequenceClassification,
        # 'base_model': DebertaForSequenceClassification.deberta
    },
}
CONFIG_NAME = 'NLIWrapper_config.json'

logger = logging.getLogger(__name__)



def pause():
    programPause = input("Press the <ENTER> key to continue...")
class NLIWrapperConfig(object):
    """A configuration for a :class:`NLIModelWrapper`."""

    def __init__(self, model_type: str, model_name_or_path: str, wrapper_type: str, dataset_name: str, max_seq_length: int, max_num_relvec: int,
                 relations_data_dir: str, relations_data_name: str, use_cuda: bool = True, verbose: bool = False, prompt_type: str = None, relvec_construct_mode: str="from_manT",
                 negative_threshold: float = 0.8, negative_idx: int = 13, max_activations = np.inf, valid_conditions=None, use_rel_embedding=False, 
                 rel_id2embeddings_id_path: str = None, rel_embeddings_path: str = None, use_marker=False, marker_position=None, marker_name=None):
        """
        Create a new config.

        :param model_type: the model type (e.g., 'bert', 'roberta', 'albert')
        :param model_name_or_path: the model name (e.g., 'roberta-large') or path to a pretrained model
        :param wrapper_type: the wrapper type (one of 'mlm', 'plm' and 'sequence_classifier') # 本实验只有sequence_classifier，即NLIWRAPPER
        :param dataset_name: the task to solve
        :param max_seq_length: the maximum number of tokens in a sequence
        :param label_list: the all relations in natural language
        :param use_cuda: whether use GPU
        :param verbose: 没用？
        :param negative_threshold: 没调这个函数所以没用
        :param max_activations: 没调这个函数所以没用
        :param negative_idx: 没调这个函数所以没用
        :param valid_conditions: 在a2t.tacred.py,调用的.py需要复制在那个文件
        """
        self.model_type = model_type
        self.model_name_or_path = model_name_or_path
        self.wrapper_type = wrapper_type
        self.dataset_name = dataset_name 
        self.max_seq_length = max_seq_length
        # self.label_list = label_list # 不需要符合input的顺序
        self.relations_data_dir = relations_data_dir 
        self.relations_data_name = relations_data_name
        self.use_cuda = use_cuda
        self.verbose = verbose
        self.negative_threshold = negative_threshold
        if dataset_name == "retacred":
            negative_idx = 0
        self.negative_idx = negative_idx
        self.max_activations = max_activations
        self.valid_conditions = valid_conditions
        self.label2id = {
            "contradiction": 0,
            "neutral": 1,
            "entailment": 2
        } # 这个是为了保险，从model主页上的config里复制的

        # for PT
        self.relvec_construct_mode = relvec_construct_mode
        self.prompt_type = prompt_type
        self.max_num_relvec = max_num_relvec

        # self.construct_NLI_labeled_data = construct_NLI_labeled_data

        self.use_rel_embedding = use_rel_embedding
        self.rel_id2embeddings_id_path = rel_id2embeddings_id_path
        self.rel_embeddings_path = rel_embeddings_path

        self.use_marker = use_marker
        self.marker_position = marker_position
        self.marker_name = marker_name
        
class NLIRelationWrapper():
    """A wrapper around a Transformer-based NLI language model."""
    def __init__(self, config: NLIWrapperConfig):
        """Create a new wrapper from the given config."""
        self.config = config
        config_class = MODEL_CLASSES[self.config.model_type]['config'] # 用于获取model自身的config.json, 以得到label2id
        tokenizer_class = MODEL_CLASSES[self.config.model_type]['tokenizer']
        model_class = MODEL_CLASSES[self.config.model_type][self.config.wrapper_type]

        model_config = config_class.from_pretrained(config.model_name_or_path) 
        self.tokenizer = tokenizer_class.from_pretrained(config.model_name_or_path)  
        self.model = model_class.from_pretrained(config.model_name_or_path)
        if self.config.model_type == 'roberta':
            self.base_model = self.model.roberta
        elif self.config.model_type == 'albert':
            self.base_model = self.model.albert
        elif self.config.model_type == 'deberta':
            self.base_model = self.model.deberta
        elif self.config.model_type == 'bart':
            self.base_model = self.model.bart

        self.marker_types = ["entity_mask", "entity_marker", "entity_marker_punct", "typed_marker", "typed_marker_punct"]
        self.all_relations = get_relations(self.config.relations_data_dir, self.config.relations_data_name) # 含有'NA'关系
        
        # 若为从manT初始化得到template，则在model的embeding层对应初始化
        if self.config.prompt_type == "relvec" and self.config.relvec_construct_mode == "from_manT":
            self.tot_new_tokens = 0

            # 得到original_vocab_size以save_optiPrompt
            self.original_vocab_size = len(list(self.tokenizer.get_vocab()))
            # 将[V1]~[V_max]加进vocab

            if self.config.use_rel_embedding:
                assert(self.config.rel_id2embeddings_id_path != None and self.config.rel_embeddings_path != None), "rel_id2embeddings_id_path or rel_embeddings_path == None"
                self.rel_embeddings = np.load(self.config.rel_embeddings_path)
                with open(self.config.rel_id2embeddings_id_path, 'r', encoding='utf-8') as f:
                    self.rel_embedddings_idx = json.load(f)

            new_tokens = [get_new_token(i+1, self.config.max_num_relvec) for i in range(self.config.max_num_relvec)]
            self.tokenizer.add_tokens(new_tokens)
            ebd = self.model.resize_token_embeddings(len(self.tokenizer))
            logger.info('# vocab after adding new tokens: %d'%len(self.tokenizer))

            def assign_embedding(new_token, token):
                """
                assign the embedding of token to new_token
                """
                logger.info('Tie embeddings of tokens: (%s, %s)'%(new_token, token))
                id_a = self.tokenizer.convert_tokens_to_ids([new_token])[0]
                id_b = self.tokenizer.convert_tokens_to_ids([token])[0]
                with torch.no_grad():
                    # print(new_token + "before:")
                    # print(self.base_model.embeddings.word_embeddings.weight[id_a])
                    self.base_model.embeddings.word_embeddings.weight[id_a] = self.base_model.embeddings.word_embeddings.weight[id_b].detach().clone()
                    # print(new_token + "after:")
                    # print(self.base_model.embeddings.word_embeddings.weight[id_a])
            self.model.eval() # 需要有这个才能进torch.no_grad()?
            for i in range(len(self.all_relations)):
                if self.all_relations[i].label == "NA" or self.all_relations[i].label == "no_relation":
                    # NA 不需要templates
                    continue
                now_templates = self.all_relations[i].templates
                new_pts = []
                for tpl in now_templates:
                    prompt = []
                    for word in tpl.split():
                        if word in ['{subj}', '{obj}']:
                            prompt.append(word)
                        else:
                            tokens = self.tokenizer.tokenize(' ' + word)
                            for token in tokens:
                                self.tot_new_tokens += 1 # [V1]开始
                                prompt.append(get_new_token(self.tot_new_tokens, self.config.max_num_relvec))
                                assign_embedding(get_new_token(self.tot_new_tokens, self.config.max_num_relvec), token) 
                    
                    # if self.config.use_rel_embedding == True:
                    #     # not use this
                    #     for idx in self.rel_embedddings_idx[str(self.all_relations[i].ID)]:
                    #         prompt.append(get_new_token(self.tot_new_tokens, self.config.max_num_relvec)) 
                    #         # 将该[V]的embedding赋值成对应的rel_embedding
                    #         with torch.no_grad():
                    #             self.base_model.embeddings.word_embeddings.weight[self.tot_new_tokens] = torch.Tensor(self.rel_embeddings[idx])
                    #         self.tot_new_tokens += 1
                    pt =  ' '.join(prompt)
                    print(tpl)
                    print(pt)
                    new_pts.append(pt)
                self.all_relations[i].templates = new_pts

        # pause()        

        # self.train_NL_data = {} # 自然语言的输入文本
        # self.train_data_cnt = 0

        self.template_list = [] # all_templates
        self.relation_name_list =  [] # relation_dataset共有多少个relation(in NL==>r.label) List[realtion.label]
        self.template_mapping = defaultdict(list) # Dict[relation(str,in NL): List[template(str, in NL)]]
        self.rel2id = {}
        for r in self.all_relations:
            self.template_list.extend(r.templates)
            self.relation_name_list.append(r.label)
            self.template_mapping[r.label] = r.templates
            self.rel2id[r.label] = r.ID

        # print("len(relation_name_list):")
        # print(len(self.relation_name_list))

        self.tot_new_tokens = 0

        self.ent_pos = model_config.label2id.get("ENTAILMENT", self.config.label2id.get("entailment", None))
        if self.ent_pos is None:
            raise ValueError("The model config must contain ENTAILMENT label in the label2id dict.")
        else:
            self.ent_pos = int(self.ent_pos)

        # feature 应用的template对应的relation在self.template_mapping_reverse查
        self.template_mapping_reverse = defaultdict(list) # Dict[template(str, in NL), List[该template对应的relation(str, in NL)]] 
        for key, value in self.template_mapping.items():
            for v in value:
                self.template_mapping_reverse[v].append(key)

        
        self.template2id = {t: i for i, t in enumerate(self.template_list)} # Dict[template(str): idx(int)]
        self.mapping = defaultdict(list) # Dict[relation(str, in NL): List[template_id(int)]]
        for key, value in self.template_mapping.items():
            if key == 'NA' or key == "no_relation":
                continue
            self.mapping[key].extend([self.template2id[v] for v in value])
        

        self.negative_threshold = self.config.negative_threshold
        self.negative_idx = self.config.negative_idx
        self.max_activations = self.config.max_activations
        self.n_rel = len(self.relation_name_list)

        if self.config.valid_conditions:
            self.valid_conditions = {} # condition：[0, 0, 1, 0, 1, 0] (num_rel)该relation是否被每个relation允许
            rel2id = self.rel2id
            for relation, conditions in self.config.valid_conditions.items():
                if relation not in rel2id:
                    continue
                for condition in conditions:
                    if condition not in self.valid_conditions:
                        # 该condition 一定对NA 适用
                        self.valid_conditions[condition] = np.zeros(self.n_rel)
                        assert(self.config.dataset_name in ["TACRED", "retacred"])
                        if self.config.dataset_name == "TACRED":
                            self.valid_conditions[condition][rel2id["NA"]] = 1.0
                        elif self.config.dataset_name == "retacred":
                            self.valid_conditions[condition][rel2id["no_relation"]] = 1.0
                    self.valid_conditions[condition][rel2id[relation]] = 1.0

        else:
            self.valid_conditions = None

        def idx2rel(idx):
            return self.relation_name_list[idx]

        self.idx2label = np.vectorize(idx2rel) # idx(int)下标：relation(str, in NL) 


    
    def construct_list_of_featrues_with_batched_inputids(self, batchEncoding, example: InputExample)->List[InputFeatures]:
        features = []
        # batchEncoding: input_ids[[], [], ...(共feature_num个), []]
        # print(self.tokenizer)
        # print(batchEncoding.keys())
        for idx in range(len(batchEncoding["input_ids"])):
            i = batchEncoding["input_ids"][idx]
            a = batchEncoding["attention_mask"][idx]
            if "token_type_ids" not in batchEncoding.keys():
                t = None
            else:
                t = batchEncoding["token_type_ids"][idx]
            label = self.rel2id[example.label]

            train_label = example.train_label[idx] if example.train_label else [-1]
            logits = example.logits  if example.logits else [-999999]
            features.append(InputFeatures(corresponce_to_InputExample_idx=example.idx, input_ids=i, token_type_ids=t, attention_mask=a, label=label, logits=logits, train_label=train_label))
        return features

    
    
    def example_processor(self, example: InputExample, mode: int) -> List[InputFeatures]:
        """ complete examples (get all raw_texts_to_tokenizer) and get List[InputFeatures]"""
        example.raw_texts_to_tokenizer = []
        features = []
        # 加入marker
        if self.config.use_marker:
            assert((self.config.marker_name is not None) and (self.config.marker_position is not None))
            assert(self.config.marker_name in self.marker_types and self.config.marker_position in ["context", "both"])
            examples.context = self.get_marked_sentence(example, self.config.marker_position)[0]
        if mode == 0:
            # 不tuning soft_template
            for template in self.template_list:
                res = f"{example.context} {self.tokenizer.sep_token} {template.format(subj=example.subj, obj=example.obj)}." 
                example.raw_texts_to_tokenizer.append(res)
        else:
            example.train_label = []
            if example.label != "NA" or example.label != "no_relation":
                # 该样本是positive example
                # 构造2样本
                cor_rel_idx = -1
                for i, rel in enumerate(self.all_relations):
                    if example.label == rel.label:
                        cor_rel_idx == i
                        for tpl in rel.templates:
                            res = f"{example.context} {self.tokenizer.sep_token} {tpl.format(subj=example.subj, obj=example.obj)}." 
                            example.raw_texts_to_tokenizer.append(res)
                            example.train_label.append(2)
                            # self.train_data_cnt += 1
                            # self.train_NL_data[self.train_data_cnt] = (res, 2)
                        break
                # 构造neural 样本
                for _ in range(len(example.raw_texts_to_tokenizer)):
                    idx_other_rel = -1
                    while(True):
                        idx_other_rel = random.randint(0, self.n_rel-1)
                        if ((len(self.all_relations[idx_other_rel].templates) > 0) and (idx_other_rel != cor_rel_idx)):
                            break
                    assert(idx_other_rel != -1)
                    assert(self.all_relations[idx_other_rel].label != "NA")
                    assert(self.all_relations[idx_other_rel].label != "no_relation")

                    
                    # idx_tpl = 0 if len(self.all_relations[idx_other_rel].templates) == 1 else random.randint(0, len(self.all_relations[idx_other_rel].templates)-1)
                    # if len(self.all_relations[idx_other_rel].templates) == 0:
                    #     print(self.all_relations[idx_other_rel].label)
                    #     pause()
                    idx_tpl = random.randint(0, len(self.all_relations[idx_other_rel].templates)-1)
                    tpl = self.all_relations[idx_other_rel].templates[idx_tpl]
                    res = f"{example.context} {self.tokenizer.sep_token} {tpl.format(subj=example.subj, obj=example.obj)}." 
                    example.raw_texts_to_tokenizer.append(res)
                    example.train_label.append(1)
                    # self.train_data_cnt += 1
                    # self.train_NL_data[self.train_data_cnt] = (res, 1)

                # 构造ctd_sample and labels    
                tpl_for_NA = "{subj} and {obj} are not related"
                ctd = f"{example.context} {self.tokenizer.sep_token} {tpl_for_NA.format(subj=example.subj, obj=example.obj)}."
                example.raw_texts_to_tokenizer.append(ctd)
                example.train_label.append(0)
                # self.train_data_cnt += 1
                # self.train_NL_data[self.train_data_cnt] = (ctd, 0)
            else:
                # negative examples: relation == "NA"
                # 构造2样本
                tpl_for_NA = "{subj} and {obj} are not related"
                res = f"{example.context} {self.tokenizer.sep_token} {tpl_for_NA.format(subj=example.subj, obj=example.obj)}."
                example.raw_texts_to_tokenizer.append(res)
                example.train_label.append(2)
                # self.train_data_cnt += 1
                # self.train_NL_data[self.train_data_cnt] = (res, 2)
                # 构造0样本
                idx_other_rel = -1
                while(True):
                    idx_other_rel = random.randint(0, self.n_rel-1)
                    if (len(self.all_relations[idx_other_rel].templates) > 0):
                        break
                assert(idx_other_rel != -1)
                idx_tpl = random.randint(0, len(self.all_relations[idx_other_rel].templates)-1)
                tpl = self.all_relations[idx_other_rel].templates[idx_tpl]
                res = f"{example.context} {self.tokenizer.sep_token} {tpl.format(subj=example.subj, obj=example.obj)}."
                example.raw_texts_to_tokenizer.append(res)
                example.train_label.append(0)
                # self.train_data_cnt += 1
                # self.train_NL_data[self.train_data_cnt] = (res, 0)
        
        # print(example.raw_texts_to_tokenizer)  
        # pause()
        batchEncoding = self.tokenizer(example.raw_texts_to_tokenizer, padding='max_length', truncation=True, max_length=self.config.max_seq_length) # , return_tensors='pt'
            
            # batchEncoding = self.tokenizer(example.raw_texts_to_tokenizer, padding=self.config.max_seq_length, truncation=self.config.max_seq_length, return_tensors='pt') 
            # batchEncoding：example_num个Dict['input_ids', 'token_type_id', 'attention_mask']
        features.extend(self.construct_list_of_featrues_with_batched_inputids(batchEncoding, example))
        # print("one example's feature_num: ")
        # print(len(features))

        # batchEncoding = self.tokenizer(example.raw_texts_to_tokenizer, padding='max_length', truncation=True, max_length=self.config.max_seq_length) # , return_tensors='pt'
        # features.extend(self.construct_list_of_featrues_with_batched_inputids(batchEncoding, example))
        return features
        

    def save(self, path: str) -> None:
        """Save a pretrained wrapper."""
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model # 仅保存一部分，怎么调用？⭐
        model_to_save.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        self._save_config(path)

    def _save_config(self, path: str) -> None:
        with open(os.path.join(path, CONFIG_NAME), 'w') as f:
            f.write(jsonpickle.encode(self.config))

    @staticmethod
    def _load_config(path: str) -> NLIWrapperConfig:
        with open(os.path.join(path, CONFIG_NAME), 'r') as f:
            return jsonpickle.decode(f.read())

    # for getting features to model

    def _convert_examples_to_features(self, examples: List[InputExample], mode) -> List[InputFeatures]:
        features = []
        for example in examples:
            features.extend(self.example_processor(example, mode)) 
        return features

    def _generate_dataset(self, data: List[InputExample], mode) -> DictDataset:
        """used in train()/eval() """
        features = self._convert_examples_to_features(data, mode)
        
        feature_dict = {
            'input_ids': torch.tensor([f.input_ids for f in features], dtype=torch.long),
            'attention_mask': torch.tensor([f.attention_mask for f in features], dtype=torch.long),
            'labels': torch.tensor([f.label for f in features], dtype=torch.long), # 这里是按feture的编号顺序构造的
            'train_labels': torch.tensor([f.train_label for f in features], dtype=torch.long), # 这里是按feture的编号顺序构造的
            'logits': torch.tensor([f.logits for f in features], dtype=torch.float),
            'corresponce_to_InputExample_idx': torch.tensor([f.corresponce_to_InputExample_idx for f in features], dtype=torch.long),
        }
        if features[0].token_type_ids is not None:
            feature_dict['token_type_ids'] =  torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        return DictDataset(**feature_dict)

    def generate_default_inputs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Generate the default inputs required by almost every language model. called by train/eval_step()"""
        inputs = {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']}
        if 'token_type_ids' in batch.keys():
            inputs['token_type_ids'] = batch['token_type_ids']
        # Roberta的模型不需要token_type_ids这个部分            
        return inputs

    # train and eval
    # for tuning prompt
    def train_step(self, batch: Dict[str, torch.Tensor], use_logits: bool = False, temperature: float = 1, **_) -> torch.Tensor:
        """Perform a sequence classifier training step."""
        inputs = self.generate_default_inputs(batch) # 从InputFeatures构造一批batch，再把这批batch转换成batch_inputs_ids
        inputs['labels'] = batch['train_labels'] # 因为train的时候model里需要用到labels，所以直接加进inputs里
        # outputs = self.model(torch.inputcoord, **inputs) 
        outputs = self.model(**inputs) 
        return outputs.loss
    

    def NLI_eval_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform a sequence classifier evaluation step."""
        inputs = self.generate_default_inputs(batch)
        inputs['labels'] = batch['train_labels']
        # 因为eval的时候model里不需要用到labels，所以在eval()里再构造成labels: List即可
        outputs = self.model(**inputs)
        return outputs.loss, outputs.logits # ⭐同train_step，确认output[0]就是结果 √
    
    def eval_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        inputs = self.generate_default_inputs(batch)
        outputs = self.model(**inputs)
        return outputs.logits

    # not use
    def train(self, task_train_data: List[InputExample], device, per_gpu_train_batch_size: int = 8, n_gpu: int = 1,
              num_train_epochs: int = 3, gradient_accumulation_steps: int = 1, weight_decay: float = 0.0,
              learning_rate: float = 5e-5, adam_epsilon: float = 1e-8, warmup_steps=0, max_grad_norm: float = 1,
              logging_steps: int = 50, per_gpu_unlabeled_batch_size: int = 8, unlabeled_data: List[InputExample] = None,
              lm_training: bool = False, use_logits: bool = False, alpha: float = 0.8, temperature: float = 1,
              max_steps=-1, **_):
        """
        Train the underlying language model.

        :param task_train_data: the training examples to use
        :param device: the training device (cpu/gpu)
        :param per_gpu_train_batch_size: the number of training examples per batch and gpu
        :param n_gpu: the number of gpus to use
        :param num_train_epochs: the number of epochs to train
        :param gradient_accumulation_steps: the number of gradient accumulation steps before performing an update
        :param weight_decay: the weight decay to use
        :param learning_rate: the learning rate to use
        :param adam_epsilon: epsilon parameter for the Adam optimizer
        :param warmup_steps: the number of warmup steps
        :param max_grad_norm: the maximum norm for the gradient
        :param logging_steps: the number of steps after which logging information is printed
        :param per_gpu_unlabeled_batch_size: the number of unlabeled examples per batch and gpu
        :param unlabeled_data: the unlabeled examples to use
        :param lm_training: whether to perform auxiliary language modeling (only for MLMs) 是否使用辅助LM
        :param use_logits: whether to use the example's logits instead of their labels to compute the loss ---用于给unlabeled数据集标注---
        :param alpha: the alpha parameter for auxiliary language modeling
        :param temperature: the temperature for knowledge distillation
        :param max_steps: the maximum number of training steps, overrides ``num_train_epochs``
        :return: a tuple consisting of the total number of steps and the average training loss 返回一个元组(总训练步数，平均训练loss)
        """

        train_batch_size = per_gpu_train_batch_size * max(1, n_gpu)
        train_dataset = self._generate_dataset(task_train_data)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size, num_workers=8, pin_memory=True)

        unlabeled_dataloader, unlabeled_iter = None, None

        if lm_training or use_logits:
            # we need unlabeled data both for Data Augment setting
            assert unlabeled_data is not None
            unlabeled_batch_size = per_gpu_unlabeled_batch_size * max(1, n_gpu)
            unlabeled_dataset = self._generate_dataset(unlabeled_data, labelled=False)
            unlabeled_sampler = RandomSampler(unlabeled_dataset)
            unlabeled_dataloader = DataLoader(unlabeled_dataset, sampler=unlabeled_sampler,
                                              batch_size=unlabeled_batch_size, num_workers=8, pin_memory=True)
            unlabeled_iter = unlabeled_dataloader.__iter__()

        if use_logits:
            train_dataloader = unlabeled_dataloader

        if max_steps > 0:
            t_total = max_steps
            num_train_epochs = max_steps // (max(1, len(train_dataloader) // gradient_accumulation_steps)) + 1
        else:
            t_total = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        # named_parameters()和parameters()，前者给出网络层的名字和参数的迭代器，而后者仅仅是参数的迭代器。
        # weight_decay:权重衰减，防止过拟合
        no_decay = ['bias', 'LayerNorm.weight'] # 不更新这些网络部分的权重 why?对比bitfit中更新的部分，可能仅更新bias⭐
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total) # 在num_warmup_steps内升到峰值lr，在num_training_steps内将为0

        # multi-gpu training
        if n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        step = 0
        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        self.model.zero_grad()

        train_iterator = trange(int(num_train_epochs), desc="Epoch") # train_iterator 是epoch，封装在tqdm.trange里显示进度条

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for _, batch in enumerate(epoch_iterator):
                self.model.train()
                unlabeled_batch = None

                batch = {k: t.to(device) for k, t in batch.items()} # batch是一个字典dict, key: k(index), val: t(放入gpu,Inputfeature)

                if lm_training:
                    while unlabeled_batch is None:
                        try:
                            unlabeled_batch = unlabeled_iter.__next__()
                        except StopIteration:
                            logger.info("Resetting unlabeled dataset")
                            unlabeled_iter = unlabeled_dataloader.__iter__()

                    lm_input_ids = unlabeled_batch['input_ids']
                    unlabeled_batch['input_ids'], unlabeled_batch['mlm_labels'] = self._mask_tokens(lm_input_ids)
                    unlabeled_batch = {k: t.to(device) for k, t in unlabeled_batch.items()}

                train_step_inputs_config = {
                    'unlabeled_batch': unlabeled_batch, 'lm_training': lm_training, 'alpha': alpha,
                    'use_logits': use_logits, 'temperature': temperature
                }
                
                loss = self.train_step(batch, **train_step_inputs_config)

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                loss.backward() # 计算梯度

                tr_loss += loss.item()
                if (step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step() # 参数更新（基于当前的梯度，梯度*学习率lr(α)）
                    scheduler.step() # 对学习率lr的更新
                    self.model.zero_grad() # 梯度归零
                    global_step += 1 # 使用梯度累积时的step

                    if logging_steps > 0 and global_step % logging_steps == 0:
                        logs = {}
                        loss_scalar = (tr_loss - logging_loss) / logging_steps
                        learning_rate_scalar = scheduler.get_lr()[0]
                        logs['learning_rate'] = learning_rate_scalar
                        logs['loss'] = loss_scalar
                        logging_loss = tr_loss

                        print(json.dumps({**logs, **{'step': global_step}}))

                if 0 < max_steps < global_step:
                    epoch_iterator.close()
                    break
                step += 1
            if 0 < max_steps < global_step:
                train_iterator.close()
                break

        return global_step, (tr_loss / global_step if global_step > 0 else -1)


    def eval(self, eval_data: List[InputExample], device: str, per_gpu_eval_batch_size: int, n_gpu: int = 1, multiclass: bool = True) -> Dict:
        """
        Evaluate the underlying language model. or as forward() in zero-shot setting

        :param eval_data: the evaluation examples to use
        :param device: the evaluation device (cpu/gpu) # 用self.config.use_cuda 控制
        :param per_gpu_eval_batch_size: the number of evaluation examples per batch and gpu
        :param n_gpu: the number of gpus to use
        :param multiclass: the result whether can be multiple relations
        :return: a dictionary of numpy arrays containing the indices, logits, labels, and (optional) question_ids for
                 each evaluation example.这里的结果output仅限model的rawoutput + metrics，后续转换可读的结果由predict()完成
        """

        eval_dataset = self._generate_dataset(eval_data, 0)
        eval_batch_size = per_gpu_eval_batch_size * max(1, n_gpu)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size, num_workers=8, pin_memory=True)

        if n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        preds: np.array = None # len(examples) * 3: label_nums
        all_indices, out_label_ids, question_ids = None, None, None

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            self.model.eval()

            batch = {k: t.to(device) for k, t in batch.items()}
            labels = batch['labels'] # 一列，所有的feature的label按顺序的列表
            indices = batch['corresponce_to_InputExample_idx'] # 一列，所有的feature对应的example的idx按feature排列顺序的列表
            with torch.no_grad():
                logits = self.eval_step(batch)

            if preds is None:
                preds = logits.detach().cpu().numpy() # 返回一个新的tensor，其requires_grad为false，从当前计算图中分离下来的，仍指向原变量的存放位置
                out_label_ids = labels.detach().cpu().numpy()
                all_indices = indices.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)
                all_indices = np.append(all_indices, indices.detach().cpu().numpy(), axis=0)

        outputs = preds
        if multiclass:
            outputs = np.exp(outputs) / np.exp(outputs).sum(-1, keepdims=True)  # 按顺序得到 contradiction、neutral、entailment的概率(softmax): np矩阵
            
            # input_ids = batch['input_ids'].detach().cpu().numpy() # tensor -> numpy
            
            # outputs = outputs[..., self.ent_pos].reshape(input_ids.shape[0] // len(self.labels), -1) 
            # 再reshape得到每个example对每个template的entailment概率：num_InputFeatures(=num_InputExamples * num_relation)*1->num_InputExamples * num_relation（一个example构造了relaions_num*num_template个inputfeature,）
            # ⭐outputs: num_InputExamples * num_relation (该example使用该template后对应的entailment概率)
            
            
            outputs = outputs[..., self.ent_pos].reshape(-1, len(self.template_list)) # example_num * template_num
            # outputs = outputs[..., self.ent_pos].reshape(len(eval_data), -1) # example_num * template_num
            print(outputs.shape)
            # print(len(eval_data))
            
            """
            原始数据集：input_example ---- relation(lable)
            NLImodle: sentence</s>sentence ---- 3(constract/entailment/neural) -->当前的output：n_rel*template_num
            还原可读结果：input_example ---- relation(res) 
            
            outputs整个(当前)：relation_num * template_num
            outputs[:, self.mapping[label]]:
            ------------------------------------------------------------------------
                     |example1在r1的template1上的entailment概率|在T2上的概率|在Tm上的概率
            relation1---------------------------------------------------------------
                     |example2在r1的template1上的entailment概率|在T2上的概率|在Tm上的概率
                     ---------------------------------------------------------------
                     |examplen在r1的template1上的entailment概率|在T2上的概率|在Tm上的概率
            -------------------------------------------------------------------------
                     |example1在r2的template1上的entailment概率|在T2上的概率|在Tm上的概率
            relation2---------------------------------------------------------------
                     |example2在r2的template1上的entailment概率|在T2上的概率|在Tm上的概率
                     ---------------------------------------------------------------
                     |examplen在r2的template1上的entailment概率|在T2上的概率|在Tm上的概率
            -------------------------------------------------------------------------
            ......
            跨列（横着的）每行的最大值作为结果(概率，其对应的relation_in_NL由predict(得到))
            """
            # 和labels (relation_name_list)比较得metrics and results
            # for label in self.relation_name_list:
            #     if label in self.mapping:
            #         print(outputs[:, self.mapping[label]])
            #     else:
            #         print(label + " is not in self.mapping.")
            outputs = np.hstack(
                [
                    np.max(outputs[:, self.mapping[label]], axis=-1, keepdims=True)
                    if label in self.mapping
                    else np.zeros((outputs.shape[0], 1))
                    for label in self.relation_name_list # label == 'NA'时没有template，output[:, []]结果是空，np.max会报错无法在空array里找最大值
                ]
            ) # 在每个relation中，哪个template的entailment得分最高(hstack(水平方向的堆叠，堆num_relation个num_example*1的东西))：，上面的ouputs和当前的outputs不一样，当前output（用原来的outputs构造的）维度：num_example*num_relation
        if not multiclass:
            outputs = np_softmax(outputs) # 每个feature对应的constract/neural/entailment的概率 但是一般不用，保证multiclass=True

        if self.valid_conditions:
            outputs = self._apply_valid_conditions(outputs, eval_data) # features->InputExample

        outputs = self._apply_negative_threshold(outputs) # 这个必须用，因为会有NA这样的关系，NA没有manual_template
        print("After convert 2nd dimension:")
        print(outputs.shape)
        return outputs # num_example * num_relation(num_relation除去‘NA’,在_apply_valid_conditions里加入NA的一列零) outputs: numpy

    def _apply_negative_threshold(self, probs: np) -> np:
        # probs维度：num_example * num_relation
        # ? 目前没有用这个方法
        activations = (probs >= self.negative_threshold).sum(-1).astype(np.int) # 对每个example找到概率大于threshold的relation的个数
        idx = np.logical_or(
            activations == 0, activations >= self.max_activations
        )  # If there are no activations then is a negative example, if there are too many, then is a noisy example
        probs[idx, self.negative_idx] = 1.00 # self.negative_idx=0, 见utils.apply_threshold
        return probs

    def _apply_valid_conditions(self, probs, features: List[InputExample]):
        # 每个inputExample的pair_type被哪些relation允许，即该example能输出的candidate_relations有哪些，允许的是1，维度：example_num * n_rel
        # mask_matrix = np.stack(
        #     [self.valid_conditions.get(feature.pair_type, np.zeros(self.n_rel)) for feature in features],
        #     axis=0,
        # )
        
        mask_matrix = None
        flag = False
        for feature in features:
            row = np.zeros(self.n_rel)
            if feature.pair_type == None:
                if flag == False:
                    mask_matrix = row
                    flag = True
                else:
                    mask_matrix = np.vstack((mask_matrix, row))
                continue
            for pairType in feature.pair_type:
                now = np.array([self.valid_conditions.get(pairType, np.zeros(self.n_rel))])
                row = np.logical_or(row, now)
            if flag == False:
                mask_matrix = row
                flag = True
            else:
                mask_matrix = np.vstack((mask_matrix, row))
        
        # dict.get(key, 若没有该key则返回的值[这里是全0])
        assert(mask_matrix.shape == probs.shape), "mask_matrix.size() != probs.size()"
        probs = probs * mask_matrix # 矩阵对应位置相乘， probs维度：num_example * num_relation
        return probs 


    def predict(self, output: np, topk: int, return_labels: bool = True, return_confidences: bool = True):
        """ get readable output 
        """
        # output = self.eval(eval_data, device, per_gpu_eval_batch_size)
        topics = np.argsort(output, -1)[:, ::-1][:, :topk] # argsort()返回数组值从小到大的索引值,沿最后一维 -> 矩阵，表示每条样本的预测top-k和个relation id
        # topics: candidata_relations
        if return_labels:
            topics = self.idx2label(topics) # 确认最终的relation (id->str)
            
        if return_confidences:
            topics = np.stack((topics, np.sort(output, -1)[:, ::-1][:, :topk]), -1).tolist()
            topics = [
                [(int(label), float(conf)) if not return_labels else (label, float(conf)) for label, conf in row]
                for row in topics
            ]  
        else:
            topics = topics.tolist()
        if topk == 1:
            topics = [row[0] for row in topics]
        return topics

    
    # for PT
    

    def save_optiprompt(self, output_dir: str, original_vocab_size):
        logger.info("Saving OptiPrompt's [V]s..")
        # vs = self.model.embeddings.word_embeddings.weight[original_vocab_size:].detach().cpu().numpy()
        vs = self.base_model.embeddings.word_embeddings.weight[original_vocab_size:].detach().cpu().numpy()
        
        with open(os.path.join(output_dir, 'prompt_vecs.npy'), 'wb') as f:
            np.save(f, vs)

        
    # def init_indices_for_filter_logprobs(self, vocab_subset, logger=None):
    #     index_list = []
    #     new_vocab_subset = []
    #     for word in vocab_subset:
    #         tokens = self.tokenizer.tokenize(' '+word)
    #         if (len(tokens) == 1) and (tokens[0] != self.UNK):
    #             index_list.append(self.tokenizer.convert_tokens_to_ids(tokens)[0])
    #             new_vocab_subset.append(word)
    #         else:
    #             msg = "word {} from vocab_subset not in model vocabulary!".format(word)
    #             if logger is not None:
    #                 logger.warning(msg)
    #             else:
    #                 logger.info("WARNING: {}".format(msg))

    #     indices = torch.as_tensor(index_list)
    #     return indices, index_list

    # def using_common_vocab(self):
    #     # no use
    #     if self.config.common_vocab_filename is not None:
    #         vocab_subset = load_vocab(self.config.common_vocab_filename)
    #         logger.info('Common vocab: %s, size: %d'%(self.config.common_vocab_filename, len(vocab_subset)))
    #         filter_indices, index_list = self.init_indices_for_filter_logprobs(vocab_subset) # 把不在交集vocab中的word删去
    #     else:
    #         filter_indices = None
    #         index_list = None
    #     return filter_indices, index_list


    def tuning_train(self, train_data: List[InputExample], dev_data: List[InputExample], device, learning_rate: float, eval_step: int, prompt_type: str,
                     warmup_proportion: float, save_optiprompt_data_dir: str, max_step: int, eval_batch_size: int, 
                     num_train_epochs: int, train_batch_size: int, check_step: int, gradient_accumulation_steps: int = 1, topk: int = 1):
        # get_train_batch
        train_dataset = self._generate_dataset(train_data, mode=1)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size, num_workers=8, pin_memory=True)
        # wandb.log({
        #     "NLI_data": self.train_NL_data
        # })
        
        _, best_result = self.tuning_eval(dev_data, device, eval_batch_size, top_k=topk) # 先得manT初始化得到的prompt的结果
        logger.info('!!! Best result for NLI train objective: %.2f .'%(best_result * 100))
        # Add word embeddings to the optimizer
        # optimizer = AdamW([{'params': self.model.embeddings.word_embeddings.parameters()}], lr=learning_rate, correct_bias=False)

        optimizer = AdamW([{'params': self.base_model.embeddings.word_embeddings.parameters()}], lr=learning_rate, correct_bias=False)
        t_total = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs # 更新参数(embeddings)的次数
        scheduler = get_linear_schedule_with_warmup(optimizer, int(t_total*warmup_proportion), t_total)

        # Train!!!
        step = 0
        global_step = 0 # optimizer.zero_grad()的次数
        tr_loss = 0

        eval_step = len(train_dataloader) // 3 # 每个epoch进行3次保存softprompt

        nb_tr_examples = 0

        train_iterator = trange(int(num_train_epochs), desc="Epoch") # train_iterator 是epoch，封装在tqdm.trange里显示进度条

        for epoch in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for i, batch in enumerate(epoch_iterator):
                if i == 0:
                    print(batch["input_ids"].size())
                self.model.train()
                batch = {k: t.to(device) for k, t in batch.items()} # batch是一个字典dict, key: k(index), val: t(放入gpu,Inputfeature)

                loss = self.train_step(batch)
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += len(batch.items())
                step += 1
                # global_step += 1
                
                
                
                if (step + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    self.model.zero_grad() # ?optiT里的部分
                    global_step += 1
                    wandb.log({
                        # "epoch": epoch,
                        # "global_step": global_step,
                        "train_loss": tr_loss / nb_tr_examples,
                    }) 
                    tr_loss = 0
                    nb_tr_examples = 0
                    # if eval_step > 0 and (step + 1) % eval_step != 0:
                    #     tr_loss = 0
                    #     nb_tr_examples = 0
                    
                    
                    # set normal tokens' gradients to be zero ?optiT里的部分
                    for p in self.base_model.embeddings.word_embeddings.parameters():
                        # only update new tokens
                        p.grad[:self.original_vocab_size, :] = 0.0
                    # for p in self.model.embeddings.word_embeddings.parameters():
                    #     p.grad[:original_vocab_size, :] = 0.0

                if check_step > 0 and ((step + 1) % check_step == 0) and nb_tr_examples > 0:
                    logger.info('Epoch=%d, iter=%d, loss=%.5f'%(_, i, tr_loss / nb_tr_examples))
                    # sys.stdout.flush()
                    # tr_loss = 0
                    # nb_tr_examples = 0

                if eval_step > 0 and (step + 1) % eval_step == 0:
                    # Eval during training
                    logger.info('step=%d, evaluating...'%(step))
                    dev_loss, precision = self.tuning_eval(dev_data, device, eval_batch_size) # evaluate()
                    if precision > best_result:
                        wandb.run.summary["best_accuracy"] = precision
                        best_result = precision
                        logger.info('!!! Best valid for NLI_objective (epoch=%d): %.2f' %
                            (_, best_result * 100))
                        self.save_optiprompt(save_optiprompt_data_dir, self.original_vocab_size)
                    wandb.log({
                        # "epoch": epoch,
                        # "step": step,
                        "dev_loss": dev_loss,
                        "dev_precision":precision
                    }) 
                    # "embeddings": self.base_model.embeddings.word_embeddings.parameters()
                     
                    # if (step + 1) % gradient_accumulation_steps == 0:
                    #     tr_loss = 0
                    #     nb_tr_examples = 0

                if 0 < max_step < global_step:
                    epoch_iterator.close()
                    break
            # 每个epoch结束后进行测试，⭐ 处理提前结束
            logger.info('step=%d, evaluating...'%(step))
            dev_loss, precision = self.tuning_eval(dev_data, device, eval_batch_size) # evaluate()
            print("evaluate train_label in dev data is: ")
            print(precision)
            if precision > best_result:
                wandb.run.summary["best_accuracy"] = precision
                best_result = precision
                logger.info('!!! Best valid for NLI_objective (epoch=%d): %.2f' %
                    (_, best_result * 100))
                self.save_optiprompt(save_optiprompt_data_dir, self.original_vocab_size)
                # "embeddings": self.base_model.embeddings.word_embeddings.parameters()
                wandb.log({
                    "train_loss": tr_loss / nb_tr_examples,
                    "dev_loss": dev_loss,
                    "dev_precision":precision
                })
            # 每次有embedding的准确率成为best时，进行RE eval，查看正确率变化及错误的样例
            # 
            

            if 0 < max_step < global_step:
                train_iterator.close()
                break

        logger.info('Best Valid for NLI_objective: %.2f'%(best_result*100))
        
    

    def tuning_eval(self, dev_data: List[InputExample], device: str, eval_batch_size: int, top_k: int = 1, n_gpu=1):
        # def evaluate(model, samples_batches, sentences_batches, filter_indices=None, index_list=None, output_topk=None):
        # 调NLIforward()得到结果Dict，再把precisionhe 
        # 输入参数里 加上self.eval()的参数

        eval_dataset = self._generate_dataset(dev_data, mode=1)
        eval_batch_size = eval_batch_size * max(1, n_gpu)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size, num_workers=8, pin_memory=True)

        if n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        preds: np.array = None # len(examples) * 3: label_nums
        all_loss = 0
        all_indices, out_label_ids, question_ids = None, None, None

        for batch in tqdm(eval_dataloader, desc="EvaluatingforPromptTuning"):
            self.model.eval()

            batch = {k: t.to(device) for k, t in batch.items()}
            labels = batch['train_labels'] # 一列，所有的feature的label按顺序的列表
            indices = batch['corresponce_to_InputExample_idx'] # 一列，所有的feature对应的example的idx按feature排列顺序的列表
            with torch.no_grad():
                loss, logits = self.NLI_eval_step(batch)
                all_loss += loss

            if preds is None:
                preds = logits.detach().cpu().numpy() # 返回一个新的tensor，其requires_grad为false，从当前计算图中分离下来的，仍指向原变量的存放位置
                out_label_ids = labels.detach().cpu().numpy()
                all_indices = indices.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)
                all_indices = np.append(all_indices, indices.detach().cpu().numpy(), axis=0)

        outputs = preds # features_num * 3
        # print(outputs.shape) # 
        outputs = np.exp(outputs) / np.exp(outputs).sum(-1, keepdims=True)  # 按顺序得到 contradiction、neutral、entailment的概率(softmax): np矩阵
        outputs = np.argmax(outputs, axis=1).reshape(-1, 1) # feartures_nums * 1
        # print(outputs.shape) # 
        # print(train_labels.shape) 
        # print(out_label_ids.shape) 
        train_labels = out_label_ids.reshape(-1, 1) 
        # train_labels = np.argmax(out_label_ids, axis=1).reshape(-1, 1)
        
        
        assert((train_labels.shape[0] == outputs.shape[0]) and (train_labels.shape[1] == outputs.shape[1])), "train_labels.shape() != outputs.shape()"
        
        precision = (train_labels == outputs).sum()/outputs.shape[0]
        return all_loss, precision
    
    

