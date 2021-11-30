""" This code load dataset and construct coarse InputExample from original dataset """

import json
from operator import truediv
import os
import random
from abc import ABC, abstractmethod # python stdlib, "Abstract Base Classes", 抽象基类，为一组子类定义公共API
from collections import defaultdict, Counter
from typing import List, Dict, Callable, Optional, Union
import logging

from numpy.core.numeric import allclose 

from utils import InputExample, get_marked_sentence
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__) 

rules = defaultdict
# 增加dataset时这里要改 …… hard code
rules = {
        "per:alternate_names": ["PERSON:PERSON", "PERSON:MISC"],
        "per:date_of_birth": ["PERSON:DATE"],
        "per:age": ["PERSON:NUMBER", "PERSON:DURATION"],
        "per:country_of_birth": ["PERSON:COUNTRY"],
        "per:stateorprovince_of_birth": ["PERSON:STATE_OR_PROVINCE"],
        "per:city_of_birth": ["PERSON:CITY"],
        "per:origin": [
            "PERSON:NATIONALITY",
            "PERSON:COUNTRY",
            "PERSON:LOCATION",
        ],
        "per:date_of_death": ["PERSON:DATE"],
        "per:country_of_death": ["PERSON:COUNTRY"],
        "per:stateorprovince_of_death": ["PERSON:STATE_OR_PROVICE"],
        "per:city_of_death": ["PERSON:CITY", "PERSON:LOCATION"],
        "per:cause_of_death": ["PERSON:CAUSE_OF_DEATH"],
        "per:countries_of_residence": ["PERSON:COUNTRY", "PERSON:NATIONALITY"],
        "per:stateorprovinces_of_residence": ["PERSON:STATE_OR_PROVINCE"],
        "per:cities_of_residence": ["PERSON:CITY", "PERSON:LOCATION"],
        "per:schools_attended": ["PERSON:ORGANIZATION"],
        "per:title": ["PERSON:TITLE"],
        "per:employee_of": ["PERSON:ORGANIZATION"],
        "per:religion": ["PERSON:RELIGION"],
        "per:spouse": ["PERSON:PERSON"],
        "per:children": ["PERSON:PERSON"],
        "per:parents": ["PERSON:PERSON"],
        "per:siblings": ["PERSON:PERSON"],
        "per:other_family": ["PERSON:PERSON"],
        "per:charges": ["PERSON:CRIMINAL_CHARGE"],
        "org:alternate_names": [
            "ORGANIZATION:ORGANIZATION",
            "ORGANIZATION:MISC",
        ],
        "org:political/religious_affiliation": [
            "ORGANIZATION:RELIGION",
            "ORGANIZATION:IDEOLOGY",
        ],
        "org:top_members/employees": ["ORGANIZATION:PERSON"],
        "org:number_of_employees/members": ["ORGANIZATION:NUMBER"],
        "org:members": ["ORGANIZATION:ORGANIZATION", "ORGANIZATION:COUNTRY"],
        "org:member_of": [
            "ORGANIZATION:ORGANIZATION",
            "ORGANIZATION:COUNTRY",
            "ORGANIZATION:LOCATION",
            "ORGANIZATION:STATE_OR_PROVINCE",
        ],
        "org:subsidiaries": [
            "ORGANIZATION:ORGANIZATION",
            "ORGANIZATION:LOCATION",
        ],
        "org:parents": ["ORGANIZATION:ORGANIZATION", "ORGANIZATION:COUNTRY"],
        "org:founded_by": ["ORGANIZATION:PERSON"],
        "org:founded": ["ORGANIZATION:DATE"],
        "org:dissolved": ["ORGANIZATION:DATE"],
        "org:country_of_headquarters": ["ORGANIZATION:COUNTRY"],
        "org:stateorprovince_of_headquarters": ["ORGANIZATION:STATE_OR_PROVINCE"],
        "org:city_of_headquarters": [
            "ORGANIZATION:CITY",
            "ORGANIZATION:LOCATION",
        ],
        "org:shareholders": [
            "ORGANIZATION:PERSON",
            "ORGANIZATION:ORGANIZATION",
        ],
        "org:website": ["ORGANIZATION:URL"],
        "org:city_of_branch": [
            "ORGANIZATION:CITY",
            "ORGANIZATION:LOCATION",
        ],
        "org:country_of_branch": ["ORGANIZATION:COUNTRY"],
        "org:stateorprovince_of_branch": ["ORGANIZATION:STATE_OR_PROVINCE"],
        "per:identity": ["PERSON:PERSON", "PERSON:MISC"],
    }



def _shuffle_and_restrict(examples: List[InputExample], num_examples: int, seed: int = 42) -> List[InputExample]:
    """
    Shuffle a list of examples and restrict it to a given maximum size.

    :param examples: the examples to shuffle and restrict
    :param num_examples: the maximum number of examples, if the val > len(examples), return all shuffled examples
    :param seed: the random seed for shuffling
    :return: the first ``num_examples`` elements of the shuffled list
    """
    random.Random(seed).shuffle(examples)
    if 0 < num_examples < len(examples):
        examples = examples[:num_examples]
    return examples


class DataProcessor(ABC):
    """
    Abstract class that provides methods for loading training, testing, development and unlabeled examples for a given
    task
    """

    @abstractmethod
    def get_train_examples(self, data_dir) -> List[InputExample]:
        """Get a collection of `InputExample`s for the train set."""
        pass

    @abstractmethod
    def get_dev_examples(self, data_dir) -> List[InputExample]:
        """Get a collection of `InputExample`s for the dev set."""
        pass

    @abstractmethod
    def get_test_examples(self, data_dir) -> List[InputExample]:
        """Get a collection of `InputExample`s for the test set."""
        pass

    @abstractmethod
    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        """Get a collection of `InputExample`s for the unlabeled set."""
        pass

    @abstractmethod
    def get_labels(self) -> List[str]:
        """Get the list of labels for this data set."""
        pass


class TACREDProcessor(DataProcessor):
    """Processor for the TACRED, Re-TACRED dataset."""

    def get_train_examples(self, data_dir) -> List[InputExample]:
        return self._create_examples(TACREDProcessor._read_json(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir) -> List[InputExample]:
        return self._create_examples(TACREDProcessor._read_json(os.path.join(data_dir, "dev.txt")), "dev")

    def get_test_examples(self, data_dir) -> List[InputExample]:
        return self._create_examples(TACREDProcessor._read_json(os.path.join(data_dir, "test.txt")), "test")

    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        return self.get_train_examples(data_dir)

    def get_labels(self) -> List[str]:
        # 不是该样本所有可能的relation：（multiple labels）
        labels = []
        for key in rules.keys():
            if key not in labels:
                labels.append(key)
        return labels

    @staticmethod
    def _create_examples(lines: List[Dict[str, str]], set_type: str) -> List[InputExample]:
        examples = []
        all_entity_marker_new_tokens = []
        all_entity_marker_punct_new_tokens = []
        all_typed_marker_new_tokens = []
        all_typed_marker_punct_new_tokens = []
        for (i, line) in enumerate(lines):
            subj = line["h"]["name"]
            obj = line["t"]["name"]
            ori_tokens = line["token"]
            context = " ".join(line["token"]).replace("-LRB-", "(").replace("-RRB-", ")").replace("-LSB-", "[").replace("-RSB-", "]").replace("-LCB-", "{").replace("-RCB-", "}") # line["token"]: List[token]
            def convert_token(token):
                """ Convert PTB tokens to normal tokens """
                if (token.lower() == '-lrb-'):
                    return '('
                elif (token.lower() == '-rrb-'):
                    return ')'
                elif (token.lower() == '-lsb-'):
                    return '['
                elif (token.lower() == '-rsb-'):
                    return ']'
                elif (token.lower() == '-lcb-'):
                    return '{'
                elif (token.lower() == '-rcb-'):
                    return '}'
                return token
            label = line["relation"] # example with multiple label,数据集中每条数据仅有一个label，但运算结果可显示多个可能的label ；若有些数据集中是多label的，在这里label: List[str]
            pair_type = rules.get(label, None)
        
            
            subj_st = line["h"]["pos"][0] 
            subj_ed = line["h"]["pos"][1]
            obj_st = line["t"]["pos"][0]
            obj_ed = line["t"]["pos"][1]
            
            subj_coarse_grained_type = line["subj_coarse_grain_type"].lower()
            obj_coarse_grained_type = line["obj_coarse_grain_type"].lower()

            subj_fine_grained_res = line["subj_fine_res"]
            obj_fine_grained_res = line["obj_fine_res"]

            # 若存在description: 使用标注出的第一个description
            subj_description: str = ""
            if len(subj_fine_grained_res) > 0:
                subj_description = subj_fine_grained_res[0][2]
            obj_description: str = ""
            if len(obj_fine_grained_res) > 0:
                obj_description = obj_fine_grained_res[0][2]

            # 若存在fine-grained_type: 全都要 会引入noise?
            subj_fine_grained_type = []
            for it in subj_fine_grained_res:
                for ty in it:
                    subj_fine_grained_type.append(ty)
            obj_fine_grained_type = []
            for it in obj_fine_grained_res:
                for ty in it:
                    obj_fine_grained_type.append(ty)
            
            # 这里只使用粗粒度的类别
            entity_marker_context_list, entity_marker_new_tokens = get_marked_sentence(ori_tokens, subj_st, subj_ed, obj_st, obj_ed, subj_coarse_grained_type, obj_coarse_grained_type, "entity_marker")
           
            entity_marker_punct_context_list, entity_marker_punct_new_tokens = get_marked_sentence(ori_tokens, subj_st, subj_ed, obj_st, obj_ed, subj_coarse_grained_type, obj_coarse_grained_type, "entity_marker_punct")
            
            typed_marker_context_list, typed_marker_new_tokens = get_marked_sentence(ori_tokens, subj_st, subj_ed, obj_st, obj_ed, subj_coarse_grained_type, obj_coarse_grained_type, "typed_marker")
            
            typed_marker_punct_context_list, typed_marker_punct_new_tokens = get_marked_sentence(ori_tokens, subj_st, subj_ed, obj_st, obj_ed, subj_coarse_grained_type, obj_coarse_grained_type, "typed_marker_punct")
            
            # 后续list换成str的时候需要使用这样的replace函数
            # print(" ".join(entity_marker_context).replace("-LRB-", "(").replace("-RRB-", ")").replace("-LSB-", "[").replace("-RSB-", "]"))
            # print(" ".join(entity_marker_punct_context).replace("-LRB-", "(").replace("-RRB-", ")").replace("-LSB-", "[").replace("-RSB-", "]"))
            # print(" ".join(typed_marker_context).replace("-LRB-", "(").replace("-RRB-", ")").replace("-LSB-", "[").replace("-RSB-", "]"))
            # print(" ".join(typed_marker_punct_context).replace("-LRB-", "(").replace("-RRB-", ")").replace("-LSB-", "[").replace("-RSB-", "]"))
            # print("--------------")
            # 这里需要都存进InputExample里
            # "entity_marker_new_tokens": entity_marker_new_tokens,
            # "entity_marker_punct_new_tokens": entity_marker_punct_new_tokens,
            # "typed_marker_new_tokens": typed_marker_new_tokens,
            # "typed_marker_punct_new_tokens": typed_marker_punct_new_tokens,
            meta = {
                "subj_pos": line["h"]["pos"],
                "obj_pos": line["t"]["pos"],
                "subj_description": subj_description,
                "obj_description": obj_description,
                "subj_coarse_grained_type": subj_coarse_grained_type,
                "obj_coarse_grained_type": obj_coarse_grained_type,
                "entity_marker_context_list": entity_marker_context_list,
                "entity_marker_punct_context_list": entity_marker_punct_context_list,
                "typed_marker_context_list": typed_marker_context_list,
                "typed_marker_punct_context_list": typed_marker_punct_context_list,
            }
            example = InputExample(i, subj, obj, ori_tokens, context, label, pair_type=pair_type, meta=meta)
            examples.append(example)
            
            for token in entity_marker_new_tokens:
                if token not in all_entity_marker_new_tokens:
                    all_entity_marker_new_tokens.append(token)
            for token in entity_marker_punct_new_tokens:
                if token not in all_entity_marker_punct_new_tokens:
                    all_entity_marker_punct_new_tokens.append(token)
            for token in typed_marker_new_tokens:
                if token not in all_typed_marker_new_tokens:
                    all_typed_marker_new_tokens.append(token)
            for token in typed_marker_punct_new_tokens:
                if token not in all_typed_marker_punct_new_tokens:
                    all_typed_marker_punct_new_tokens.append(token)

        new_tokens = {
            "all_entity_marker_new_tokens": all_entity_marker_new_tokens,
            "all_entity_marker_punct_new_tokens": all_entity_marker_punct_new_tokens,
            "all_typed_marker_new_tokens": all_typed_marker_new_tokens,
            "all_typed_marker_punct_new_tokens": all_typed_marker_punct_new_tokens
        }
        return examples, new_tokens

    @staticmethod
    def _read_json(input_file):
        with open(input_file, 'r', encoding='UTF-8') as f:
            lines = []
            for i in f.readlines():
                d = json.loads(i) # .txt文件里最后一行后面不能有别的空行，否则json.loads无法处理空行
                lines.append(d)
            return lines


class ReTACREDProcessor(DataProcessor):
    """Processor for the TACRED, Re-TACRED dataset."""

    def get_train_examples(self, data_dir) -> List[InputExample]:
        return self._create_examples(ReTACREDProcessor._read_json(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir) -> List[InputExample]:
        return self._create_examples(ReTACREDProcessor._read_json(os.path.join(data_dir, "dev.txt")), "dev")

    def get_test_examples(self, data_dir) -> List[InputExample]:
        return self._create_examples(ReTACREDProcessor._read_json(os.path.join(data_dir, "test.txt")), "test")

    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        return self.get_train_examples(data_dir)

    def get_labels(self) -> List[str]:
        # 不是该样本所有可能的relation：（multiple labels）
        labels = []
        for key in rules.keys():
            if key not in labels:
                labels.append(key)
        return labels

    @staticmethod
    def _create_examples(lines: List[Dict[str, str]], set_type: str) -> List[InputExample]:
        examples = []
        all_entity_marker_new_tokens = []
        all_entity_marker_punct_new_tokens = []
        all_typed_marker_new_tokens = []
        all_typed_marker_punct_new_tokens = []
        for (i, line) in enumerate(lines):
            subj = line["h"]["name"]
            obj = line["t"]["name"]
            ori_tokens = line["token"]
            context = " ".join(line["token"]).replace("-LRB-", "(").replace("-RRB-", ")").replace("-LSB-", "[").replace("-RSB-", "]") # line["token"]: List[token]
            label = line["relation"] # example with multiple label,数据集中每条数据仅有一个label，但运算结果可显示多个可能的label ；若有些数据集中是多label的，在这里label: List[str]
            pair_type = rules.get(label, None)
        
            
            subj_st = line["h"]["pos"][0] 
            subj_ed = line["h"]["pos"][1]
            obj_st = line["t"]["pos"][0]
            obj_ed = line["t"]["pos"][1]
            
            subj_coarse_grained_type = line["subj_coarse_grain_type"].lower()
            obj_coarse_grained_type = line["obj_coarse_grain_type"].lower()

            subj_fine_grained_res = line["subj_fine_res"]
            obj_fine_grained_res = line["obj_fine_res"]

            # 若存在description: 使用标注出的第一个description
            subj_description: str = ""
            if len(subj_fine_grained_res) > 0:
                subj_description = subj_fine_grained_res[0][2]
            obj_description: str = ""
            if len(obj_fine_grained_res) > 0:
                obj_description = obj_fine_grained_res[0][2]

            # 若存在fine-grained_type: 全都要 会引入noise?
            subj_fine_grained_type = []
            for it in subj_fine_grained_res:
                for ty in it:
                    subj_fine_grained_type.append(ty)
            obj_fine_grained_type = []
            for it in obj_fine_grained_res:
                for ty in it:
                    obj_fine_grained_type.append(ty)
            
            # 这里只使用粗粒度的类别
            entity_marker_context_list, entity_marker_new_tokens = get_marked_sentence(ori_tokens, subj_st, subj_ed, obj_st, obj_ed, subj_coarse_grained_type, obj_coarse_grained_type, "entity_marker")
           
            entity_marker_punct_context_list, entity_marker_punct_new_tokens = get_marked_sentence(ori_tokens, subj_st, subj_ed, obj_st, obj_ed, subj_coarse_grained_type, obj_coarse_grained_type, "entity_marker_punct")
            
            typed_marker_context_list, typed_marker_new_tokens = get_marked_sentence(ori_tokens, subj_st, subj_ed, obj_st, obj_ed, subj_coarse_grained_type, obj_coarse_grained_type, "typed_marker")
            
            typed_marker_punct_context_list, typed_marker_punct_new_tokens = get_marked_sentence(ori_tokens, subj_st, subj_ed, obj_st, obj_ed, subj_coarse_grained_type, obj_coarse_grained_type, "typed_marker_punct")
            
            # 后续list换成str的时候需要使用这样的replace函数
            # print(" ".join(entity_marker_context).replace("-LRB-", "(").replace("-RRB-", ")").replace("-LSB-", "[").replace("-RSB-", "]"))
            # print(" ".join(entity_marker_punct_context).replace("-LRB-", "(").replace("-RRB-", ")").replace("-LSB-", "[").replace("-RSB-", "]"))
            # print(" ".join(typed_marker_context).replace("-LRB-", "(").replace("-RRB-", ")").replace("-LSB-", "[").replace("-RSB-", "]"))
            # print(" ".join(typed_marker_punct_context).replace("-LRB-", "(").replace("-RRB-", ")").replace("-LSB-", "[").replace("-RSB-", "]"))
            # print("--------------")
            # 这里需要都存进InputExample里
            # "entity_marker_new_tokens": entity_marker_new_tokens,
            # "entity_marker_punct_new_tokens": entity_marker_punct_new_tokens,
            # "typed_marker_new_tokens": typed_marker_new_tokens,
            # "typed_marker_punct_new_tokens": typed_marker_punct_new_tokens,
            meta = {
                "subj_pos": line["h"]["pos"],
                "obj_pos": line["t"]["pos"],
                "subj_description": subj_description,
                "obj_description": obj_description,
                "subj_coarse_grained_type": subj_coarse_grained_type,
                "obj_coarse_grained_type": obj_coarse_grained_type,
                "entity_marker_context_list": entity_marker_context,
                "entity_marker_punct_context_list": entity_marker_punct_context,
                "typed_marker_context_list": typed_marker_context,
                "typed_marker_punct_context_list": typed_marker_punct_context,
            }
            example = InputExample(i, subj, obj, ori_tokens, context, label, pair_type=pair_type, meta=meta)
            examples.append(example)
            
            for token in entity_marker_new_tokens:
                if token not in all_entity_marker_new_tokens:
                    all_entity_marker_new_tokens.append(token)
            for token in entity_marker_punct_new_tokens:
                if token not in all_entity_marker_punct_new_tokens:
                    all_entity_marker_punct_new_tokens.append(token)
            for token in typed_marker_new_tokens:
                if token not in all_typed_marker_new_tokens:
                    all_typed_marker_new_tokens.append(token)
            for token in typed_marker_punct_new_tokens:
                if token not in all_typed_marker_punct_new_tokens:
                    all_typed_marker_punct_new_tokens.append(token)
                    
        new_tokens = {
            "all_entity_marker_new_tokens": all_entity_marker_new_tokens,
            "all_entity_marker_punct_new_tokens": all_entity_marker_punct_new_tokens,
            "all_typed_marker_new_tokens": all_typed_marker_new_tokens,
            "all_typed_marker_punct_new_tokens": all_typed_marker_punct_new_tokens
        }
        return examples, new_tokens

    @staticmethod
    def _read_json(input_file):
        with open(input_file, 'r', encoding='UTF-8') as f:
            lines = []
            for i in f.readlines():
                i = json.dumps(eval(i))
                # print(i)
                d = json.loads(i) # .txt文件里最后一行后面不能有别的空行，否则json.loads无法处理空行
                lines.append(d)
            return lines



TRAIN_SET = "train"
DEV_SET = "dev"
TEST_SET = "test"
ALL_SET = "all"
FEW_SHOT_SET = "few_shot"

# SET_TYPES = [TRAIN_SET, DEV_SET, TEST_SET, FEW_SHOT_SET, ALL_SET]
SET_TYPES = [TRAIN_SET, DEV_SET, TEST_SET]

PROCESSORS = {
    "TACRED": TACREDProcessor,
    "tacrev": ReTACREDProcessor,
    "retacred": ReTACREDProcessor,
    # "few_rel": few_relProcessor,
    # "TREX-1p": ,
    # "TREX-2p": ,
}


def load_examples(dataset_name, data_dir_parent: str, set_type: str, num_examples: int = None,
                  num_train_examples: int = 0, num_dev_examples: int = 0, seed: int = 42, mode: str = None, 
                  use_marker=False, marker_name=None, marker_position=None, sample_num_per_rel=None) -> List[InputExample]:
    """Load examples for a given dataset_name.
    because of few-shot setting, num_example & num_example_per_label need to be set
    REdataset的存储方式必须和TACRED格式、文件目录、文件名一样
    """
    
    data_dir = os.path.join(data_dir_parent, dataset_name)
    processor = PROCESSORS[dataset_name]()
    ex_str = f"num_examples={num_examples}" if num_examples is not None \
        else f"all_examples"
    logger.info(
            f"Creating examples from dataset file at {data_dir} ({ex_str}, set_type={set_type})"
        )
    # if set_type == ALL_SET:
    #     # assert((num_examples is not None) and (num_examples > 0)), "must set num_samples."
    #     all_examples = []
    #     all_examples.extend(processor.get_dev_examples(data_dir)) # 使用append()导致：[[], [], []],不是把元素直接合并为同一个list
    #     all_examples.extend(processor.get_test_examples(data_dir))
    #     all_examples.extend(processor.get_train_examples(data_dir))
    #     for i, ex in enumerate(all_examples):
    #         ex.idx = i
    #     if ((num_examples is not None) and (num_examples > 0)):
    #         assert(num_examples <= len(all_examples)), "num_examples should <= len(all_examples)"
    #         examples = _shuffle_and_restrict(all_examples, num_examples, seed) # 这之后就不再shuffle了，保持该顺序
    #     else:
    #         examples = _shuffle_and_restrict(all_examples, len(all_examples), seed)
    #     label_list = []
    #     for example in examples:
    #         label_list.append(example.label)
    #     label_distribution = Counter(label_list) # label分布：Counter对象，统计xxx有几个的dict
    #     logger.info(f"Returning {len(examples)} {set_type} examples with label dist.: {list(label_distribution.items())}")
    #     return examples
    if set_type == DEV_SET:
        examples, new_tokens = processor.get_dev_examples(data_dir)
        if mode == "small_dataset":
            # small data 不应该shuffle，因为每次用一样的数据才能看出来model 方法 有没有work
            assert(sample_num_per_rel is not None and sample_num_per_rel > 0)
            new_examples = []
            label_samples = {}
            for ex in examples:
                l = ex.label
                if l not in label_samples.keys():
                    label_samples[l] = []
                if len(label_samples[l]) >= sample_num_per_rel:
                    continue
                label_samples[l].append(ex)

            for k, v in label_samples.items():
                new_examples.extend(v)

            label_list = []
            for example in new_examples:
                label_list.append(example.label)
            label_distribution = Counter(label_list) # label分布：Counter对象，统计xxx有几个的dict
            logger.info(f"small_data: Returning {len(new_examples)} {set_type} examples with label dist.: {list(label_distribution.items())}")
            return new_examples, new_tokens

        if ((num_examples is not None) and (num_examples > 0)):
            assert(num_examples <= len(examples)), "num_examples should <= len(all_dev_examples)"
            examples = _shuffle_and_restrict(examples, num_examples, seed)
        else:
            examples = _shuffle_and_restrict(examples, len(examples), seed)
        label_list = []
        for example in examples:
            label_list.append(example.label)
        label_distribution = Counter(label_list) # label分布：Counter对象，统计xxx有几个的dict
        logger.info(f"Returning {len(examples)} {set_type} examples with label dist.: {list(label_distribution.items())}")
        return examples, new_tokens
    elif set_type == TRAIN_SET:
        examples, new_tokens = processor.get_train_examples(data_dir)
        if mode == "small_dataset":
            # small data 不应该shuffle，因为每次用一样的数据才能看出来model 方法 有没有work
            assert(sample_num_per_rel is not None and sample_num_per_rel > 0)
            new_examples = []
            label_samples = {}
            for ex in examples:
                l = ex.label
                if l not in label_samples.keys():
                    label_samples[l] = []
                if len(label_samples[l]) >= sample_num_per_rel:
                    continue
                label_samples[l].append(ex)

            for k, v in label_samples.items():
                new_examples.extend(v)

            label_list = []
            for example in new_examples:
                label_list.append(example.label)
            label_distribution = Counter(label_list) # label分布：Counter对象，统计xxx有几个的dict
            logger.info(f"small_data: Returning {len(new_examples)} {set_type} examples with label dist.: {list(label_distribution.items())}")
            return new_examples, new_tokens

        if ((num_examples is not None) and (num_examples > 0)):
            assert(num_examples <= len(examples)), "num_examples should <= len(all_train_examples)"
            examples = _shuffle_and_restrict(examples, num_examples, seed)
        else:
            examples = _shuffle_and_restrict(examples, len(examples), seed)
        label_list = []
        for example in examples:
            label_list.append(example.label)
        label_distribution = Counter(label_list) # label分布：Counter对象，统计xxx有几个的dict
        logger.info(f"Returning {len(examples)} {set_type} examples with label dist.: {list(label_distribution.items())}")
        return examples, new_tokens

    elif set_type == TEST_SET:
        examples, new_tokens = processor.get_test_examples(data_dir)
        if mode == "small_dataset":
            # small data 不应该shuffle，因为每次用一样的数据才能看出来model 方法 有没有work
            assert(sample_num_per_rel is not None and sample_num_per_rel > 0)
            new_examples = []
            label_samples = {}
            for ex in examples:
                l = ex.label
                if l not in label_samples.keys():
                    label_samples[l] = []
                if len(label_samples[l]) >= sample_num_per_rel:
                    continue
                label_samples[l].append(ex)

            for k, v in label_samples.items():
                new_examples.extend(v)

            label_list = []
            for example in new_examples:
                label_list.append(example.label)
            label_distribution = Counter(label_list) # label分布：Counter对象，统计xxx有几个的dict
            logger.info(f"small_data: Returning {len(new_examples)} {set_type} examples with label dist.: {list(label_distribution.items())}")
            return new_examples, new_tokens

        if ((num_examples is not None) and (num_examples > 0)):
            assert(num_examples <= len(examples)), "num_examples should <= len(all_test_examples)"
            examples = _shuffle_and_restrict(examples, num_examples, seed)
        else:
            examples = _shuffle_and_restrict(examples, len(examples), seed)
        label_list = []
        for example in examples:
            label_list.append(example.label)
        label_distribution = Counter(label_list) # label分布：Counter对象，统计xxx有几个的dict
        logger.info(f"Returning {len(examples)} {set_type} examples with label dist.: {list(label_distribution.items())}")
        return examples, new_tokens
    # elif set_type == FEW_SHOT_SET:
    #     all_examples = []
    #     all_examples.extend(processor.get_dev_examples(data_dir)) # 使用append()导致：[[], [], []],不是把元素直接合并为同一个list
    #     all_examples.extend(processor.get_test_examples(data_dir))
    #     all_examples.extend(processor.get_train_examples(data_dir))
    #     for i, ex in enumerate(all_examples):
    #         ex.idx = i
        
    #     assert(num_train_examples > 0 and num_dev_examples > 0 and (num_train_examples + num_dev_examples < len(all_examples))), "num_train_examples should > 0 and num_dev_examples should > 0"
    #     all_examples = _shuffle_and_restrict(all_examples, len(all_examples), seed)
    #     train_examples = all_examples[: num_train_examples]
    #     dev_examples = all_examples[num_train_examples: num_train_examples+num_dev_examples]
    #     test_examples = all_examples[num_train_examples+num_dev_examples:]
    #     return train_examples, dev_examples, test_examples
    else:
        raise ValueError(f"'set_type' must be one of {SET_TYPES}, got '{set_type}' instead")


