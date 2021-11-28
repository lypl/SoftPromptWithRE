""" This code define template from different dataset """

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List
import json
import os

import numpy as np
import logging
from numpy.lib.npyio import load

from numpy.lib.type_check import real

from utils import relation, get_new_token

logger = logging.getLogger(__name__)


class Basic_relation_loader(ABC):
    """
    This class contains functions to load relation data/template and information 
    """
    
    @abstractmethod
    def get_all_relation(self, data_dir) -> List[relation]:
        """Get a collection of `InputExample`s for the train set."""
        pass

    
# done
class TACRED_manual_realtion_loader(Basic_relation_loader):
    """
    This class contains functions to load TACRED relation data/template and information 把a2t的手动构造的template写进一个文件里，构造时注意tempalte写成列表，该类读取这个文件
    """
    
    def get_all_relation(self, data_dir, relation_name: str) -> List[relation]:
        data_path = os.path.join(data_dir, relation_name + ".txt") 
        return self._create_relation(self._read_file(data_path))
    
    @staticmethod
    def _create_relation(lines: List[Dict[str, str]]) -> List[relation]: 
        relations = []
        for line in lines:
            # template = [template for template in line["template"]] # 注意和构造的文件里格式符合 :{"relation": str, "template": [str], "id": int(见rel2id数据文件)}
            templates = line["templates"] # 文件里line["template"]已经是list了
            ID = line["ID"]
            label = line["label"]
            meta = {
                "pair_type": line["meta"]["pair_type"],
            }
            assert(templates != None)
            relations.append(relation(label, ID=ID, templates=templates, meta=meta))

        return relations
    

    @staticmethod
    def _read_file(data_path):
        with open(data_path, 'r', encoding='UTF-8') as f:
            lines = []
            for i in f.readlines():
                d = json.loads(i)
                lines.append(d)
            return lines


RELATION_LOADERS = {
    'TACRED_manT': TACRED_manual_realtion_loader,
    'retacred_manT': TACRED_manual_realtion_loader,
    'tacrev_manT':TACRED_manual_realtion_loader,
}


# 构造soft-prompt这个部分放在wrapper里，不要在这里写
# def construct_other_type_prompt():
#     pass
# def get_soft_template_with_positions(positions: List[str], tot_new_tokens, max_num_relvec, relvec_num_F: int=2, relvec_num_M: int=2, relvec_num_E: int=2):
#     """ relvec_num: 每个位置上填充的个数 """
#     res = []
#     for position in positions:
#         ss = ''
#         if 'F' in position:
#             for k in range(relvec_num_F):
#                 ss = ss + get_new_token(tot_new_tokens, max_num_relvec) + ' ' 
#                 tot_new_tokens += 1
        
#         ss += '{subj}' + ' '
#         if 'M' in position:
#             for k in range(relvec_num_M):
#                 ss = ss + get_new_token(tot_new_tokens, max_num_relvec) + ' '
#                 tot_new_tokens += 1

#         ss += '{obj}'
#         if 'E' in position:
#             for k in range(relvec_num_E):
#                 ss = ss + ' ' + get_new_token(tot_new_tokens, max_num_relvec)
#                 tot_new_tokens += 1
#         assert(tot_new_tokens <= max_num_relvec), "tot_new_tokens > max_num_relvec, change relvec_num_F/M/E"
#         res.append(ss)
#     return res, tot_new_tokens


# def constuct_relvec(config: Dict[str, str], tot_new_tokens, max_num_relvec) -> List[str]:
#     """ 得到str类型的relvec，并把base_model里面的word_embedding初始化好了"""
#     if config["relvec_construct_mode"] == 'from_manT':
#         # 从mantT中构造relvec，把template中原token对应的embedding赋给[Vi],并确定会使用多少个[Vi]
#         # 放在wrapper里面重写
#         pass
#         # assert(config["manual_template"] is not None), "prompt_building_config['manual_template'] should not be None"
#         # soft_templates, tot_new_tokens = get_soft_template_from_a_manT(config["manual_template"], tot_new_tokens, max_num_relvec)
#     elif config["relvec_construct_mode"] == 'with_position':
#         # FME, 不需要给word_embeddin层赋值
#         assert((config["positions"] is not None) and (config["tokenizer"] is not None)), "prompt_building_config['positions'] should not be None and tokenizer should not None"
#         soft_templates, tot_new_tokens = get_soft_template_with_positions(config["tokenizer"], config["positions"], tot_new_tokens, max_num_relvec, relvec_num_F=config["relvec_num_F"], relvec_num_M=config["relvec_num_M"], relvec_num_E=config["relvec_num_E"])
#     else:
#         # 其他构造relvec的方法
#         pass

#     # debug
#     # print(soft_templates)
#     return soft_templates, tot_new_tokens
    
# def build_prompt(relation: relation, prompt_type: str, tot_new_tokens, max_num_relvec, relvec_construct_mode) -> List[str]:
#     """ return [prompt1, prompt2, ...] """
#     prompt_building_config: Dict[str, str] = None
#     if prompt_type == "relvec":
#         # "relvec_construct_mode": ["from_manT", 'with_position'("positions": ["F", 'M', 'E', 'FM', 'FE', 'ME', 'FME']), ...]
#         if relvec_construct_mode == "from_manT":
#             # 放在wrapper里面重写
#             pass
#             # prompt_building_config = {
#             #     "relvec_construct_mode": "from_manT",
#             #     "manual_template": relation.templates[0] # 这里选哪个manT作为prompt的初始化，需要进一步探讨⭐
#             #     "tokenizer": 
#             # }
            
#             # return constuct_relvec(prompt_building_config, tot_new_tokens, max_num_relvec)
#         elif relvec_construct_mode == "with_position":
#             prompt_building_config = {
#                 "relvec_construct_mode": "with_position",
#                 "positions": "FME",
#                 "relvec_num_F": 2,
#                 "relvec_num_M": 2,
#                 "relvec_num_E": 2
#             }
#             return constuct_relvec(prompt_building_config, tot_new_tokens, max_num_relvec)
#     else:
#         # 其他构造prompt的方法 如span_prompt\transfor_prompt等
#         pass
    

def get_relations(data_dir: str, dataset_name: str) -> List[relation]:
    """This function must be call to get relations, which will feed to wrapper"""
    relation_loader = RELATION_LOADERS[dataset_name]()
    return relation_loader.get_all_relation(data_dir, dataset_name)

    # if template_type == "manual_template" or (template_type != "manual_template" and relvec_construct_mode == "from_manT"):
    #     return relation_loader.get_all_relation(data_dir, dataset_name)
    # else:
    #     # 构建prompt，重写relation.templates内容
    #     assert((prompt_type is not None) and (max_num_relvec is not None) and (relvec_construct_mode is not None))
    #     all_relations = relation_loader.get_all_relation(data_dir, dataset_name)
    #     tot_new_tokens = 0
    #     for i in range(len(all_relations)):
    #         prompt_list, tot_new_tokens = build_prompt(all_relations[i], prompt_type, tot_new_tokens, max_num_relvec, relvec_construct_mode) # 将tot_new_tokens作为全局变量传给每个构建relation.prompt的过程
    #         all_relations[i].templates = prompt_list
    #     return all_relations
