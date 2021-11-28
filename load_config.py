import json
import os

import torch

from NLImodelsWrapper import NLIWrapperConfig
from using_wrapper_method import NLIEvalConfig, NLITrainConfig
from typing import ContextManager, Dict, List

NLIEVAL_CONFIG = "NLIEvalConfig"
NLITRAIN_CONFIG = "NLITrainConfig"
NLIWRAPPER_CONFIG = "NLIWrapperConfig"

CONFIG_TYPES = [NLIEVAL_CONFIG, NLITRAIN_CONFIG, NLIWRAPPER_CONFIG]
def construct_config(config_type: str, content: Dict):
    if config_type == NLIEVAL_CONFIG:
        device = content["device"]
        per_gpu_eval_batch_size = content["per_gpu_eval_batch_size"]
        metrics = eval(content["metrics"])
        topk = content["topk"]
        return NLIEvalConfig(device=device, topk=topk, per_gpu_eval_batch_size=per_gpu_eval_batch_size, metrics=metrics)
    elif config_type == NLITRAIN_CONFIG:
        device = content["device"]
        learning_rate = content["learning_rate"] 
        eval_step = content["eval_step"]
        warmup_proportion = content["warmup_proportion"]
        save_optiprompt_dir_parent = eval(content["save_optiprompt_dir_parent"])# "os.path.join(os.getcwd(), 'experiments_results')"
        # save_optiprompt_dir = os.path.join(save_optiprompt_dir_parent, "experiment") # 这里最后在run_experiments.py里补充
        max_step = content["max_step"]
        num_train_epoch = content["num_train_epoch"]
        train_batch_size = content["train_batch_size"]
        eval_batch_size = content["eval_batch_size"]
        check_step = content["check_step"] 
        gradient_accumulation = content["gradient_accumulation"]
        seed = content["seed"]
        # template_type = content["template_type"]
        # relvec_construct_mode = content["relvec_construct_mode"]
        prompt_type = content["prompt_type"]
        # max_num_relvec = content["max_num_relvec"]

        return NLITrainConfig(device=device, save_optiprompt_dir=save_optiprompt_dir_parent, 
                              prompt_type=prompt_type, train_batch_size=train_batch_size, 
                              eval_batch_size=eval_batch_size, eval_step=eval_step, max_steps=max_step, num_train_epoch=num_train_epoch, 
                              gradient_accumulation_steps=gradient_accumulation, check_step=check_step, learning_rate=learning_rate, 
                              warmup_proportion=warmup_proportion, seed=seed)
    elif config_type == NLIWRAPPER_CONFIG:
        model_type = content["model_type"]
        model_name_or_path = content["model_name_or_path"]
        wrapper_type = content["wrapper_type"]
        dataset_name = content["dataset_name"]
        max_seq_length = content["max_seq_length"]
        relations_data_dir_parent = eval(content["relations_data_dir_parent"]) #  格式
        relations_data_name = content["relations_data_name"] 
        relations_data_dir = os.path.join(relations_data_dir_parent, dataset_name, "for_template")
        valid_conditions = content["valid_conditions"]
        max_num_relvec = content["max_num_relvec"]
        # template_type = content["template_type"]
        relvec_construct_mode = content["relvec_construct_mode"]
        prompt_type = content["prompt_type"] 
        use_rel_embedding = eval(content["use_rel_embedding"])
        rel_id2embeddings_id_path = eval(content["rel_id2embeddings_id_path"])
        rel_embeddings_path = eval(content["rel_embeddings_path"])

        use_marker = eval(content["use_marker"])
        marker_position = content["marker_position"]
        marker_name = content["marker_name"]

        return NLIWrapperConfig(model_type=model_type, model_name_or_path=model_name_or_path, wrapper_type=wrapper_type, dataset_name=dataset_name, max_seq_length=max_seq_length, 
                                max_num_relvec=max_num_relvec, relations_data_dir=relations_data_dir, use_rel_embedding=use_rel_embedding, rel_id2embeddings_id_path=rel_id2embeddings_id_path,
                                rel_embeddings_path=rel_embeddings_path,
                                relations_data_name=relations_data_name, prompt_type=prompt_type, relvec_construct_mode=relvec_construct_mode, valid_conditions=valid_conditions,
                                use_marker=use_marker, marker_position=marker_position, marker_name=marker_name)
    else:
        raise ValueError(f"'config_type' must be one of {CONFIG_TYPES}, got '{config_type}' instead")


def load_config_from_file(path: str, config_type: str):
    path = eval(path)
    with open(path, 'r', encoding='UTF-8') as f:
        config_content = json.load(f)
        config = construct_config(config_type, config_content)
        return config_content, config