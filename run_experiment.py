""" This code use A2T 4 settings to run experiments ---! not comlete! ---"""

# import argparse
import json
import argparse
import os
import sys
import posixpath
from pprint import pprint
from collections import Counter
from numpy.lib.npyio import load
from abc import ABC
from typing import List, Dict

import torch
import wandb

from utils import save_result_for_a_experiment
from load_config import NLIEVAL_CONFIG, NLITRAIN_CONFIG, NLIWRAPPER_CONFIG, load_config_from_file, construct_config
from load_REdata2example import load_examples, PROCESSORS, SET_TYPES
from load_relation import get_relations
from using_wrapper_method import NLIforward, NLIEvalConfig, prompt_tuning, NLITrainConfig, marker_tuning
import numpy as np
from NLImodelsWrapper import NLIRelationWrapper, NLIWrapperConfig
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

import logging

os.environ["TOKENIZERS_PARALLELISM"] = "true" # 不设置这个会warning：The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks，这个问题的原因是tokenizer版本高，并需要指定这个设置

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)

def pause():
    programPause = input("Press the <ENTER> key to continue...")

def load_optiprompt(save_prompt_dir: str, WrapperConfig: NLIWrapperConfig):
    # load bert model (pre-trained)
    wrapper = NLIRelationWrapper(WrapperConfig)
    original_vocab_size = wrapper.original_vocab_size
    # prepare_for_dense_prompt() # 在wrapper.init里面做好
    
    logger.info("Loading OptiPrompt's [V]s..")
    with open(os.path.join(save_prompt_dir, 'prompt_vecs.npy'), 'rb') as f:
        vs = np.load(f)
        print(vs.shape)
    
    # copy fine-tuned new_tokens to the pre-trained model
    with torch.no_grad():
        wrapper.base_model.embeddings.word_embeddings.weight[original_vocab_size:] = torch.Tensor(vs)
    return wrapper

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:3', help='device idx')
    parser.add_argument('--num_train_epoch', type=int, default=None, help='num_train_epoch')
    parser.add_argument('--experiment_id', type=str, default='3', help='run which experiment')
    parser.add_argument('--save_init_result', type=bool, default=False, help='save manT init res in wrong samples fold')
    parser.add_argument('--marker_name', type=str, default='entity_marker', help='use which marker')
    parser.add_argument('--marker_position', type=str, default='context', help='where use marker') 
    parser.add_argument('--marker_num_train_epoch', type=int, default=8, help='marker tuning epoch num') 
    new_args = parser.parse_args() # args.xxx

    experiments_dir = os.path.join(os.getcwd(), "experiments_configs")
    experiments: List[str] = os.listdir(experiments_dir) # 得到该目录下的每个文件名称

    for experiment in experiments:
        exprmt_dir = os.path.join(experiments_dir, experiment)
        config_path = os.path.join(exprmt_dir, "experiment_config.json")
        with open(config_path, 'r', encoding='UTF-8') as f:
            args = json.load(f)
            if args["experiment_info"]["name"] != ("experiment_" + new_args.experiment_id):
                continue
            if args["experiment_info"]["get_template_method"] == "template_already_in_datasets":
                # step1: load config.json -> for all varience to control the experiments
        
                # step2: load REdataset -> use datasetProcessor get InputExamples
                REDataset_dir = eval(args["REdataset_dir_parent"])
                # 加载test.txt里的所有数据
                RE_dataset_examples = load_examples(args["REdataset_name"], REDataset_dir, args["load_REdata_type"]) # args.load_REdata_type=all, args.REdata_num_corespnd_to_type=不用传，函数里能做
                # for debug
                # test_data = load_examples(args["REdataset_name"], REDataset_dir, "test", mode="NA_1/3")
                # RE_dataset_examples = test_data[:20]
                # step3: use wrapper with relations to get complete examples and get features
                REWrapper_args: NLIWrapperConfig = load_config_from_file(args["NLIWrapper_config_file_path"], NLIWRAPPER_CONFIG)
                REWrapper: NLIRelationWrapper = NLIRelationWrapper(REWrapper_args)
                NLIEvalConfig_args:NLIEvalConfig = load_config_from_file(args["NLIEvalConfig_config_file_path"], NLIEVAL_CONFIG)
                
                forward_result = NLIforward(REWrapper, RE_dataset_examples, NLIEvalConfig_args)

                # step4: use wrapper to forward the NLI model -> get result in natrual language and coresponding label and entailment probability 到一个命名详细的文件夹里，包括实验config、结果、指标等
                save_result_for_a_experiment(forward_result, eval(args["RE_result_dir"]), args["experiment_info"]["name"])

            if args["experiment_info"]["get_template_method"] == "prompt":
                """ 使用soft prompt作为代替manT """
                """ 说明:流程：构造train \ dev \ test InputExample -> 在完善InputExample的texts_to_tokenizer时，
                使用构造的prompt_template,    tuning prompt的本质是：tuning [Vi] 的embedding，让这些embedding 的特定组合序列（即prompt_template）更适合让NLImodel得到更高的结果
                ⭐(让该relation的该prompt/prompts更能反映该relation的信息 or (让该relation下的所有example的结果更高?)，这个prompt可能需要和context相关联，下一步改进计划？)
                因此：每个relation 有一个prompt(或一组prompts，下一步考虑)，让该relation下的所有example的结果更高？， 而是让所有example在用其对应的relation的prompt时entailment概率更高，用其他relation的prompt时不高⭐ 而不是每个example有一个prompt
                注意：relation1: [V1]~[V4], relation2: [V5]~[V10], relation3: [V11]~[V20] ....
                prompt分类：1）使用manT初始化prompt_template， 多个的时候使用哪个manT？——涉及到prompt mining等问题，但可以先使用每个relation的第一个manT
                      2）使用位置信息，则随机初始化（或者把[Vi]加进vocab中以后就不用管它的初始化部分）
                      3）spanPrompt等，
                      * 这些改进的方法：探讨为什么一些prompt的效果更好 -> 修改prompt的方法本质是prompt初始化方式，即搜索的图（few-shot learning 综述（在paper/slide里面）的搜索的三个图：优化搜索原/起点、剪枝搜索空间、...后续看这个图）"""
                REDataset_dir = eval(args["REdataset_dir_parent"])
                
                
                # 使用small_data进行model方法检测,
                train_data, train_new_tokens = load_examples(args["REdataset_name"], REDataset_dir, "train", mode="small_dataset", sample_num_per_rel=100)
                dev_data, dev_new_tokens = load_examples(args["REdataset_name"], REDataset_dir, "dev", mode="small_dataset", sample_num_per_rel=100)
                test_data, test_new_tokens = load_examples(args["REdataset_name"], REDataset_dir, "test", mode="small_dataset", sample_num_per_rel=100)
                # train_data = train_data[:100]
                # dev_data = dev_data[:50]
                # test_data = test_data[:30]

                # pause()
                
                # REWrapper_args: NLIWrapperConfig, Wrapper_config_to_log: Dict
                Wrapper_config_to_log, REWrapper_args = load_config_from_file(args["NLIWrapper_config_file_path"], NLIWRAPPER_CONFIG)
                if REWrapper_args.use_marker:
                    tr_new_tokens = None
                    dv_new_tokens = None
                    te_new_tokens = None
                    ty = None
                    if REWrapper_args.marker_name == "entity_marker":
                        ty = "all_entity_marker_new_tokens"
                    elif REWrapper_args.marker_name == "entity_marker_punct":
                        ty = "all_entity_marker_punct_new_tokens"
                    elif REWrapper_args.marker_name == "typed_marker":
                        ty = "all_typed_marker_new_tokens"
                    elif REWrapper_args.marker_name == "typed_marker_punct":
                        ty = "all_typed_marker_punct_new_tokens"
                    tr_new_tokens = train_new_tokens[ty]
                    dv_new_tokens = dev_new_tokens[ty]
                    te_new_tokens = test_new_tokens[ty]

                    new_tokens = [] # train/dev/test里的 把所有可能的new_token都弄出来
                    for token in tr_new_tokens:
                        if token not in new_tokens:
                            new_tokens.append(token)
                    for token in dv_new_tokens:
                        if token not in new_tokens:
                            new_tokens.append(token)
                    for token in te_new_tokens:
                        if token not in new_tokens:
                            new_tokens.append(token)
                REWrapper: NLIRelationWrapper = NLIRelationWrapper(REWrapper_args, new_tokens)
                # NLITrainConfig_args:NLITrainConfig
                Train_config_to_log, NLITrainConfig_args = load_config_from_file(args["NLITrainConfig_config_file_path"], NLITRAIN_CONFIG)
                # hard code
                NLITrainConfig_args.device = new_args.device
                REWrapper_args.marker_name = new_args.marker_name
                REWrapper_args.marker_position = new_args.marker_position
                NLITrainConfig_args.marker_num_train_epoch = new_args.marker_num_train_epoch
                NLITrainConfig_args.num_train_epoch = new_args.num_train_epoch
                Train_config_to_log["num_train_epoch"] = new_args.num_train_epoch
                Train_config_to_log["marker_name"] = new_args.marker_name
                Train_config_to_log["marker_position"] = new_args.marker_position
                Train_config_to_log["marker_num_train_epoch"] = new_args.marker_num_train_epoch

                NLITrainConfig_args.save_optiprompt_dir = os.path.join(NLITrainConfig_args.save_optiprompt_dir, args["experiment_info"]["name"])
                NLITrainConfig_args.marker_save_model_dir = os.path.join(NLITrainConfig_args.marker_save_model_dir, REWrapper_args.marker_name, args["REdataset_name"])
                
                # NLIEvalConfig_args:NLIEvalConfig
                Eval_config_to_log, NLIEvalConfig_args = load_config_from_file(args["NLIEvalConfig_config_file_path"], NLIEVAL_CONFIG)
                # hard code
                NLIEvalConfig_args.device = new_args.device

                wdb_config = {}
                wdb_config["NA_threshold"] = 0.8
                for k, v in Wrapper_config_to_log.items():
                    if ("relations" in k) or ("embedding" in k) or ("valid" in k):
                        continue
                    wdb_config[k] = v
                for k, v in Train_config_to_log.items():
                    if k == "save_optiprompt_dir_parent" or k == "device" or k == "max_step":
                        continue
                    wdb_config[k] = v
                for k, v in Eval_config_to_log.items():
                    if k == "device":
                        continue
                    wdb_config[k] = v
                # print(wdb_config)
                # pause()
                wandb.init(
                    project="Verbalize_RE",
                    config=wdb_config,
                    notes="marker_tuning",
                    name=REWrapper_args.marker_name
                )
                # init_exp_info = {
                #     "dataset_name": args["REdataset_name"],
                #     "epoch_num": new_args.num_train_epoch,
                #     "tuning_or_init": "init",
                # }
                # tuning_exp_info = {
                #     "dataset_name": args["REdataset_name"],
                #     "epoch_num": new_args.num_train_epoch,
                #     "tuning_or_init": "tuning",
                # }
                exp_info = {
                    "marker_name": REWrapper_args.marker_name,
                    "dataset_name": args["REdataset_name"],
                    "epoch_num": NLITrainConfig_args.marker_num_train_epoch
                }

                # debug
                if new_args.save_init_result:
                    forward_result = NLIforward(REWrapper, test_data, NLIEvalConfig_args)
                    save_result_for_a_experiment(forward_result, eval(args["RE_result_dir"]), args["experiment_info"]["name"], exp_info) # just for debug
                    # pause()
                # prompt_tuning(REWrapper, train_data, dev_data, test_data, NLITrainConfig_args, NLIEvalConfig_args) # tuning softprompt, wrapperConfig里指定构造哪种prompt
                marker_result = marker_tuning(REWrapper, train_data, dev_data, test_data, NLITrainConfig_args, NLIEvalConfig_args)
                
                save_result_for_a_experiment(marker_result, eval(args["RE_result_dir"]), args["experiment_info"]["name"], exp_info, tag="marker_tuning")
                wandb.finish()
                

            if args["experiment_info"]["get_template_method"] == "test_prompt_init":
                # experiment_4
                REDataset_dir = eval(args["REdataset_dir_parent"])
                # 加载test.txt中所有的data
                RE_dataset_examples = load_examples(args["REdataset_name"], REDataset_dir, args["load_REdata_type"]) # args.load_REdata_type=all, args.REdata_num_corespnd_to_type=不用传，函数里能做
                # for debug
                # RE_dataset_examples = RE_dataset_examples[:100]
                #
                REWrapper_args: NLIWrapperConfig = load_config_from_file(args["NLIWrapper_config_file_path"], NLIWRAPPER_CONFIG)
                REWrapper: NLIRelationWrapper = NLIRelationWrapper(REWrapper_args)
                
                NLIEvalConfig_args:NLIEvalConfig = load_config_from_file(args["NLIEvalConfig_config_file_path"], NLIEVAL_CONFIG)

                best_result_for_task = NLIforward(REWrapper, RE_dataset_examples, NLIEvalConfig_args)
                print("after init the template embeddings, precision is: ")
                print(best_result_for_task['scores']["top-1"])
                save_result_for_a_experiment(best_result_for_task, eval(args["RE_result_dir"]), args["experiment_info"]["name"])

                pass
        


"""
命令行： python run_experiment.py --experiment_id experiment_3 --device "cuda:3" --num_train_epoch 6 --save_init_result True
"""