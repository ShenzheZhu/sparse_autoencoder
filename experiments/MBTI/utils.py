import json
import os
from decimal import Decimal

import ijson
import torch


def extract_activations(prompt, tokens, latent_activations, top_k=32):
    activations_dict = {}
    prompt_key = prompt  # 根据需要设置不同的 prompt 标识符

    # 遍历所有 feature
    for feature_index in range(latent_activations.shape[1]):
        # 获取该 feature 的所有激活值
        feature_activations = latent_activations[:, feature_index]

        # 仅提取 top k 非零激活值
        non_zero_activations = feature_activations[feature_activations != 0]
        if non_zero_activations.numel() == 0:
            continue

        top_k_values, top_k_indices = torch.topk(non_zero_activations, min(top_k, non_zero_activations.numel()))

        # 构建特征激活字典
        feature_key = f"Feature {feature_index}"
        activations_dict[feature_key] = {prompt_key: {}}

        for value, index in zip(top_k_values, top_k_indices):
            token_index = (feature_activations == value).nonzero(as_tuple=True)[0].item()
            token = tokens[token_index]
            activations_dict[feature_key][prompt_key][f"{token}"] = value.item()

    return activations_dict


def decimal_default(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError

def update_json_file(filename, new_activations):
    activations_dict = {}
    # 尝试读取现有的 JSON 文件
    try:
        with open(filename, "r") as json_file:
            parser = ijson.items(json_file, '')
            for item in parser:
                activations_dict.update(item)
                break
    except FileNotFoundError:
        pass

    # 合并新激活值到现有字典中
    for new_feature_key, new_feature_data in new_activations.items():
        if new_feature_key not in activations_dict:
            activations_dict[new_feature_key] = new_feature_data
        else:
            for new_prompt_key, new_prompt_data in new_feature_data.items():
                if new_prompt_key not in activations_dict[new_feature_key]:
                    activations_dict[new_feature_key][new_prompt_key] = new_feature_data[new_prompt_key]
                else:
                    activations_dict[new_feature_key][new_prompt_key].update(new_feature_data[new_prompt_key])

    # 保存更新后的字典到 JSON 文件
    with open(filename, "w") as json_file:
        json.dump(activations_dict, json_file, indent=4, default=decimal_default)


def count_activations(filename):
    try:
        with open(filename, "r") as json_file:
            activations_dict = json.load(json_file)
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return 0

    activation_count = 0

    # 遍历 JSON 结构，统计激活值的数量
    for feature_key, feature_data in activations_dict.items():
        for prompt_key, prompt_data in feature_data.items():
            activation_count += len(prompt_data)

    return activation_count
