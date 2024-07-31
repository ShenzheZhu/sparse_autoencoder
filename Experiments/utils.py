import json
import os
import re
from decimal import Decimal
import blobfile as bf
import ijson
import numpy as np
import pandas as pd
import torch
import transformer_lens
import sparse_autoencoder
from transformers import GPT2LMHeadModel, AutoTokenizer, GPT2Tokenizer

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

def update_csv_file(filename, new_activations):
    # 尝试读取现有的 CSV 文件
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        df = pd.DataFrame(columns=['Feature', 'Index', 'SubIndex', 'Value'])

    # 将新的激活值转换为数据帧
    new_data = {
        'Feature': [],
        'Index': [],
        'SubIndex': [],
        'Value': []
    }

    for new_feature_key, new_feature_data in new_activations.items():
        for new_prompt_key, new_prompt_data in new_feature_data.items():
            for sub_index, value in new_prompt_data.items():
                new_data['Feature'].append(new_feature_key)
                new_data['Index'].append(int(new_prompt_key))
                new_data['SubIndex'].append(sub_index)
                new_data['Value'].append(value)

    new_df = pd.DataFrame(new_data)
    # 删除空值或全为NA值的列
    new_df.dropna(axis=1, how='all', inplace=True)
    # 合并新的数据帧到现有的数据帧中
    updated_df = pd.concat([df, new_df], ignore_index=True)
    # 删除重复项，保留最新值
    updated_df.drop_duplicates(subset=['Feature', 'Index', 'SubIndex'], keep='last', inplace=True)

    updated_df.sort_values(by=['Feature', 'Index', 'SubIndex'], inplace=True)
    # 保存更新后的数据帧到 CSV 文件
    updated_df.to_csv(filename, index=False)

def extract_feature_number(feature_str):
    # 提取 feature 字符串中的数字部分
    match = re.search(r'\d+', feature_str)
    return int(match.group()) if match else float('inf')

def sort_csv_file(filename):
    # 尝试读取现有的 CSV 文件
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return
    # 提取Feature列中的数字部分
    df['FeatureNumber'] = df['Feature'].apply(extract_feature_number)
    # 按 FeatureNumber 从小到大排序，然后按 Value 从大到小排序
    df.sort_values(by=['FeatureNumber', 'Value'], ascending=[True, False], inplace=True)
    # 删除临时的 FeatureNumber 列
    df.drop(columns=['FeatureNumber'], inplace=True)
    # 保存排序后的数据帧到 CSV 文件
    df.to_csv(filename, index=False)
    print(f"File {filename} has been sorted and saved.")

def update_numpy_file(filename, new_activations):
    # 尝试读取现有的 NumPy 文件
    try:
        data = np.load(filename, allow_pickle=True).item()
        df = pd.DataFrame(data)
    except (FileNotFoundError, OSError):
        df = pd.DataFrame(columns=['Feature', 'Index', 'SubIndex', 'Value'])
    # 将新的激活值转换为数据帧
    new_data = {
        'Feature': [],
        'Index': [],
        'SubIndex': [],
        'Value': []
    }
    for new_feature_key, new_feature_data in new_activations.items():
        for new_prompt_key, new_prompt_data in new_feature_data.items():
            for sub_index, value in new_prompt_data.items():
                new_data['Feature'].append(new_feature_key)
                new_data['Index'].append(int(new_prompt_key))
                new_data['SubIndex'].append(sub_index)
                new_data['Value'].append(value)

    new_df = pd.DataFrame(new_data)
    # 删除空值或全为NA值的列
    new_df.dropna(axis=1, how='all', inplace=True)
    # 合并新的数据帧到现有的数据帧中
    updated_df = pd.concat([df, new_df], ignore_index=True)
    # 删除重复项，保留最新值
    updated_df.drop_duplicates(subset=['Feature', 'Index', 'SubIndex'], keep='last', inplace=True)
    # 保存更新后的数据帧到 NumPy 文件
    data_dict = updated_df.to_dict('list')
    np.save(filename, data_dict)
    print(f"File {filename} has been updated and saved.")

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


# 加载模型
def load_model(model_name, center_writing_weights=False):
    model = transformer_lens.HookedTransformer.from_pretrained(model_name, center_writing_weights=center_writing_weights)
    device = next(model.parameters()).device
    return model, device

# 处理输入
def process_input(model, prompt):
    tokens_id = model.to_tokens(prompt)  # (1, n_tokens)
    tokens_str = model.to_str_tokens(prompt)
    with torch.no_grad():
        logits, activation_cache = model.run_with_cache(tokens_id, remove_batch_dim=True)
    return tokens_id, tokens_str, activation_cache

# 提取激活
def get_activation(activation_cache, layer_index=6, location="resid_post_mlp"):
    transformer_lens_loc = {
        "mlp_post_act": f"blocks.{layer_index}.mlp.hook_post",
        "resid_delta_attn": f"blocks.{layer_index}.hook_attn_out",
        "resid_post_attn": f"blocks.{layer_index}.hook_resid_mid",
        "resid_delta_mlp": f"blocks.{layer_index}.hook_mlp_out",
        "resid_post_mlp": f"blocks.{layer_index}.hook_resid_post",
    }[location]
    return activation_cache[transformer_lens_loc]

def load_autoencoder_from_local(layer_index, device, size=32):
    save_path = f"model/gpt2_sae/sae_state_{size}k_layer_{layer_index}.pt"  # 本地文件路径
    state_dict = torch.load(save_path, map_location=device)  # 从本地加载状态字典
    autoencoder = sparse_autoencoder.Autoencoder.from_state_dict(state_dict)
    autoencoder.to(device)  # 将模型移到指定的设备
    return autoencoder

def download_autoencoder(location, layer_index, size=32):
    if size == 32:
        path = sparse_autoencoder.paths.v5_32k(location, layer_index)
    else:
        path = sparse_autoencoder.paths.v5_128k(location, layer_index)
    directory = 'model/gpt2_sae'
    if not os.path.exists(directory):
        os.makedirs(directory)
    save_path = f"model/gpt2_sae/sae_state_{size}k_layer_{layer_index}.pt"  # 定义本地保存路径
    with bf.BlobFile(path, mode="rb") as f:
        print(f"Downloading SAE from: {path}")
        state_dict = torch.load(f)
        torch.save(state_dict, save_path)  # 保存状态字典到本地文件
        print(f"State dictionary saved to {save_path}")

# 编码和解码激活张量
def encode_decode(autoencoder, input_tensor):
    with torch.no_grad():
        latent_activations, info = autoencoder.encode(input_tensor)
        reconstructed_activations = autoencoder.decode(latent_activations, info)
    return latent_activations, reconstructed_activations

# 计算误差并打印结果
def calculate_normalized_mse(input_tensor, reconstructed_activations):
    normalized_mse = (reconstructed_activations - input_tensor).pow(2).sum(dim=1) / (input_tensor).pow(2).sum(dim=1)
    return normalized_mse

def extract_activations(prompt, tokens, latent_activations, top_k=32, activation_threshold=3):
    activations_dict = {}
    prompt_key = prompt  # 根据需要设置不同的 prompt 标识符

    total_activations_count = 0
    
    # 遍历所有 feature
    for feature_index in range(latent_activations.shape[1]):
        # 获取该 feature 的所有激活值
        feature_activations = latent_activations[:, feature_index]
        
        # 仅提取 top k 非零激活值
        non_zero_activations = feature_activations[(feature_activations != 0) & (feature_activations >= activation_threshold)]
        if non_zero_activations.numel() == 0:
            continue
        top_k_values, top_k_indices = torch.topk(non_zero_activations, min(top_k, non_zero_activations.numel()))

        # 构建特征激活字典
        feature_key = f"Feature {feature_index}"
        activations_dict[feature_key] = {prompt_key: {}}
        for value, index in zip(top_k_values, top_k_indices):
            nonzero_indices = (feature_activations == value).nonzero(as_tuple=True)
            if len(nonzero_indices[0]) == 1:  # 确保只有一个元素
                token_index = nonzero_indices[0].item()
                token = tokens[token_index]
                activations_dict[feature_key][prompt][token] = [f"idx:{token_index}",value.item()]
            else:
                print(f"Skipping ambiguous token index: {nonzero_indices}")

        total_activations_count += len(top_k_values)

    # Print the total number of activations extracted
    print(f"Total activations extracted: {total_activations_count}")

    # Optionally, return the total number of activations
    return activations_dict

# 加载模型
def load_model_hf(model_name):
    model = GPT2LMHeadModel.from_pretrained(model_name, output_hidden_states=True)
    auto_tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, auto_tokenizer, device

# 处理输入
def process_input_hf(model, tokenizer, prompt):
    tokens_id = tokenizer(prompt, return_tensors='pt').input_ids.to(model.device)
    tokens_str = tokenizer.convert_ids_to_tokens(tokens_id[0])
    with torch.no_grad():
        outputs = model(tokens_id)
    activation_cache = outputs.hidden_states
    return tokens_id, tokens_str, activation_cache


def get_activation_hf(activation_cache, layer_index=6):
    return activation_cache[layer_index][0]