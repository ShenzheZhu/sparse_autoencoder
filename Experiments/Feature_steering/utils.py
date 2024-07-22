# 设置环境变量
import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 导入库
import torch
import blobfile as bf
import transformer_lens
import sparse_autoencoder

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


# 加载自编码器
def load_autoencoder(location, layer_index, device, size=32):
    if size == 32 :
        path = sparse_autoencoder.paths.v5_32k(location, layer_index)
    else:
        path = sparse_autoencoder.paths.v5_128k(location, layer_index)
    print(path)
    with bf.BlobFile(path, mode="rb") as f:
        state_dict = torch.load(f)
        autoencoder = sparse_autoencoder.Autoencoder.from_state_dict(state_dict)
        autoencoder.to(device)
    return autoencoder


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
                activations_dict[feature_key][prompt][token] = value.item()
            else:
                print(f"Skipping ambiguous token index: {nonzero_indices}")

        total_activations_count += len(top_k_values)

    # Print the total number of activations extracted
    print(f"Total activations extracted: {total_activations_count}")

    # Optionally, return the total number of activations
    return activations_dict