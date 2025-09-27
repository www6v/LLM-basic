from transformers.models.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM,Qwen3MoeModel
import torch

def run_qwen3_moe():
    qwen3_moe_config = Qwen3MoeConfig(
        # 基础参数
        vocab_size=151936,
        hidden_size=2048,
        intermediate_size=1000,
        num_hidden_layers=6,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=64,
        max_position_embeddings=40960,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        
        # 注意力机制参数
        attention_bias=False,
        attention_dropout=0.0,
        use_sliding_window=False,
        sliding_window=None,
        max_window_layers=2,
        
        # 分词器相关
        bos_token_id=151643,
        eos_token_id=151645,
        
        # 初始化参数
        initializer_range=0.02,
        
        # 模型类型和数据类型
        model_type="qwen3_moe",
        torch_dtype="bfloat16",
        
        # RoPE旋转位置编码
        rope_theta=1000000.0,
        rope_scaling=None,
        
        # 嵌入层参数
        tie_word_embeddings=False,
        
        # MoE混合专家参数
        decoder_sparse_step=2,
        moe_intermediate_size=768,
        num_experts=4,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        
        # 其他参数
        mlp_only_layers=[5],
        use_cache=True,
    )
    model = Qwen3MoeModel(config=qwen3_moe_config)

    input_ids = torch.randint(low=0, high=qwen3_moe_config.vocab_size,size=(2,18))

    res = model(input_ids=input_ids)

    print(res)


if __name__ == '__main__':
    run_qwen3_moe()