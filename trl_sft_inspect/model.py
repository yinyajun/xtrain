from transformers import PreTrainedTokenizerBase, PreTrainedModel


def qwen25_smoke(tokenizer: PreTrainedTokenizerBase) -> PreTrainedModel:
    # 构造一个随机初始化的小 Qwen2 CausalLM，结构真实、权重很小，适合观察数据流。
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=512,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return Qwen2ForCausalLM(config)
