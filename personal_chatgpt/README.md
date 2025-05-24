# personal_chatgpt


| Stage                    | Pre-Training                     | Supervised Fine-Tuning               | Reward Modeling                | Reinforcement Learning                     |
|--------------------------|----------------------------------|--------------------------------------|--------------------------------|--------------------------------------------|
| Training Data            | Trillions of tokens from websites, books, etc. | Prompt-Response Pairs for various tasks | Response Preferences           | Prompts                                     |
| Modeling Method          | Language Modeling (negative log-likelihood) | Language Modeling (negative log-likelihood) | Binary Classification or Regression | Reinforcement Learning (Using PPO Method)  |
| Model                    | Base Model                       | SFT Model                            | Reward Model                    | RL Model                                    |


- 语言模型是怎么被训练出来的
    - 1. pre-train：无监督预训练，海量的文本；（学前时代）
    - 2. Alignment (如下的2和3，严格意义上都是 Alignment，都属于对齐技术)
        - 2. SFT：supervised fine-tuning：有监督训练（学生时代），少量有标注；
        - 3. RLHF：真实的人类反馈，强化学习训练； 


## llama 源码阅读


## LoRA & PEFT


## TRL (Transformer Reinforcement Learning)


up:五道口纳什