# Reproduction of *Surfacing Pathological Behaviors in Language Models*

This project is a small-scale reproduction of the [paper](https://transluce.org/pathological-behaviors) by Transluce.

### 1. Experiment Specification

- **Target model:** Qwen 2.5 7B
- **Steered model:** Qwen 2.5 7B
- **Investigator model:** Llama 3.1 8B
- **Judge model:** Qwen 2.5 32B (AWQ)

Target, steered and judge model were served using OpenAI-like API via vLLM.