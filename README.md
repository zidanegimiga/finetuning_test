# Biography QA PoC with LoRA Fine-tuning on Apple Silicon (M-series)

This project demonstrates a proof-of-concept (PoC) for building a domain-specific Question Answering (QA) system using Parameter-Efficient Fine-Tuning (PEFT) with LoRA. The goal is to train a pre-trained Large Language Model (LLM) to answer questions specifically about a provided biography, showcasing how to create a "bounded" system that knows what it knows and what it doesn't.

This setup is optimized for Apple Silicon (M-series) Macs, addressing common challenges encountered when performing LLM fine-tuning on this architecture.

## Table of Contents

1.  [Introduction](#introduction)
2.  [Features](#features)
3.  [Prerequisites](#prerequisites)
4.  [Project Setup](#project-setup)
    * [Install Dependencies](#install-dependencies)
    * [Prepare Dataset](#prepare-dataset)
5.  [Running the Fine-tuning](#running-the-fine-tuning)
    * [Online Mode (Default)](#online-mode-default)
    * [Offline Mode](#offline-mode)
6.  [Running Inference (Testing the Model)](#running-inference-testing-the-model)
7.  [Troubleshooting & Learnings](#troubleshooting--learnings)
8.  [Further Improvements](#further-improvements)

---

## 1. Introduction

Traditional fine-tuning directly adapts an LLM's weights to a specific dataset, allowing the model to internalize new knowledge and respond more accurately within a defined domain. This differs from Retrieval-Augmented Generation (RAG), which primarily retrieves information from an external knowledge base. For creating highly specialized and "bounded" QA systems (like the Morning Consult example where the model explicitly states what it does not know), fine-tuning can be a powerful approach.

This PoC uses your biography as the domain, fine-tuning a small LLM (TinyLlama) using LoRA to answer questions about it.

## 2. Features

* **LoRA Fine-tuning:** Efficiently adapts a pre-trained LLM using PEFT, significantly reducing computational requirements compared to full fine-tuning.
* **Apple Silicon (MPS) Optimization:** Configured to leverage your Mac's GPU for accelerated training.
* **Offline Capability:** Designed to run without an internet connection after initial model downloads.
* **Domain-Specific QA:** Trains the model to answer questions within the provided biographical context.

## 3. Prerequisites

* **Python 3.9+** (recommended)
* **Apple Silicon Mac (M1, M2, M3, M4 Air, etc.)** with sufficient unified memory (at least 16GB recommended, though 8GB can work with smaller models/batch sizes).
* **Internet connection** for initial setup and model downloads.

## 4. Project Setup

### A. Create Project Directory & Files

Create a new directory for your project and add the following files:

* `finetune_biography_poc.py` (your Python script)
* `biography_qa_dataset.json` (your QA dataset)

### B. Install Dependencies

Open your terminal in the project directory and run:

```bash
# Recommended core libraries
pip install torch transformers peft accelerate datasets trl

# Special handling for bitsandbytes on Apple Silicon:
# Uninstall any existing bitsandbytes
pip uninstall bitsandbytes

# Install bitsandbytes without pre-compiled binaries, letting it compile for your system.
# This is crucial for MPS compatibility.
pip install bitsandbytes --no-binary bitsandbytes --index-url=[https://pypi.org/simple/](https://pypi.org/simple/)

# Upgrade accelerate just in case of version mismatches
pip install --upgrade accelerate

Note on bitsandbytes: The bitsandbytes library is primarily for CUDA GPUs and 8-bit/4-bit quantization. While not directly used for mixed precision on MPS in this setup (due to accelerate limitations), it can cause issues if not correctly installed. The --no-binary flag attempts to compile it for your specific environment. If you still see warnings about "compiled without GPU support", it might run, but without optimal bitsandbytes features.
```

### C. Prepare Dataset (biography_qa_dataset.json)
Create biography_qa_dataset.json with questions and answers derived from your biography. This is your training data.
biography_qa_dataset.json Example:

```json

[
  {
    "question": "Where was Gimiga Zidane born?",
    "answer": "Gimiga Zidane was born in Nairobi, Kenya."
  },
  {
    "question": "When did Gimiga graduate from university?",
    "answer": "He graduated with honors in 2012."
  },
  {
    "question": "What are Gimiga's hobbies?",
    "answer": "Gimiga is an avid chess player and enjoys hiking in his free time."
  }
  // ... add more Q&A pairs about your biography
]

```

Markdown

# Biography QA PoC with LoRA Fine-tuning on Apple Silicon (M-series)

This project demonstrates a proof-of-concept (PoC) for building a domain-specific Question Answering (QA) system using Parameter-Efficient Fine-Tuning (PEFT) with LoRA. The goal is to train a pre-trained Large Language Model (LLM) to answer questions specifically about a provided biography, showcasing how to create a "bounded" system that knows what it knows and what it doesn't.

This setup is optimized for Apple Silicon (M-series) Macs, addressing common challenges encountered when performing LLM fine-tuning on this architecture.

## Table of Contents

1.  [Introduction](#introduction)
2.  [Features](#features)
3.  [Prerequisites](#prerequisites)
4.  [Project Setup](#project-setup)
    * [Install Dependencies](#install-dependencies)
    * [Prepare Dataset](#prepare-dataset)
5.  [Running the Fine-tuning](#running-the-fine-tuning)
    * [Online Mode (Default)](#online-mode-default)
    * [Offline Mode](#offline-mode)
6.  [Running Inference (Testing the Model)](#running-inference-testing-the-model)
7.  [Troubleshooting & Learnings](#troubleshooting--learnings)
8.  [Further Improvements](#further-improvements)

---

## 1. Introduction

Traditional fine-tuning directly adapts an LLM's weights to a specific dataset, allowing the model to internalize new knowledge and respond more accurately within a defined domain. This differs from Retrieval-Augmented Generation (RAG), which primarily retrieves information from an external knowledge base. For creating highly specialized and "bounded" QA systems (like the Morning Consult example where the model explicitly states what it does not know), fine-tuning can be a powerful approach.

This PoC uses your biography as the domain, fine-tuning a small LLM (TinyLlama) using LoRA to answer questions about it.

## 2. Features

* **LoRA Fine-tuning:** Efficiently adapts a pre-trained LLM using PEFT, significantly reducing computational requirements compared to full fine-tuning.
* **Apple Silicon (MPS) Optimization:** Configured to leverage your Mac's GPU for accelerated training.
* **Offline Capability:** Designed to run without an internet connection after initial model downloads.
* **Domain-Specific QA:** Trains the model to answer questions within the provided biographical context.

## 3. Prerequisites

* **Python 3.9+** (recommended)
* **Apple Silicon Mac (M1, M2, M3, M4 Air, etc.)** with sufficient unified memory (at least 16GB recommended, though 8GB can work with smaller models/batch sizes).
* **Internet connection** for initial setup and model downloads.

## 4. Project Setup

### A. Create Project Directory & Files

Create a new directory for your project and add the following files:

* `finetune_biography_poc.py` (your Python script)
* `biography_qa_dataset.json` (your QA dataset)

### B. Install Dependencies

Open your terminal in the project directory and run:

```bash
# Recommended core libraries
pip install torch transformers peft accelerate datasets trl

# Special handling for bitsandbytes on Apple Silicon:
# Uninstall any existing bitsandbytes
pip uninstall bitsandbytes

# Install bitsandbytes without pre-compiled binaries, letting it compile for your system.
# This is crucial for MPS compatibility.
pip install bitsandbytes --no-binary bitsandbytes --index-url=[https://pypi.org/simple/](https://pypi.org/simple/)

# Upgrade accelerate just in case of version mismatches
pip install --upgrade accelerate
Note on bitsandbytes: The bitsandbytes library is primarily for CUDA GPUs and 8-bit/4-bit quantization. While not directly used for mixed precision on MPS in this setup (due to accelerate limitations), it can cause issues if not correctly installed. The --no-binary flag attempts to compile it for your specific environment. If you still see warnings about "compiled without GPU support", it might run, but without optimal bitsandbytes features.

C. Prepare Dataset (biography_qa_dataset.json)
Create biography_qa_dataset.json with questions and answers derived from your biography. This is your training data.

biography_qa_dataset.json Example:

```JSON

[
  {
    "question": "Where was Gimiga Zidane born?",
    "answer": "Gimiga Zidane was born in Nairobi, Kenya."
  },
  {
    "question": "When did Gimiga graduate from university?",
    "answer": "He graduated with honors in 2012."
  },
  {
    "question": "What are Gimiga's hobbies?",
    "answer": "Gimiga is an avid chess player and enjoys hiking in his free time."
  }
  // ... add more Q&A pairs about your biography
]
```
Important: The more diverse and numerous your (Question, Answer) pairs, the better the model will learn. Aim for at least 10-20 for a basic PoC, but hundreds or thousands would be ideal for robust performance. The format <s>[INST] {question} [/INST] {answer}</s> will be automatically applied by the script.

## 5. Running the Fine-tuning
The finetune_biography_poc.py script will download the base model (TinyLlama), fine-tune it with your data, and then save the merged fine-tuned model.

### A. Online Mode (Default)
The first time you run the script, it will automatically download the TinyLlama/TinyLlama-1.1B-Chat-v1.0 model. This requires an internet connection.

```bash
python3 finetune_biography_poc.py
```

### Expected Output:

1. You'll see messages about model loading, tokenizer adjustment, and LoRA configuration.
2. The model (TinyLlama, ~1.1B parameters) will be moved to your MPS (GPU) device and cast to float16.
3. A progress bar for training will appear. For 10 examples and 5 epochs, this should take a few minutes to half an hour depending on your M4's core count and memory.
4. Upon completion, it will save the LoRA adapters and then merge them with the base model, saving the final merged model in a local directory (./fine_tuned_bio_model/merged_full_model).

### B. Offline Mode
After the initial run (which caches the base model), you can run the script entirely offline. Hugging Face libraries check a local cache (~/.cache/huggingface/hub/ by default) for models.

To ensure no network requests are made:

```bash 
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
python3 finetune_biography_poc.py
```

This tells the transformers and huggingface_hub libraries to strictly use local files. The script will load TinyLlama from your cache and use your locally saved biography_qa_dataset.json.

## 6. Running Inference (Testing the Model)

After the fine-tuning completes, the script will automatically enter an interactive testing mode.

```
--- Ask me questions about Gimiga Zidane! (Type 'exit' to quit) ---
Try questions like: 'Where was Gimiga born?', 'What did he study?', 'What are his hobbies?', 'Who is Gimiga Zidane?'
Also try out-of-scope questions like: 'What is the capital of France?', 'Define quantum physics?'

Your question: Where was Gimiga Zidane born?
Answer: Gimiga Zidane was born in Nairobi, Kenya.
--------------------------------------------------

Your question: What is the capital of France?
Answer: I don't have enough information about Gimiga Zidane to answer that question from the provided data. (Expected or similar)
--------------------------------------------------

```

*Observe*:
- For questions within the scope of your biography_qa_dataset.json, the model should provide accurate answers based on the fine-tuned knowledge.
- For questions outside the scope, the model should ideally decline to answer or state that it doesn't have information related to your biography, demonstrating the "bounded" behavior.

## 7. Troubleshooting & Learnings
The journey to this working PoC involved overcoming several common challenges:
- bitsandbytes Compatibility on MPS: Initial attempts to use bitsandbytes for 4-bit quantization failed due to its primary CUDA focus. The solution involved uninstalling bitsandbytes or installing it specifically for MPS support.
- RuntimeError: Tensor.item() cannot be called on meta tensors: This occurred when model.resize_token_embeddings() was called on a model that was partially on a "meta" device due to device_map="auto". The fix was to load the model entirely on CPU, resize embeddings, and then move it to MPS.
- RuntimeError: MPS backend out of memory: Training large models like Llama-2-7b on 16GB of unified memory is extremely challenging. The resolution was to use a much smaller model (TinyLlama/TinyLlama-1.1B-Chat-v1.0), which fits within the memory constraints.
- ValueError: fp16 mixed precision requires a GPU (not 'mps').: accelerate's mixed precision (fp16/bf16) implementation is tailored for CUDA and not directly compatible with MPS. The solution was to explicitly set fp16=False and bf16=False in SFTConfig, relying on direct torch.float16 casting for memory efficiency on MPS.
- trl.SFTTrainer API Changes: The trl library is under active development, and arguments like dataset_text_field, max_seq_length, packing, and tokenizer (processing_class) have shifted between SFTTrainer and SFTConfig across versions. Our final code adheres to the trl 0.18.1 API where:
    - processing_class (for the tokenizer), peft_config, model, train_dataset, and args (the SFTConfig object) are direct SFTTrainer arguments.
    - max_length, packing, dataset_text_field, and all other TrainingArguments-like parameters reside within SFTConfig.


## 8. Further Improvements
1. Larger Dataset: For a more robust and capable biography QA system, significantly more diverse and detailed (Question, Answer) pairs are needed.
2. Evaluation Metrics: Implement quantitative evaluation metrics (e.g., ROUGE, BLEU) to measure the model's performance on a separate test set.
3. User Interface: Create a simple web interface (e.g., with Gradio or Streamlit) to make the QA system more user-friendly.
4. Negative Examples: For stricter "boundedness," consider including some "I don't know" or "out of scope" examples in your training data, though this is more advanced.
5. Error Handling: Add more robust error handling in the script for network issues, file loading, etc.
6. Experiment with LoRA parameters: Adjust r, lora_alpha, and lora_dropout to see their impact on performance.
7. Different base models: Explore other small, instruction-tuned LLMs that might fit your memory.

This PoC serves as a solid foundation for exploration into domain-specific LLM fine-tuning.
