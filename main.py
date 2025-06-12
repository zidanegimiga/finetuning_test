# Flake8: noqa E501
import os
import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer, SFTConfig


#  Configuration 
BIOGRAPHY_DATASET_FILE = "biography_qa_dataset.json"
# BASE_MODEL_NAME = "EleutherAI/pythia-160m-deduped"
BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
FINE_TUNED_MODEL_DIR = "./fine_tuned_bio_model"
MAX_SEQ_LENGTH = 512


if torch.backends.mps.is_available():
    DEVICE = "mps"
    print("Using Apple Silicon (MPS) for training.")
elif torch.cuda.is_available():
    DEVICE = "cuda"
    print("Using NVIDIA CUDA for training.")
else:
    DEVICE = "cpu"
    print("No GPU found. Training will run on CPU and will be very slow.")


def load_dataset_from_json(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)

    formatted_data = []
    for item in qa_data:
        formatted_text = (
            f"<s>[INST] {item['question'].strip()} [/INST] {item['answer'].strip()}</s>"
        )
        formatted_data.append({"text": formatted_text})

    print(f"Loaded {len(formatted_data)} examples from {file_path}")
    return Dataset.from_list(formatted_data)


def load_base_model(model_name: str):
    print(f"Loading base model: {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Load as float32 on CPU first for safe resizing
        device_map=None
    )

    model.resize_token_embeddings(len(tokenizer))
    print("Tokenizer adjusted and embeddings resized on CPU.")

    model_dtype_for_mps = torch.float16  # Explicitly set to float16 for MPS

    if DEVICE == "mps":
        model = model.to("mps").to(model_dtype_for_mps)
        print(f"Model moved to MPS and cast to {model_dtype_for_mps} for training.")
    elif DEVICE == "cuda":
        model_dtype_for_cuda = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = model.to("cuda").to(model_dtype_for_cuda)
        print(f"Model moved to CUDA and cast to {model_dtype_for_cuda} for training.")
    else:  # CPU
        print("Model remains on CPU.")

    # Enable input grads for gradient checkpointing with PEFT
    # This is CRITICAL for the RuntimeError: element 0 of tensors does not require grad
    model.enable_input_require_grads()
    model.config.use_cache = False

    print("Base model loaded and ready.")
    return model, tokenizer


def configure_lora():
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    print("LoRA configured.")
    return lora_config


def setup_training_args(output_dir: str, max_length_val: int):
    """Sets up SFT training arguments using SFTConfig."""
    print("Setting up training arguments using SFTConfig.")
    training_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=50,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        optim="adamw_torch",  # if GPU use bitsandbytes
        save_steps=50,
        logging_steps=10,
        learning_rate=2e-4,
        weight_decay=0.001,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",
        dataloader_num_workers=0,

        # MPS-Specific Settings
        # Disable fp16 and bf16 explicitly for MPS, as accelerate's mixed precisions primarily designed for CUDA and is not compatible with MPS's backend.
        fp16=False,
        bf16=False,
        use_mps_device=True,  # Explicitly tell accelerate to use MPS
        max_length=max_length_val,
        packing=False,
        dataset_text_field="text",
    )
    print("Training arguments set up.")
    return training_config


def run_fine_tuning():
    dataset = load_dataset_from_json(BIOGRAPHY_DATASET_FILE)

    model, tokenizer = load_base_model(BASE_MODEL_NAME)

    lora_config = configure_lora()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_config = setup_training_args(FINE_TUNED_MODEL_DIR, MAX_SEQ_LENGTH)

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
        args=training_config,
    )

    print("\nStarting model training...")
    trainer.train()
    print("Model training finished.")

    trainer.save_model(FINE_TUNED_MODEL_DIR)
    print(f"LoRA adapters saved to {FINE_TUNED_MODEL_DIR}")

    print("Merging LoRA adapters with base model...")
    del model
    if DEVICE == "mps":
        torch.mps.empty_cache()
    elif DEVICE == "cuda":
        torch.cuda.empty_cache()

    base_model_merged = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float32,
        device_map=None,
    )
    if DEVICE == "mps":
        base_model_merged = base_model_merged.to("mps").to(torch.float16)
        print("Base model reloaded on MPS for merging.")
    elif DEVICE == "cuda":
        base_model_merged = base_model_merged.to("cuda").to(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        print("Base model reloaded on CUDA for merging.")

    tokenizer_merged = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tokenizer_merged.pad_token is None:
        tokenizer_merged.add_special_tokens({'pad_token': '[PAD]'})

    merged_model = PeftModel.from_pretrained(base_model_merged, FINE_TUNED_MODEL_DIR)
    merged_model = merged_model.merge_and_unload()

    merged_model_save_path = os.path.join(FINE_TUNED_MODEL_DIR, "merged_full_model")
    merged_model.save_pretrained(merged_model_save_path)
    tokenizer_merged.save_pretrained(merged_model_save_path)
    print(f"Merged full model saved to {merged_model_save_path}")

    return merged_model_save_path, tokenizer_merged


def test_model(model_path: str, tokenizer):
    print(f"\nLoading fine-tuned model from {model_path} for inference...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    model.eval()
    print("Model loaded. Ready for questions.")

    system_prompt_template = (
        "<<SYS>>You are an AI assistant specialized in providing information exclusively about Zidane Gimiga's biography. "
        "If a question is outside this domain or if the information is not explicitly present in Zidane Gimiga's biography, "
        "state clearly 'I do not have information about that in Zidane Gimiga's biography.' Do not answer questions outside this domain.<<SYS>>\n\n"
    )

    print("\n--- Ask me questions about Zidane Gimiga! (Type 'exit' to quit) ---")
    print("Try questions like: 'Where was Zidane Gimiga born?', 'What did Zidane Gimiga study?', 'What are Zidane Gimiga's hobbies?', 'Who is Zidane Gimiga?'")
    print("Also try out-of-scope questions like: 'What is the capital of France?', 'Define quantum physics?'")

    while True:
        question = input("\nYour question: ")
        if question.lower() == 'exit':
            break

        prompt = f"<s>[INST] {system_prompt_template}{question.strip()} [/INST]"


        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.01,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        if "[/INST]" in generated_text:
            answer_start = generated_text.find("[/INST]") + len("[/INST]")
            answer = generated_text[answer_start:].strip()
            if answer.endswith("</s>"):
                answer = answer[:-4].strip()
            if answer.startswith("<<SYS>>"):
                answer = answer.replace("<<SYS>>", "").strip()
            if answer.startswith("<s>"):
                answer = answer.replace("<s>", "").strip()
        else:
            answer = generated_text.strip()
            if answer.endswith("</s>"):
                answer = answer[:-4].strip()
            if answer.startswith("<<SYS>>"):
                answer = answer.replace("<<SYS>>", "").strip()
            if answer.startswith("<s>"):
                answer = answer.replace("<s>", "").strip()

        print("\nAnswer:")
        print(answer)
        print("-" * 50)

if __name__ == "__main__":
    if torch.backends.mps.is_available():
        print("Apple Silicon (MPS) is available. Using GPU for training.")
    elif torch.cuda.is_available():
        print("CUDA is available. Using GPU for training.")
        if torch.cuda.get_device_properties(0).major >= 8:
            print("GPU supports bfloat16.")
        else:
            print("GPU does not support bfloat16. Using float16 instead.")
    else:
        print("No GPU found. Training will run on CPU and will be very slow.")

    merged_model_path, tokenizer_for_test = run_fine_tuning()

    test_model(merged_model_path, tokenizer_for_test)
