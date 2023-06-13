# Copyright 2023 Databricks, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from functools import partial
from pathlib import Path
import typing as t

import click
import numpy as np
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
import torch
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
# not sure if this will work outside of a deepspeed context
from deepspeed.runtime.lr_schedules import WarmupLR
import pathlib

from training.consts import (
    DEFAULT_INPUT_MODEL,
    DEFAULT_SEED,
    PROMPT_WITH_INPUT_FORMAT,
    PROMPT_NO_INPUT_FORMAT,
    END_KEY,
    INSTRUCTION_KEY,
    RESPONSE_KEY_NL,
    DEFAULT_TRAINING_DATASET,
)

logger = logging.getLogger(__name__)
ROOT_PATH = Path(__file__).parent.parent


class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    def torch_call(self, examples: t.List[t.Union[t.List[int], t.Any, t.Dict[str, t.Any]]]) -> t.Dict[str, t.Any]:
        batch = super().torch_call(examples)

        # The prompt ends with the response key plus a newline.  We encode this and then try to find it in the
        # sequence of tokens.  This should just be a single token.
        response_token_ids = self.tokenizer.encode(RESPONSE_KEY_NL)

        labels = batch["labels"].clone()

        for i in range(len(examples)):

            response_token_ids_start_idx = None
            for idx in np.where(batch["labels"][i] == response_token_ids[0])[0]:
                response_token_ids_start_idx = idx
                break

            if response_token_ids_start_idx is None:
                raise RuntimeError(
                    f'Could not find response key {response_token_ids} in token IDs {batch["labels"][i]}'
                )

            response_token_ids_end_idx = response_token_ids_start_idx + 1

            # Make pytorch loss function ignore all tokens up through the end of the response key
            labels[i, :response_token_ids_end_idx] = -100

        batch["labels"] = labels

        return batch


def preprocess_batch(batch: t.Dict[str, t.List], tokenizer: AutoTokenizer, max_length: int) -> dict:
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )


def load_training_dataset(path_or_dataset: str = DEFAULT_TRAINING_DATASET) -> Dataset:
    logger.info(f"Loading dataset from {path_or_dataset}")
    dataset = load_dataset(path_or_dataset)["train"]
    logger.info("Found %d rows", dataset.num_rows)

    def _add_text(rec):
        instruction = rec["instruction"]
        response = rec["response"]
        context = rec.get("context")

        if not instruction:
            raise ValueError(f"Expected an instruction in: {rec}")

        if not response:
            raise ValueError(f"Expected a response in: {rec}")

        # For some instructions there is an input that goes along with the instruction, providing context for the
        # instruction.  For example, the input might be a passage from Wikipedia and the instruction says to extract
        # some piece of information from it.  The response is that information to extract.  In other cases there is
        # no input.  For example, the instruction might be open QA such as asking what year some historic figure was
        # born.
        if context:
            rec["text"] = PROMPT_WITH_INPUT_FORMAT.format(instruction=instruction, response=response, input=context)
        else:
            rec["text"] = PROMPT_NO_INPUT_FORMAT.format(instruction=instruction, response=response)
        return rec

    dataset = dataset.map(_add_text)

    return dataset


def load_tokenizer(pretrained_model_name_or_path: str = DEFAULT_INPUT_MODEL) -> PreTrainedTokenizer:
    logger.info(f"Loading tokenizer for {pretrained_model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": [END_KEY, INSTRUCTION_KEY, RESPONSE_KEY_NL]})
    return tokenizer

# copy of model.print_trainable_parameters() which uses the logger
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logger.info(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )



def load_model(
    pretrained_model_name_or_path: str = DEFAULT_INPUT_MODEL, *, gradient_checkpointing: bool = False
) -> AutoModelForCausalLM:
    logger.info(f"Loading model for {pretrained_model_name_or_path}")

    # not QLoRA yet - get regular LoRA working first!
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        trust_remote_code=True,
        use_cache=not gradient_checkpointing,
        device_map="auto",
        # quantization_config=bnb_config,  # for QLoRA
    )

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)

    # The dimension used by the LoRA update matrices
    LORA_R = 8
    # Scaling factor
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05

    # r and alpha together control the total number of final trainable parameters when using LoRA, giving you the flexibility to balance a trade-off between end performance and compute efficiency.
    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",  # Specifies if the bias parameters should be trained
        task_type=TaskType.CAUSAL_LM,
        target_modules=["query_key_value"],
        # inference_mode=False  # default is False
    )
    model = get_peft_model(model, config)

    # not sure if this is necessary
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if logger:
        print_trainable_parameters(model)
    else:
        model.print_trainable_parameters()

    return model


def get_model_tokenizer(
    pretrained_model_name_or_path: str = DEFAULT_INPUT_MODEL, *, gradient_checkpointing: bool = False
) -> t.Tuple[AutoModelForCausalLM, PreTrainedTokenizer]:
    tokenizer = load_tokenizer(pretrained_model_name_or_path)
    model = load_model(pretrained_model_name_or_path, gradient_checkpointing=gradient_checkpointing)
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int, seed=DEFAULT_SEED, training_dataset: str = DEFAULT_TRAINING_DATASET) -> Dataset:
    """Loads the training dataset and tokenizes it so it is ready for training.

    Args:
        tokenizer (AutoTokenizer): Tokenizer tied to the model.
        max_length (int): Maximum number of tokens to emit from tokenizer.

    Returns:
        Dataset: HuggingFace dataset
    """

    dataset = load_training_dataset(training_dataset)

    logger.info("Preprocessing dataset")
    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=["instruction", "context", "response", "text", "category"],
    )

    # Make sure we don't have any truncated records, as this would mean the end keyword is missing.
    logger.info("Processed dataset has %d rows", dataset.num_rows)
    dataset = dataset.filter(lambda rec: len(rec["input_ids"]) < max_length)
    logger.info("Processed dataset has %d rows after filtering for truncated records", dataset.num_rows)

    logger.info("Shuffling dataset")
    dataset = dataset.shuffle(seed=seed)

    logger.info("Done preprocessing")

    return dataset


def train(
    *,
    input_model: str,
    output_dir: str,
    epochs: int,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    lr: float,
    seed: int,
    gradient_checkpointing: bool,
    bf16: bool,
    logging_steps: int,
    save_steps: int,
    eval_steps: int,
    test_size_ratio: float,
    save_total_limit: int,
    warmup_steps: int,
    training_dataset: str = DEFAULT_TRAINING_DATASET,
    repo_name: str = None,
    hf_token: str = None,
):
    set_seed(seed)

    accelerator = Accelerator()

    model, tokenizer = get_model_tokenizer(
        pretrained_model_name_or_path=input_model, gradient_checkpointing=gradient_checkpointing
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # https://deepspeed.readthedocs.io/en/latest/schedulers.html#warmuplr
    scheduler = WarmupLR(optimizer, warmup_num_steps=warmup_steps)
    # try this if the deepspeed WarmupLR doesn't work outside of the deepspeed context
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    model, optimizer, scheduler = accelerator.prepare(
        model, optimizer, scheduler
    )

    # Use the same max length that the model supports.  Fall back to 1024 if the setting can't be found.
    # The configuraton for the length can be stored under different names depending on the model.  Here we attempt
    # a few possible names we've encountered.
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            logger.info(f"Found max lenth: {max_length}")
            break
    if not max_length:
        max_length = 1024
        logger.info(f"Using default max length: {max_length}")

    processed_dataset = preprocess_dataset(tokenizer=tokenizer, max_length=max_length, seed=seed, training_dataset=training_dataset)

    split_dataset = processed_dataset.train_test_split(test_size=test_size_ratio, seed=seed)

    logger.info("Train data size: %d", split_dataset["train"].num_rows)
    logger.info("Test data size: %d", split_dataset["test"].num_rows)

    data_collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer, mlm=False, return_tensors="pt", pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
    )

    # enable fp16 if not bf16
    fp16 = not bf16

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        fp16=fp16,
        bf16=bf16,
        learning_rate=lr,
        num_train_epochs=epochs,
        gradient_checkpointing=gradient_checkpointing,
        logging_dir=f"{output_dir}/runs",
        logging_strategy="steps",
        logging_steps=logging_steps,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        load_best_model_at_end=False,
        report_to="tensorboard",
        disable_tqdm=True,
        remove_unused_columns=False,
        warmup_steps=warmup_steps,
        scheduler=scheduler,
        optimizer=optimizer,
    )

    logger.info("Instantiating Trainer")

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        data_collator=data_collator,
    )

    logger.info("Training")
    trainer.train()

    logger.info(f"Saving Model to {output_dir}")
    trainer.save_model(output_dir=output_dir)

    logger.info(f"Pushing model to HuggingFace Hub, repo_name={repo_name}")
    if repo_name:
        if hf_token:
            from huggingface_hub import login
            login(token=hf_token)
        model.push_to_hub(repo_name, private=True, use_auth_token=hf_token)

    logger.info("Done.")


@click.command()
@click.option("--input-model", type=str, help="Input model to fine tune", default=DEFAULT_INPUT_MODEL)
@click.option("--output-dir", type=str, help="Sync data to this path")
@click.option("--epochs", type=int, default=3, help="Number of epochs to train for.")
@click.option("--per-device-train-batch-size", type=int, default=8, help="Batch size to use for training.")
@click.option("--per-device-eval-batch-size", type=int, default=8, help="Batch size to use for evaluation.")
@click.option(
    "--test-size-ratio", type=float, default=None, help="Ratio of test records for evaluation."
)
@click.option("--warmup-steps", type=int, default=None, help="Number of steps to warm up to learning rate")
@click.option("--logging-steps", type=int, default=10, help="How often to log")
@click.option("--eval-steps", type=int, default=50, help="How often to run evaluation on test records")
@click.option("--save-steps", type=int, default=400, help="How often to checkpoint the model")
@click.option("--save-total-limit", type=int, default=10, help="Maximum number of checkpoints to keep on disk")
@click.option("--lr", type=float, default=1e-5, help="Learning rate to use for training.")
@click.option("--seed", type=int, default=DEFAULT_SEED, help="Seed to use for training.")
@click.option("--training-dataset", type=str, default=DEFAULT_TRAINING_DATASET, help="Path to dataset for training")
@click.option(
    "--gradient-checkpointing/--no-gradient-checkpointing",
    is_flag=True,
    default=True,
    help="Use gradient checkpointing?",
)
@click.option("--bf16", type=bool, default=None, help="Whether to use bf16 (preferred on A100's).")
@click.option("--repo-name", type=str, default=None, help="Repo name on HF Hub to push model to.")
@click.option("--hf-token", type=str, default=None, help="HF token to use for pushing model to HF Hub.")
def main(**kwargs):
    train(**kwargs)


if __name__ == "__main__":
    """run with:

    PYTHONPATH=. accelerate launch \
        --config_file config/accelerate_config.yaml \
        training/trainer.py \
        --input-model EleutherAI/pythia-12b \
        --epochs 2 \
        --output-dir output \
        --per-device-train-batch-size 2 \
        --per-device-eval-batch-size 2 \
        --logging-steps 10 \
        --save-steps 200 \
        --save-total-limit 20 \
        --eval-steps 50 \
        --warmup-steps 50 \
        --test-size-ratio 0.1 \
        --lr 5e-6 \
        --bf16 True

    """
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )
    try:
        main()
    except Exception:
        logger.exception("main failed")
        raise
