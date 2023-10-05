#based on https://colab.research.google.com/drive/1Ly01S--kUwkKQalE-75skalp-ftwl0fE?usp=sharing 
import getpass
import locale; locale.getpreferredencoding = lambda: "UTF-8"
import logging
import os
import yaml

# from google.colab import data_table; data_table.enable_dataframe_formatter() # verify if this is required.
import numpy as np; np.random.seed(123)
import pandas as pd

from ludwig.api import LudwigModel

# Make sure the below global variables are defined during the env build
os.environ["HUGGING_FACE_HUB_TOKEN"] = "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"
assert os.environ["HUGGING_FACE_HUB_TOKEN"]


# Import The Code Generation Dataset

df = pd.read_json("https://raw.githubusercontent.com/sahil280114/codealpaca/master/data/code_alpaca_20k.json")

# We're going to create a new column called `split` where:
# 90% will be assigned a value of 0 -> train set
# 5% will be assigned a value of 1 -> validation set
# 5% will be assigned a value of 2 -> test set
# Calculate the number of rows for each split value
total_rows = len(df)
split_0_count = int(total_rows * 0.9)
split_1_count = int(total_rows * 0.05)
split_2_count = total_rows - split_0_count - split_1_count

# Create an array with split values based on the counts
split_values = np.concatenate([
    np.zeros(split_0_count),
    np.ones(split_1_count),
    np.full(split_2_count, 2)
])

# Shuffle the array to ensure randomness
np.random.shuffle(split_values)

# Add the 'split' column to the DataFrame
df['split'] = split_values
df['split'] = df['split'].astype(int)

# For this usecase, we will take just 1000 rows of this dataset.
# df = df.head(n=1000)


'''
    Quantization-Based Fine-Tuning (QLoRA):

    Involves reducing the precision of model parameters (e.g., converting 32-bit floating-point values to 8-bit or 4-bit integers). This reduces the amount of CPU and GPU memory required by either 4x if using 8-bit integers, or 8x if using 4-bit integers.
    Typically, since we're changing the weights to 8 or 4 bit integers, we will lose some precision/performance.
    This can lead to reduced memory usage and faster inference on hardware with reduced precision support.
    Particularly useful when deploying models on resource-constrained devices, such as mobile phones or edge devices.
'''

qlora_fine_tuning_config = yaml.safe_load(
"""
model_type: llm
base_model: meta-llama/Llama-2-7b-hf

input_features:
  - name: instruction
    type: text

output_features:
  - name: output
    type: text

prompt:
  template: >-
    Below is an instruction that describes a task, paired with an input
    that provides further context. Write a response that appropriately
    completes the request.

    ### Instruction: {instruction}

    ### Input: {input}

    ### Response:

generation:
  temperature: 0.1
  max_new_tokens: 512

adapter:
  type: lora

quantization:
  bits: 4

preprocessing:
  global_max_sequence_length: 512
  split:
    type: random
    probabilities:
    - 1
    - 0
    - 0

trainer:
  type: finetune
  epochs: 1
  batch_size: 1
  eval_batch_size: 2
  gradient_accumulation_steps: 16
  learning_rate: 0.0004
  learning_rate_scheduler:
    warmup_fraction: 0.03
"""
)

model = LudwigModel(config=qlora_fine_tuning_config, logging_level=logging.INFO)
results = model.train(dataset=df)
# LudwigModel.upload_to_hf_hub("your_org/model_name", "path/to/model")
model.upload_to_hf_hub("oli2/ludwig_finetuned", "/results/api_experiment_run")


# What's next once the model has been trained?

# upload to huggingface
# LudwigModel.upload_to_hf_hub("your_org/model_name", "path/to/model")