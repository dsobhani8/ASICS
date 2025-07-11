# Language Models can Categorize System Inputs for Performance Analysis

Under construction

## General

Automated System Input Categorization Scheme (ASICS) takes in text samples, usually LM user queries or questions from a dataset, and the performance of a system on each system according to some metric (usually a human preference label or classification accuracy). It returns ```{dataset_name}_significance_results``` which contains a categories a LM discovered within the text samples along with the performance of the system on samples that match that category description.   

## Running ASICS

Before running any of the ASICS scripts, you need to set the following variables:

```bash
export OPENAI_API_KEY="your_openai_api_key_here"
export MODEL_NAME="open_ai_model_here"
```

To run ASICS on the full Human Preference Dataset (HPD) use:

```bash
git clone https://github.com/dsobhani8/ASICS.git
cd ASICS
chmod +x asics/run_asics.py
python asics/run_asics.py --input_dataset hpd
```

## Dataset Structure

To upload your own dataset ensure your file is a csv with 'system_input' and 'is_correct' as the variable names. The value for is_correct should be True or False.

- In the HPD is_correct=True represents whether Llama was preferred to Claude.

- In the Deepseek dataset is_correct=True represents whether deepseek answered a given query from BigCodeBench correctly. Deepseek-coder-33B results were obtained from the BigCodeBench repo in the summer of 2024. Use
  ```--input_dataset deepseek``` to run asics on this dataset.

## Changing LM Components

Currently our default LM is ```gpt-4.1-mini```. If you want to use different language models, you can implement the inference in ```asics/query.py```. If you want to use different prompts, you can modify the templates in ```asics/prompts/```.
