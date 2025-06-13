# Language Models can Categorize System Inputs for Performance Analysis

Under construction

## Running ASICS

Before running any of the ASICS scripts, you need to set the following variables:

```bash
export OPENAI_API_KEY="your_openai_api_key_here"
export MODEL_NAME="gpt-4.1-mini-2025-04-14"
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

In the HPD is_correct=True represents whether Llama won.
