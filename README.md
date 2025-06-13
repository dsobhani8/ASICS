# Language Models can Categorize System Inputs for Performance Analysis

Under construction


To upload your own dataset ensure your file is a csv with 'system_input' and 'is_correct' as the variable names. 

In the HPD is_correct=True represents whether Llama won.

Environment variables that you will need to specifiy are OPENAI_API_KEY and MODEL_NAME.

To run ASICS on the full Human Preference Dataset (HPD) use:

```bash
git clone https://github.com/dsobhani8/ASICS.git
cd ASICS
chmod +x asics/run_asics.py
python asics/run_asics.py --input_dataset hpd
```
