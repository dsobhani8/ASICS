# Language Models can Categorize System Inputs for Performance Analysis

Under construction

To run specify the following args:
    parser.add_argument("--difference-threshold", type=float, default=0.05,
                        help="Minimum difference (category rate - overall rate) to keep a category.")
    parser.add_argument("--cosine-sim", type=float, default=0.82,
                        help="Cosine similarity threshold for deduplication.")
    parser.add_argument("--percentage", type=float, default=100.0,
                        help="Percentage of the input dataset to use (0 < percentage ≤ 100). Default 100.")
    parser.add_argument("--input_dataset", type=str, required=True,
                        help="Dataset name (without .csv). 'hpd' uses hpd_llama_vs_claude.csv.")
    parser.add_argument("--output_csv", type=str, required=False,
                        help="output dataset")

To run ASICS on the full Human Preference Dataset (HPD) use:

```bash
git clone https://github.com/dsobhani8/ASICS.git
cd ASICS
chmod +x asics/run_asics.py
python asics/run_asics.py --input_dataset hpd

To upload your own dataset ensure your file is a csv with 'system_input' and 'is_correct' as the variable names. 

In the HPD is_correct=True represents whether Llama won.

Environment variables that you will need to specifiy are OPENAI_API_KEY and MODEL_NAME.
