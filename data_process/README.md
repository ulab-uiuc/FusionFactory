# 🔧 Data Collection

## Argument Descriptions for LLM Data Collection Script

This script uses OpenAI-compatible APIs to collect LLM-generated data. The process can be repeated in multiple rounds to avoid API instability or temporary failures.

| Argument                 | Type  | Default                                  | Description                                                                                                                             |
| ------------------------ | ----- | ---------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `--split`                | `str` | `"train"`                                | Specifies whether to generate training or test data. Options: `"train"`, `"test"`.                                                      |
| `--case_num`             | `int` | `50`                                     | Number of data samples to collect for each dataset per round.                                                                           |
| `--seed`                 | `int` | `42`                                     | Random seed for reproducibility.                                                                                                        |
| `--round`                | `int` | `5`                                      | Number of repeated collection rounds. Increases robustness by retrying failed samples due to API errors.                                |
| `--llm_description_path` | `str` | `"./data_process/LLM_Descriptions.json"` | Path to the JSON file describing different LLMs. Example reference: `./data_process/LLM_Descriptions.json`. |
| `--cache_save_path`      | `str` | `"./dataset/cache"`                      | Path to save intermediate cache files during data collection.                                                                           |
| `--csv_save_path`        | `str` | `"./dataset/router_data.csv"`            | Path to save the final collected dataset in CSV format.                                                                                 |
| `--api_base`             | `str` | `"[YOUR_API_BASE]"`                      | The base URL of the OpenAI-compatible API endpoint (e.g., `https://api.openai.com/v1`).                                                 |
| `--api_key`              | `str` | `"[YOUR_API_KEY]"`                       | Your OpenAI API key or compatible provider's key.                                                                                       |

## LLM Judge Scoring

The `add_llm_judge.py` script adds quality scores to the collected data using an LLM judge. This helps evaluate the quality of responses for training purposes.

### Usage

```bash
python add_llm_judge.py
```

The script will:
1. Read the input CSV file (default: `./dataset/router_data.csv`)
2. Add LLM judge scores to each response
3. Save the results to a new CSV file (default: `./dataset/router_data_with_judge.csv`)

The script includes features like:
- Parallel processing with configurable number of workers
- Automatic resumption from the last checkpoint
- Rate limiting to respect API constraints
- Retry logic for failed requests

Make sure to set your API credentials in the script before running:
- `YOUR_API_BASE_URL`: The base URL for your OpenAI-compatible API
- `YOUR_API_KEY`: Your API key
- `YOUR_MODEL_NAME`: The name of the model to use for judging
