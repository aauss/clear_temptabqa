import argparse
import csv
import importlib.util
import json
import os

from tqdm import tqdm

ABS_PATH = os.path.dirname(__file__)


def find_function(directory, function_name):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                module_name = file_path.replace("/", ".").replace("\\", ".")[:-3]

                spec = importlib.util.spec_from_file_location(module_name, file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                if hasattr(module, function_name):
                    return getattr(module, function_name)

    raise ImportError(f"Function {function_name} not found in {directory}")


def read_tables() -> list[str]:
    tables: list[str] = []
    file_index = 0
    while True:
        file_name = f"{file_index}.json"
        file_path = os.path.join(ABS_PATH, "data/temptabqa_v2/tables/json/", file_name)
        if not os.path.exists(file_path):
            break
        with open(file_path, "r") as file:
            json_data = json.load(file)
            table_str = ""
            for key in json_data.keys():
                if isinstance(json_data[key], dict):
                    table_str += key + "\n"
                    for sub_key in json_data[key].keys():
                        if sub_key == key:
                            table_str += str(json_data[key][sub_key]) + "\n"
                        else:
                            table_str += sub_key + "\t" + json_data[key][sub_key] + "\n"
                    table_str += "\n"
                else:
                    table_str += key + "\n" + json_data[key] + "\n\n"
            tables.append(table_str)
        file_index += 1
    return tables


def read_questions() -> dict[str, list[dict[str, str | int]]]:
    questions = {}
    for split in ["dev", "train", "head", "tail", "head-temp"]:
        file_path = os.path.join(
            ABS_PATH, f"data/temptabqa_v2/qapairs/{split}-set/{split}-set.json"
        )
        with open(file_path, "r") as file:
            questions[split] = json.load(file)
    return questions


def main(model: str, prompt: str, split: str):
    model_function = find_function("models", model)
    prompt_function = find_function("prompts", prompt)
    tables = read_tables()
    questions = read_questions()

    data = []
    for qid, question in tqdm(enumerate(questions[split])):
        input_data = prompt_function(
            tables=tables, questions=questions, split=split, question_id=qid
        )
        output = model_function(input_data)
        data.append(
            [
                qid,
                question["question"],
                question["answer"],
                question["category"],
                question["table_id"],
                output,
            ]
        )
        # break
    with open(f"output_{model}_{prompt}_{split}.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "question", "answer", "category", "table_id", "output"])
        for row in data:
            writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model with prompt")
    parser.add_argument(
        "--model",
        required=True,
        help="The model function to call",
        choices=[
            "gpt3",
            "gpt4",
            "gemini",
            "llama",
            "llama_ct",
            "palm",
            "timo",
            "timo_ct",
            "phi",
            "phi_ct",
        ],
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="The prompt function to call",
        choices=[
            "few_shot_cot",
            "zero_shot_cot",
            "zero_shot_cot_ct",
            "few_shot_cot_ct",
            "clear",
            "few_shot_faithful_cot",
            "zero_shot_faitful_cot",
            "few_shot_pot",
            "zero_shot_pot",
            "question_decomposition",
            "evidence_extraction",
        ],
    )
    parser.add_argument(
        "--split",
        required=True,
        help="The dataset split to run on",
        choices=["dev", "train", "head", "tail", "head-temp"],
    )

    args = parser.parse_args()

    main(model=args.model, prompt=args.prompt, split=args.split)
