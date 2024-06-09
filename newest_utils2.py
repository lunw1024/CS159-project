import glob
import pandas as pd
import re
from tqdm.notebook import tqdm


def extract_answer(response, use_code):
    # extract answer based on whether code is used
    if use_code:
        try:
            code_match = re.search(r"###code###(.*?)###endcode###", response, re.DOTALL)
            code_to_execute = code_match.group(1).strip()
            local_env = {}
            exec(code_to_execute, {}, local_env)
            return local_env.get(
                "result", "No result produced"
            )  # Assume the code defines 'result'
        except Exception as e:
            return str(e)
    else:
        ans = re.search(r"\*+(.+?)\*+", response[::-1])
        extracted = ans.group(1).strip() if ans else "No answer found"
        return extracted[::-1]


from pathlib import Path


def evaluate_file(
    file,
    model,
    leading_prompt,
    trailing_prompt,
    use_code,
    suffix="",
    store_intermediate=True,
    save=True,
    max_iters=float("inf"),
    pbar=None,
    filenum=None,
    tot_files=None,
    debug=False
):
    
    data_df = pd.read_json(file, lines=True)
    columns = (
        [
            "Question",
            "Expected Answer",
            "Model Response",
            "Extracted Answer",
            "Is Correct",
        ]
        if store_intermediate
        else ["Is Correct"]
    )
    results_df = pd.DataFrame(columns=columns)

    total_qs = min(data_df.shape[0], max_iters)
    for index, row in data_df.iterrows():
        if pbar is not None:
          pbar.set_description(f"File {filenum}/{tot_files} | Question {index+1}/{total_qs}...")
        if index >= max_iters:
            break
        question = leading_prompt + row["question"] + trailing_prompt
        expected_answer = row["answer"].lower().removesuffix(".")
        response = model.predict(question)
        extracted_answer = str(extract_answer(response, use_code))
        is_correct = expected_answer in extracted_answer.lower() # extracted_answer.lower() == expected_answer

        if store_intermediate:
            results_df.loc[index] = [
                row["question"],
                expected_answer,
                response,
                extracted_answer,
                is_correct,
            ]
        else:
            results_df.loc[index] = [is_correct]
        
        if debug:
            break
        # correct_or_not = "correct" if is_correct else "wrong"
        # pbar.set_description(f"Question {index} is {correct_or_not}")

    if save:
        model_name = type(model).__name__
        subfolder_path = Path(f"/content/drive/MyDrive/CS 159/{model_name}")
        subfolder_path.mkdir(parents=True, exist_ok=True)

        file_mode = "code" if use_code else "no_code"
        file_name = f"{Path(file).stem}_evaluation_{file_mode}_{suffix}.csv"
        save_path = subfolder_path / file_name

        results_df.to_csv(save_path, index=False)
        print(f"Intermediate results saved to {save_path}")

    return results_df


def evaluate_across_files(
    model,
    leading_prompt,
    trailing_prompt,
    data_path,
    suffix="",
    which_files="all",
    use_code=False,
    save=True,
    num_runs_self_consistency=1,
):
    map_global_files = glob.glob(
        data_path + "map_global/*"
    )  # describe the map top down
    map_local_files = glob.glob(data_path + "map_local/*")  # don't

    if which_files == "all":
        these_files = map_global_files + map_local_files
    elif which_files == "local":
        these_files = map_local_files
    elif which_files == "global":
        these_files = map_global_files
    elif which_files == "debug":
        these_files = [
            "/content/drive/MyDrive/data/raw/SpatialEvalLLM/map_global/type-ring_size-12_steps-8_seed-12_n-100.jsonl"
        ]
    else:
        raise ValueError(f"which_files={which_files} not supported")

    # evaluate each file and store the results
    scores_df = pd.DataFrame(columns=["Filename", "Correct", "Total", "Accuracy"])

    i = 0
    pbar = tqdm(these_files, leave=True, desc="Evaluating files")
    for file in pbar:
        i += 1
        max_iters = float("inf")
        if which_files == "debug":
            max_iters = 1
        if num_runs_self_consistency > 1:
            results = pd.DataFrame()
            for run in range(num_runs_self_consistency):
                this_result = evaluate_file(
                    file,
                    model,
                    leading_prompt,
                    trailing_prompt,
                    use_code,
                    suffix,
                    store_intermediate=True,
                    save=save,
                    max_iters=max_iters,
                    pbar=pbar,
                    filenum=i,
                    tot_files=len(these_files)
                )
                if run == 0:
                    results["Expected Answer"] = this_result["Expected Answer"].values
                results[f"output_{run}"] = this_result["Extracted Answer"].values
            results["voting_answer"] = (
                results[[f"output_{run}" for run in range(num_runs_self_consistency)]]
                .mode(axis=1)
                .values[:, 0]
            )
            results["correct"] = (
                results["voting_answer"] == results["Expected Answer"]
            ).values
            num_correct = results["correct"].sum()
            num_total = results.shape[0]
        else:
            result = evaluate_file(
                file,
                model,
                leading_prompt,
                trailing_prompt,
                use_code,
                suffix,
                max_iters=max_iters,
                pbar=pbar,
                filenum=i
            )
            num_correct = result["Is Correct"].sum()
            num_total = len(result)
        scores_df.loc[len(scores_df)] = [
            file,
            num_correct,
            num_total,
            num_correct / num_total,
        ]

    return scores_df
