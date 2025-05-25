import nbformat
import pandas as pd
import base64
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from difflib import SequenceMatcher
from datetime import datetime
from zoneinfo import ZoneInfo

# File paths and metadata
problem_notebook_path = "/Users/rakeshdevarakonda/Documents/Auto_Eval/Auto_eval2/Retail_sales_classification_Tredence_problem.ipynb"
solution_notebook_path = "/Users/rakeshdevarakonda/Documents/Auto_Eval/Auto_eval2/Retail_sales_classification_Tredence_solution.ipynb"
user_email = "user@example.com"
attempt_id = "1"
project = "python_problem1"

# Task weightage
task_weightage = {
    "read_dataset": 2,
    "df_shape": 0.5,
    "df_dtypes": 0.5,
    "drop_columns": 2,
    "outlier_treatment": 3,
    "transpose_1": 0.5,
    "treat_outliers_iqr": 6,
    "transpose_2": 0.5,
    "missing_value": 2,
    "remove_missing_value": 2
}

# Extract function outputs
def extract_function_outputs_with_presence(notebook_path, valid_functions):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    output_dict = {}
    current_func = None
    for cell in nb.cells:
        if cell.cell_type == 'code':
            lines = cell.source.split('\n')
            for line in lines:
                if line.strip().startswith("def "):
                    current_func = line.split("(")[0].replace("def ", "").strip()
            if current_func and current_func in valid_functions:
                captured_output = None
                for output in cell.get('outputs', []):
                    if output.output_type == "execute_result" and "text/plain" in output.data:
                        captured_output = output.data["text/plain"]
                    elif output.output_type == "stream" and output.name == "stdout":
                        captured_output = output.text.strip()
                output_dict[current_func] = captured_output if captured_output is not None else "__NO_OUTPUT__"

    return pd.DataFrame([
        {"Function": func, "Output": out} for func, out in output_dict.items()
    ])

# Enhanced string similarity comparison with line normalization
def similarity(a, b):
    norm_a = "\n".join([line.strip() for line in a.strip().splitlines()])
    norm_b = "\n".join([line.strip() for line in b.strip().splitlines()])
    return SequenceMatcher(None, norm_a, norm_b).ratio()

# Compare outputs and assign scores
def compare_outputs(problem_df, solution_df, task_weightage, threshold=0.95):
    scores = []
    problem_dict = dict(zip(problem_df["Function"], problem_df["Output"]))
    solution_dict = dict(zip(solution_df["Function"], solution_df["Output"]))

    for function_name in task_weightage.keys():
        prob_out = problem_dict.get(function_name, "").strip()
        sol_out = solution_dict.get(function_name, "").strip()

        if prob_out and sol_out:
            sim = similarity(prob_out, sol_out)
            if sim >= threshold:
                score = task_weightage[function_name]
                remark = "Success"
            else:
                score = 0
                remark = "Mismatch"
        else:
            score = 0
            remark = "Missing output"

        scores.append({
            "method_name": function_name,
            "score_gained": score,
            "remarks": remark
        })

    return pd.DataFrame(scores)

# Extract and compare image output for outlier_treatment
def extract_plot_image(notebook_path, function_name):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    for cell in nb.cells:
        if cell.cell_type == 'code' and function_name in cell.source:
            for output in cell.get('outputs', []):
                if output.output_type == 'display_data' and 'image/png' in output.data:
                    image_data = base64.b64decode(output.data['image/png'])
                    return Image.open(pd.io.common.BytesIO(image_data)).convert('RGB')
    return None

# Run evaluation
def evaluate_notebooks(problem_path, solution_path, task_weightage):
    problem_outputs = extract_function_outputs_with_presence(problem_path, task_weightage.keys())
    solution_outputs = extract_function_outputs_with_presence(solution_path, task_weightage.keys())

    output_scores = compare_outputs(problem_outputs, solution_outputs, task_weightage)
    task_df = pd.DataFrame(list(task_weightage.items()), columns=['method_name', 'max_score'])
    merged_df = output_scores.merge(task_df, on='method_name', how='inner')

    # Add metadata
    merged_df['UserEmail'] = user_email
    merged_df['attempt_id'] = attempt_id
    merged_df['timestamp'] = datetime.now(ZoneInfo("Asia/Kolkata"))
    merged_df['project'] = project

    # Evaluate image similarity for outlier_treatment
    problem_img = extract_plot_image(problem_path, "outlier_treatment")
    solution_img = extract_plot_image(solution_path, "outlier_treatment")

    if problem_img and solution_img:
        size = (400, 400)
        p_img = problem_img.resize(size).convert("L")
        s_img = solution_img.resize(size).convert("L")
        sim_score, _ = ssim(np.array(p_img), np.array(s_img), full=True)
        if sim_score >= 0.95:
            merged_df.loc[merged_df["method_name"] == "outlier_treatment", "score_gained"] = task_weightage["outlier_treatment"]
            merged_df.loc[merged_df["method_name"] == "outlier_treatment", "remarks"] = "Success"

    return merged_df[['UserEmail', 'attempt_id', 'method_name', 'score_gained',
                      'max_score', 'timestamp', 'remarks', 'project']]

# Generate final score dataframe
score_df = evaluate_notebooks(problem_notebook_path, solution_notebook_path, task_weightage)
score_df.reset_index(drop=True, inplace=True)
#score_df



# In[21]:


import sys
import pandas as pd
import mysql.connector
 
# Get user email from Node.js (passed as an argument)
if len(sys.argv) < 2:
    print("User email not provided")
    sys.exit(1)
 


# In[22]:


# MySQL Connection Setup
db_config = {
    "host": 'arshniv.cuceurst1z3t.us-east-1.rds.amazonaws.com',
    "user": 'admin',
    "password": 'arshnivdb',
    "database":'autovmharbor'
}


# In[23]:


def insert_results_into_db(df):
    try:
        # Connect to MySQL
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # Insert each row into MySQL
        for _, row in df.iterrows():
            sql = """INSERT INTO assignment_results 
                     (UserEmail, attempt_id, method_name, score_gained, max_score,timestamp,remarks,project)
                     VALUES (%s, %s, %s, %s, %s, %s, %s,%s)"""
            values = (
                row["UserEmail"], 
                row["attempt_id"], 
                row["method_name"], 
                row["score_gained"], 
                row["max_score"],
                row["timestamp"],
                row['remarks'],
                row['project']
            )
            cursor.execute(sql, values)

        # Commit and close
        conn.commit()
        cursor.close()
        conn.close()
        print("Results inserted into database successfully.")

    except Exception as e:
        print("Error inserting results into database:", e)


# In[24]:


insert_results_into_db(score_df)


# In[5]:


def describe_table(table_name):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute(f"DESCRIBE {table_name};")
        result = cursor.fetchall()
        for row in result:
            print(row)
        cursor.close()
        conn.close()
    except Exception as e:
        print("Error describing table:", e)

describe_table("assignment_results")


# In[ ]:




