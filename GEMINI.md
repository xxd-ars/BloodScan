# BloodScan Project Gemini AI Assistant Development Guidelines

This document outlines the best practices and standards for interacting with the Gemini AI assistant within the BloodScan project. Adhering to these guidelines will improve efficiency, reduce errors, and ensure consistency.

## 1. Core Principles

- **Code-First:** Prioritize generating or modifying code over descriptive text. Instead of asking "How do I do X?", ask "Implement X for me." Gemini should directly perform the task.

- **Iterative Development:** Break down complex tasks into smaller, manageable steps. For example, instead of "Build the entire UI," start with "Create the main window and a button." Then, iteratively add more components. This makes debugging easier and keeps the context clear.

- **Leverage Context:** Use the `@` symbol to reference specific files or directories. This provides Gemini with the necessary context, leading to more accurate and relevant responses.

- **Clear Instructions:** Be as specific as possible in your requests. For instance, instead of "Fix the bug in this file," describe the bug and the expected behavior: "In `@src/utils/func.py`, the `calculate_iou` function fails when masks are empty. Modify it to return 0 in such cases."

## 2. Workflow

### 2.1. Understand & Plan
- **Goal:** Before writing code, ensure Gemini understands the context.
- **Action:** Use commands like `ls -R` to list the project structure or ask Gemini to read the contents of relevant files (`@<file_path>`).
- **Example:** "First, read `@dual_yolo/d_model_evaluate.py` and `@dual_yolo/d_dataset_config.py` to understand the current evaluation flow."

### 2.2. Execute & Verify
- **Goal:** Implement the planned changes and verify they work.
- **Action:** Request specific code changes, additions, or file creations. Immediately follow up by asking Gemini to run tests or execute the script to validate the changes.
- **Example:** "Now, add a new function to `d_model_evaluate.py` to calculate the F1-score. After that, run the script to ensure there are no errors."

### 2.3. Refactor & Optimize
- **Goal:** Improve the quality and efficiency of the code.
- **Action:** Once a feature is working, ask Gemini to refactor it. This could involve improving readability, optimizing performance, or ensuring it adheres to project conventions.
- **Example:** "The new F1-score function is working. Please refactor it to be more concise and add comments explaining the logic."

## 3. Common Command Examples

### 3.1. Code Generation & Modification
- **To create a new function:** "In `@llm/json_utils.py`, create a function named `save_to_json` that takes a dictionary and a file path, and saves the dictionary to the specified JSON file."
- **To modify an existing function:** "Modify the `train` function in `@blue_yolo/yolo_train.py`. Change the number of epochs from 50 to 100."
- **To add a new class:** "Create a new Python script at `@src/utils/image_processor.py` containing a class `ImageProcessor` with methods for resizing and rotating images."

### 3.2. File Operations
- **To create a file:** "Create a new file named `README.md` in the `@dual_yolo/evaluation_results_aug/` directory."
- **To read a file:** "What are the contents of `@setup.py`?"
- **To list files:** "Show me all the `.yaml` configuration files in the `@dual_yolo/models/` directory."

### 3.3. Testing
- **To write a test:** "Write a pytest unit test for the `CrossModalAttention` module in `@ultralytics/nn/modules/fusion.py`. The test should verify the output tensor shape."
- **To run tests:** "Run all tests in the `tests/` directory."

### 3.4. Analysis & Explanation
- **To explain code:** "Explain the purpose of the `d_model_evaluate.py` script in the `@dual_yolo` folder."
- **To analyze results:** "Analyze the `results.csv` file in `@dual_yolo/runs/segment/dual_modal_train_crossattn-30epoch/` and summarize the model's performance."
- **To generate a document:** "Based on the scripts in `@dual_yolo`, generate a presentation outline detailing the workflow and key technical decisions."

## 4. Important Notes

- **API Keys & Sensitive Information:** Never include API keys, passwords, or other sensitive information in your prompts. Use environment variables or a secure configuration method.
- **Large File Handling:** For large files, ask Gemini to read or analyze specific sections or functions instead of the entire file to avoid context overload.
- **Environment Dependencies:** If you encounter an error, remember to inform Gemini about the project's dependencies (e.g., from `requirements.txt`) so it can provide compatible code.
- **Stay in Sync:** After Gemini makes changes, ensure your local editor reflects these changes to avoid context discrepancies in subsequent requests.
