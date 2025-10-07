<div align="center">

# 🧠 The LLM Teacher Framework 🎓

*An end-to-end framework for fine-tuning Small Language Models (SLMs) using a Large Language Model (LLM) as a teacher and judge.*

</div>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-orange.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT">
</p>

This project demonstrates a complete, real-world workflow for creating a specialized AI model. It goes beyond simple training scripts to include a full, iterative development loop: synthetic data generation, parameter-efficient fine-tuning, quantitative evaluation, and a targeted refinement strategy.

---

### **The "LLM Teacher" Framework** 💡

The core idea is to use a large, powerful "Teacher" LLM (like Cohere's Command R) to mentor a smaller, more efficient "Student" SLM (like Microsoft's Phi-3 Mini). This mentorship happens in three key stages:

1.  **Curriculum Design (Data Generation):** The Teacher LLM generates a high-quality, diverse dataset of questions and answers on a specific topic (in this case, RAG systems).
2.  **Study Session (Fine-Tuning):** The Student SLM is fine-tuned on this custom dataset using Parameter-Efficient Fine-Tuning (LoRA), allowing it to become an expert without the need for massive computational resources.
3.  **Final Exam (Evaluation):** The Teacher LLM acts as an impartial judge, scoring the Student's performance on a hold-out test set to provide a quantitative measure of its new expertise.

*[Optional: Create a simple workflow diagram in a tool like Excalidraw or PowerPoint, save it as an image in your project, and embed it here with this line: `![Workflow Diagram](path/to/your/diagram.png)`]*

---

### **Key Features** ✨

* **Synthetic Data Generation:** A robust, resumable Python script to generate high-quality training data using a powerful Teacher LLM.
* **Parameter-Efficient Fine-Tuning (PEFT):** Utilizes **LoRA** to fine-tune the Student SLM efficiently on a consumer-grade Mac with Apple Silicon.
* **LLM-as-a-Judge Evaluation:** Implements a quantitative evaluation framework where the Teacher LLM scores the Student's performance, achieving an **8.88 / 10** average score.
* **Iterative Refinement Loop:** The entire process is designed as a loop. By analyzing the evaluation results, we can identify knowledge gaps and generate a "booster pack" of data to patch them, demonstrating a true machine learning iteration cycle.

---

### **Tech Stack** 🛠️

* **Models:** `microsoft/phi-3-mini-4k-instruct`, `cohere/command-r`
* **Orchestration:** `Ollama` for local model serving
* **Fine-Tuning:** `PyTorch`, Hugging Face `transformers`, `peft` (LoRA), `trl` (`SFTTrainer`)
* **Environment:** Python 3.11+, `venv`, `git`

---

### **How to Run This Project** 🚀

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/LLM-Teacher-Framework.git](https://github.com/YourUsername/LLM-Teacher-Framework.git)
    cd LLM-Teacher-Framework
    ```
2.  **Set up the environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```
3.  **Download Ollama models:**
    ```bash
    ollama pull command-r
    ollama pull phi3:mini
    ```
4.  **Run the pipeline scripts in order:**
    * `python generate_rag_dataset.py`  *(to create the dataset)*
    * `python finetune_model.py`          *(to fine-tune the model)*
    * `python test_expert_model.py`       *(to chat with your new expert)*
    * *...and so on for evaluation.*

---

### **What Sets This Project Apart: My Key Learnings & Insights** 🧠

This project was more than just writing code; it was a deep dive into the practical realities of building with local LLMs. Here’s what differentiates this work:

* **Beyond the Happy Path:** I didn't just run a script. I navigated a complex debugging process involving hangs, library version mismatches (`AttributeError`, `TypeError`), and silent out-of-memory errors. The final scripts in this repo are hardened and resilient because they were forged through solving these real-world problems.

* **The Primacy of Data Quality:** My first fine-tuned model was a failure. It only memorized, it couldn't generalize (e.g., it didn't know what "RAG" was). This taught me the most critical lesson: **the quality and diversity of the training data are more important than anything else.** I iterated on the data generation prompt to create a more varied "curriculum," which directly led to the final model's success.

* **Iterative, Data-Driven Improvement:** This project implements the full machine learning loop. We didn't just train a model; we **evaluated it quantitatively** (achieving 8.88/10), **analyzed its specific weaknesses** (e.g., lack of depth on "re-rankers"), and designed a **targeted plan to fix them**. This demonstrates a mature, data-driven approach to model improvement.

This repository isn't just a collection of scripts; it's a testament to a persistent, problem-solving journey through the entire lifecycle of creating a specialized AI model.