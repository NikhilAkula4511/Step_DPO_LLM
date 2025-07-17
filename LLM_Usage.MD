# 🤖 LLM Usage Guide with Stepwise DPO – Deep Explanation

This document explains how to **load, interact with, and understand** a language model fine-tuned using **Stepwise Direct Preference Optimization (DPO)**, particularly in reasoning tasks that require multi-step logical deductions.

---

## 🎯 Objective

After training a model using **Stepwise DPO**, the goal is to:

- Generate high-quality, **step-by-step logical answers**.
- Encourage **intermediate reasoning steps** during generation.
- Evaluate how well the model adheres to **logical consistency**.

---

## 🧩 Model Overview

- **Base model:** `mistralai/Mistral-7B` (or similar, depending on your config)
- **Fine-tuning method:** Stepwise Direct Preference Optimization (DPO)
- **Training objective:** Prefer completions that demonstrate correct step-by-step reasoning.
- **Output behavior:** The model is encouraged to not just jump to the final answer but to **verbalize its thought process**.

---

## 🛠️ 1. Environment Setup

Before using the model:

```bash
pip install -r requirements.txt
