# Multi-Agent Debate for LLM Truthfulness

This repository implements a prompt-based multi-agent debate framework
(Proposer → Critic → Judge) to improve truthfulness and reduce hallucinations
in Large Language Models.

## Motivation
LLMs often generate confident but incorrect answers.
Inspired by human debate, we study whether structured interaction between
LLM agents improves factual accuracy and calibration.

## Features
- Multi-agent debate pipeline
- Strong baselines: single-agent, self-critique, self-consistency
- Evaluation on TruthfulQA and TriviaQA
- Hallucination and calibration analysis

## Installation
```bash
pip install -r requirements.txt
