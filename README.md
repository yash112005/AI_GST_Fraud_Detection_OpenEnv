---
title: AI GST Fraud Detection Env
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: docker
app_file: server.app:main 
pinned: false
---

# AI GST Fraud Detection Environment

## About the Project
This project is a simple simulation of a GST fraud detection system.

It uses invoice data (like amount and payment delay) to decide whether a transaction is fraud or not. The idea is to create an environment where an agent can learn how to detect fraud step by step.

---

## How it Works
- The environment gives a **state** (invoice details)
- The agent takes an **action**:
  - `0` → Not fraud  
  - `1` → Fraud  
- Based on the action, the agent gets a **reward**

The goal is to take better decisions and get higher rewards.

---

## Difficulty Levels

### Easy
- Basic level  
- Correct answer → good reward  

### Medium
- Slightly harder  
- Wrong answers can give negative reward  

### Hard
- More realistic  
- Reward depends on how risky the transaction is  

---

## What Makes It Interesting
- Uses simple ML model (RandomForest) for prediction  
- Includes real-world logic like:
  - payment delays  
  - high invoice amounts  
- Reward system changes based on difficulty  
- Designed for reinforcement learning experiments  

---

## API Endpoints

- `/reset` → gives a new starting state  
- `/step` → takes action and returns result  

Example:
```json
{
  "action": 1
}