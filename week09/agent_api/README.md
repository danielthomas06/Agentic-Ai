# Week 9 — FastAPI Agent API

This project exposes the Week 7 multi-agent supervisor through a REST API.

## Architecture

Client
   ↓
FastAPI
   ↓
service.py
   ↓
Week 7 LangGraph
   ↓
Supervisor
   ↓
Research → Analyst → Writer
   ↓
Final Answer

## Start the API

From the repository root:

```powershell
uvicorn week09.agent_api.main:app --reload