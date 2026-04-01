# Agentic-AI-Travel-Planner: Travel AI Concierge

An intelligent, multi-agent travel concierge application built with a modern React/Vite frontend and a powerful Python Flask/LangGraph backend.

## Architecture

* **Frontend:** Built with React, Vite, and Tailwind CSS. The frontend provides a responsive chat-like interface, interactive system traces, detailed itinerary views, and an automatically generated visual calendar.
* **Backend:** A robust Python API driven by Flask. It acts as an orchestrator using LangChain and LangGraph to manage complex reasoning, routing, search, calculation, and safety-checking steps for every user query. Let your LLM autonomously search the internet, calculate budgets, generate files, and get accurate real-time weather information locally!

## Key Features

* **Real-time Agentic Tracing:** Visually track the execution of each underlying tool or LLM reasoning stage.
* **Contextual Conversational Logic:** Intelligent interpretation of destination, origins, budget scale, and diet preferences with a unified "memory".
* **Visual Itinerary & Extracted Calendar:** Converts multi-day unstructured outputs mathematically into structured JSON dates/schedules, mapping them onto a dynamic React calendar or generating an `.ics` export file.
* **Live Tool Actions:** Agents autonomously access a series of real-time tools including `WebSearch` with DuckDuckGo, `GetWeather` via `wttr.in`, currency conversion, and algebraic budget-checking!

## Setup and Running Environment

This project expects Node for modern Vite capabilities and Python 3.10+ for LangGraph API capabilities.

### Prerequisites & Dependencies
1. Make sure you have python `requirements.txt` / pip modules installed globally or within your respective Python virtual environment:
   ```bash
   pip install duckduckgo_search flask flask_cors icalendar langchain langchain_core langchain_openai langgraph
   ```
2. For your Node application, make sure all packages are resolved:
   ```bash
   cd travel-agent-ai
   npm install
   ```

### Execution

1. First, export your Groq API key (to enable running fast open-source models).
   ```bash
   export GROQ_API_KEY="your_api_key_here"
   ```

2. Start the Backend API. This will run locally on port `8001`:
   ```bash
   python3 travel_agent.py
   ```

3. Ensure you start up your React development server. It will run on port `3000` and accurately proxy `/api/` calls out of the UI and into the `8001` backend. You can open a new terminal pane from the `travel-agent-ai` directory:
   ```bash
   npm run dev
   ```

You are now ready to visit [http://localhost:3000](http://localhost:3000) and deploy your travel agent trips!
