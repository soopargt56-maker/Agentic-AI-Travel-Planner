# Agentic-AI-Travel-Planner: Travel AI Concierge

An intelligent, multi-agent travel concierge application built with a modern React/Vite frontend and a powerful Python Flask/LangGraph backend.

## Architecture

* **Frontend:** Built with React, Vite, and Tailwind CSS. The frontend provides a responsive chat-like interface, interactive system traces, detailed itinerary views, a fully navigable visual calendar, trip history browser, and persistent user memory display.
* **Backend:** A robust Python API driven by Flask. It acts as an orchestrator using LangChain and LangGraph to manage complex reasoning, routing, search, calculation, and safety-checking steps for every user query. Powered by Groq's blazing-fast LLM inference with Llama 3.3 70B.

## Key Features

* **Real-time Agentic Tracing:** Visually track the execution of each underlying tool or LLM reasoning stage.
* **Contextual Conversational Logic:** Intelligent interpretation of destination, origins, budget scale, and diet preferences with a unified "memory".
* **Full Visual Calendar:** Navigate between months with Previous/Next/Today buttons. Trip events are mapped directly onto calendar dates with highlighted event days.
* **Trip History Panel:** Browse, select, and delete past trip itineraries from the left sidebar.
* **User Memory Panel:** View the agent's learned traveler profile — home city, nationality, diet, budget style, last destination, and past trip history.
* **Schedule Overlap Detection:** The planner automatically checks for date conflicts with existing trips and warns before planning.
* **Enhanced Itinerary Generation:** When tool-calling fails, the agent falls back to direct LLM calls enriched with web search data for specific, real-world itineraries.
* **Live Tool Actions:** Agents autonomously access real-time tools including `WebSearch` with DuckDuckGo, `GetWeather` via `wttr.in`, currency conversion, and algebraic budget-checking.
* **Calendar Export:** Generate and download `.ics` calendar files for any planned trip.

## Setup and Running Environment

This project expects Node for modern Vite capabilities and Python 3.10+ for LangGraph API capabilities.

### Prerequisites & Dependencies
1. Make sure you have python `requirements.txt` / pip modules installed globally or within your respective Python virtual environment:
   ```bash
   pip install -r requirements.txt
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

3. Start the React development server. It will run on port `3000` and proxy `/api/` calls to the backend:
   ```bash
   cd travel-agent-ai
   npm run dev
   ```

You are now ready to visit [http://localhost:3000](http://localhost:3000) and deploy your travel agent trips!

## UI Overview

| Panel | Description |
|-------|-------------|
| **Trace** | Real-time LangGraph execution log with node timings |
| **History** | Browse and manage all past trip plans |
| **Memory** | View learned traveler preferences and past trip summary |
| **Itinerary Map** | Full markdown-rendered itinerary with budget table |
| **Visual Calendar** | Month-by-month calendar with trip events highlighted |
