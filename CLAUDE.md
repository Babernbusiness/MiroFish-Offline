# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

MiroFish-Offline is a multi-agent social media simulation engine. Upload a document, and it generates hundreds of AI agents with unique personalities that simulate public reaction on social media. The entire stack runs locally via Neo4j + Ollama.

## Architecture

```
frontend (Vue 3, Vite) → Flask API → Service Layer → GraphStorage (Neo4j + Ollama)
```

**Workflow stages:**
1. **Graph Build** — NER extraction via Ollama LLM, embed chunks with nomic-embed-text, store in Neo4j
2. **Env Setup** — OASIS profile generator creates agent personas (PoliticalLeader, Diplomat, MediaOutlet, etc.)
3. **Simulation** — OASIS simulation runner manages agent interactions (posts, replies, arguments, opinion shifts)
4. **Report** — ReportAgent interviews agents post-simulation, searches Neo4j for evidence, generates structured analysis

**Key services:**
- `GraphBuilderService` — orchestrates entity extraction and graph construction
- `SimulationManager` / `SimulationRunner` — manages simulation lifecycle, runs OASIS as subprocess
- `ReportAgent` — generates post-simulation analysis via LLM + Neo4j graph queries
- `Neo4jStorage` — abstract graph storage backed by Neo4j CE 5.x
- `EmbeddingService` — Ollama nomic-embed-text for vector search
- `NERExtractor` — local LLM for named entity extraction
- `LLMClient` — unified OpenAI-compatible LLM client for all services

**API endpoints (Flask blueprints):**
- `/api/graph` — knowledge graph operations
- `/api/simulation` — simulation lifecycle (create, start, stop, status)
- `/api/report` — report generation

## Commands

```bash
# Start all services (Neo4j + Ollama + MiroFish)
docker compose up -d

# Pull required models
docker exec mirofish-ollama ollama pull qwen2.5:32b
docker exec mirofish-ollama ollama pull nomic-embed-text

# Run backend only (manual)
cd backend && python run.py

# Run frontend only (manual)
cd frontend && npm run dev

# Backend runs on http://localhost:5001
# Frontend on http://localhost:3000
```

## Configuration

All settings in `.env` (copy from `.env.example`):
- `LLM_BASE_URL` / `LLM_MODEL_NAME` — Ollama endpoint and model
- `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` — Neo4j connection
- `EMBEDDING_MODEL` / `EMBEDDING_BASE_URL` — embedding service

## Model Notes

- Qwen3 series models (`qwen3.5:9b`, etc.) have a **thinking mode** enabled by default — they output reasoning traces before the actual answer. This causes JSON-mode API responses to be malformed. Always pass `think: false` when using JSON mode with Qwen3 models via the OpenAI-compatible endpoint.
- Native Ollama `/api/generate` endpoint respects `think: false` parameter directly.
- `qwen2.5` models do not have this issue.
