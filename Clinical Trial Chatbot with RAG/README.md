# Clinical Trial Discovery Chatbot

An AI-powered assistant that helps patients discover clinical trials using natural language. The system combines a Retrieval-Augmented Generation (RAG) pipeline with BioBERT embeddings, Pinecone vector search, and a Streamlit interface to deliver simplified, patient-friendly explanations of complex trial information.

---
# Overview
This repository contains the frontend chatbot application, the data ingestion pipeline, and supporting workflow assets (pipeline diagram + demo video).
The core system works by:
- Ingesting and preprocessing trial data from ClinicalTrials.gov
- Generating BioBERT embeddings for all unstructured text (titles, descriptions, inclusion/exclusion criteria)
- Indexing vectors in Pinecone for fast semantic search
- Storing full structured metadata in Firebase Firestore
- Exposing a patient-facing interface via Streamlit that supports trial search, summarization, and eligibility reasoning  

The main user-facing file is app.py, which loads embeddings, retrieves relevant trials, and generates responses through an LLM.

---
# AI & NLP Stack
- BioBERT for medical-domain embeddings
- Pinecone for semantic vector search
- OpenBioLLM for generation
- Streamlit for frontend deployment
- Firebase Firestore for trial metadata and bookmarking, user sessions and logins
- Google Cloud Run + Scheduler for automated ingestion jobs

---
# Demo

A short demo video is included in the repo demonstrating:
- Natural-language trial search
- AI-generated plain-language summaries
- Eligibility reasoning
- Trial bookmarking and retrieval

---
# Project Goals

This system aims to reduce barriers to clinical trial discovery by enabling patients to search using natural language instead of navigating complex medical datasets. It also generates clear, accessible summaries of trial criteria and supports preliminary eligibility reasoning. Ultimately, the chatbot improves discoverability and comprehension of clinical trial options while adhering to privacy principles and secure design.
