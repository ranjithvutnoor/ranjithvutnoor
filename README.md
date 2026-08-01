<div align="center">

# Hi, I'm Ranjith Vutnoor 👋

### AI/ML Engineer · Generative AI Practitioner · GPU-Accelerated ML Research · IIT Bhilai Alumnus

**M.Tech in Data Science and Artificial Intelligence**

I build and explore intelligent systems at the intersection of **machine learning, information retrieval, large language models, GPU-accelerated computing, software engineering, and applied research**.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge\&logo=linkedin\&logoColor=white)](YOUR_LINKEDIN_URL)
[![LeetCode](https://img.shields.io/badge/LeetCode-Profile-FFA116?style=for-the-badge\&logo=leetcode\&logoColor=black)](YOUR_LEETCODE_URL)
[![Kaggle](https://img.shields.io/badge/Kaggle-Profile-20BEFF?style=for-the-badge\&logo=kaggle\&logoColor=white)](YOUR_KAGGLE_URL)
[![Email](https://img.shields.io/badge/Email-Contact-EA4335?style=for-the-badge\&logo=gmail\&logoColor=white)](mailto:YOUR_EMAIL_ADDRESS)

</div>

---

## About Me

I am an AI and machine learning professional with an **M.Tech in Data Science and Artificial Intelligence from the Indian Institute of Technology Bhilai**.

My work and technical interests focus on building intelligent systems that can retrieve information, reason over context, interact with tools, and generate reliable responses.

I have hands-on experience with:

* Large Language Model applications
* Retrieval-Augmented Generation systems
* Semantic search and information retrieval
* Vector databases and embedding models
* Retrieval and reranking pipelines
* Agentic AI workflows
* Structured model outputs
* AI system evaluation
* Model fine-tuning and experimentation
* Local LLM and vision-language model inference
* Backend integration for AI applications
* Industrial time-series anomaly detection
* CUDA and GPU-accelerated deep learning
* Custom PyTorch extensions and performance benchmarking

I enjoy taking technically complex ideas and converting them into practical, measurable, and maintainable software systems.

---

## What I Work On

### Generative AI Applications

I work on applications that combine Large Language Models with enterprise data, search systems, external tools, and structured workflows.

My areas of focus include:

* Prompt and context engineering
* Structured response generation
* Tool-integrated AI systems
* Multi-step reasoning workflows
* Retrieval quality improvement
* Response validation
* Hallucination reduction
* Model migration and evaluation
* Failure handling and fallback strategies

---

### Retrieval-Augmented Generation

I have worked extensively on retrieval pipelines for answering questions from technical and enterprise documentation.

My RAG-related work includes:

* Converting documents into retrieval-ready content
* Generating dense vector embeddings
* Query expansion using multiple search formulations
* Semantic similarity search
* Retrieval threshold tuning
* Candidate deduplication
* Document and chunk reranking
* Context selection
* Retrieval retries when the available evidence is insufficient
* Citation and source mapping
* Incremental document ingestion

I am especially interested in improving RAG systems through systematic evaluation rather than relying only on subjective response quality.

---

### AI Evaluation

I work with retrieval and ranking metrics to understand whether an AI system is actually improving.

Metrics and evaluation concepts I have explored include:

* Mean Reciprocal Rank
* Normalized Discounted Cumulative Gain
* Hit rate
* Precision and recall
* Retrieval coverage
* Relevance scoring
* Human evaluation
* LLM-assisted evaluation
* Context sufficiency
* Response grounding
* Latency and quality trade-offs

---

## Selected Engineering Work

### Enterprise RAG and Semantic Search Platform

Contributed to the development and improvement of an enterprise Retrieval-Augmented Generation system for searching and answering questions from technical documentation.

#### Work involved

* Dense embedding-based document retrieval
* Query expansion into multiple search phrases
* Retrieval of candidate chunks from a vector database
* Similarity threshold tuning
* Candidate aggregation and deduplication
* Reranking of retrieved content
* Selection of the strongest evidence for the final response
* Evaluation using ranking and retrieval metrics
* Iterative retrieval when initial context was insufficient
* Incremental document update handling

#### Technical themes

`Python` · `LLMs` · `Embeddings` · `Qdrant` · `Semantic Search` · `Reranking` · `RAG Evaluation`

---

### Document Ingestion and Knowledge Processing

Worked on ingestion workflows for converting technical documentation into structured, searchable knowledge.

Areas explored include:

* HTML document parsing
* Topic and section identification
* Processing tables, figures, and code blocks
* Stable content identifiers
* Document-to-URL mapping
* Metadata preservation
* Change detection
* Incremental indexing
* Updating only changed documentation
* Maintaining traceability between retrieved content and source pages

---

### Reflective Retrieval Workflow

Explored a reflective retrieval approach in which an AI system evaluates whether the retrieved evidence is sufficient before generating a final answer.

The workflow can:

1. Analyse the user’s question
2. Generate multiple search formulations
3. Retrieve relevant evidence
4. Evaluate whether the evidence is sufficient
5. Reformulate the search when necessary
6. Retrieve additional information
7. Produce a grounded response from the strongest available context

This approach is useful for reducing premature answers and improving coverage for complex questions.

---

### AI Agent and Tool Integration

Worked with systems where a language model can select and interact with external tools or data sources.

Areas explored include:

* Tool definition and selection
* Structured function arguments
* Multi-step execution
* Context management
* Tool result validation
* Error recovery
* Response synthesis
* Model Context Protocol concepts
* Agent observability
* Agent evaluation

---

## Research and Experimentation

### M.Tech Thesis — Industrial Time-Series Anomaly Detection and GPU Acceleration

**Title:** *Anomaly Detection of Multivariate Time Series Data and Acceleration Using GPUs*  
**Institution:** Indian Institute of Technology Bhilai  
**Department:** Computer Science and Engineering  
**Degree:** M.Tech in Data Science and Artificial Intelligence  
**Completed:** June 2023  
**Supervisor:** Dr. Vishwesh Jatala

My thesis addressed anomaly detection in high-volume industrial sensor data collected from the **Bar and Rod Mill at Bhilai Steel Plant**. The objective was to detect abnormal operating behaviour—particularly cobble-related events—while reducing the computational cost of multivariate deep-learning analysis.

#### Research scope

* Analysed approximately **1,860 PLC-generated industrial signals**
* Studied univariate and multivariate time-series anomaly patterns
* Detected HMD flickering through Roll-ON/Roll-OFF period analysis
* Analysed looper-signal abnormalities using moving averages, lag-based comparisons, and percentage deviation
* Modelled overshoot and undershoot events using the **MSCRED** framework
* Used an unsupervised **ConvLSTM encoder-decoder** to capture temporal and inter-signal dependencies
* Generated reconstruction-based anomaly scores for multivariate signals

#### CUDA programming, im2col, and custom PyTorch GPU extension

The multivariate model contained computationally expensive convolution operations across the encoder, ConvLSTM, and decoder. I investigated this bottleneck by transforming Conv2D into a **GEMM-based operation using lowering/im2col**, then accelerating the matrix multiplication with CUDA.

Key implementation areas included:

* CUDA programming for parallel matrix multiplication
* GEMM-based Conv2D acceleration
* GPU global-memory and shared-memory analysis
* GPU shared-memory optimisation to reduce high-latency global-memory access
* A custom PyTorch extension for GPU computation
* CPU-versus-GPU performance benchmarking
* Scalability analysis for larger time-series workloads

The experiments were conducted on an **NVIDIA RTX A6000 GPU with 48 GB memory**, alongside a 16-core Intel Xeon Gold 5218 CPU and 64 GB RAM.

#### Reported benchmark

For a benchmark with **36,000 data points, 30 features, batch size 128, and 32 threads**:

| Component | CPU | GPU without shared memory | GPU with shared memory |
|---|---:|---:|---:|
| Encoder | 300 s | 27.492 s | **23.664 s** |
| ConvLSTM | 1,200 s | 909.208 s | **847.260 s** |
| Total execution | 2,700 s | 1,594 s | **1,481 s** |
| F1 score | 0.930 | 0.936 | **0.941** |

The shared-memory GPU implementation achieved approximately:

* **12.68× faster encoder execution** than the CPU implementation
* **1.82× faster total execution**, equivalent to about a **45.1% reduction in total runtime**
* Comparable predictive quality across implementations, with F1 remaining around **0.93–0.94**

#### Research themes

`Time-Series Analysis` · `Anomaly Detection` · `Industrial AI` · `MSCRED` · `ConvLSTM` · `PyTorch` · `CUDA` · `GEMM` · `Conv2D` · `GPU Shared Memory` · `High-Performance Computing`

> The raw industrial dataset should not be published publicly unless the data owner has explicitly authorised its release. A public thesis repository can instead contain the abstract, architecture, synthetic examples, implementation notes, benchmark tables, and non-confidential code.

---

### NVIDIA Nemotron Model Reasoning Challenge

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-NVIDIA%20Nemotron%20Reasoning%20Challenge-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge)

Participated in NVIDIA's structured-reasoning competition using the **Nemotron-3-Nano-30B** open model. The objective was to improve reasoning accuracy and submit a compatible **LoRA adapter with rank no greater than 32**, evaluated through deterministic vLLM inference and final-answer extraction from `\boxed{}` output.

#### Result

* Improved the competition score from approximately **0.53 to 0.668** through repeated data, prompting, training, and packaging iterations
* Finished at approximately **public leaderboard rank 3,345** and **private leaderboard rank 3,020**
* Gained approximately **325 positions** during the public-to-private leaderboard transition
* Competed in a field of more than **4,100 teams** and over **71,000 submissions**

> Leaderboard positions can change when late submissions or display rules are applied. Keep a screenshot of the final leaderboard result in the project repository as supporting evidence.

#### Problem statement

The benchmark tested whether a language model could infer hidden rules from worked examples and apply the discovered rule to a new query. The tasks required more than general language generation: the model needed to identify the task family, infer the latent transformation, verify its hypothesis against the examples, perform the final computation, and return the exact answer in the required format.

The challenge allowed prompting, data curation, synthetic-data generation, reinforcement learning, and lightweight fine-tuning. My approach focused on **reasoning-oriented synthetic data engineering and QLoRA-style supervised fine-tuning** rather than full-parameter fine-tuning or reinforcement learning.

#### My approach

1. **Analysed the benchmark structure** and separated problems into reusable reasoning families, including bit manipulation, physics and gravity, unit conversion, substitution ciphers, numeral systems, and custom arithmetic/operator transformations.
2. **Designed deterministic reasoning procedures** for each family instead of relying on unconstrained free-form chain-of-thought generation.
3. **Created and curated synthetic reasoning traces** containing task classification, the key diagnostic, the inferred rule, verification against examples, application to the query, and a strict boxed answer.
4. **Compressed the reasoning traces** to approximately 50–150 tokens where possible so that training capacity was spent on the successful reasoning path instead of verbose or repetitive text.
5. **Applied 4-bit QLoRA-style parameter-efficient fine-tuning** to the Nemotron base model using the Hugging Face ecosystem.
6. **Resolved model-architecture and low-precision compatibility problems** involving Mamba kernels, RMSNorm fallbacks, Mixture-of-Experts dtype mismatches, and 4-bit execution.
7. **Evaluated multiple checkpoints and iterations**, selected the strongest adapter checkpoint, validated the required files, and packaged the submission in Kaggle's expected format.
8. **Used leaderboard feedback and error analysis** to refine the data distribution, reasoning format, and final-answer reliability.

#### Reasoning-data design

The training traces followed a compact structure:

```text
<think>
Type: <reasoning category>
Key observation or diagnostic
Hypothesis and verification across examples
RULE FOUND: <concise transformation>
Applying to query: <calculation>
</think>
\boxed{<final answer>}
```

Important design decisions included:

* Teaching the model to **classify before computing**
* Selecting the simplest hypothesis that explains all examples
* Rejecting a hypothesis immediately when an example fails
* Verifying the selected rule on multiple examples
* Adding numerical sign, magnitude, precision, and formatting checks
* Keeping only the successful reasoning path in fine-tuning data
* Ensuring that `\boxed{}` contains only the final answer

#### Problem families and solver logic

**Bit manipulation**

* XOR, AND, OR, NOT, rotations, reversals, shifts, and nibble swaps
* Compositions such as rotation followed by XOR with a constant
* Neighbour and majority/choice Boolean functions
* Per-bit GF(2) analysis as a systematic fallback when simpler whole-byte rules failed

**Physics and numerical reasoning**

* Recovering hidden constants from examples
* Applying physical relationships such as quadratic distance-time behaviour
* Full-precision intermediate arithmetic with final-step rounding
* Interval-based estimation around rounded observations

**Unit conversion and numeral systems**

* Identification and application of conversion factors
* Binary, octal, hexadecimal, Roman-numeral, and custom representations
* Explicit output-format matching, including leading zeros and precision

**Cipher and transformation problems**

* Building substitution mappings from demonstrations
* Elimination over unused characters
* Testing candidate mappings against vocabulary and all examples
* Inferring custom arithmetic operators through difference tables, algebraic decomposition, and hypothesis competition

#### QLoRA-style fine-tuning pipeline

This was **parameter-efficient fine-tuning**, not full-model retraining:

* Loaded the Nemotron model for **4-bit low-memory training**
* Used **PEFT LoRA adapters** so that only a small set of adapter parameters was trained
* Used the Hugging Face stack: `Transformers`, `PEFT`, `bitsandbytes`, `TRL`, `Accelerate`, and `Safetensors`
* Performed supervised fine-tuning on structured synthetic reasoning traces
* Preserved the competition requirement that the final adapter rank be no greater than 32
* Selected `checkpoint-1056` as the final adapter checkpoint in the submission workflow
* Cast trainable adapter weights to `float16` for compact packaging
* Verified the presence of `adapter_config.json` and `adapter_model.safetensors`
* Packaged only the required adapter files into `submission.zip`

> Exact values for LoRA rank, alpha, dropout, learning rate, batch size, gradient accumulation, sequence length, and epoch count should be added from the final training script or `adapter_config.json`. They should not be guessed from the competition limits.

#### Engineering problems solved

**Hybrid model and 4-bit compatibility**

The Nemotron architecture required additional engineering for QLoRA-style execution. I prepared a reproducible RunPod setup and applied compatibility patches that:

* Replaced unavailable fused RMSNorm behaviour with a PyTorch fallback
* Disabled a fused Mamba execution path that was incompatible with 4-bit weights
* Corrected dtype mismatches in Mixture-of-Experts aggregation
* Corrected dummy-expert dtype handling
* Cleared cached model modules so that patched source was loaded consistently
* Validated all downloaded `safetensors` shards before training

**Reasoning quality and answer extraction**

* Reduced verbose reasoning that wasted sequence capacity
* Added explicit rule summaries and cross-validation
* Prevented explanations from leaking into the final boxed answer
* Matched the competition's deterministic decoding and exact-answer evaluation behaviour

**Submission reliability**

* Loaded the selected checkpoint explicitly rather than relying on the last training state
* Reduced adapter size using `float16`
* Validated mandatory adapter files before zipping
* Removed intermediate checkpoint directories from the final package

#### Technologies

`Python` · `PyTorch` · `Transformers` · `PEFT` · `QLoRA` · `LoRA` · `bitsandbytes` · `TRL` · `Accelerate` · `Safetensors` · `vLLM` · `Synthetic Data` · `Reasoning Evaluation` · `RunPod` · `NVIDIA GPUs`

---

### Local LLM Experimentation

I experiment with running language models locally to understand their quality, speed, memory requirements, and suitability for practical applications.

Models and model families explored include:

* Mistral
* DeepSeek distilled models
* Qwen
* Vision-language models
* Quantized GGUF models

Tools used for experimentation include:

* LM Studio
* Ollama
* Hugging Face
* Jupyter Notebook

Areas of investigation include:

* Quantization levels
* Model size versus response quality
* Prompt behaviour
* Inference latency
* Memory utilisation
* Local privacy
* CPU and GPU constraints
* Vision-language understanding
* Application integration

---

### Model and Retrieval Evaluation Dataset

Worked with a large evaluation dataset containing thousands of questions for comparing retrieval and response quality.

Areas evaluated include:

* Search relevance
* Retrieved context quality
* Ranking performance
* Answer completeness
* Citation accuracy
* Failure patterns
* Model consistency
* Performance across different question categories

---

## Personal Software Projects

### Desktop AI Companion and Productivity Application

Developed and experimented with an Electron-based desktop companion application designed to combine animation, interaction, and productivity.

The application concept includes:

* A desktop puppy that moves around the screen
* Cursor-following behaviour
* Natural movement and momentum
* Rest and play states
* Interactive animations
* Multiple companion characters
* Menu-bar controls
* Persistent preferences
* Low-resource operation
* Pomodoro-based focus sessions
* Task completion prompts
* Productivity reminders
* Desktop notifications

I have also explored how such an application could be extended with:

* Natural-language interaction
* Voice responses
* AI-generated behaviour
* User authentication
* Payment workflows
* Cloud persistence
* Subscription management
* Landing-page deployment

#### Technologies

`Electron` · `JavaScript` · `Desktop Applications` · `UI Interaction` · `Productivity Tools`

---

### AI and Developer Tool Experiments

I regularly build small prototypes and experiments involving:

* LLM-powered assistants
* Retrieval pipelines
* Local model interfaces
* Prompt evaluation tools
* Data processing utilities
* Desktop productivity software
* API integrations
* Interactive AI applications

Some experiments are research-oriented, while others focus on learning how complete software products are designed, deployed, monitored, and improved.

---

## Technical Skills

### Programming

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square\&logo=python\&logoColor=white)
![SQL](https://img.shields.io/badge/SQL-4479A1?style=flat-square\&logo=mysql\&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat-square\&logo=javascript\&logoColor=black)
![Git](https://img.shields.io/badge/Git-F05032?style=flat-square\&logo=git\&logoColor=white)
![Linux](https://img.shields.io/badge/Linux-FCC624?style=flat-square\&logo=linux\&logoColor=black)

### AI and Machine Learning

![Machine Learning](https://img.shields.io/badge/Machine%20Learning-1F6FEB?style=flat-square)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-8A2BE2?style=flat-square)
![Generative AI](https://img.shields.io/badge/Generative%20AI-412991?style=flat-square)
![NLP](https://img.shields.io/badge/NLP-008080?style=flat-square)
![LLMs](https://img.shields.io/badge/Large%20Language%20Models-000000?style=flat-square)
![RAG](https://img.shields.io/badge/RAG-FF6F00?style=flat-square)

### GPU Computing and Deep Learning

![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat-square&logo=nvidia&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![NVIDIA](https://img.shields.io/badge/NVIDIA%20GPU-76B900?style=flat-square&logo=nvidia&logoColor=white)
![High Performance Computing](https://img.shields.io/badge/High--Performance%20Computing-4B5563?style=flat-square)

* CUDA-based GPU computation
* GPU global and shared memory
* GEMM and matrix-multiplication optimisation
* Lowering/im2col-based convolution
* Conv2D and ConvLSTM acceleration
* Custom PyTorch extensions
* CPU/GPU benchmarking
* Throughput, latency, memory, and scalability analysis

### AI Systems and Platforms

![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=flat-square\&logo=openai\&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-DC244C?style=flat-square)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=flat-square\&logo=huggingface\&logoColor=black)
![Ollama](https://img.shields.io/badge/Ollama-000000?style=flat-square)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square\&logo=jupyter\&logoColor=white)

### Concepts

* Large Language Models
* Retrieval-Augmented Generation
* Natural Language Processing
* Dense vector retrieval
* Embedding models
* Vector databases
* Semantic search
* Query expansion
* Reranking
* Prompt engineering
* Model evaluation
* Fine-tuning
* LoRA
* Quantized inference
* AI agents
* Model Context Protocol
* Structured outputs
* Information retrieval
* Document processing
* REST API integration
* Software architecture
* Industrial time-series analysis
* Anomaly detection
* ConvLSTM and encoder-decoder models
* CUDA and GPU acceleration
* GEMM-based convolution
* GPU shared-memory optimisation
* High-performance computing

---

## Data Structures and Algorithms

I consistently practise data structures and algorithms to improve problem-solving ability, coding precision, and interview readiness.

I have solved more than **240 problems** across areas such as:

* Arrays and strings
* Hash maps
* Two pointers
* Sliding window
* Binary search
* Linked lists
* Stacks and queues
* Trees and binary search trees
* Recursion and backtracking
* Dynamic programming
* Graphs
* Intervals
* Matrix traversal
* Greedy algorithms

Representative problems include:

* Add Two Numbers
* Longest Substring Without Repeating Characters
* Longest Palindromic Substring
* Container With Most Water
* 3Sum
* 4Sum
* Search in Rotated Sorted Array
* Rotate Image
* Group Anagrams
* Maximum Subarray
* Spiral Matrix
* Merge Intervals
* Set Matrix Zeroes
* Word Search
* Validate Binary Search Tree
* Recover Binary Search Tree

[![LeetCode](https://img.shields.io/badge/View%20My%20LeetCode-FFA116?style=for-the-badge\&logo=leetcode\&logoColor=black)](YOUR_LEETCODE_URL)

---

## Areas I Am Currently Strengthening

* Advanced data structures and algorithms
* Distributed systems
* System design
* AI application architecture
* Advanced CUDA kernel optimisation
* GPU inference systems and model serving
* Model serving
* Scalable retrieval infrastructure
* AI observability
* Evaluation-driven development
* Backend engineering
* Cloud deployment
* Production-grade agent systems

---

## Education

### Indian Institute of Technology Bhilai

**Master of Technology in Data Science and Artificial Intelligence**

`2021 – 2023`

Academic and research areas included:

* Machine learning
* Deep learning
* Artificial intelligence
* Data science
* Statistical modelling
* Optimisation
* Natural language processing
* Research methodology
* Computational problem-solving

My postgraduate experience strengthened my ability to approach engineering problems through a combination of theoretical understanding, experimentation, evaluation, and implementation.


**M.Tech Thesis:** *Anomaly Detection of Multivariate Time Series Data and Acceleration Using GPUs*

The thesis combined industrial time-series analytics, unsupervised deep learning, and GPU computing. It used statistical methods for univariate anomaly detection, MSCRED with ConvLSTM for multivariate signals, and CUDA-accelerated GEMM-based convolution with GPU shared-memory optimisation.

---

## Engineering Approach

I prefer building systems that are:

* Measurable rather than based only on intuition
* Reliable rather than impressive only in demonstrations
* Grounded in evidence
* Modular and maintainable
* Observable and testable
* Designed around real user problems
* Evaluated using clearly defined metrics
* Improved through structured experimentation

I believe strong AI engineering requires more than connecting an application to a model API. It requires careful work across data quality, retrieval, prompts, evaluation, architecture, latency, cost, safety, and user experience.

---

## Analytical Interests

Beyond implementation, I enjoy analysing complex decisions using data, assumptions, comparisons, and scenario-based reasoning.

Topics I frequently explore include:

* AI industry trends
* Emerging technologies
* Software products
* Model and hardware comparisons
* Personal computing
* Technology investments
* Real-estate market dynamics
* Infrastructure development
* Long-term economic and technology trends
* Product feasibility
* Risk and return trade-offs
* Consumer technology evaluation

I enjoy breaking large decisions into measurable factors, challenging assumptions, and comparing multiple scenarios before arriving at a conclusion.

---

<details>
<summary><strong>Beyond Engineering</strong></summary>

<br>

Outside my primary engineering work, I am interested in several areas that influence how I think about technology, people, and decision-making.

### Technology Exploration

I enjoy experimenting with:

* Personal computers and laptops
* Local AI models
* Consumer electronics
* Productivity tools
* Desktop applications
* Smart-home technology
* Software product design
* New developer platforms

### Travel and Cultural Exploration

I enjoy travelling and visiting culturally and historically significant places, particularly temples and heritage destinations across India.

Some of the places I have explored include:

* Tirupati
* Arunachalam
* Kanchipuram
* Kukke Subramanya
* Shirdi
* Nashik
* Simhachalam
* Annavaram
* Vijayawada
* Basar
* Vemulawada
* Muramalla
* Amalapuram

### Productivity and Continuous Learning

I am interested in:

* Focus and Pomodoro systems
* Habit formation
* Structured interview preparation
* Technical reading
* Learning through projects
* Tracking measurable progress
* Building tools that improve everyday productivity

### Entertainment

I enjoy intelligent detective stories, investigative dramas, technology-related content, and narratives centred on observation, reasoning, and problem-solving.

</details>

---

## Selected Project Ideas and Future Work

Some projects I plan to develop or expand include:

### Production-Ready RAG Evaluation Platform

A platform for comparing:

* Embedding models
* Chunking strategies
* Search configurations
* Reranking methods
* Prompt templates
* Language models
* Retrieval metrics
* Response quality

---

### Local AI Knowledge Assistant

A privacy-focused assistant that:

* Runs with local language models
* Searches personal documents
* Supports citations
* Works offline
* Compares multiple models
* Tracks response latency and quality

---

### Intelligent Desktop Companion

An extended version of my desktop companion project with:

* Natural-language conversations
* Voice interaction
* Productivity coaching
* Local AI inference
* Custom behaviour
* Task tracking
* Cross-device preference synchronisation

---

### AI System Design Repository

A collection of architecture notes and practical implementations covering:

* RAG architecture
* Vector search
* Agent systems
* Model serving
* Evaluation pipelines
* Caching
* Observability
* Scaling
* Cost optimisation
* Failure recovery

---

## Featured Repositories

Replace the following placeholders with your strongest repositories.

### GPU-Accelerated Industrial Anomaly Detection

Research implementation and documentation based on my IIT Bhilai M.Tech thesis on anomaly detection in industrial multivariate time-series data and CUDA acceleration of GEMM-based convolution.

**Technologies:** Python, PyTorch, CUDA, MSCRED, ConvLSTM, GEMM, time-series analysis, anomaly detection

[View Repository](YOUR_THESIS_REPOSITORY_URL) · [Read Thesis](YOUR_THESIS_PDF_URL)

> Publish only code and data that you are authorised to share. Use synthetic or anonymised data when the original industrial dataset is restricted.

---

### NVIDIA Nemotron Reasoning Fine-Tuning

A reproducible project documenting synthetic reasoning-data generation, compact chain-of-thought formatting, 4-bit QLoRA-style adapter training, architecture compatibility fixes, checkpoint selection, and Kaggle submission packaging.

**Result:** Score improved from approximately **0.53 to 0.668**; final private leaderboard rank approximately **3,020**.

**Technologies:** Python, PyTorch, Transformers, PEFT, QLoRA, bitsandbytes, TRL, Accelerate, Safetensors, vLLM

[View Repository](YOUR_NEMOTRON_REPOSITORY_URL) · [View Competition](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge)

---

### Enterprise RAG Evaluation

An evaluation-oriented RAG project demonstrating document ingestion, vector search, reranking, context selection, and retrieval metrics.

**Technologies:** Python, Qdrant, embeddings, LLMs, information retrieval

[View Repository](YOUR_RAG_REPOSITORY_URL)

---

### Desktop AI Companion

An interactive Electron desktop application combining animated companion behaviour with productivity and Pomodoro features.

**Technologies:** Electron, JavaScript, desktop UI, productivity systems

[View Repository](YOUR_DESKTOP_APP_REPOSITORY_URL)

---

### Local LLM Experiments

A collection of notebooks and experiments comparing local language models, quantisation settings, latency, and response quality.

**Technologies:** Python, Ollama, LM Studio, Hugging Face, Jupyter

[View Repository](YOUR_LOCAL_LLM_REPOSITORY_URL)

---

### Machine Learning and Data Science Projects

A collection of academic and independent machine-learning experiments, notebooks, evaluations, and implementation notes.

**Technologies:** Python, machine learning, data analysis, Jupyter

[View Repository](YOUR_ML_REPOSITORY_URL)

---

## GitHub Statistics

<!-- Replace YOUR_USERNAME with your exact GitHub username. -->

<div align="center">

<img
src="https://github-readme-stats.vercel.app/api?username=YOUR_USERNAME&show_icons=true&hide_border=true&include_all_commits=true&count_private=false"
alt="Ranjith's GitHub statistics"
height="165"
/>

<img
src="https://github-readme-stats.vercel.app/api/top-langs/?username=YOUR_USERNAME&layout=compact&hide_border=true"
alt="Most used programming languages"
height="165"
/>

</div>

> Language statistics are calculated from public repository content and do not represent my complete technical proficiency.

---

## Connect With Me

I am open to conversations about:

* Artificial intelligence

* Machine learning engineering

* Generative AI

* Retrieval-Augmented Generation

* AI agents

* Information retrieval

* System design

* Technical collaboration

* Research and experimentation

* AI engineering opportunities
* GPU computing and CUDA optimisation
* Industrial AI and time-series anomaly detection

* **LinkedIn:** [Connect with me](https://www.linkedin.com/in/ranjithvutnoor/)

* **Email:** [YOUR_EMAIL_ADDRESS](mailto:ranjithvutnoor@gmail.com)

* **LeetCode:** [View my profile](https://leetcode.com/u/ranjithvutnoor/)

* **Kaggle:** [View my profile](https://www.kaggle.com/ranjithvutnoor)

* **Portfolio:** [Visit my portfolio](https://ranjithvutnoor.github.io/portfolio/)

---

<div align="center">

### Building intelligent systems through research, engineering, experimentation, and measurable improvement.

</div>
