# Lesson 03: Deep Dive into LLMs and Introduction to Decentralized Computing

In this lesson, we will dive deeper into the capabilities of LLMs and how to use them to build more complex applications. We are going to study more of what is happening behind the scenes of the LLMs and how to avoid their common pitfalls.

We are going to cover important aspects of the GPTs components, and how the _Transformers_ play a crucial role in the GPTs architecture.

In this lesson we will see the pros and cons of using online API providers for AI Inference, and w will cover how to run local models in your own hardware using some helpful tools like Ollama and GPT4All.

By the end we are going to see how decentralized computing may solve some real problems of the AI industry, and how we can use it to build more reliable and scalable applications.

## Prerequisites

- Proficiency in using a shell/terminal/console/bash on your device
  - Familiarity with basic commands like `cd`, `ls`, and `mkdir`
  - Ability to execute packages, scripts, and commands on your device
- Installation of Python tools on your device
  - [Python](https://www.python.org/downloads/)
  - [Pip](https://pip.pypa.io/en/stable/installation/)
- Proficiency in using `python` and `pip` commands
  - Documentation: [Python](https://docs.python.org/3/)
  - Documentation: [Pip](https://pip.pypa.io/en/stable/)
- Proficiency in using `venv` to create and manage virtual environments
  - Documentation: [Python venv](https://docs.python.org/3/library/venv.html)
- Node.js installed on your device
  - [Node.js](https://nodejs.org/en/download/)
- Proficiency with `npm` and `npx` commands
  - Documentation: [npm](https://docs.npmjs.com/)
  - Documentation: [npx](https://www.npmjs.com/package/npx)
- Understanding of `npm install` and managing the `node_modules` folder
  - Documentation: [npm install](https://docs.npmjs.com/cli/v10/commands/npm-install)
- Git CLI installed on your device
  - [Git](https://git-scm.com/downloads)
- Proficiency with `git` commands for cloning repositories
  - Documentation: [Git](https://git-scm.com/doc)
- Basic knowledge of JavaScript programming language syntax
  - [JavaScript official tutorial](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide)
  - [Learn X in Y minutes](https://learnxinyminutes.com/docs/javascript/)
- Basic knowledge of TypeScript programming language syntax
  - [TypeScript official tutorial](https://www.typescriptlang.org/docs/)
  - [Learn X in Y minutes](https://learnxinyminutes.com/docs/typescript/)
- An account at [OpenAI Platform](https://platform.openai.com/)
  - To run API Commands on the platform, set up [billing](https://platform.openai.com/account/billing/overview) and add at least **5 USD** credits to your account
- Create an account at [Hugging Face](https://huggingface.co/)
- Create an account at [Google Colab](https://colab.research.google.com)

## Introduction to Transformers

Transformer was first introduced in the [Attention is All You Need](https://dl.acm.org/doi/10.5555/3295222.3295349) paper in 2017 by Vaswani et al.

> > _We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely._
> > _Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train_.

- Transformer models operate on the principle of **next-word prediction**
  - Given a text prompt from the user, the model can _infer_ which is the most probable next word that will follow this input
- Transformers use **self-attention mechanisms** to process entire sequences and capture long-range dependencies
- Transformers need to be **pre-trained** on a large dataset to properly provide accurate predictions
  - This is why we use Generative **Pre-trained** Transformers models for handling AI tasks
- Architecture of a Transformer model:
  - **Embedding**:
    - Text input is divided into **tokens**
    - Tokens can be words or sub-words
    - Tokens are converted into **embeddings**
    - Embeddings are numerical vectors
    - They capture **semantic meaning** of words
  - **Transformer Block**:
    - Processes and transforms input data
    - Each block includes:
      - **Attention Mechanism**:
        - Allows tokens to **communicate**
        - Captures **contextual information**
        - Identifies **relationships** between words
      - **MLP (Multilayer Perceptron) Layer**:
        - A **feed-forward network**
          - Processes information in one direction, from input to output, without loops or feedback connections
        - Operates on each token independently
        - **Routes information** between tokens
        - **Refines** each token's representation
  - **Output Probabilities**:
    - Uses **linear** and **softmax** layers
    - Transforms processed embeddings
    - Generates probabilities
    - Enables **predictions** for next tokens
- Visualization of a Transformer model
  - The [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) application is great for understanding the inner components of the Transformer architecture
- Transformers are much more capable of understanding semantic relationships than traditional neural networks
  - Example: [Google's BERT for search](https://blog.google/products/search/search-language-understanding-bert/)
  - Example: [DeepMind's AlphaFold 2 for protein structure prediction](https://deepmind.com/blog/article/alphafold-a-solution-to-a-50-year-old-grand-challenge-in-biology)
  - Example: [Meta's NLLB for machine translation](https://ai.facebook.com/blog/nllb-200-high-quality-machine-translation/)

## Experimenting with Transformers

Instead of diving into the deep technical details of transformers, we will use frameworks, tools, and libraries that abstract away the complexities of the computational, mathematical, and statistical work.

In fact, we're going to use pre-made models and shortcuts that make it as simple as calling a function to execute tasks over data passed as parameters.

> Note: It is important to explore these concepts in depth later, so you understand exactly what is happening under the hood. For now, to build functional AI applications as quickly as possible, we will focus on the practical aspects of using these abstractions and simplifications.

- Machine Learning frameworks and tools:
  - [TensorFlow](https://www.tensorflow.org/)
  - [PyTorch](https://pytorch.org/)
  - [JAX](https://jax.readthedocs.io/en/latest/index.html)
- Using a library to abstract away complexities:
  - [Transformers](https://github.com/huggingface/transformers)
- Getting started with a simple Python script
- Using `Pipelines`:
  - Downloading models
  - Using sample data
- Using `Tokenizer` and `Model` shortcuts
- Working with sample `datasets`
- Following a tutorial for an NLP pipeline

## Getting Started with Transformers

Hugging Face's Transformers library can abstract most of the complexities of using Machine Learning and other AI techniques, making it simple to apply these models to real-world problems.

The only concepts you need to fully understand when interacting with this library are: the _configuration_ itself, the _model_ you are using, and the required _processor_ for the task you are trying to accomplish.

- Using [Transformers](https://github.com/huggingface/transformers) Library from Hugging Face
- Using the [Pipelines](https://huggingface.co/docs/transformers/main_classes/pipelines) API for running pre-trained tasks
  - Running these tasks requires almost no previous knowledge in AI or Machine Learning or even programming, thanks to Hugging Face's [Philosophy](https://huggingface.co/docs/transformers/main/en/philosophy)

- Practical exercises:
  - Exercise 1: Getting started with [Google Colab](https://colab.research.google.com)
  - Exercise 2: Running a **Sentiment Analysis** model using Hugging Face's Transformers library with an [example notebook](https://colab.research.google.com/drive/1G4nvWf6NtytiEyiIkYxs03nno5ZupIJn?usp=sharing)
    - Create a Hugging Face [Access Token](https://huggingface.co/settings/tokens) for using with Google Colab
    - Add the token to the notebook's environment variables
      - Open the "Secrets" section in the sidebar of the notebook
      - Click on `+ Add new secret`
      - Enter the name `HF_TOKEN` and paste your secret token in the value field
    - Click on `Grant access` to grant the notebook access to your Hugging Face token to download models
      - The token is required when downloading models that require authentication
  - Exercise 3: Getting started with Hugging Face's [Transformers](https://huggingface.co/transformers/) library with an [example notebook](https://colab.research.google.com/github/huggingface/education-toolkit/blob/main/03_getting-started-with-transformers.ipynb#scrollTo=mXAlr2u76bkg)
  - Exercise 4: Understanding the role of transformers in the GPT Architecture with the [Transformer Explainer](https://poloclub.github.io/transformer-explainer/)

## Introduction to GPTs

- GPTs (Generative Pre-trained Transformers) are a type of LLMs (Large Language Models) trained to generate **outputs** (inference) from **inputs** (prompts) based on their **training data** (pre-training)
- Although GPTs can follow complex instructions, like chatting, answering questions, generating stories, and completing code, they do not possess true intelligence or "understanding" of the **inputs** and **outputs** they process
  - GPTs don't actually "understand" the **inputs** and **outputs** they process; they learn from the training data (using **Machine Learning**) how to generate outputs from inputs based on **patterns** and **probabilities** observed in the training data
  - GPTs don't retain information they process after the training phase; they "forget" the **inputs** and **outputs** after processing
    - GPTs can store the chat history in **context** while processing tasks, but this is only temporary storage for the current chat/inference session
    - When the chat/inference session is deleted, the **context** is lost, and the GPT "forgets" everything it processed
  - GPTs don't "learn" or "improve" each time they process an inference, regardless of how many times they process similar **inputs** and **outputs**
    - However, GPTs can be **fine-tuned** to process specific **inputs** and **outputs** after the initial training phase to perform better on specific tasks and/or datasets

### GPT Applications

Some examples of applications using GPTs:

- [ChatGPT](https://chat.openai.com/)
- [Google Gemini](https://gemini.google.com/) (formerly known as Bard)
- [Bing AI](https://www.bing.com/chat) (also known as Copilot)
- [Claude AI Chat](https://claude.ai/chats)
- [Pi](https://pi.ai/talk)
- [Grok](https://grok.x.ai/)

All of these applications use various techniques and models to process different types of tasks (also known as modalities), but they all share similar limitations and capabilities of GPTs, to varying degrees and in different ways.

There are ongoing discussions about evolving these applications to reach [AGI](https://en.wikipedia.org/wiki/Artificial_general_intelligence) (Artificial General Intelligence) capabilities using GPTs and similar models. As of mid-2024, this remains a complex and long-term goal that is not yet feasible with the current state of the art in AI and Machine Learning (and may never be).

## Running GPTs for Inference

- The resources required to run _inference_ tasks with GPTs are much smaller than the ones required to run _training_ tasks
  - Still bigger models may require several GPUs and/or a lot of time to run simple textual inference tasks
- Running inference through APIs can be a good option for many applications, but it may not be the best option for some use cases, like when:
  - Using sensitive data
  - Needs reliable uptime
  - Needs ultra low latency
  - Running tasks without internet connection
  - Using custom models
- API providers may offer a convenient way to run inference tasks, but they may offer limitations, especially in the growth phase of AI applications
  - Pricing changes
  - Feature changes
  - Rate limits
  - Vendor lock-in
  - Downtime
- Running inference tasks locally, although more complex, may remove most of these limitations and offer a more reliable and customizable experience for your applications

## Tooling for Local LLM Serving

Using Hugging Face's Transformers, we can run many models using tools like `pyTorch` and `TensorFlow`, while configuring the pipelines, models, inputs, and outputs by invoking them inside a Python script. However:

- Configuring these tools and models to work properly within scripts is not always trivial or straightforward
- This process can be overwhelming for beginners
- Numerous other concerns require coding and implementation before we can effectively use the models:
  - Handling server connections
  - Managing model parameters
  - Dealing with caches and storage
  - Fine-tuning
  - Prompt parsing

Several tools can abstract away these concerns and simplify the process for users and developers to use GPT models on their own devices:

- Some of these have binary releases that can be installed and run like any other common software
- Here are a few examples:

1. [GPT4All](https://github.com/nomic-ai/gpt4all): An ecosystem of open-source chatbots and language models that can run locally on consumer-grade hardware.

2. [Ollama](https://github.com/ollama/ollama): A lightweight framework for running, managing, and deploying large language models locally.

3. [Vllm](https://github.com/vllm-project/vllm): A high-throughput and memory-efficient inference engine for LLMs, optimized for both single-GPU and distributed inference.

4. [H2OGPT](https://github.com/h2oai/h2ogpt): An open-source solution for running and fine-tuning large language models locally or on-premise.

5. [LMStudio](https://lmstudio.ai/): A desktop application for running and fine-tuning language models locally, with a user-friendly interface.

6. [LocalAI](https://github.com/go-skynet/LocalAI): A self-hosted, community-driven solution for running LLMs locally with an API compatible with OpenAI.

7. [Text Generation WebUI](https://github.com/oobabooga/text-generation-webui): A gradio web UI for running large language models locally, supporting various models and extensions.

8. [LlamaGPT](https://github.com/getumbrel/llama-gpt): A self-hosted, offline, ChatGPT-like chatbot, powered by Llama 2, that can run on a Raspberry Pi.

These tools offer various features such as model management, optimized inference, fine-tuning capabilities, and user-friendly interfaces, making it easier to work with LLMs locally.

## Loading Models and Running Inference Tasks

Tools like [GPT4All](https://github.com/nomic-ai/gpt4all) simplify the process of loading models and running inference tasks, even for non-developers. These tools abstract away many configurations, leaving room for basic settings such as CPU thread usage, device selection, and simple sampling options.

### Running a local model with GPT4All

- Download GPT4All from their [official website](https://www.nomic.ai/gpt4all)
- Install the correct version for your operating system
- Run the application
- Download models from the web using the `Models` tab
- Chat with the local model using the `Chat` tab

## Downloading Models

The default models packed with your model loaders are a good starting point for experimentation. But if you want to experiment it further, you can download open-source models from the web, from places like the [Hugging Face Model Hub](https://huggingface.co/models) and others.

- Notable accounts for model exploration and download:
  - [Mistral](https://huggingface.co/mistralai)
  - [Stability AI](https://huggingface.co/stabilityai)
  - [OpenAI](https://huggingface.co/openai)
  - [Google](https://huggingface.co/google)
  - [Intel](https://huggingface.co/Intel)
  - [Microsoft](https://huggingface.co/microsoft)
  - [Meta](https://huggingface.co/meta-llama) and [Facebook](https://huggingface.co/facebook)
  - [xAI](https://huggingface.co/xai-org)
  - [ByteDance](https://huggingface.co/ByteDance)
  - [Salesforce](https://huggingface.co/salesforce)
  - [Anthropic](https://huggingface.co/anthropic)
  - [Databricks](https://huggingface.co/databricks)
  - [NVIDIA](https://huggingface.co/nvidia)
  - [Cohere](https://huggingface.co/cohere)
  - [Hugging Face](https://huggingface.co/huggingface)
  - [EleutherAI](https://huggingface.co/EleutherAI)
  - [BigCode](https://huggingface.co/bigcode)
  - [BigScience](https://huggingface.co/bigscience)
- Model sizes indicating data usage and processing:
  - 1B: Uses 1 billion parameters, suitable for low-resource devices, can't handle complex tasks
  - 7B: Uses 7 billion parameters, suitable for mid-resource devices, can handle low complexity tasks
  - 13B: Uses 13 billion parameters, suitable for mid-resource devices, can handle medium complexity tasks
  - 30B: Uses 30 billion parameters, suitable for high-resource devices, can handle high complexity tasks
  - 70B: Uses 70 billion parameters, suitable for datacenter-grade devices, can handle very high complexity tasks
    - Examples in model names:
      - `Dr_Samantha-7B-GGUF`
      - `CodeLlama-70B-Python-GPTQ`
- Model types and compatibility considerations:
  - Some models are modified with quantization, pruning, or other techniques for hardware compatibility
  - Models marked with `GGUF`, `GPTQ`, `GGML`, `AWQ`, and similar tags may require specific configurations or tweaking for proper functionality
- Model fine-tuning:
  - Some models are fine-tuned for specific tasks or domains
  - Examples:
    - `Llama-3.1-8B-Instruct-GGUF`
- Multi-modal models:
  - Some models are designed to handle both text and image inputs
  - Examples:
    - `Llama-3.2-90B` which is derived from `Llama-3.1-70B` with additional `20B` parameters for handling images

## Limitations of Local LLM Serving

- Most consumer devices have limited resources, so running large models may not be feasible
- Small Language Models may be a suitable option in some cases, but still they do require some elevated resources to run, and their performance is still very far from the 10B+ LLMs
- Being able to run models remotely without relying on unsafe third-party is still a big challenge for the whole AI industry (as of 2025)
  - Decentralized Computing may pose as the perfect solution for this challenge

## Introduction to Web3

Web3 represents the next evolution of the internet, characterized by decentralization, blockchain technology, and token-based economics. It provides a framework for creating trustless, permissionless applications that can interact with AI Agents.

### Key Concepts of Web3

- Decentralization and distributed systems
- Blockchain technology and its role in Web3
- Cryptocurrency and tokenomics
- Decentralized applications (dApps)
- Interoperability and cross-chain communication

### Web3 Infrastructure

- Ethereum and other smart contract platforms
- Layer 2 scaling solutions
- Decentralized storage (e.g., IPFS, Filecoin)
- Decentralized identity systems

### Web3 Development Tools

- Web3.js and Ethers.js libraries
- Truffle and Hardhat development frameworks
- MetaMask and other wallet integrations
- IPFS and Pinata for decentralized file storage

## Smart Contracts

Smart contracts are self-executing contracts with the terms of the agreement directly written into code. They run on blockchain networks and can interact with AI Agents to create complex, automated systems.

### Understanding Smart Contracts

- Definition and characteristics of smart contracts
- How smart contracts work on blockchain networks
- Benefits and limitations of smart contracts
- Popular smart contract languages (e.g., Solidity, Vyper)

### Deploying your First Smart Contract

- Create a Fungible Token with [OpenZeppelin Wizard](https://wizard.openzeppelin.com/)
  - Select the `Premint` checkbox and set the value higher than 0
- Open the contract in [Remix IDE](https://remix.ethereum.org/)
- Deploy the contract to the `sepolia` network using your `injected provider` as environment
- Call the `transfer()` function to transfer tokens to another address

### Key Aspects of Decentralized Applications

- Where the code is hosted?
- Who has control over the application?
- Where the data is stored?
- What if someone wants to manipulate or censor this application?

## Introduction to Decentralized Computing

Decentralized Computing is a new paradigm that allows users to run applications on a decentralized network of computers, instead of a centralized one.

In the next lessons we will cover more examples of Decentralized Computing for solving real problems in the AI industry.

## Exercise

- Pick your group from previous lesson
- Describe the AI and Web3 components of your project in the `README.md` file
- Try to come up with at least one really useful feature for each of these technologies for solving real problems within the project that you have envisioned
