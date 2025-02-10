# Lesson 03: Deep Dive into LLMs

In this lesson, we will dive deeper into the capabilities of LLMs and how to use them to build more complex applications. We are going to study more of what is happening behind the scenes of the LLMs and how to avoid their common pitfalls.

In this lesson we are going to cover an overview about Machine Learning and how the GPT models can perform tasks like reasoning, planning, and more using the LLM's capabilities.

We are going to cover important aspects of the GPTs architectures, learning how to handle context limitations and how to use the prompt engineering to improve the results.

By the end of the lesson we will cover how to run local models in your own hardware using some helpful tools like Ollama and GPT4All.

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

  ## Introduction to Machine Learning

- How can a computer "learn"?
- [Machine learning](https://en.wikipedia.org/wiki/Machine_learning) is a broad terminology for a set of algorithms that can learn from and/or make predictions on data
- There are many forms of Machine Learning:
  - **[Supervised learning](https://en.wikipedia.org/wiki/Supervised_learning)**: The most common form of machine learning, which consists of learning a function that maps an input to an output based on example input-output pairs
    - Requires a **training dataset** with input-output pairs
    - The algorithm learns from the dataset and can make/extrapolate predictions on new data
  - **[Unsupervised learning](https://en.wikipedia.org/wiki/Unsupervised_learning)**: A type of machine learning that looks for previously undetected patterns in a dataset with no pre-existing labels
    - Requires a **training dataset** with input data only
    - The algorithm learns from the dataset and can make predictions on new data
  - **[Reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning)**: A type of machine learning that enables an agent to learn in an interactive environment by trial and error using feedback from its own actions and experiences
    - Requires a **training dataset** with input data and feedback
    - The algorithm learns from the dataset and can make predictions on new data
  - Other models and techniques that can be applied/extended:
    - Semi-supervised learning
    - Self-supervised learning
    - Multi-task learning
    - Transfer learning
    - Meta learning
    - Online learning
    - Active learning
    - Ensemble learning
    - Bayesian learning
    - Inductive learning
    - Instance-based learning
    - And many others

These models have been evolving and improving over the years, aiming to output some form of "intelligence" from the data, mimicking human-like behavior.

- For example, some advanced Machine Learning algorithms use [Neural Networks](https://en.wikipedia.org/wiki/Neural_network) to compute complex functions and make predictions on data in ways that a "normal" program would take billions or more lines of code to accomplish.

This "brain-like" computational approach has been used to extend the capabilities of AI exponentially, far beyond what traditional computing could achieve.

- An example of this is [Deep Learning](https://en.wikipedia.org/wiki/Deep_learning), a subset of machine learning that uses neural networks with many [layers](https://en.wikipedia.org/wiki/Artificial_neural_network#Deep_neural_networks) to learn from data and make much more complex predictions.
- Neural Networks like [CNNs](https://en.wikipedia.org/wiki/Convolutional_neural_network) and [RNNs](https://en.wikipedia.org/wiki/Recurrent_neural_network) have been used to power many AI applications for tasks such as image and text recognition and generation, computer vision, and many others.
- Currently (as of mid 2024), the most advanced form of Deep Learning is the [Transformers](<https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)>) architecture, which has been used to power many AI applications, including the GPT models.
  - Unlike traditional neural networks, transformers can process data in parallel, making them much faster and more efficient.
  - This technical advancement, aligned with favorable market/investment conditions in recent years, has made the current Generative AI boom possible.

To better experiment with and understand how transformers work, we will use samples from the [Hugging Face tutorials](https://huggingface.co/docs/transformers/index), which make it simple and straightforward to start using these tools and techniques without needing to understand the deep technical details of the models beforehand.

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

### How GPTs Work

- Model training
  - GPTs are pre-trained models that are trained on large and diverse datasets of texts to compute **patterns** and **probabilities** between the contents present in the training data
    - For example, a model trained on terabytes of chemical and biological texts may be able to generate correlations between chemical component names and illness symptoms with much more accuracy and reliability than a model trained on terabytes of literature and poetry texts
    - Generally, the more data and the more diverse the data, the better the model will be able to generate outputs based on the **patterns** and **probabilities** observed in the training data
    - Current state-of-the-art models are trained on datasets with hundreds of terabytes of texts and other content, and the training process can take weeks or even months to complete even on powerful hardware
  - The process of training these models involves sophisticated **Machine Learning** techniques, as we studied in the previous lesson
- Tokens encoding and decoding
  - All calculations in these models are performed on **tokens** that are _encoded_ and _decoded_ from character strings, rather than on the raw character strings themselves
  - Each word (or fragment of letters) is broken down into tiny pieces ([n-grams](https://en.wikipedia.org/wiki/N-gram)) that are used to compute the **patterns** and **probabilities** between the contents present in the training data
  - The _encoding_ process converts character strings into these **tokens**, and the _decoding_ process converts these **tokens** back into character strings
  - This tokenization process allows the model to handle a wide range of languages and character sets efficiently
- Transformers
  - The process of relating tokens to each other in the training data is done by [Transformer](<https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)>) neural networks, which we saw in the previous lesson
  - These transformers use self-attention mechanisms to capture complex relationships between different parts of the input data
  - Transformers are used both in the training phase and in the inference phase of these models to calculate the relations between tokens and generate outputs
- Inference
  - The process of generating outputs from inputs in a transformer is called inference, and can be accomplished much more quickly for **prompting** than the training phase, since the **patterns** and **probabilities** between the contents present in the training data are already computed and stored in the model
  - For every **prompt** provided, the model calculates the most probable **output** based on the model data using various decoding strategies
  - Since the model doesn't "understand" the concepts of _correct_ or _incorrect_, _meaningful_ or _nonsensical_, _helpful_ or _toxic_, it is up to the person or system passing the **prompt** to the model to _judge_ the **output** generated by the model according to their own criteria and standards
- Prompting
  - After the model has been trained, it won't execute anything on its own unless it's _prompted_ to do so
  - The **prompt** being passed to the model may vary greatly in format, depending on the task being executed and the model configurations
    - For example, a **prompt** for a chatbot may be a question like "_What is your name?_", while a **prompt** for a code generator may be a code snippet like "_for i in range(10): print(i)_"
    - Most models have certain _structures_ for prompts, allowing for multiple _actors_ to interact inside the model, usually with a "system" actor for answering things that a "user" actor requests from it
  - The **prompt** is the input that the model will process to generate the output, and it can be as simple as a single word or as complex as a full book
  - The **prompt** is the most important part of the process of generating outputs from inputs, and it can be _engineered_ in many different ways to _guide_ the model to generate more _meaningful_ outputs
- Context
  - GPTs have a limited _context_ maximum length for processing **prompts**, and they can store the chat history in **context** while processing tasks, but this is only temporary storage for the current chat/inference session
  - When passing too much data in a **prompt**, the model may not be able to process it all correctly, and may even disregard some parts of the **prompt** or of the **training data** based on the model configurations and parameters
  - The context size usually ranges from a few hundred to a few thousand tokens, with some exceptions up to one million tokens context size
    - This means that the more information you include in the **prompt**, the less _token space_ the model will have to evaluate the trained data against what is being asked
  - Managing context effectively is crucial for obtaining accurate and relevant responses from the model
- Fine tuning
  - If GPTs trained on random internet text were not adjusted, they would merely regurgitate random internet texts, giving responses that are completely irrelevant, inaccurate, or even nonsensical
  - To save space in the **context** while improving the accuracy of the outputs, the **fine tuning** process can be used to _guide_ the model to generate more _meaningful_ outputs
  - In the **fine tuning** process, the model is trained to prioritize reliable sources and accurate information, reducing the likelihood of generating answers based on misinformation or low-quality internet texts
  - These processes often include incorporating human feedback to correct errors and biases in the model's responses. This iterative process helps in aligning the model's outputs with human values and expectations
  - Another common practice is to apply targeted training, which involves fine-tuning the model on a specific dataset that is relevant to the desired application
    - This targeted approach helps the model learn context and nuances pertinent to particular topics or industries
- RAG
  - The **RAG** (Retrieval-Augmented Generation) process is a technique that combines a language model's generative abilities with an external information retrieval system
  - Instead of passing all the context for a question in the prompt, RAG first identifies relevant information from a large database, then incorporates this data into the language model's query
  - This method significantly improves the model's precision and accuracy, particularly for queries requiring specific information
  - This is more efficient than passing all the information in the prompt, and is also much simpler than fine-tuning the model with new data
  - RAG allows models to access up-to-date information without requiring constant retraining, making it particularly useful for applications that need to provide current and accurate information

### Limitations of GPTs

To better understand the capabilities of GPTs, let's highlight some of the most impactful implications of their limitations:

- Lack of "understanding"
  - Questions like "_Who are you?_" and "_What is your name?_" are not "understood" by GPTs; they simply generate outputs based on patterns and probabilities observed in the training data
  - If the training data contains texts like "_I am a chatbot_" and "_My name is ChatGPT_" frequently associated with these questions, the GPT will generate outputs like "_I am a chatbot_" and "_My name is ChatGPT_" when asked
  - If the training data is some amount of text where these questions are frequently associated with other words and phrases, the GPT will generate outputs based on these associations
    - For example, a GPT trained on the Don Quixote book might generate outputs like "_I am Don Quixote_" and "_My name is Alonso Quijano_" when asked these questions
  - There are techniques like **Fine Tuning**, **HFRL** (Human Feedback Reinforcement Learning), **Prompt Engineering**, **RAG** (Retrieval-Augmented Generation), and **Control Codes** that can be used to _guide_ GPTs to generate more _meaningful_ outputs, but these are not _learning_ processes; they are _tweaking_ processes
  - Ultimately, there's no current technique that can make a GPT actually "understand" and "reason" about the **inputs** and **outputs** they're dealing with, as a human mind would do
- Sensitivity to input
  - GPTs are very sensitive to the **inputs** they process, and they can generate very different **outputs** based on small changes in input _phrasing_ (or **prompting**)
  - Asking the same question in different word orders, or even using different languages, can lead to very different outputs, sometimes even conflicting or entirely unrelated
    - For example, asking "_What is the capital of France?_" and "_France's capital is_" can lead to very different outputs, depending on the model's training
    - This happens because even if the **semantic** meaning of these questions is almost the same in these two phrasings, the **syntactic** meaning is different, and GPTs are very sensitive to these differences
    - In practice, every _character_ difference in a **prompt** can move the coordinates in the _text space_ function that computes the output value over that input, regardless of whether that character changes the _semantic_ meaning of the **prompt**
      - As a loose metaphor, we could imagine a function `f(x)` that gives possibly very different outputs if we give `x = 1`, `x = 1.00000001`, or `x = 1.000000000000...001`, even if the **semantic** meaning of these inputs is almost the same
- Inconsistency
  - GPTs may be inconsistent in generating outputs for the same inputs, even if the **inputs** are identical
  - Since most operations in the **inference** process are based on **probabilities** and **patterns** observed in the training data, the **outputs** can be very different for the same **inputs** based on the **randomness** of the **sampling** process
  - The results can be more or less deterministic based on the model **configurations** and **parameters** used in the **inference** process
    - There are parameters like **Temperature**, **Top-K**, **Top-P**, and **Nucleus Sampling** that can be used to **control** the **randomness** of the **sampling** process
  - A model operating with **greedy search** will generate the same outputs for the same input, while a model operating with **random sampling** (or **temperature** sampling) might often generate different outputs for the same input
- Incorrect or Nonsensical Outputs (and hallucinations)
  - As we have seen in previous lessons, GPTs (as we have nowadays) lack on **neural symbolic** reasoning capabilities, thus they are simply outputting whatever is most probable according to the **patterns** and **probabilities** observed in the training data
  - GPTs can generate incorrect or nonsensical outputs based on the **inputs** they process, especially when the **inputs** are very _ambiguous_ or _open-ended_
  - This is not necessarily a defect of GPTs, but a consequence of the **patterns** and **probabilities** observed in the training data
    - For example, a GPT trained on human anatomy may answer the question "How many eyes does a spider have?" with "Two" because it has learned from the training data that most subjects have two eyes, and it might not be clear in the training data that spiders are not humans
      - Even though this information is incorrect (according to the common definition of spiders), it might be _statistically_ correct to associate "number of eyes (of a living thing)" with "two" based on the training data of that model
  - This is very common with ambiguous topics, such as person names, locations, historical events, dates, quantities, and other subjects where there may be many different _correct_ answers based on the **context** of the **inputs**
    - The question "Who is James?" may vary greatly depending on the **context** of the question and the training data of the model
      - For example, if the training data contains many texts about James Bond, the GPT may answer "James Bond" to this question, even if the question is about another James
  - The term [Hallucination](https://en.wikipedia.org/wiki/Hallucination_(artificial_intelligence)) is often used in AI to describe these incorrect or nonsensical outputs because they are generated based on **patterns** and **probabilities** observed in the training data, not based on "understanding" or "reasoning" about the **inputs** and **outputs** they're processing
    - These hallucinations can be dangerous, as GPTs may provide very _convincing_ and _realistic_ outputs based on **patterns** and **probabilities** observed in the training data, even if these outputs are completely incorrect or nonsensical
- Bias and toxicity
  - GPTs can generate outputs that are biased and toxic based on the **inputs** they process, especially when the information in the **training data** is biased and toxic
  - Techniques like **Fine Tuning** and **HFRL** (Human Feedback Reinforcement Learning) can help mitigate this effect by tweaking the responses to be as _helpful_, _harmless_, and _honest_ as possible
    - However, even human feedback and direct oversight of the training data cannot guarantee that GPTs will generate unbiased and non-toxic outputs, as the very definitions of "helpfulness", "harmlessness", and "honesty" can be highly subjective, depending on various social, cultural, contextual, and personal factors
  - Current AI models (as of mid 2024) have a strong inclination to generate outputs based on _mild-leftist_ biases, as we have in the conclusions of studies like the [Political Preferences of LLMs](https://arxiv.org/pdf/2402.01789) paper by David Rozado and the [Tracking AI](https://trackingai.org/) website currently maintained by Maxim Lott
- No (native) access to real-time information
  - Since GPTs don't "learn" or "improve" when processing inferences, they can't access any information that occurs after their training, unless explicitly provided with this information in the **inputs** or through techniques like **RAG** (Retrieval-Augmented Generation) or similar methods
  - This means that GPTs can't provide real-time information about anything, such as news, events, weather, stock prices, sports results, etc.
    - It is possible to provide GPTs with real-time information, but this is separate from the training phase and is not a true learning process; it's simply passing data to the model with the expectation that it will process and return it correctly

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

## Exercise

- Pick your group from previous lesson
- Create an account on [Hugging Face](https://huggingface.co/)
- Navigate to the [Model Hub](https://huggingface.co/models)
- Find a text generation model to download and experiment with
- Try to run the model locally using Hugging Face Transformers or GPT4All (or any similar tool), at your preference
