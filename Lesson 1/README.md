# Lesson 01: Introduction to AI Development

In this lesson, we will introduce the basic concepts of AI, its capabilities and limitations, while debunking some myths and misconceptions about it. We will also discuss practical uses of AI with an example of using AI APIs for text generation, and also how to run it on your own hardware.

## What is AI?

- AI before November 30, 2022
  - First formal mention of AI was in [1956 by John McCarthy](https://home.dartmouth.edu/about/artificial-intelligence-ai-coined-dartmouth)
  - First decades of AI focused on rule-based systems
    - AI was seen as a way to **automate** tasks that required human intelligence
  - By early 2000s, AI applications became more common, especially in digital environments
    - AIs for marketing, customer service, and other digital tasks
    - AIs for search engines, recommendation systems, and other digital services
    - AIs for games, simulations, and other digital entertainment
- November 30, 2022 -> ChatGPT launch date
- AI now
  - With ChatGPT (and all similar generative AI tools that followed), AI has entered an age of **mass adoption**
  - AI is now used to **create** content, not just to **automate** tasks
  - AI is now used to (attempt to) _"understand"_ content, not just to **process** data
  - AI is now used to **generate** brand new content, not just to **recommend** or slightly **modify** existing content

## Learning AI

- Common prerequisites for working with AI
  - Basic programming skills
    - Python
    - Python development tools
    - Libraries and dependencies
    - Defining and calling functions
    - Classes, variables, and objects
    - Dictionaries, lists, and sets
    - Loops and conditionals
  - Basic understanding of statistics
    - Mean, median, mode, and outliers
    - Standard deviation
    - Probability
    - Distributions
  - Basic understanding of algebra
    - Variables, coefficients, and functions
    - Linear equations
    - Logarithms and logarithmic equations
    - Exponential equations
    - Matrix operations
    - Sigmoid functions
  - Basic understanding of calculus
    - Derivatives
    - Partial derivatives and gradients
    - Integrals
    - Limits
    - Sequences and series

## Learning Practical AI Applications

- Prerequisites
  - A computer or similar device
  - Internet connection
  - Dedication
  - Time
- Everything else will be covered as we progress through this bootcamp

## Introduction to Generative AI

- Programming a system or application to solve a specific task can take a lot of time and effort, depending on the complexity of the task and the number of edge cases that need to be considered
  - Imagine programming a system to translate text from one language to another by looking up words in a dictionary and applying grammar rules, comparing contexts, and considering idiomatic expressions for every single word in every single variation of their usage
    - Such applications are simply not feasible to be programmed by hand, not even by a huge team of programmers
- For these situations, **AI Models** can be **trained** for statistically solving the task without necessarily handling the actual _"reasoning"_ or _"understanding"_ of the task
  - The **training** process is done by **feeding** the model with **examples** of the task and the **correct** answers
  - The **model** then _"learns"_ the **patterns** and **rules** that **statistically** solve the task
  - The **model** can then **predict** the **correct** answer for new **examples** that it has never seen before

### Examples of AI Tasks

- Natural Language Processing (NLP)
  - Question answering
  - Feature extraction
  - Text classification (e.g., Sentiment Analysis)
  - Text generation (e.g., Text Summarization and Text Completions)
  - Fill-Mask
  - Translation
  - Zero-shot classification
- Computer Vision (CV)
- Image Generation
- Audio processing
- Multi-modal tasks

## Generative AI

- Generative AI is a type of AI that can generate new content based on a given input
  - The generated content can be in form of text, images, audio, or any other type of data
  - The input (in most cases) is a text prompt, which is a short text that the user writes to ask the AI to do something
    - Ideally the AI should be able to handle prompts in natural language (i.e. in a way that is similar to how humans communicate), without requiring domain-specific knowledge from the user
    - Together with the prompt, the user can also provide images or other types of data to guide the AI in generating the content
- Example of Generative AI application: [ChatGPT](https://chat.openai.com/)

> Did someone program the application to understand and generate text for each single word in each single language?
>
> > No, the application was **trained** to _"understand"_ and generate text

- How to **train** an application?
  - Applications are pieces of **software** that run on a **computer**
  - How do we **train** a **computer**?
- Machine learning

> "A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E." [Tom M Mitchell et al](https://www.cs.cmu.edu/~tom/mlbook.html)

- Does it mean that the computer is _"learning"_?
- Does it mean that the computer is _"thinking"_?
- Does it mean that the computer is _"conscious"_?

## Capabilities and Limitations of Generative AI

- [Stochastic parrot](https://en.wikipedia.org/wiki/Stochastic_parrots)
  - A critique view of current LLMs (large language models)
    - LLMs are limited by the data they are trained by and are simply stochastically repeating contents of datasets
    - Because they are just "making up" outputs based on training data, LLMs do not _understand_ if they are saying something incorrect or inappropriate
- [The Chinese Room](https://en.wikipedia.org/wiki/Chinese_room) philosophical experiment presented by John Searle in 1980
  - The notion of machine _understanding_
  - The implementation of syntax alone would constitute semantics?
  - A _simulation_ of mentality can be considered as a **replication** of mentality?
- Can AI truly "understand" language?
  - What is, indeed, "understanding"?
    - [Aristotle. (350 BCE). Metaphysics.](https://classics.mit.edu/Aristotle/metaphysics.html): Knowledge of causes
    - [Locke, J. (1690). An Essay Concerning Human Understanding.](https://www.gutenberg.org/files/10615/10615-h/10615-h.htm): Perception of agreement or disagreement of ideas
    - [Dilthey, W. (1900). The Rise of Hermeneutics.](https://www.degruyter.com/document/doi/10.1515/9780691188706-006/html?lang=en): Interpretive process of making meaning
    - [Ryle, G. (1949). The Concept of Mind.](https://archive.org/details/conceptofmind0000ryle): Ability to apply knowledge in various contexts
    - [Chalmers, D. (1996). The Conscious Mind.](https://personal.lse.ac.uk/ROBERT49/teaching/ph103/pdf/Chalmers_The_Conscious_Mind.pdf): Functional role in cognitive system
- The current capabilities of AI models
  - Limited to "statistical" reasoning
  - Infer answers based on patterns and correlations in data
    - Often the correct answer is very similar to wrong answers (hallucinations)
  - The architectures of the current most popular models (as of mid 2024) are not able to process [neuro-symbolic](https://en.wikipedia.org/wiki/Neuro-symbolic_AI) parameters
- **Weak AI** vs **Strong AI**
  - Weak AI: Designed for specific tasks, lacks general intelligence
  - Strong AI: Hypothetical AI with human-like general intelligence
- Concept of **AGI** (Artificial General Intelligence)
  - AI with human-level cognitive abilities across various domains
  - Ability to transfer knowledge between different tasks
  - Potential to surpass human intelligence in many areas
- Symbol processing
  - Able to _reason_ beyond the connectionist approaches in current popular AI models
    - Manipulation of symbolic representations
    - Logical inference and rule-based reasoning
    - Explicit representation of knowledge through linking symbols
    - Formal manipulation of symbols to derive conclusions
    - Ability to handle abstract concepts and relationships
- Meta consciousness
  - Claude-3 Opus [needle-in-the-haystack experiment](https://medium.com/@ignacio.serranofigueroa/on-the-verge-of-agi-97556c35692e)
    - Impression of consciousness due to the **instruction following** fine tuning
  - Hermes 3 405B [empty system prompt response](https://venturebeat.com/ai/meet-hermes-3-the-powerful-new-open-source-ai-model-that-has-existential-crises/)
    - Impression of consciousness due to the amount of similar data present in the training set (possibly role-playing game texts)

## Practical Uses of Generative AI

- Dealing with text inputs
  - What is **Natural Language Processing (NLP)**
  - Much more than just **replying** to word commands
    - Example: [Zork](https://en.wikipedia.org/wiki/Zork) text input processor
  - **NLP** AI Models are able to process text inputs by relating the **concepts** related to the textual inputs with the most probable **concepts** in the training set
    - Ambiguity in textual definitions
    - Contextual variations
    - Cultural variations
    - Semantic variations
- Dealing with image inputs
  - What is **Computer Vision (CV)**
  - Dealing with elements inside an image
  - Dealing with the _"meaning"_ of an image
- Dealing with audio inputs
  - What is **Automatic Speech Recognition (ASR)**
  - Dealing with spoken commands
  - Categorizing noises and sounds
  - Translating speech/audio to text/data elements
- Generating **text outputs**
- Generating **image outputs**
- Generating **audio/speech outputs**
- Generating **actions**
  - API calls
  - Integrations
    - Interacting with the real world through robotics

## Getting Started with Generative AI for Text-to-Text Tasks

- Using the [OpenAI Platform](https://platform.openai.com/)
  - [Docs](https://platform.openai.com/docs/introduction)

### Using OpenAI Chat Playground

1. Go to [OpenAI Chat Playground](https://platform.openai.com/playground?mode=chat)
2. Use the following parameters:

   - System settings:

     "_You are a knowledgeable and resourceful virtual travel advisor, expertly equipped to assist with all aspects of travel planning. From suggesting hidden gems and local cuisines to crafting personalized itineraries, you provide insightful, tailored travel advice. You adeptly navigate through various travel scenarios, offering creative solutions and ensuring a delightful planning experience for every traveler._"

   - User prompt:

     "_Hello! I'm dreaming of an adventure and need your help. I want to explore a place with breathtaking landscapes, unique culture, and delicious food. Surprise me with a destination I might not have thought of, and let's start planning an unforgettable trip!_"

   - Configurations:
     - Model: `gpt-4`
     - Temperature: `0.75`
     - Max tokens: `500`
     - Top p: `0.9`
     - Frequency penalty: `0.5`
     - Presence penalty: `0.6`

3. Click on `Submit`
4. Wait for the response from `Assistant`
5. Ask a follow-up question like "_What are the best amusements for kids there?_" or similar
6. Click on `Submit`
7. Wait for the response from `Assistant`, which should use the context from the previous messages to generate a response
8. Keep experimenting with other messages and **parameters**

## Parameters

- **Agent description**: This plays a crucial role in guiding the AI's behavior and response style. Different descriptions can set the tone, personality, and approach of the model.

  - Example: "You are a creative storyteller" would prompt the AI to adopt a more imaginative and narrative style, whereas "You are a technical expert" might lead to more detailed and specific technical information in responses.

- **Temperature**: Controls the randomness of the responses.

  - Lower temperature (0.0-0.3): More predictable, conservative responses, ideal for factual or specific queries.
  - Higher temperature (0.7-1.0): More creative and varied responses, useful for brainstorming or creative writing.

- **Max Tokens (Length)**: Sets the maximum length of the response.

  - Lower range (50-100 tokens): Suitable for concise, straightforward answers.
  - Higher range (150-500 tokens): Suitable for more detailed explanations or narratives.

- **Stop Sequence**: A list of up to four sequences of tokens that, when generated, signal the model to stop generating text. Useful for controlling response length or preventing off-topic content.

- **Top P (Nucleus Sampling)**: Determines the breadth of word choices considered by the model.

  - Lower setting (0.6-0.8): More predictable text, good for formal or factual writing.
  - Higher setting (0.9-1.0): Allows for more creativity and divergence, ideal for creative writing or generating unique ideas.

- **Frequency Penalty**: Reduces the likelihood of the model repeating the same word or phrase.

  - Lower setting (0.0-0.5): Allows some repetition, useful for emphasis in writing or speech.
  - Higher setting (0.5-1.0): Minimizes repetition, helpful for generating diverse and expansive content.

- **Presence Penalty**: Discourages the model from mentioning the same topic or concept repeatedly.
  - Lower setting (0.0-0.5): Suitable for focused content on a specific topic.
  - Higher setting (0.5-1.0): Encourages the model to explore a wider range of topics, useful for brainstorming or exploring different aspects of a subject.

> Learn more about these parameters at [OpenAI API Reference](https://platform.openai.com/docs/api-reference/chat/create)

### Reflections for Generative AI

- Does it sound like a _"real person"_?
- Does it sound _"intelligent"_?
- How is it possible that the model can answer like a _"real person"_?
  - Did someone program it to answer properly for every possible question?
- Where the model is running?
- How the **context** is handled?

## Exercise: Build a simple python script to chat with an LLM using the OpenAI API

Let's start this exercise by setting up the development environment according to your operating system.

### Linux Setup

- Open terminal (Ctrl+Alt+T)
- Check if Python is installed: `python3 --version`
- If not installed:

  ```bash
  sudo apt update
  sudo apt install python3
  sudo apt install python3-pip
  ```

### Windows Setup

- Download Python installer from [python.org](https://www.python.org/downloads/)
- Run the installer
- **Important**: Check "Add Python to PATH" during installation
- Verify installation in Command Prompt:

  ```cmd
  python --version
  ```

### macOS Setup

- Install Homebrew if not already installed:

  ```bash
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  ```

- Install Python:

  ```bash
  brew install python
  ```

- Verify installation:

  ```bash
  python3 --version
  ```

### Project Setup

Now let's configure a folder and organize the project structure there.

1. Create a folder for the project

   ```bash
   mkdir ai_project
   ```

   > You can create the folder normally with Explorer or Finder in Windows and macOS, or using the terminal with `mkdir` in Linux. You can name it as you want instead of `ai_project`.

2. Navigate to the folder in your terminal

   ```bash
   cd ai_project
   ```

3. Create a virtual environment

   ```bash
   python3 -m venv .venv
   ```

4. Activate the virtual environment

   ```bash
   . .venv/bin/activate
   ```

   > You can activate the virtual environment in Windows using `.\venv\Scripts\activate`. Read more about virtual environments in [Python documentation](https://docs.python.org/3/library/venv.html).

5. Install the dotenv package

   ```bash
   pip install python-dotenv
   ```

Then let's setup the environment variables for the OpenAI API.

1. Create a `.env` file in your project directory

   ```plaintext
   OPENAI_API_KEY=sk-proj-your_api_key_here
   ```

2. Create a `test.py` file in your project directory and open it with your favorite text editor or IDE.

3. Verify setup by running a test script.

   - Paste the following code into the `test.py` file:

   ```python
   from dotenv import load_dotenv
   import os

   load_dotenv()

   api_key = os.getenv("OPENAI_API_KEY")

   if api_key is None:
       print("Error: API key not found")
   else:
       print("API key loaded successfully")
   ```

4. Run the script

   ```bash
   python test.py
   ```

5. Common troubleshooting:
   - If `pip` commands fail, try using `pip3` instead
   - If you get permission errors, try adding `sudo` (Linux/macOS) or running as administrator (Windows)
   - If Python command not found, verify PATH environment variables
   - For virtual environment issues, ensure you're in the correct directory and it's activated

### Configuring the OpenAI API

1. Go to [OpenAI API](https://platform.openai.com/api-keys)
2. Click on `Create new secret key`
3. Copy the API key
4. Open the `.env` file and paste the API key into the `OPENAI_API_KEY` variable
5. Install the OpenAI package

   ```bash
   pip install openai
   ```

### Creating a CLI Chatbot

1. Create a `chatbot.py` file in your project directory and open it with your favorite text editor or IDE.
2. Paste the following code into the `chatbot.py` file:

   ```python
   from dotenv import load_dotenv
   import os
   import openai

   load_dotenv()

   api_key = os.getenv("OPENAI_API_KEY")

   if api_key is None:
       print("Error: API key not found")
   else:
       print("API key loaded successfully")

   # Initialize the OpenAI client
   openai.api_key = api_key

   # Define the model and prompt
   model = "gpt-4o-mini"

   messages = []
   while True:
       user_input = input("You: ")
       messages.append({"role": "user", "content": user_input})
       if user_input.lower() == "exit":
           print("Exiting...")
           break
       else:
           response = openai.chat.completions.create(
               model=model,
               messages=messages
           )
           print("Assistant: ", response.choices[0].message.content)
           messages.append({"role": "assistant", "content": response.choices[0].message.content})
   ```

3. Run the script

   ```bash
   python chatbot.py
   ```

## Project Building Guidelines

- Major problems:
  - Bad coordination
    - Your biggest enemy
  - Bad idea
    - Hidden threat that can take you by surprise
  - Bad (technical) execution
    - A relatively smaller problem
    - What most people focus on

### Solving Coordination Problems

1. Team formation
2. Planning
3. Communication

### Solving Ideation Problems

1. Technical ideation vs Business ideation
2. Prototyping and the importance of a lean MVP
3. Listening to feedback carefully

### Solving Execution Problems

1. Architecture and documentation
2. Execution planning and monitoring
3. Presentation and delivery

## Project Building Exercise

- Form groups of 2-5 people
- Compose a project idea together
- Create a GitHub repository for the project
- Create a README.md file for the project
  - Write down a very short description of the project
- Create a plan for the project
  - Write down the steps to complete the project in a WIP (Work In Progress) Pull Request
- Implement the project with your team
- Try to come up with a working demo of your project within the week, following the exercises and appending your idea as needed
- Submit your project in the [Projects Discussion Category](https://github.com/Wagademy/agent-camp/discussions/categories/projects) in this repository to be eligible for the demo day at the end of the bootcamp
  - Template for the initial post:

  ```text
  # <Project Name>

  <Your short description here - Keep it under a twitter post length>

  ## Team

  <List the team members here - Please include name, email, and if possible social media links>

  ## Description

  <Project longer description here - Two to five paragraphs about the project you have built>

  ## Roadmap

  <List the current planning of the roadmap here>

  ## Demo

  <Relevant links to the project here, like repository, links, demo video, etc>
  ```

- You can edit your project as much as you want during the bootcamp
  - You may create several projects with different ideas to help brainstorming with your team or even to find other teammates
  - If you pivot from your initial idea after building some working code, you can create a new project and add it to the discussion, and in the future you or someone else could revisit your older idea and build from there
