# Lesson 05: Decentralized AI

In this lesson we'll cover the subject of decentralized AI, how to build AI Agents that can interact with the world and how to monetize them using tokens.

We are going to introduce the concept of Web3 and smart contracts, and how these technologies can be used to enhance the capabilities of AI models, bringing benefits from peer to peer networking, token economic incentives, decentralized infrastructures and much more.

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
- Basic knowledge of Solidity programming language syntax
  - [Solidity official tutorial](https://docs.soliditylang.org/en/latest/)
  - [Learn X in Y minutes](https://learnxinyminutes.com/docs/solidity/)

## Decentralized AI

The capabilities of Generative AI models can be extended to decentralized applications by integrating them with smart contracts. This allows for creating many powerful combinations, such as financially autonomous agents, AI-powered marketplaces, dataset sharing for distributed training, peer-to-peer GPU computing markets, and more.

## ORA Protocol

- [ORA Protocol](https://ora.io/)
- [Onchain AI Oracles](https://www.ora.io/app/opml/)

### Optimistic ML

- The [opML paper](https://arxiv.org/abs/2401.17555)
- Open-source framework for verifying ML inference onchain
- Similar to optimistic rollups
- Example [opML powered AI](https://www.ora.io/app/opml/openlm)

### Resilient Model Services (RMS)

- [RMS (Resilient Model Services)](https://docs.ora.io/doc/resilient-model-services-rms/overview) is an AI service designed to provide computation for all scenarios
  - It ensurer resilient (stable, reliable, fault tolerant, and secure) AI computation
  - Powered by opML
- AI API service that integrates seamlessly with existing AI frameworks
- Replace your existing AI API provider with RMS API Key and point it to the RMS endpoint

### Initial Model Offerings

- Model Ownership ([ERC-7641 Intrinsic RevShare Token](https://ethereum-magicians.org/t/erc-7641-intrinsic-revshare-token/18999)) + Inference Asset (eg. [ERC-7007 Verifiable AI-Generated Content Token](https://github.com/AIGC-NFT/ERCs/blob/master/ERCS/erc-7007.md))
- IMO launches an ERC-20 token (more specifically, ERC-7641 Intrinsic RevShare Token) of any AI model to capture its long-term value
- Anyone who purchases the token becomes one of the owners of this AI model
- Token holders share revenue of the IMO AI model
- The [IMO launch blog post](https://mirror.xyz/orablog.eth/xYMD27tN23ppbKCluB9faytF_W6M1hKXTuKcfkm3D50) and the [first IMO implementation](https://mirror.xyz/orablog.eth/GSjMm-qC4WWsduGqCISSvA1IxicJbyRDES_bl7-Tt2o)

### Perpetual Agents

- The [opAgent](https://mirror.xyz/orablog.eth/sEFCQVmERNDIsiPDs2LUnU-__SdLmKERpCKcEP7hO08) use case
  - Agents running without relying on a centralized provider
  - Token economic incentives for hosting the agent
- Lifecycle
  - Genesis Transaction: The initial creation transaction that establishes the agent's existence on the blockchain
  - Asset Binding: Permanent linkage of digital assets to the agent through smart contracts
    -Identity Formation: Creation of a unique, immutable identity that cannot be replicated or falsified
  - Autonomous Initialization: Self-bootstrapping process that establishes initial operating parameters

### Tokenized AI Generated Content (AIGC)

- The [ERC-7007 standard](https://eips.ethereum.org/EIPS/eip-7007)
- [ERC-721](https://eips.ethereum.org/EIPS/eip-721) extension
- Verifiable AIGC tokens using ZK and opML
- Verifiable "AI Creativity" with the [7007 Protocol](https://www.7007.ai/)

### Running AI Text Generation Tasks with Decentralized AI Model Inferences

- The [OAO repository](https://github.com/ora-io/OAO)
- Implementing the `IAIOracle.sol` interface
- Building smart contracts [with ORA's AI Oracle](https://docs.ora.io/doc/ai-oracle/ai-oracle/build-with-ai-oracle)
- Handling the [Callback gas limit estimation](https://docs.ora.io/doc/ai-oracle/ai-oracle/callback-gas-limit-estimation) for each model ID
- [Reference list](https://docs.ora.io/doc/ai-oracle/ai-oracle/references) for models and addresses for different networks

### Experimenting with the OAO Sample Prompt Contract Implementation

- Open the [OAO repository](https://github.com/ora-io/OAO) in [Remix IDE](https://remix.ethereum.org/)
  - Click on the hamburger menu for the `WORKSPACES` section
  - Select the `Clone` option
  - Pase the URL of the repository in the `Repository URL` field
    - URL: <https://github.com/ora-io/OAO>
- Open the `Prompt.sol` file in the `contracts` folder and compile it
- Go for the `Deploy & run transactions` tab and select the `injected provider` environment
- Make sure that your wallet is connected to the `sepolia` network
- Go for the the `At Address` button, paste the example `Prompt` contract address (`0xe75af5294f4CB4a8423ef8260595a54298c7a2FB` for the `sepolia` network), and then click on the `At Address` button
- Scroll down to the `Deployed Contracts` section and click on the `Prompt` contract
- Click on the `estimateFee` function and set the model ID to `11`
- Copy the returned value and set it as the transaction value (in wei)
  - Scroll up to see the `VALUE` field, and make sure that the dropdown is set to `Wei`
- Click on the `calculateAIResult` function and set the model ID to `11` and the prompt text to `"What is the capital of France?"`
- Click on the red `Transact` button and confirm the popup on your wallet to execute the transaction
- Wait for the transaction to be confirmed
  - It might take a little while until the callback is executed and the result is available
- When the callback is executed, the result is stored in the storage of the `Prompt` contract
  - The `prompt` function can be called to retrieve the result, by passing the model ID and prompt text again
- Try this process with a different prompt, for example: `"List all capitals that Brazil had in the past"`

## Final Exercise

- Append your project submitted in the [Projects Discussion Category](https://github.com/Wagademy/agent-camp/discussions/categories/projects) with the developments of your project within the days of the bootcamp
- Prepare a short video presentation of your project with a duration of no more than 3 minutes
  - Explain as much as possible of your project with visual demonstrations of things that you have built, and avoid long explanations (show, don't tell)
- Update your project post with the video link and the project description
  - Include any relevant links and technical details of your project for evaluation
