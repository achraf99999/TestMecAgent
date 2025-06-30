# CAD Query Code Generation from Images

This repository explores and implements two distinct approaches for generating CAD (Computer-Aided Design) Query code directly from images of mechanical parts. The overarching goal is to automate the creation of 3D models, significantly streamlining the design process by leveraging advancements in Vision-Language Models (VLMs) and Large Language Models (LLMs).

## Project Overview

The project aims to develop robust pipelines that can interpret visual information from mechanical part images and translate it into precise, executable CADquery Python code. This capability has the potential to accelerate prototyping, design iteration, and enhance accessibility for CAD modeling.

## Implemented Approaches

### 1. Vision-Language Model (VLM) Pipeline: ViT Encoder + CodeGPT Decoder

This approach focuses on an end-to-end vision-to-code generation, combining a powerful image encoder with a code-generating language model.

* **Architecture:**
    * **Encoder:** A **Vision Transformer (ViT)** is employed to process the input image. It's responsible for extracting high-level semantic and geometric features from the visual data.
    * **Decoder:** **CodeGPT Small** acts as the decoder. It takes the contextually rich embeddings produced by the ViT encoder and generates the corresponding CADquery Python code.
* **Fine-tuning:** Both the ViT encoder and CodeGPT decoder were fine-tuned on a subset of the **CADCODER/GenCAD-Code** dataset. This dataset provides paired images and their respective CAD code, crucial for teaching the model the intricate relationship between visual features and CAD commands.
* **Implementation:** For the detailed implementation, including data loading, model architecture setup, and the fine-tuning process, please refer to the `finetynining vit+codegpt.ipynb` notebook.

### 2. Dual Large Language Model (LLM) Pipeline: Description Generation + Code Generation

This pipeline adopts a modular, two-stage approach, leveraging the strengths of different LLMs for specific tasks.

* **Stage 1: Image Description Generation:**
    * The **Gemini API** (specifically the `gemini-1.5-flash` model) is used to analyze the input image. It's prompted to act as an expert mechanical engineer and CAD programmer, extracting comprehensive details such as the primary shape, precise dimensions (length, width, height, diameter), geometric features (holes, slots, chamfers, fillets), relative positions, and a step-by-step construction plan.
    * The prompt is carefully engineered to encourage structured JSON output, making the extracted information easily parsable and ready for the next stage.
* **Stage 2: CAD Code Generation:**
    * The rich, structured mechanical description obtained from Gemini is then fed into a specialized large language model. This model's task is to translate the textual description into executable CADquery Python code.
    * **Models Explored:**
        * **OurSFTQwen2.5-3B:** A fine-tuned language model that was tested for its ability to generate CADquery code.
        * **BlenderLLM:** This model, detailed in the paper "[BlenderLLM: Training Large Language Models for Computer-Aided Design with Self-improvement](https://arxiv.org/abs/2412.14203)", was evaluated due to its reported superior performance in CAD code generation tasks.
* **Implementation:** The Python script for this approach (integrated into `good-luck.ipynb` or as a standalone script) demonstrates the interaction with the Gemini API, the structured data parsing, and the subsequent CAD query code generation using the fine-tuned LLMs.

## How to Improve and Future Work

The field of AI-driven CAD generation is rapidly evolving, and there are numerous promising directions for further improvement:

* **Advanced Fine-tuning Strategies:**
    * **Larger and More Diverse Datasets:** The current fine-tuning uses a portion of `CADCODER/GenCAD-Code`. Expanding this with more varied and complex mechanical parts, especially those with intricate CADquery constructions, will significantly enhance the models' generalizability and accuracy.
    * **Multi-view Datasets:** Incorporating multi-view images of the same object during training could provide richer 3D understanding for the VLM approach.
* **Exploring Cutting-Edge Models:**
    * **Dedicated CAD-VLMs:** Investigate and integrate models specifically designed for CAD tasks, such as **CAD-Coder** (as introduced in the paper "[Text-to-CadQuery: A New Paradigm for CAD Generation with Scalable Large Model Capabilities](https://arxiv.org/abs/2505.06507)" and mentioned in "[Unleashing the Power of Vision-Language Models in Computer-Aided Design: A Survey](https://arxiv.org/abs/2505.14646)"). CAD-Coder is an open-source VLM explicitly fine-tuned to generate editable CAD code (CadQuery Python) from visual input, showcasing superior performance over general VLMs.
    * **Newer LLMs:** Continuously evaluate the latest LLMs for their code generation capabilities and adaptability to CAD-specific tasks. Models with stronger logical reasoning and spatial understanding are particularly valuable.
* **Refined Prompt Engineering:** For the dual-LLM pipeline, invest further in sophisticated prompt engineering techniques. This includes:
    * Providing more detailed, diverse, and complex in-context learning examples in the prompts.
    * Using chain-of-thought prompting to guide the LLM through logical steps of CAD construction.
    * Implementing self-correction mechanisms where the LLM can evaluate its own generated code against certain criteria.
* **Robust Post-processing and Validation:**
    * **Static Code Analysis:** Implement more rigorous static analysis of the generated CADquery code to catch syntax errors, missing imports, or logical inconsistencies.
    * **Dynamic Validation:** Develop a system to execute the generated CADquery code and potentially render the 3D model. This allows for programmatic verification of the output, checking for common CAD errors (e.g., non-manifold geometry, self-intersections).
    * **Similarity Metrics:** Utilize metrics like 3D solid similarity and Chamfer Distance (as mentioned in relevant research papers) to quantitatively evaluate the generated CAD models against ground truth, providing concrete targets for improvement.
* **Incorporating Geometric Constraints/Parameters:** Future work could involve allowing users to specify geometric constraints or parameters (e.g., "make this hole diameter 10mm") to guide the generation process, moving towards more parametric design.
* **Interactive Refinement:** Developing an interactive interface where users can provide feedback on incorrect generations, allowing for rapid model fine-tuning or reinforcement learning from human feedback (RLHF).

## Setup and Usage

To set up and run the code in this repository:

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/achraf99999/TestMecAgent.git]
    ```

2.  **Install Dependencies:**
    You will need Python 3.9+ and the following libraries. It's highly recommended to use a virtual environment.
    ```bash
    pip install -U transformers torch google-generativeai cadquery jupyterlab
    # Additional dependencies might be needed based on specific model requirements (e.g., accelerate, bitsandbytes for quantization)
    ```

3.  **Configure Google Gemini API Key:**
    For the dual-LLM pipeline, you need a Google API Key. Set it as an environment variable:
    ```bash
    export GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"
    # Or, in your Python script, you can set:
    # os.environ["GOOGLE_API_KEY"] = "YOUR_GEMINI_API_KEY"
    ```
    **Remember to keep your API key secure and never hardcode it directly in publicly accessible files.**

4.  **Run the Code:**
    * **For the VLM Pipeline:** Open and run the `finetynining vit+codegpt.ipynb` notebook in a Jupyter environment. This notebook contains the training and potentially inference code for the ViT+CodeGPT model.
    * **For the Dual LLM Pipeline:** The core logic is present in the Python code you provided (e.g., `MechanicalPartAnalyzer` class). You can integrate this into a script (e.g., `main.py`) and run it:
        ```bash
        python main.py
        ```
        Make sure to update the `image_path` variable in your `main()` function to point to your desired input image.

## Repository Structure
