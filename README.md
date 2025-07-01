# CAD Query Code Generation from Images

This project explores and implements two distinct approaches for generating CAD (Computer-Aided Design) Query code directly from images of mechanical parts. The overarching goal is to automate the creation of 3D models, significantly streamlining the design process by leveraging advancements in Vision-Language Models (VLMs) and Large Language Models (LLMs).

---

## Project Overview

The project aims to develop robust pipelines that can interpret visual information from mechanical part images and translate it into precise, executable CADquery Python code. This capability has the potential to accelerate prototyping, design iteration, and enhance accessibility for CAD modeling.

---

## Implemented Approaches

### 1. Vision-Language Model (VLM) Pipeline: ViT Encoder + CodeGPT Decoder

This approach focuses on an end-to-end vision-to-code generation, combining a powerful image encoder with a code-generating language model.

- **Architecture:**
    - **Encoder:** A **Vision Transformer (ViT)** was used here primarily due to its simplicity and because I have extensive experience fine-tuning it. ViT reliably extracts high-level semantic and geometric features from images. For future iterations, replacing it with a ViT + Q‑Former stack (e.g. BLIP‑2) would likely yield better results. BLIP‑2 significantly boosts vision-language alignment and performance—achieving 8.7 % higher accuracy on zero-shot VQA compared to Flamingo80B using 54× fewer trainable parameters—by using a lightweight Q‑Former bridge between the vision encoder and a frozen LLM .
    - **Decoder:** **CodeGPT Small** acts as the decoder. It takes the contextually rich embeddings produced by the ViT encoder and generates the corresponding CADquery Python code. Here, I chose CodeGPT Small not because it’s necessarily the best, but because it’s lightweight and easy to integrate. For future improvements, I plan to explore larger models like LLaMaCode, CodeT5, or SFTQwen2.5-3B. Additionally, it’s possible to fine-tune the decoder model to generate code directly from ViT embeddings as input.

- **Fine-tuning:** Both the ViT encoder and CodeGPT decoder were fine-tuned on a subset of the **CADCODER/GenCAD-Code** dataset. This dataset provides paired images and their respective CAD code, crucial for teaching the model the intricate relationship between visual features and CAD commands.

- **Implementation:** For the detailed implementation, including data loading, model architecture setup, and the fine-tuning process, please refer to the `vit+codegptsmall.py` notebook.

---

### 2. Dual Large Language Model (LLM) Pipeline: Description Generation + Code Generation

This pipeline adopts a modular, two-stage approach, leveraging the strengths of different LLMs for specific tasks.

- **Stage 1: Image Description Generation:**

    - The **Gemini API** (specifically the `gemini-1.5-flash` model) is used to analyze the input image. It's prompted to act as an expert mechanical engineer and CAD programmer, extracting comprehensive details such as the primary shape, precise dimensions (length, width, height, diameter), geometric features (holes, slots, chamfers, fillets), relative positions, and a step-by-step construction plan. Here  WE CAN ALSO USE LLama-Vision as it's open source and a state of art in VLms domain  but the problem is that he is lourd with big size trhat make his used in my local machine a bite hard .

    - The prompt is carefully engineered to encourage structured JSON output, making the extracted information easily parsable and ready for the next stage.

- **Stage 2: CAD Code Generation:**

    - The rich, structured mechanical description obtained from Gemini is then fed into a specialized large language model. This model's task is to translate the textual description into executable CADquery Python code.

    - **Models Explored:**
        - **OurSFTQwen2.5-3B:** A fine-tuned language model that was tested for its ability to generate CADquery code.
        - **BlenderLLM:** This model, detailed in the paper "[BlenderLLM: Training Large Language Models for Computer-Aided Design with Self-improvement](https://arxiv.org/abs/2412.14203)", was evaluated due to its reported superior performance in CAD code generation tasks.

- **Implementation:** The Python script for this approach (integrated into `Agents+prompt.py` or as a standalone script) demonstrates the interaction with the Gemini API, the structured data parsing, and the subsequent CAD query code generation using the fine-tuned LLMs.

---

---
I had thinking about another approches that's ** Hybrid Pipeline: VLM → RAG → LLM Refinement **

### Overview

1. **Extract structured description**  
   Use a **Vision-Language Model (VLM)** to generate a detailed description from the image—for example:  
   *“LLamaVision ”*

2. **Identify part type via NER**  
   Apply a lightweight NER model like **GLiNER**, which is fast and efficient, to determine the mechanical part category mentioned in the description (e.g., Bearings, Shafts, Keys).

3. **Perform vector search in RAG system**  
   - Maintain a **vector DB** containing paired text + CadQuery code examples, categorized by part type (e.g., Bearings · Shafts · Keys · Couplings · Fasteners · Gears).  
   - Search within the relevant category to fetch the most semantically similar code snippets based on the description.

4. **Refine code using LLM**  
   Pass the retrieved CadQuery snippet and the original VLM-generated description to an LLM , in this step we will already have a code that's closely represent  the descritiopn so here we  will just  add or adjust details to ensure the final script closely matches the intended shape and specifications.

---

### 🔍 Why This Pipeline?

- **Precise grounding**: Vector retrieval anchors code to real, tested examples—reducing hallucinations and boosting relevance.  
- **Scalable architecture**: New part types can be added by simply inserting new text/code pairs into corresponding collections.  
- **Efficient processing**: GLiNER quickly identifies the correct category, and vector search narrows down to candidate snippets before invoking the LLM.  
- **Improved accuracy**: The LLM’s refinement ensures the final script aligns with both input image and description.

---

### 🔧 Components

- **VLM** – Extracts structured descriptions from images.  
- **NER (GLiNER)** – Lightweight, BERT-based model to detect part categories fast and accurately :contentReference[oaicite:1]{index=1}.  
- **Vector DB** – Stores embeddings of (description, CadQuery code) pairs. Collections are organized by part type.  
- **Retriever** – Searches Collection using semantic similarity (e.g. : Chroma,  Qdrant) :contentReference[oaicite:2]{index=2}.  
- **LLM Refiner** – Refines retrieved code to include missing features and align with the specific input description.

---

### 🔄 Workflow Recap

1. 🖼️ Input image → VLM → structured description.  
2. 🏷️ GLiNER NER → identify part type.  
3. 🔎 Vector retrieval → fetch similar CadQuery code snippet.  
4. 🛠️ LLM takes code + description → outputs final, refined CadQuery script.

---

### 🚀 Benefits & Next Steps

- **Grounded script generation**: Reduces hallucination risk by retrieving existing code.  
- **Modular expansion**: Easily support new mechanical part types via additional collections.  
- **Hybrid strengths**: Combines grounded retrieval with generative flexibility.  
- **Future direction**: Add evaluation—execute generated code, measure metrics like IoU/F1, and iterate on refinement quality.

---

This VLM → RAG → LLM pipeline intelligently bridges visual understanding, semantic retrieval, and generative refinement to produce structured, high-fidelity CadQuery scripts.  
Feel free to reach out if you'd like help setting up the vector DB schema or LLM refinement prompts!


---


## How to Improve and Future Work

The field of AI-driven CAD generation is rapidly evolving, and there are numerous promising directions for further improvement:

- **Advanced Fine-tuning Strategies:**

    - **Larger and More Diverse Datasets:** The current fine-tuning uses a portion of `CADCODER/GenCAD-Code`. Expanding this with more varied and complex mechanical parts, especially those with intricate CADquery constructions, will significantly enhance the models' generalizability and accuracy.

    - **Multi-view Datasets:** Incorporating multi-view images of the same object during training could provide richer 3D understanding for the VLM approach.

- **Exploring Cutting-Edge Models:**

    - **Dedicated CAD-VLMs:** Investigate and integrate models specifically designed for CAD tasks, such as **CAD-Coder** (as introduced in the paper "[Text-to-CadQuery: A New Paradigm for CAD Generation with Scalable Large Model Capabilities](https://arxiv.org/abs/2505.06507)" and mentioned in "[Unleashing the Power of Vision-Language Models in Computer-Aided Design: A Survey](https://arxiv.org/abs/2505.14646)"). CAD-Coder is an open-source VLM explicitly fine-tuned to generate editable CAD code (CadQuery Python) from visual input, showcasing superior performance over general VLMs.

    - **Newer LLMs:** Continuously evaluate the latest LLMs for their code generation capabilities and adaptability to CAD-specific tasks. Models with stronger logical reasoning and spatial understanding are particularly valuable.

- **Refined Prompt Engineering:** For the dual-LLM pipeline, invest further in sophisticated prompt engineering techniques. This includes:
    - Providing more detailed, diverse, and complex in-context learning examples in the prompts.
    - Using chain-of-thought prompting to guide the LLM through logical steps of CAD construction.
    - Implementing self-correction mechanisms where the LLM can evaluate its own generated code against certain criteria.

- **Robust Post-processing and Validation:**

    - **Static Code Analysis:** Implement more rigorous static analysis of the generated CADquery code to catch syntax errors, missing imports, or logical inconsistencies.

    - **Dynamic Validation:** Develop a system to execute the generated CADquery code and potentially render the 3D model. This allows for programmatic verification of the output, checking for common CAD errors (e.g., non-manifold geometry, self-intersections).

    - **Similarity Metrics:** Utilize metrics like 3D solid similarity and Chamfer Distance (as mentioned in relevant research papers) to quantitatively evaluate the generated CAD models against ground truth, providing concrete targets for improvement.

- **Incorporating Geometric Constraints/Parameters:** Future work could involve allowing users to specify geometric constraints or parameters (e.g., "make this hole diameter 10mm") to guide the generation process, moving towards more parametric design.

- **Interactive Refinement:** Developing an interactive interface where users can provide feedback on incorrect generations, allowing for rapid model fine-tuning or reinforcement learning from human feedback (RLHF).

---

## Conclusion

This is truly a project with many exciting directions and possibilities. Beyond the methods already implemented, we could also explore Reinforcement Learning to further enhance code generation quality and decision-making in complex CAD workflows.

I believe this project is a perfect opportunity to showcase my skills and deepen my expertise in AI. It reflects my genuine interest in the field and demonstrates how ready I am to invest significant time and effort into working on innovative, cutting-edge solutions in AI and CAD automation.



