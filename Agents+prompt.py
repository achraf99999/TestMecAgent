import os
import json
import re
import requests
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import google.generativeai as genai
from typing import Optional, Dict, Any, List
import logging
from pathlib import Path
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MechanicalPartAnalyzer:
    """
    A comprehensive pipeline for analyzing mechanical parts from images and generating CADquery code.
    """

    def __init__(self, gemini_api_key: str, model_path: str = "FreedomIntelligence/BlenderLLM"):
        """Initialize the analyzer with API keys and model paths."""
        self.gemini_api_key = gemini_api_key
        self.model_path = model_path
        self.gemini_client = None
        self.cad_model = None
        self.tokenizer = None

        # Configure Gemini
        genai.configure(api_key=self.gemini_api_key)
        self.gemini_client = genai.GenerativeModel("gemini-1.5-flash")

        logger.info("MechanicalPartAnalyzer initialized successfully")

    def upload_file_to_gemini(self, file_path: str) -> Optional[str]:
        """Upload a file to Gemini and return the File object name."""
        if not Path(file_path).exists():
            logger.error(f"File not found: {file_path}")
            return None

        logger.info(f"Uploading file: {file_path}")
        try:
            my_file = genai.upload_file(file_path)
            logger.info(f"File uploaded successfully: {my_file.name}")
            logger.info(f"File URI: {my_file.uri}")
            logger.info(f"File MIME type: {my_file.mime_type}")

            # Wait for file processing to complete
            import time
            time.sleep(2)

            # Verify file is ready
            file_info = genai.get_file(my_file.name)
            logger.info(f"File status: {file_info.state}")

            return my_file.name
        except Exception as e:
            logger.error(f"Error uploading file: {e}")
            return None

    def get_mechanical_details_from_image(self, file_resource_name: str) -> Optional[Dict[str, Any]]:
        """Extract mechanical details from image using Gemini with structured output."""
        logger.info("Analyzing image with Gemini for mechanical details")

        # Enhanced prompt for better CADquery code generation
        improved_prompt = """
Vous êtes un ingénieur mécanique expert et programmeur CAD. Analysez cette image de pièce mécanique et extrayez les détails spécifiquement pour la génération de code CADquery.

Concentrez-vous sur :
1. **Géométrie de base** : Forme générale (rectangulaire, cylindrique, complexe), dimensions principales
2. **Caractéristiques** : Trous, fentes, chanfreins, congés, filetages, rainures
3. **Éléments d'assemblage** : Trous de montage, brides, connecteurs, joints
4. **Propriétés du matériau** : Type, épaisseur, tailles standard
5. **Détails pertinents pour CAD** : Symétries, motifs, points de référence, système de coordonnées

Soyez précis avec les mesures et spécifiez :
- Dimensions principales (longueur, largeur, hauteur, diamètre)
- Emplacements des caractéristiques par rapport aux bords/centre
- Tailles de trous standard, spécifications de filetage
- Rayons de congé/chanfrein
- Motifs répétés ou matrices

Répondez UNIQUEMENT en JSON valide avec cette structure exacte :
{
  "overall_description": "description générale de la pièce",
  "basic_geometry": {
    "shape_type": "type de forme",
    "primary_dimensions": ["dimension 1", "dimension 2"],
    "material_thickness": "épaisseur"
  },
  "features": [
    {
      "type": "type de caractéristique",
      "location": "emplacement",
      "dimensions": "dimensions",
      "description": "description"
    }
  ],
  "mechanical_details": {
    "dimensions": ["liste des dimensions"],
    "tolerances": ["liste des tolérances"],
    "material": "matériau",
    "surface_finish": "finition de surface",
    "functionality": "fonctionnalité",
    "manufacturing_process": "processus de fabrication"
  }
}
"""

        try:
            # Use the simpler approach with current Gemini model
            uploaded_file = genai.get_file(file_resource_name)

            # Create the model
            model = genai.GenerativeModel("gemini-2.5-flash")

            # Generate content with simpler configuration
            response = model.generate_content(
                [uploaded_file, improved_prompt],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=1500,
                )
            )

            if response.text:
                # Clean and parse JSON response
                json_text = response.text.strip()

                # Remove markdown formatting if present
                if json_text.startswith('```json'):
                    json_text = json_text.replace('```json', '').replace('```', '').strip()
                elif json_text.startswith('```'):
                    json_text = json_text.replace('```', '').strip()

                # Remove any text before the first {
                if '{' in json_text:
                    json_text = json_text[json_text.find('{'):]

                # Remove any text after the last }
                if '}' in json_text:
                    json_text = json_text[:json_text.rfind('}') + 1]

                try:
                    parsed_data = json.loads(json_text)
                    logger.info("Successfully parsed mechanical details from Gemini")
                    return parsed_data
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON parsing failed: {e}")
                    logger.info("Creating fallback response from text")
                    return self._create_fallback_response(response.text)
            else:
                logger.error("Empty response from Gemini")
                return None

        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            return None

    def _create_fallback_response(self, text_response: str) -> Dict[str, Any]:
        """Create a structured response when JSON parsing fails."""
        return {
            "overall_description": text_response[:500] + "..." if len(text_response) > 500 else text_response,
            "basic_geometry": {
                "shape_type": "Unknown - see description",
                "primary_dimensions": ["Not specified"],
                "material_thickness": "Not specified"
            },
            "features": [],
            "mechanical_details": {
                "dimensions": ["Not specified"],
                "tolerances": [],
                "material": "Not specified",
                "surface_finish": "Not specified",
                "functionality": "Not specified",
                "manufacturing_process": "Not specified"
            }
        }

    def format_description_for_cad_query(self, parsed_data: Dict[str, Any]) -> str:
        """Format the parsed mechanical description for CADquery generation."""
        if not parsed_data:
            return "No mechanical details available for CAD query generation."

        description_parts = []

        # Overall description
        overall_desc = parsed_data.get("overall_description", "")
        if overall_desc and overall_desc.lower() != "n/a":
            description_parts.append(f"Part Description: {overall_desc}")

        # Basic geometry
        basic_geo = parsed_data.get("basic_geometry", {})
        if basic_geo:
            shape_type = basic_geo.get("shape_type", "")
            if shape_type:
                description_parts.append(f"Basic Shape: {shape_type}")

            dimensions = basic_geo.get("primary_dimensions", [])
            if dimensions:
                description_parts.append(f"Primary Dimensions: {'; '.join(dimensions)}")

        # Features
        features = parsed_data.get("features", [])
        if features:
            feature_descriptions = []
            for feature in features:
                feat_type = feature.get("type", "")
                location = feature.get("location", "")
                dims = feature.get("dimensions", "")
                desc = feature.get("description", "")

                feat_desc = f"{feat_type}"
                if location: feat_desc += f" at {location}"
                if dims: feat_desc += f" ({dims})"
                if desc: feat_desc += f" - {desc}"
                feature_descriptions.append(feat_desc)

            description_parts.append(f"Features: {'; '.join(feature_descriptions)}")

        return " | ".join(description_parts)

    def load_cad_model(self):
        """Load the CADquery generation model."""
        if self.cad_model is not None:
            return True

        logger.info(f"Loading CAD generation model: {self.model_path}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                use_fast=False,
                model_max_length=2048  # Increased for longer CAD descriptions
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"

            self.cad_model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            self.cad_model.eval()
            logger.info("CAD generation model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Error loading CAD model: {e}")
            return False

    def generate_cad_query_code(self, description: str) -> Optional[str]:
        """Generate CADquery code from the mechanical description."""
        if not self.load_cad_model():
            return None

        # Enhanced prompt for better CADquery code generation
        cad_prompt = f"""Generate complete, executable CADquery code to create this mechanical part:

{description}

Requirements:
1. Import cadquery as cq
2. Create a complete, functional CADquery script
3. Use proper CADquery syntax and methods
4. Include comments explaining each step
5. End with .val() to return the solid
6. Handle edge cases and provide reasonable defaults for unclear dimensions

Example structure:
```python
import cadquery as cq

# Create base geometry
result = (cq.Workplane("XY")
    .box(length, width, height)
    # Add features
    .faces(">Z").workplane()
    .hole(diameter)
    # Additional operations
)

# Return the result
result
```

Generate the CADquery code:"""

        try:
            messages = [{"role": "user", "content": cad_prompt}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.cad_model.device)

            logger.info("Generating CADquery code...")
            generated_ids = self.cad_model.generate(
                model_inputs.input_ids,
                max_new_tokens=1024,  # Increased for longer code
                do_sample=True,
                temperature=0.5,  # Balance between creativity and consistency
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )

            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in
                zip(model_inputs.input_ids, generated_ids)
            ]

            response_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            logger.info("CADquery code generated successfully")
            return response_text.strip()

        except Exception as e:
            logger.error(f"Error generating CADquery code: {e}")
            return None

    def post_process_cad_code(self, cad_code: str) -> str:
        """Post-process the generated CADquery code for better quality."""
        if not cad_code:
            return ""

        # Clean up common issues
        lines = cad_code.split('\n')
        cleaned_lines = []

        for line in lines:
            # Remove extra whitespace
            line = line.strip()
            if not line:
                continue

            # Ensure proper import
            if 'import cadquery' in line and 'as cq' not in line:
                line = 'import cadquery as cq'

            cleaned_lines.append(line)

        # Ensure import is at the top
        if cleaned_lines and 'import cadquery as cq' not in cleaned_lines[0]:
            cleaned_lines.insert(0, 'import cadquery as cq')
            cleaned_lines.insert(1, '')

        return '\n'.join(cleaned_lines)

    def analyze_part(self, image_path: str) -> Dict[str, Any]:
        """Complete pipeline to analyze a mechanical part and generate CADquery code."""
        results = {
            'success': False,
            'image_path': image_path,
            'file_resource_name': None,
            'mechanical_details': None,
            'formatted_description': None,
            'cad_code': None,
            'error': None
        }

        try:
            # Step 1: Upload image
            file_resource_name = self.upload_file_to_gemini(image_path)
            if not file_resource_name:
                results['error'] = "Failed to upload image"
                return results
            results['file_resource_name'] = file_resource_name

            # Step 2: Get mechanical details
            mechanical_details = self.get_mechanical_details_from_image(file_resource_name)
            if not mechanical_details:
                results['error'] = "Failed to extract mechanical details"
                return results
            results['mechanical_details'] = mechanical_details

            # Step 3: Format for CAD generation
            formatted_description = self.format_description_for_cad_query(mechanical_details)
            results['formatted_description'] = formatted_description

            # Step 4: Generate CADquery code
            cad_code = self.generate_cad_query_code(formatted_description)
            if not cad_code:
                results['error'] = "Failed to generate CADquery code"
                return results

            # Step 5: Post-process code
            final_cad_code = self.post_process_cad_code(cad_code)
            results['cad_code'] = final_cad_code
            results['success'] = True

            logger.info("Part analysis completed successfully")
            return results

        except Exception as e:
            logger.error(f"Error in analyze_part: {e}")
            results['error'] = str(e)
            return results

    def save_results(self, results: Dict[str, Any], output_dir: str = "output"):
        """Save analysis results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        timestamp = int(time.time())

        # Save mechanical details as JSON
        if results.get('mechanical_details'):
            json_file = output_path / f"mechanical_details_{timestamp}.json"
            with open(json_file, 'w') as f:
                json.dump(results['mechanical_details'], f, indent=2)
            logger.info(f"Mechanical details saved to: {json_file}")

        # Save CADquery code
        if results.get('cad_code'):
            cad_file = output_path / f"generated_cad_{timestamp}.py"
            with open(cad_file, 'w') as f:
                f.write(results['cad_code'])
            logger.info(f"CADquery code saved to: {cad_file}")

        # Save complete results
        results_file = output_path / f"analysis_results_{timestamp}.json"
        # Make a copy without the mechanical_details for cleaner JSON
        summary_results = {k: v for k, v in results.items() if k != 'mechanical_details'}
        with open(results_file, 'w') as f:
            json.dump(summary_results, f, indent=2)
        logger.info(f"Analysis summary saved to: {results_file}")


# Example usage
def main():
    """Example usage of the MechanicalPartAnalyzer."""
    # Configuration
    api_key = os.getenv('GOOGLE_API_KEY', "")
    image_path = "/content/cad.jpg"  # Update with your image path

    if api_key == "":
        logger.error("Please set your GOOGLE_API_KEY environment variable")
        return

    # Initialize analyzer
    analyzer = MechanicalPartAnalyzer(gemini_api_key=api_key)

    # Analyze the part
    results = analyzer.analyze_part(image_path)

    if results['success']:
        print("\n" + "="*50)
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*50)

        print(f"\nFormatted Description:")
        print("-" * 30)
        print(results['formatted_description'])

        print(f"\nGenerated CADquery Code:")
        print("-" * 30)
        print(results['cad_code'])

        # Save results
        analyzer.save_results(results)

    else:
        print(f"\nAnalysis failed: {results.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()