"""
LLM Explanation Service
Generates human-readable explanations of lameness predictions.

Key Features:
- Evidence-based summaries (no hallucination)
- Structured prompts with strict input constraints
- Executive summary, guidance, and action recommendations
- Integration with OpenAI API or local LLM
"""
import asyncio
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml
from shared.utils.nats_client import NATSClient


class LLMExplanationService:
    """
    Service for generating LLM-based explanations of lameness predictions.
    
    Constraints:
    - Only reference provided inputs (no external knowledge)
    - Explicitly state when evidence is missing or conflicting
    - Keep explanations concise and actionable
    """
    
    # System prompt template
    SYSTEM_PROMPT = """You are a veterinary AI assistant explaining lameness predictions for dairy cows.

STRICT RULES:
1. ONLY reference the data provided in the user message
2. NEVER invent or assume information not in the input
3. If evidence is missing or conflicting, explicitly say so
4. Keep explanations clear and actionable for farm staff
5. Use simple language, avoid jargon

OUTPUT FORMAT:
1. Executive Summary (2-3 sentences): Main conclusion with confidence level
2. Key Evidence: Bullet points of supporting data
3. Uncertainties: Any missing data or model disagreements  
4. Recommended Action: Clear next step"""

    EXPLANATION_TEMPLATE = """Generate an explanation for this lameness prediction:

## Final Decision
- Prediction: {prediction_label} ({probability:.1%} probability)
- Confidence: {confidence_level} ({confidence:.1%})
- Decision Mode: {decision_mode}

## Pipeline Predictions
{pipeline_summary}

## Quality Indicators
- Clip Quality: {clip_quality}
- Pose Quality: {pose_quality}
- Detection Confidence: {detection_confidence}

## Gait Features (from T-LEAP)
{gait_features}

## Top SHAP Features
{shap_features}

## Human Consensus
{human_consensus}

## Model Agreement
- Agreement Level: {agreement_level}
- Models in agreement: {models_agree}

Generate a clear explanation following the output format specified."""

    def __init__(self):
        self.config_path = Path("/app/shared/config/config.yaml")
        self.config = self._load_config()
        self.nats_client = NATSClient(str(self.config_path))
        
        # LLM configuration
        self.llm_provider = os.getenv("LLM_PROVIDER", "openai")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        # Results directory
        self.results_dir = Path("/app/data/results/explanations")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize OpenAI client if available
        self.openai_client = None
        if self.openai_api_key:
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=self.openai_api_key)
                print(f"✅ OpenAI client initialized with model: {self.openai_model}")
            except ImportError:
                print("⚠️ OpenAI library not installed")
            except Exception as e:
                print(f"⚠️ Failed to initialize OpenAI client: {e}")
        else:
            print("⚠️ No OPENAI_API_KEY set; using template-based explanations")
    
    def _load_config(self) -> dict:
        if self.config_path.exists():
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        return {}
    
    def format_pipeline_summary(self, pipeline_contributions: Dict[str, Any]) -> str:
        """Format pipeline predictions for the prompt"""
        lines = []
        for pipeline, data in pipeline_contributions.items():
            if isinstance(data, dict):
                prob = data.get("probability", 0.5)
                pred = "Lame" if prob > 0.5 else "Sound"
                uncertainty = data.get("uncertainty", 0.5)
                lines.append(f"- {pipeline.upper()}: {pred} ({prob:.1%} probability, uncertainty: {uncertainty:.1%})")
            else:
                lines.append(f"- {pipeline.upper()}: {data}")
        
        return "\n".join(lines) if lines else "No pipeline predictions available"
    
    def format_gait_features(self, tleap_features: Dict[str, Any]) -> str:
        """Format T-LEAP gait features for the prompt"""
        if not tleap_features:
            return "No gait features available"
        
        feature_descriptions = {
            "back_arch_mean": "Back arch angle",
            "back_arch_score": "Back arch severity",
            "head_bob_magnitude": "Head bobbing intensity",
            "head_bob_score": "Head bob severity",
            "front_leg_asymmetry": "Front leg asymmetry",
            "rear_leg_asymmetry": "Rear leg asymmetry",
            "lameness_score": "Overall lameness score"
        }
        
        lines = []
        for key, value in tleap_features.items():
            if key in feature_descriptions:
                severity = "High" if value > 0.7 else "Medium" if value > 0.4 else "Low"
                lines.append(f"- {feature_descriptions[key]}: {value:.2f} ({severity})")
        
        return "\n".join(lines) if lines else "No significant gait abnormalities detected"
    
    def format_shap_features(self, shap_data: Dict[str, Any]) -> str:
        """Format SHAP feature importance for the prompt"""
        if not shap_data or "top_features" not in shap_data:
            return "SHAP analysis not available"
        
        lines = []
        for feature in shap_data.get("top_features", [])[:5]:
            name = feature.get("name", "Unknown")
            value = feature.get("value", 0)
            importance = feature.get("importance", 0)
            direction = "increases" if importance > 0 else "decreases"
            lines.append(f"- {name}: {value:.2f} ({direction} lameness probability)")
        
        return "\n".join(lines) if lines else "No significant feature contributions"
    
    def format_human_consensus(self, human_data: Optional[Dict[str, Any]]) -> str:
        """Format human consensus data for the prompt"""
        if not human_data or human_data.get("num_raters", 0) == 0:
            return "No human labels available for this video"
        
        prob = human_data.get("probability", 0.5)
        confidence = human_data.get("confidence", 0.5)
        num_raters = human_data.get("num_raters", 0)
        
        consensus = "Lame" if prob > 0.5 else "Sound"
        conf_level = "High" if confidence > 0.7 else "Medium" if confidence > 0.4 else "Low"
        
        return f"Human assessors ({num_raters} raters): {consensus} with {conf_level} confidence ({confidence:.1%})"
    
    def build_prompt(self, fusion_result: Dict[str, Any], 
                     shap_data: Optional[Dict[str, Any]] = None,
                     quality_data: Optional[Dict[str, Any]] = None) -> str:
        """Build the full prompt for LLM explanation"""
        
        # Extract key information
        probability = fusion_result.get("final_probability", 0.5)
        confidence = fusion_result.get("confidence", 0.5)
        prediction_label = "Lame" if probability > 0.5 else "Sound"
        confidence_level = "High" if confidence > 0.7 else "Medium" if confidence > 0.4 else "Low"
        decision_mode = fusion_result.get("decision_mode", "unknown")
        
        # Pipeline summary
        pipeline_contributions = fusion_result.get("pipeline_contributions", {})
        pipeline_summary = self.format_pipeline_summary(pipeline_contributions)
        
        # Quality indicators
        quality_data = quality_data or {}
        clip_quality = quality_data.get("clip_quality", "Unknown")
        pose_quality = quality_data.get("pose_quality", "Unknown")
        detection_conf = quality_data.get("detection_confidence", "Unknown")
        
        # Gait features
        tleap_features = fusion_result.get("tleap_features", {})
        gait_features = self.format_gait_features(tleap_features)
        
        # SHAP features
        shap_features = self.format_shap_features(shap_data or {})
        
        # Human consensus
        human_data = pipeline_contributions.get("human", {})
        human_consensus = self.format_human_consensus(human_data)
        
        # Agreement level
        model_agreement = fusion_result.get("model_agreement", 0.5)
        unanimous = fusion_result.get("unanimous", False)
        agreement_level = "Unanimous" if unanimous else "High" if model_agreement > 0.8 else "Medium" if model_agreement > 0.5 else "Low"
        models_agree = "All models agree" if unanimous else f"{len(pipeline_contributions)} models with {agreement_level.lower()} agreement"
        
        return self.EXPLANATION_TEMPLATE.format(
            prediction_label=prediction_label,
            probability=probability,
            confidence_level=confidence_level,
            confidence=confidence,
            decision_mode=decision_mode,
            pipeline_summary=pipeline_summary,
            clip_quality=clip_quality,
            pose_quality=pose_quality,
            detection_confidence=detection_conf,
            gait_features=gait_features,
            shap_features=shap_features,
            human_consensus=human_consensus,
            agreement_level=agreement_level,
            models_agree=models_agree
        )
    
    async def generate_explanation_openai(self, prompt: str) -> str:
        """Generate explanation using OpenAI API"""
        if not self.openai_client:
            return self.generate_template_explanation(prompt)
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Low temperature for consistency
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return self.generate_template_explanation(prompt)
    
    def generate_template_explanation(self, prompt: str) -> str:
        """Generate a template-based explanation when LLM is unavailable"""
        # Parse the prompt to extract key values
        lines = prompt.split("\n")
        
        prediction = "Unknown"
        confidence_level = "Unknown"
        decision_mode = "unknown"
        
        for line in lines:
            if "Prediction:" in line:
                prediction = "Lame" if "Lame" in line else "Sound"
            if "Confidence:" in line:
                confidence_level = "High" if "High" in line else "Medium" if "Medium" in line else "Low"
            if "Decision Mode:" in line:
                decision_mode = line.split(":")[-1].strip()
        
        # Generate template explanation
        explanation = f"""## Executive Summary
The AI system predicts this cow is **{prediction}** with **{confidence_level}** confidence. """
        
        if decision_mode == "human":
            explanation += "This prediction is primarily based on human expert consensus."
        elif decision_mode == "automated":
            explanation += "This prediction is based on automated pipeline analysis with strong model agreement."
        elif decision_mode == "hybrid":
            explanation += "This prediction combines human assessments with automated analysis."
        else:
            explanation += "More data is recommended to increase prediction confidence."
        
        explanation += """

## Key Evidence
- Multiple pipelines analyzed gait patterns and visual features
- See pipeline contributions above for individual model predictions

## Uncertainties
"""
        if confidence_level == "Low":
            explanation += "- Low confidence indicates limited data or model disagreement\n"
            explanation += "- Additional human labels recommended\n"
        else:
            explanation += "- See individual pipeline uncertainties for details\n"
        
        explanation += """
## Recommended Action
"""
        if prediction == "Lame" and confidence_level == "High":
            explanation += "Recommend veterinary examination at earliest convenience."
        elif prediction == "Lame":
            explanation += "Monitor this cow closely and consider veterinary consultation."
        else:
            explanation += "Continue routine monitoring."
        
        return explanation
    
    async def generate_explanation(self, video_id: str, 
                                   fusion_result: Dict[str, Any],
                                   shap_data: Optional[Dict[str, Any]] = None,
                                   quality_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate complete explanation for a video"""
        
        # Build prompt
        prompt = self.build_prompt(fusion_result, shap_data, quality_data)
        
        # Generate explanation
        if self.openai_client:
            explanation_text = await self.generate_explanation_openai(prompt)
        else:
            explanation_text = self.generate_template_explanation(prompt)
        
        # Parse sections from explanation
        sections = {
            "executive_summary": "",
            "key_evidence": "",
            "uncertainties": "",
            "recommended_action": ""
        }
        
        current_section = None
        for line in explanation_text.split("\n"):
            if "Executive Summary" in line:
                current_section = "executive_summary"
            elif "Key Evidence" in line:
                current_section = "key_evidence"
            elif "Uncertainties" in line:
                current_section = "uncertainties"
            elif "Recommended Action" in line:
                current_section = "recommended_action"
            elif current_section:
                sections[current_section] += line + "\n"
        
        # Clean up sections
        for key in sections:
            sections[key] = sections[key].strip()
        
        result = {
            "video_id": video_id,
            "explanation": explanation_text,
            "sections": sections,
            "prompt_used": prompt,
            "llm_provider": self.llm_provider if self.openai_client else "template",
            "fusion_summary": {
                "prediction": "Lame" if fusion_result.get("final_probability", 0.5) > 0.5 else "Sound",
                "probability": fusion_result.get("final_probability", 0.5),
                "confidence": fusion_result.get("confidence", 0.5),
                "decision_mode": fusion_result.get("decision_mode", "unknown")
            }
        }
        
        # Save explanation
        output_path = self.results_dir / f"{video_id}_explanation.json"
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        
        return result
    
    async def handle_explanation_request(self, data: dict):
        """Handle incoming explanation request"""
        video_id = data.get("video_id")
        if not video_id:
            return
        
        print(f"Generating explanation for {video_id}")
        
        try:
            # Load fusion results
            fusion_path = Path(f"/app/data/results/fusion/{video_id}_fusion.json")
            if not fusion_path.exists():
                print(f"  No fusion results for {video_id}")
                return
            
            with open(fusion_path) as f:
                fusion_data = json.load(f)
            
            fusion_result = fusion_data.get("fusion_result", {})
            
            # Load SHAP data if available
            shap_path = Path(f"/app/data/results/shap/{video_id}_shap.json")
            shap_data = None
            if shap_path.exists():
                with open(shap_path) as f:
                    shap_data = json.load(f)
            
            # Generate explanation
            explanation = await self.generate_explanation(
                video_id, fusion_result, shap_data
            )
            
            print(f"  ✅ Explanation generated for {video_id}")
            
            # Publish result
            await self.nats_client.publish(
                "explanation.generated",
                {
                    "video_id": video_id,
                    "explanation_path": str(self.results_dir / f"{video_id}_explanation.json"),
                    "summary": explanation["sections"]["executive_summary"][:200]
                }
            )
            
        except Exception as e:
            print(f"  ❌ Error generating explanation: {e}")
            import traceback
            traceback.print_exc()
    
    async def start(self):
        """Start the LLM explanation service"""
        await self.nats_client.connect()
        
        # Subscribe to analysis complete events
        subject = self.config.get("nats", {}).get("subjects", {}).get(
            "analysis_complete", "analysis.complete"
        )
        print(f"LLM Explanation Service subscribing to: {subject}")
        
        await self.nats_client.subscribe(subject, self.handle_explanation_request)
        
        print("=" * 60)
        print("LLM Explanation Service Started")
        print("=" * 60)
        print(f"Provider: {self.llm_provider if self.openai_client else 'template'}")
        if self.openai_client:
            print(f"Model: {self.openai_model}")
        print("=" * 60)
        
        await asyncio.Event().wait()


async def main():
    """Main entry point"""
    service = LLMExplanationService()
    await service.start()


if __name__ == "__main__":
    asyncio.run(main())

