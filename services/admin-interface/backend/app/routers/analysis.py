"""
Analysis endpoints for pipeline results inspection
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pathlib import Path
from typing import Optional, List
import json
import csv
import io

router = APIRouter()

RESULTS_DIR = Path("/app/data/results")
VIDEOS_DIR = Path("/app/data/videos")
CANONICAL_DIR = Path("/app/data/canonical")


@router.get("/{video_id}")
async def get_analysis(video_id: str):
    """Get complete analysis results for a video"""
    fusion_file = RESULTS_DIR / "fusion" / f"{video_id}_fusion.json"
    
    if not fusion_file.exists():
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    with open(fusion_file) as f:
        fusion_data = json.load(f)
    
    # Load individual pipeline results
    pipeline_results = {}
    
    for pipeline in ["yolo", "sam3", "dinov3", "ml", "tleap"]:
        result_file = RESULTS_DIR / pipeline / f"{video_id}_{pipeline}.json"
        if result_file.exists():
            with open(result_file) as f:
                pipeline_results[pipeline] = json.load(f)
    
    return {
        "video_id": video_id,
        "fusion": fusion_data.get("fusion_result", {}),
        "pipelines": pipeline_results
    }


@router.get("/{video_id}/summary")
async def get_analysis_summary(video_id: str):
    """Get analysis summary"""
    fusion_file = RESULTS_DIR / "fusion" / f"{video_id}_fusion.json"

    if not fusion_file.exists():
        raise HTTPException(status_code=404, detail="Analysis not found")

    with open(fusion_file) as f:
        fusion_data = json.load(f)

    fusion_result = fusion_data.get("fusion_result", {})

    return {
        "video_id": video_id,
        "final_probability": fusion_result.get("final_probability", 0.5),
        "final_prediction": fusion_result.get("final_prediction", 0),
        "prediction_label": "lame" if fusion_result.get("final_prediction", 0) == 1 else "sound",
        "pipeline_contributions": fusion_result.get("pipeline_contributions", {})
    }


@router.get("/{video_id}/all")
async def get_all_pipeline_results(video_id: str):
    """Get all pipeline results for a video including Graph Transformer"""
    pipelines = ["yolo", "sam3", "dinov3", "tleap", "tcn", "transformer", "gnn", "graph_transformer", "ml", "fusion"]

    results = {
        "video_id": video_id,
        "pipelines": {}
    }

    for pipeline in pipelines:
        result_file = RESULTS_DIR / pipeline / f"{video_id}_{pipeline}.json"
        if result_file.exists():
            with open(result_file) as f:
                results["pipelines"][pipeline] = {
                    "status": "success",
                    "data": json.load(f)
                }
        else:
            results["pipelines"][pipeline] = {
                "status": "not_available",
                "data": None
            }

    # Also check for SHAP results
    shap_file = RESULTS_DIR / "shap" / f"{video_id}_shap.json"
    if shap_file.exists():
        with open(shap_file) as f:
            results["pipelines"]["shap"] = {
                "status": "success",
                "data": json.load(f)
            }

    # Check for LLM explanation
    explanation_file = RESULTS_DIR / "explanations" / f"{video_id}_explanation.json"
    if explanation_file.exists():
        with open(explanation_file) as f:
            results["pipelines"]["explanation"] = {
                "status": "success",
                "data": json.load(f)
            }
    else:
        results["pipelines"]["explanation"] = {
            "status": "not_available",
            "data": None
        }

    return results


@router.get("/{video_id}/graph_transformer")
async def get_graph_transformer_results(video_id: str):
    """Get Graph Transformer (Graphormer) results for a video"""
    result_file = RESULTS_DIR / "graph_transformer" / f"{video_id}_graph_transformer.json"

    if not result_file.exists():
        raise HTTPException(status_code=404, detail="Graph Transformer results not found")

    with open(result_file) as f:
        data = json.load(f)

    return {
        "video_id": video_id,
        "pipeline": "graph_transformer",
        "model": data.get("model", "CowLamenessGraphormer"),
        "graph_prediction": data.get("graph_prediction", 0.5),
        "node_prediction": data.get("node_prediction", 0.5),
        "uncertainty": data.get("uncertainty", 0.0),
        "prediction": data.get("prediction", 0),
        "confidence": data.get("confidence", 0.5),
        "graph_info": data.get("graph_info", {}),
        "attention_info": data.get("attention_info", {})
    }


@router.get("/{video_id}/frames/{frame_num}")
async def get_frame_data(video_id: str, frame_num: int):
    """Get per-frame data from all pipelines for a specific frame"""
    frame_data = {
        "video_id": video_id,
        "frame": frame_num,
        "detections": [],
        "pose_keypoints": [],
        "mask_coverage": None
    }

    # Get YOLO detections for this frame
    yolo_file = RESULTS_DIR / "yolo" / f"{video_id}_yolo.json"
    if yolo_file.exists():
        with open(yolo_file) as f:
            yolo_data = json.load(f)
            for det in yolo_data.get("detections", []):
                if det.get("frame") == frame_num:
                    frame_data["detections"] = det.get("detections", [])
                    break

    # Get T-LEAP pose keypoints for this frame
    tleap_file = RESULTS_DIR / "tleap" / f"{video_id}_tleap.json"
    if tleap_file.exists():
        with open(tleap_file) as f:
            tleap_data = json.load(f)
            for pose in tleap_data.get("pose_sequences", []):
                if pose.get("frame") == frame_num:
                    frame_data["pose_keypoints"] = pose.get("keypoints", [])
                    frame_data["pose_bbox"] = pose.get("bbox", [])
                    break

    # Get SAM3 mask coverage for this frame
    sam3_file = RESULTS_DIR / "sam3" / f"{video_id}_sam3.json"
    if sam3_file.exists():
        with open(sam3_file) as f:
            sam3_data = json.load(f)
            for seg in sam3_data.get("segmentations", []):
                if seg.get("frame") == frame_num:
                    frame_data["mask_coverage"] = seg.get("mask_available", False)
                    break

    return frame_data


@router.get("/{video_id}/export")
async def export_analysis(video_id: str, format: str = "json"):
    """Export all pipeline results in specified format (json, csv)"""
    if format not in ["json", "csv"]:
        raise HTTPException(status_code=400, detail="Format must be 'json' or 'csv'")

    # Collect all results
    all_results = await get_all_pipeline_results(video_id)

    if format == "json":
        content = json.dumps(all_results, indent=2)
        return StreamingResponse(
            io.BytesIO(content.encode()),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename={video_id}_analysis.json"}
        )

    elif format == "csv":
        # Flatten results to CSV format
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow(["video_id", "pipeline", "metric", "value"])

        for pipeline, data in all_results["pipelines"].items():
            if data["status"] == "success" and data["data"]:
                _flatten_to_csv(writer, video_id, pipeline, data["data"])

        output.seek(0)
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={video_id}_analysis.csv"}
        )


def _flatten_to_csv(writer, video_id: str, pipeline: str, data: dict, prefix: str = ""):
    """Recursively flatten nested dict to CSV rows"""
    for key, value in data.items():
        metric_name = f"{prefix}{key}" if prefix else key
        if isinstance(value, dict):
            _flatten_to_csv(writer, video_id, pipeline, value, f"{metric_name}.")
        elif isinstance(value, list):
            if len(value) > 0 and isinstance(value[0], dict):
                # Skip complex nested arrays
                writer.writerow([video_id, pipeline, f"{metric_name}_count", len(value)])
            else:
                writer.writerow([video_id, pipeline, metric_name, str(value)])
        else:
            writer.writerow([video_id, pipeline, metric_name, value])


@router.post("/batch")
async def get_batch_analysis(video_ids: List[str], pipelines: Optional[List[str]] = None):
    """Get analysis results for multiple videos at once"""
    if pipelines is None:
        pipelines = ["yolo", "sam3", "dinov3", "tleap", "tcn", "transformer", "gnn", "graph_transformer", "ml", "fusion"]

    results = {}
    for video_id in video_ids:
        results[video_id] = {"pipelines": {}}
        for pipeline in pipelines:
            result_file = RESULTS_DIR / pipeline / f"{video_id}_{pipeline}.json"
            if result_file.exists():
                with open(result_file) as f:
                    results[video_id]["pipelines"][pipeline] = {
                        "status": "success",
                        "data": json.load(f)
                    }
            else:
                results[video_id]["pipelines"][pipeline] = {
                    "status": "not_available",
                    "data": None
                }

    return {
        "count": len(video_ids),
        "results": results
    }


@router.get("/{video_id}/explanation")
async def get_llm_explanation(video_id: str):
    """Get LLM-generated explanation for a video analysis"""
    explanation_file = RESULTS_DIR / "explanations" / f"{video_id}_explanation.json"
    
    if not explanation_file.exists():
        # Try to generate a basic explanation from fusion results
        fusion_file = RESULTS_DIR / "fusion" / f"{video_id}_fusion.json"
        if not fusion_file.exists():
            raise HTTPException(status_code=404, detail="No analysis results found for this video")
        
        # Return a placeholder indicating LLM explanation not yet generated
        with open(fusion_file) as f:
            fusion_data = json.load(f)
        
        fusion_result = fusion_data.get("fusion_result", {})
        probability = fusion_result.get("final_probability", 0.5)
        prediction = "Lame" if probability > 0.5 else "Sound"
        confidence = fusion_result.get("confidence", 0.5)
        
        return {
            "video_id": video_id,
            "status": "pending",
            "message": "LLM explanation not yet generated. Showing basic summary.",
            "explanation": f"The AI system predicts this cow is **{prediction}** with {confidence:.0%} confidence.",
            "sections": {
                "executive_summary": f"Prediction: {prediction} ({probability:.1%} probability)",
                "key_evidence": "See pipeline contributions in the analysis results.",
                "uncertainties": "Full LLM analysis pending.",
                "recommended_action": "Monitor cow and request full analysis if needed."
            },
            "llm_provider": "none",
            "fusion_summary": {
                "prediction": prediction,
                "probability": probability,
                "confidence": confidence,
                "decision_mode": fusion_result.get("decision_mode", "unknown")
            }
        }
    
    with open(explanation_file) as f:
        return json.load(f)


@router.post("/{video_id}/explanation/generate")
async def request_explanation_generation(video_id: str):
    """Request LLM explanation generation for a video (triggers via NATS)"""
    fusion_file = RESULTS_DIR / "fusion" / f"{video_id}_fusion.json"
    
    if not fusion_file.exists():
        raise HTTPException(status_code=404, detail="No fusion results found. Run analysis first.")
    
    # In a real implementation, this would publish to NATS to trigger LLM service
    # For now, return status indicating the request was received
    return {
        "video_id": video_id,
        "status": "requested",
        "message": "Explanation generation requested. Check back shortly."
    }


@router.get("/{video_id}/{pipeline}")
async def get_pipeline_result(video_id: str, pipeline: str):
    """Get individual pipeline result for a video"""
    valid_pipelines = ["yolo", "sam3", "dinov3", "tleap", "tcn", "transformer", "gnn", "graph_transformer", "ml", "fusion", "shap", "explanation"]

    if pipeline not in valid_pipelines:
        raise HTTPException(status_code=400, detail=f"Invalid pipeline. Must be one of: {valid_pipelines}")

    # Handle explanation specially
    if pipeline == "explanation":
        return await get_llm_explanation(video_id)

    result_file = RESULTS_DIR / pipeline / f"{video_id}_{pipeline}.json"

    if not result_file.exists():
        raise HTTPException(status_code=404, detail=f"No {pipeline} results found for this video")

    with open(result_file) as f:
        return json.load(f)

