"""
FastAPI Application for Hospital Readmission Prediction MLOps Pipeline

This module provides a REST API that runs the complete MLOps pipeline
from raw data to trained model, returning training metrics and metadata.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Import the actual project components
from src.main import main as run_pipeline
from src.data_loader.data_loader import DataLoader
from src.validation.data_validator import DataValidator
from src.eda.eda import EDAAnalyzer
from src.preprocessing.preprocessor import Preprocessor
from src.features.feature_engineering import FeatureEngineer
from src.model.trainer import ModelTrainer
from src.evaluation.evaluator import ModelEvaluator
from src.inference.predict import ModelPredictor
from src.utils import load_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Hospital Readmission Prediction MLOps Pipeline API",
    description="API for running the complete MLOps pipeline from raw data to trained model with metrics",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
pipeline_running = False
pipeline_results: Optional[Dict[str, Any]] = None
pipeline_error: Optional[str] = None

# Pydantic models for request/response
class PipelineRequest(BaseModel):
    """Request model for running the MLOps pipeline."""
    force_retrain: bool = Field(False, description="Force retraining even if model exists")
    data_source: str = Field(..., description="Path to raw data file (relative or absolute)")

class PipelineResponse(BaseModel):
    """Response model for pipeline execution."""
    status: str = Field(..., description="Pipeline status")
    message: str = Field(..., description="Status message")
    timestamp: str = Field(..., description="Execution timestamp")
    results: Optional[Dict[str, Any]] = Field(None, description="Pipeline results")
    error: Optional[str] = Field(None, description="Error message if failed")

class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str = Field(..., description="Service status")
    pipeline_running: bool = Field(..., description="Whether pipeline is currently running")
    pipeline_completed: bool = Field(..., description="Whether pipeline has completed")
    timestamp: str = Field(..., description="Current timestamp")
    last_results: Optional[Dict[str, Any]] = Field(None, description="Last pipeline results")

class DataInfoResponse(BaseModel):
    """Response model for data information endpoint."""
    data_source: str = Field(..., description="Data source path")
    data_exists: bool = Field(..., description="Whether data file exists")
    data_size: Optional[str] = Field(None, description="Data file size")
    data_shape: Optional[Dict[str, int]] = Field(None, description="Data shape (rows, columns)")
    columns: Optional[List[str]] = Field(None, description="Data columns")
    timestamp: str = Field(..., description="Check timestamp")

def run_pipeline_async(data_source: str, force_retrain: bool = False) -> Dict[str, Any]:
    """Run the MLOps pipeline asynchronously."""
    global pipeline_running, pipeline_results, pipeline_error
    
    try:
        pipeline_running = True
        pipeline_error = None
        
        logger.info(f"🚀 Starting MLOps pipeline with data source: {data_source}")
        
        # Check if data file exists
        if not os.path.exists(data_source):
            raise FileNotFoundError(f"Data file not found: {data_source}")
        
        # Run the complete pipeline, passing the data_source to main if possible
        # If main() does not accept data_source, you may need to modify main.py to accept it
        results = run_pipeline(data_source=data_source) if 'data_source' in run_pipeline.__code__.co_varnames else run_pipeline()
        
        # Attach the data_source to results for caching
        if isinstance(results, dict):
            results['data_source'] = data_source
        pipeline_results = results
        pipeline_running = False
        
        logger.info("✅ Pipeline completed successfully")
        return results
        
    except Exception as e:
        pipeline_error = str(e)
        pipeline_running = False
        logger.error(f"❌ Pipeline failed: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize the API on startup."""
    logger.info("🚀 Starting Hospital Readmission Prediction MLOps Pipeline API...")
    
    # Check if data directory exists
    data_dir = Path("data/raw")
    if not data_dir.exists():
        logger.warning("Data directory not found. Creating it...")
        data_dir.mkdir(parents=True, exist_ok=True)

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Hospital Readmission Prediction MLOps Pipeline API",
        "version": "1.0.0",
        "description": "Run the complete MLOps pipeline from raw data to trained model",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "run_pipeline": "/run-pipeline",
            "data_info": "/data-info",
            "results": "/results"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        pipeline_running=pipeline_running,
        pipeline_completed=pipeline_results is not None,
        timestamp=datetime.now().isoformat(),
        last_results=pipeline_results
    )

@app.get("/data-info", response_model=DataInfoResponse)
async def get_data_info(data_source: str = "data/raw/diabetic_readmission_data.csv"):
    """Get information about the raw data. Accepts a file path as a query parameter."""
    try:
        data_exists = os.path.exists(data_source)
        data_size = None
        data_shape = None
        columns = None
        
        if data_exists:
            # Get file size
            file_size = os.path.getsize(data_source)
            data_size = f"{file_size / (1024*1024):.2f} MB"
            
            # Get data shape and columns
            try:
                df = pd.read_csv(data_source, nrows=1)  # Just read header for columns
                columns = list(df.columns)
                
                # Get total rows (efficiently)
                with open(data_source, 'r') as f:
                    total_rows = sum(1 for line in f) - 1  # Subtract header
                
                data_shape = {"rows": total_rows, "columns": len(columns)}
            except Exception as e:
                logger.warning(f"Could not read data info: {e}")
        
        return DataInfoResponse(
            data_source=data_source,
            data_exists=data_exists,
            data_size=data_size,
            data_shape=data_shape,
            columns=columns,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error getting data info: {e}")
        return DataInfoResponse(
            data_source=data_source,
            data_exists=False,
            data_size=None,
            data_shape=None,
            columns=None,
            timestamp=datetime.now().isoformat()
        )

@app.post("/run-pipeline", response_model=PipelineResponse)
async def run_mlops_pipeline(request: PipelineRequest, background_tasks: BackgroundTasks):
    """Run the complete MLOps pipeline from raw data. Requires data_source in request body."""
    global pipeline_running, pipeline_results, pipeline_error
    
    if pipeline_running:
        raise HTTPException(
            status_code=409,
            detail="Pipeline is already running. Please wait for completion."
        )
    
    # Check if data file exists
    if not os.path.exists(request.data_source):
        raise HTTPException(
            status_code=404,
            detail=f"Data file not found: {request.data_source}"
        )
    
    # Reset previous results if force retrain or data_source changes
    if request.force_retrain or (pipeline_results and pipeline_results.get('data_source') != request.data_source):
        pipeline_results = None
        pipeline_error = None
    
    # If we have recent results for this data_source and not forcing retrain, return them
    if pipeline_results is not None and not request.force_retrain and pipeline_results.get('data_source') == request.data_source:
        return PipelineResponse(
            status="completed",
            message="Using existing pipeline results",
            timestamp=datetime.now().isoformat(),
            results=pipeline_results,
            error=None
        )
    
    # Run pipeline in background with the provided data_source
    background_tasks.add_task(run_pipeline_async, request.data_source, request.force_retrain)
    
    return PipelineResponse(
        status="started",
        message="Pipeline started successfully. Check /health for status.",
        timestamp=datetime.now().isoformat(),
        results=None,
        error=None
    )

@app.get("/results", response_model=PipelineResponse)
async def get_pipeline_results():
    """Get the latest pipeline results."""
    if pipeline_running:
        return PipelineResponse(
            status="running",
            message="Pipeline is currently running. Please wait for completion.",
            timestamp=datetime.now().isoformat(),
            results=None,
            error=None
        )
    
    if pipeline_error:
        return PipelineResponse(
            status="failed",
            message="Pipeline failed",
            timestamp=datetime.now().isoformat(),
            results=None,
            error=pipeline_error
        )
    
    if pipeline_results is None:
        return PipelineResponse(
            status="not_run",
            message="Pipeline has not been run yet. Use /run-pipeline to start.",
            timestamp=datetime.now().isoformat(),
            results=None,
            error=None
        )
    
    return PipelineResponse(
        status="completed",
        message="Pipeline completed successfully",
        timestamp=datetime.now().isoformat(),
        results=pipeline_results,
        error=None
    )

@app.get("/metrics")
async def get_metrics():
    """Get detailed metrics from the latest pipeline run."""
    if pipeline_results is None:
        raise HTTPException(
            status_code=404,
            detail="No pipeline results available. Run the pipeline first."
        )
    
    metrics = pipeline_results.get("metrics", {})
    return {
        "metrics": metrics,
        "data_shapes": pipeline_results.get("data_shapes", {}),
        "model_path": pipeline_results.get("model_path", ""),
        "inference_results": pipeline_results.get("inference_results", {}),
        "comparison": pipeline_results.get("comparison", {}),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model-info")
async def get_model_info():
    """Get detailed model information from the latest pipeline run."""
    if pipeline_results is None:
        raise HTTPException(
            status_code=404,
            detail="No pipeline results available. Run the pipeline first."
        )
    
    model_path = pipeline_results.get("model_path", "")
    
    if not model_path or not os.path.exists(model_path):
        return {
            "model_exists": False,
            "model_path": model_path,
            "message": "Model file not found"
        }
    
    # Get model file info
    model_stats = os.stat(model_path)
    
    return {
        "model_exists": True,
        "model_path": model_path,
        "model_size": f"{model_stats.st_size / (1024*1024):.2f} MB",
        "model_created": datetime.fromtimestamp(model_stats.st_ctime).isoformat(),
        "model_modified": datetime.fromtimestamp(model_stats.st_mtime).isoformat(),
        "metrics": pipeline_results.get("metrics", {}),
        "data_shapes": pipeline_results.get("data_shapes", {}),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict-sample")
async def predict_sample():
    """Make a sample prediction using the trained model."""
    if pipeline_results is None:
        raise HTTPException(
            status_code=404,
            detail="No trained model available. Run the pipeline first."
        )
    
    model_path = pipeline_results.get("model_path", "")
    if not model_path or not os.path.exists(model_path):
        raise HTTPException(
            status_code=404,
            detail="Model file not found. Run the pipeline first."
        )
    
    try:
        # Load the predictor
        predictor = ModelPredictor(config_path="src/config.yaml")
        predictor.load_model(model_path)
        
        # Get a sample from the test data for prediction
        # This would require loading the test data, but for now we'll return a message
        return {
            "message": "Sample prediction endpoint ready",
            "model_loaded": True,
            "model_path": model_path,
            "note": "To make actual predictions, load test data and use predictor.predict()",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error loading model: {str(e)}"
        )

if __name__ == "__main__":
    # For local development
    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 