"""
Pydantic schemas for API request/response validation.
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from enum import Enum


class GenderEnum(str, Enum):
    """Valid gender values."""
    MALE = "male"
    FEMALE = "female"


class TransportationEnum(str, Enum):
    """Valid transportation values."""
    BUS = "Bus"
    CAR = "Car"


class AccommodationEnum(str, Enum):
    """Valid accommodation values."""
    DORMS = "Dorms"
    WITH_FAMILY = "With familly"  # Fixed: matches original data spelling


class MidExamEnum(str, Enum):
    """Valid mid exam preparation values."""
    REGULARLY = "Regularly during the semester"  # Fixed: matches original data case
    CLOSEST_DAY = "Closest day to the exam"  # Fixed: matches original data case


class TakingNotesEnum(str, Enum):
    """Valid note-taking values."""
    ALWAYS = "Always"
    SOMETIMES = "Sometimes"


class GradeEnum(str, Enum):
    """Valid grade values."""
    A = "A"
    B = "B"
    C = "C"
    D = "D"


class StudentFeatures(BaseModel):
    """
    Input schema for student features prediction request.
    """
    gender: GenderEnum = Field(
        ..., 
        description="Student's gender"
    )
    transportation: TransportationEnum = Field(
        ..., 
        description="Transportation method to university"
    )
    accommodation: AccommodationEnum = Field(
        ..., 
        description="Student's accommodation type"
    )
    mid_exam: MidExamEnum = Field(
        ..., 
        description="Mid-term exam preparation approach"
    )
    taking_notes: TakingNotesEnum = Field(
        ..., 
        description="Note-taking frequency in classes"
    )

    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "gender": "female",
                "transportation": "Bus",
                "accommodation": "Dorms",
                "mid_exam": "Regularly during the semester",
                "taking_notes": "Always"
            }
        }


class PredictionResponse(BaseModel):
    """
    Response schema for grade prediction.
    """
    predicted_grade: GradeEnum = Field(
        ..., 
        description="Predicted grade for the student"
    )
    confidence: Optional[float] = Field(
        None, 
        description="Confidence score for the prediction (0-1)",
        ge=0.0,
        le=1.0
    )
    probabilities: Optional[Dict[str, float]] = Field(
        None, 
        description="Probability distribution across all grades"
    )
    model_info: Optional[Dict[str, Any]] = Field(
        None, 
        description="Information about the model used for prediction"
    )

    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "predicted_grade": "A",
                "confidence": 0.85,
                "probabilities": {
                    "A": 0.85,
                    "B": 0.10,
                    "C": 0.03,
                    "D": 0.02
                },
                "model_info": {
                    "version": "v2",
                    "k_value": 1,
                    "accuracy": 0.89
                }
            }
        }


class ErrorResponse(BaseModel):
    """
    Error response schema.
    """
    error: str = Field(
        ..., 
        description="Error message"
    )
    detail: Optional[str] = Field(
        None, 
        description="Detailed error information"
    )

    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "error": "Validation Error",
                "detail": "Invalid input values provided"
            }
        }


class HealthResponse(BaseModel):
    """
    Health check response schema.
    """
    status: str = Field(
        ..., 
        description="API status"
    )
    model_loaded: bool = Field(
        ..., 
        description="Whether the ML model is loaded and ready"
    )
    version: str = Field(
        ..., 
        description="API version"
    )

    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "version": "1.0.0"
            }
        }