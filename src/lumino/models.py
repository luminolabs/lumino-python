from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class UserStatus(str, Enum):
    """Enumeration of possible user statuses."""
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"


class ApiKeyStatus(str, Enum):
    """Enumeration of possible API key statuses."""
    ACTIVE = "ACTIVE"
    EXPIRED = "EXPIRED"
    REVOKED = "REVOKED"


class DatasetStatus(str, Enum):
    """Enumeration of possible dataset statuses."""
    UPLOADED = "UPLOADED"
    VALIDATED = "VALIDATED"
    ERROR = "ERROR"
    DELETED = "DELETED"


class FineTuningJobStatus(str, Enum):
    """Enumeration of possible fine-tuning job statuses."""
    NEW = "NEW"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"


class BaseModelStatus(str, Enum):
    """Enumeration of possible base model statuses."""
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    DEPRECATED = "DEPRECATED"


class Pagination(BaseModel):
    """Model for pagination information."""
    total_pages: int
    current_page: int
    items_per_page: int


class UserUpdate(BaseModel):
    """Model for updating user information."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    email: Optional[EmailStr] = None


class UserResponse(BaseModel):
    """Model for user response data."""
    id: str
    created_at: datetime
    updated_at: datetime
    status: UserStatus
    name: str
    email: EmailStr


class ApiKeyCreate(BaseModel):
    """Model for creating a new API key."""
    name: str = Field(..., min_length=1, max_length=255)
    expires_at: datetime

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ApiKeyUpdate(BaseModel):
    """Model for updating an API key."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    expires_at: Optional[datetime] = None


class ApiKeyResponse(BaseModel):
    """Model for API key response data."""
    id: str
    created_at: datetime
    last_used_at: Optional[datetime]
    expires_at: datetime
    status: ApiKeyStatus
    name: str
    prefix: str


class ApiKeyWithSecretResponse(ApiKeyResponse):
    """Model for API key response data including the secret."""
    secret: str


class DatasetCreate(BaseModel):
    """Model for creating a new dataset."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)


class DatasetUpdate(BaseModel):
    """Model for updating a dataset."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)


class DatasetResponse(BaseModel):
    """Model for dataset response data."""
    id: str
    created_at: datetime
    updated_at: datetime
    status: DatasetStatus
    name: str
    description: Optional[str]
    file_name: str
    file_size: int
    errors: Optional[Dict[str, Any]]


class FineTuningJobParameters(BaseModel):
    """Model for fine-tuning job parameters."""
    batch_size: int = Field(default=2, gt=0, le=8)
    shuffle: bool = Field(default=True)
    num_epochs: int = Field(default=1, gt=0, le=10)
    use_lora: bool = Field(default=True)
    use_qlora: bool = Field(default=False)


class FineTuningJobCreate(BaseModel):
    """Model for creating a new fine-tuning job."""
    base_model_name: str
    dataset_name: str
    name: str = Field(..., min_length=1, max_length=255)
    parameters: FineTuningJobParameters


class FineTuningJobResponse(BaseModel):
    """Model for fine-tuning job response data."""
    id: str
    created_at: datetime
    updated_at: datetime
    base_model_name: str
    dataset_name: str
    status: FineTuningJobStatus
    name: str
    current_step: Optional[int]
    total_steps: Optional[int]
    current_epoch: Optional[int]
    total_epochs: Optional[int]
    num_tokens: Optional[int]


class FineTuningJobDetailResponse(FineTuningJobResponse):
    """Model for detailed fine-tuning job response data."""
    parameters: Dict[str, Any]
    metrics: Optional[Dict[str, Any]]


class BaseModelResponse(BaseModel):
    """Model for base model response data."""
    id: str
    description: Optional[str]
    hf_url: str
    status: BaseModelStatus
    name: str
    meta: Optional[Dict[str, Any]]


class FineTunedModelResponse(BaseModel):
    """Model for fine-tuned model response data."""
    id: str
    created_at: datetime
    fine_tuning_job_name: str
    name: str
    description: Optional[str]
    artifacts: Optional[Dict[str, Any]]


class UsageRecordResponse(BaseModel):
    """Model for usage record response data."""
    id: str
    created_at: datetime
    service_name: str
    usage_amount: float
    cost: float
    fine_tuning_job_name: str


class TotalCostResponse(BaseModel):
    """Model for total cost response data."""
    start_date: datetime
    end_date: datetime
    total_cost: float


class ListResponse(BaseModel):
    """Model for list response data."""
    data: List[Any]
    pagination: Pagination
