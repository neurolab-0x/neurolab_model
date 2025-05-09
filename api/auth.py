import os
import logging
import secrets
import time
import hashlib
from fastapi import APIRouter, HTTPException, Depends, status, Header
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from utils.security import JWTAuth
from api.security import require_admin_role, get_current_user
from config.settings import SECURITY_CONFIG

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize JWT authentication handler
jwt_auth = JWTAuth(token_expiry=SECURITY_CONFIG['token_expiry_hours'])

# In-memory user database for demonstration
# In production, use a proper user database
USERS_DB = {
    "admin": {
        "username": "admin",
        "password_hash": hashlib.sha256("admin_secure_password".encode()).hexdigest(),
        "roles": ["admin", "user"],
        "created_at": datetime.utcnow().isoformat(),
    },
    "user": {
        "username": "user",
        "password_hash": hashlib.sha256("user_secure_password".encode()).hexdigest(),
        "roles": ["user"],
        "created_at": datetime.utcnow().isoformat(),
    }
}

# Store refresh tokens (in production, use Redis or a database)
REFRESH_TOKENS = {}


class UserLogin(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8, max_length=100)
    
    @validator('username')
    def validate_username(cls, v):
        if not v.isalnum() and not all(c.isalnum() or c in '_-.' for c in v):
            raise ValueError("Username must contain only alphanumeric characters, underscores, hyphens, or dots")
        return v

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user_info: Dict[str, Any]

class RefreshTokenRequest(BaseModel):
    refresh_token: str

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8, max_length=100)
    roles: List[str] = Field(default=["user"])
    
    @validator('username')
    def validate_username(cls, v):
        if not all(c.isalnum() or c in '_-.' for c in v):
            raise ValueError("Username must contain only alphanumeric characters, underscores, hyphens, or dots")
        return v
    
    @validator('roles')
    def validate_roles(cls, v):
        valid_roles = ["user", "admin"]
        for role in v:
            if role not in valid_roles:
                raise ValueError(f"Invalid role: {role}. Valid roles are: {valid_roles}")
        return v


@router.post("/api/auth/login", response_model=TokenResponse)
async def login(user_data: UserLogin):
    """
    User login endpoint that returns JWT access and refresh tokens
    """
    # Find user in database
    user = USERS_DB.get(user_data.username)
    if not user:
        # Use constant time comparison to prevent timing attacks
        # Compare to a dummy hash to maintain consistent timing
        dummy_hash = hashlib.sha256(b"dummy_password")
        secrets.compare_digest(
            hashlib.sha256(user_data.password.encode()).hexdigest(), 
            dummy_hash.hexdigest()
        )
        logger.warning(f"Login attempt with non-existent username: {user_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )
    
    # Verify password using constant-time comparison
    password_hash = hashlib.sha256(user_data.password.encode()).hexdigest()
    if not secrets.compare_digest(password_hash, user["password_hash"]):
        logger.warning(f"Failed login attempt for user: {user_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )
    
    # Generate access token
    access_token = jwt_auth.create_token(user["username"], user["roles"])
    
    # Generate refresh token
    refresh_token = secrets.token_urlsafe(32)
    
    # Store refresh token
    REFRESH_TOKENS[refresh_token] = {
        "username": user["username"],
        "expires_at": datetime.utcnow() + timedelta(days=SECURITY_CONFIG['refresh_token_expiry_days']),
        "created_at": datetime.utcnow()
    }
    
    logger.info(f"User {user_data.username} logged in successfully")
    
    # Return tokens and user info
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": SECURITY_CONFIG['token_expiry_hours'] * 3600,  # in seconds
        "user_info": {
            "username": user["username"],
            "roles": user["roles"]
        }
    }

@router.post("/api/auth/refresh", response_model=TokenResponse)
async def refresh_token(request: RefreshTokenRequest):
    """
    Refresh access token using a valid refresh token
    """
    # Check if refresh token exists
    token_data = REFRESH_TOKENS.get(request.refresh_token)
    if not token_data:
        logger.warning("Invalid refresh token used")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    
    # Check if refresh token has expired
    if datetime.utcnow() > token_data["expires_at"]:
        # Remove expired token
        REFRESH_TOKENS.pop(request.refresh_token, None)
        logger.warning("Expired refresh token used")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token has expired"
        )
    
    # Get user data
    username = token_data["username"]
    user = USERS_DB.get(username)
    
    if not user:
        # This should never happen unless the user was deleted
        REFRESH_TOKENS.pop(request.refresh_token, None)
        logger.error(f"Refresh token references non-existent user: {username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user"
        )
    
    # Generate new access token
    new_access_token = jwt_auth.create_token(user["username"], user["roles"])
    
    # Generate new refresh token (rotate tokens for security)
    new_refresh_token = secrets.token_urlsafe(32)
    
    # Remove the old refresh token and store the new one
    REFRESH_TOKENS.pop(request.refresh_token, None)
    REFRESH_TOKENS[new_refresh_token] = {
        "username": username,
        "expires_at": datetime.utcnow() + timedelta(days=SECURITY_CONFIG['refresh_token_expiry_days']),
        "created_at": datetime.utcnow()
    }
    
    logger.info(f"Refreshed token for user {username}")
    
    # Return new tokens
    return {
        "access_token": new_access_token,
        "refresh_token": new_refresh_token,
        "token_type": "bearer",
        "expires_in": SECURITY_CONFIG['token_expiry_hours'] * 3600,  # in seconds
        "user_info": {
            "username": user["username"],
            "roles": user["roles"]
        }
    }

@router.post("/api/auth/logout")
async def logout(
    refresh_token: Optional[str] = Header(None),
    current_user: Dict = Depends(get_current_user)
):
    """
    Logout endpoint to invalidate refresh tokens
    """
    username = current_user["sub"]
    
    # Remove specific refresh token if provided
    if refresh_token and refresh_token in REFRESH_TOKENS:
        if REFRESH_TOKENS[refresh_token]["username"] == username:
            REFRESH_TOKENS.pop(refresh_token, None)
    
    # Remove all refresh tokens for this user (logout from all devices)
    tokens_to_remove = []
    for token, data in REFRESH_TOKENS.items():
        if data["username"] == username:
            tokens_to_remove.append(token)
    
    for token in tokens_to_remove:
        REFRESH_TOKENS.pop(token, None)
    
    logger.info(f"User {username} logged out successfully")
    
    return {"status": "success", "message": "Logged out successfully"}

@router.post("/api/auth/users", status_code=status.HTTP_201_CREATED)
async def create_user(user_data: UserCreate, current_user: Dict = Depends(require_admin_role)):
    """
    Create a new user (admin only)
    """
    # Check if username already exists
    if user_data.username in USERS_DB:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already exists"
        )
    
    # Create new user
    new_user = {
        "username": user_data.username,
        "password_hash": hashlib.sha256(user_data.password.encode()).hexdigest(),
        "roles": user_data.roles,
        "created_at": datetime.utcnow().isoformat(),
        "created_by": current_user["sub"]
    }
    
    # Add to database
    USERS_DB[user_data.username] = new_user
    
    logger.info(f"Admin {current_user['sub']} created new user: {user_data.username}")
    
    # Return user info (without password)
    return {
        "username": new_user["username"],
        "roles": new_user["roles"],
        "created_at": new_user["created_at"]
    } 