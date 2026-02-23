"""
Authentication endpoints
Handles user registration, login, logout, and token management
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime, timedelta
import uuid
import os

from app.database import get_db, User, Session, Tenant
from app.middleware.auth import (
    get_password_hash,
    verify_password,
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_token,
    get_current_user,
    UserResponse,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    REFRESH_TOKEN_EXPIRE_DAYS
)

router = APIRouter()

DEFAULT_TENANT_NAME = os.getenv("DEFAULT_TENANT_NAME", "Farm1")


async def _get_or_create_default_tenant(db: AsyncSession) -> Tenant:
    result = await db.execute(select(Tenant).where(Tenant.name == DEFAULT_TENANT_NAME))
    tenant = result.scalar_one_or_none()
    if tenant:
        return tenant

    tenant = Tenant(id=uuid.uuid4(), name=DEFAULT_TENANT_NAME, is_active=True)
    db.add(tenant)
    await db.commit()
    await db.refresh(tenant)
    return tenant


# ============== REQUEST/RESPONSE MODELS ==============

class UserCreate(BaseModel):
    """User registration request"""
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=100)
    password: str = Field(..., min_length=8)
    role: Optional[str] = "rater"


class UserLogin(BaseModel):
    """User login request"""
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    """Token response"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = ACCESS_TOKEN_EXPIRE_MINUTES * 60


class RefreshRequest(BaseModel):
    """Token refresh request"""
    refresh_token: str


class PasswordChange(BaseModel):
    """Password change request"""
    current_password: str
    new_password: str = Field(..., min_length=8)


# ============== ENDPOINTS ==============

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Register a new user account.
    Default role is 'rater'. Admin role requires admin approval.
    """
    # Check if email already exists
    result = await db.execute(
        select(User).where(User.email == user_data.email)
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    # Check if username already exists
    result = await db.execute(
        select(User).where(User.username == user_data.username)
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already taken"
        )

    # Validate role (only allow rater for self-registration)
    role = user_data.role if user_data.role in ["rater"] else "rater"

    # Create user
    default_tenant = await _get_or_create_default_tenant(db)
    user = User(
        id=uuid.uuid4(),
        email=user_data.email,
        username=user_data.username,
        password_hash=get_password_hash(user_data.password),
        role=role,
        is_active=True,
        rater_tier="bronze" if role == "rater" else None,
        tenant_id=default_tenant.id,
        created_at=datetime.utcnow()
    )

    db.add(user)
    await db.commit()
    await db.refresh(user)

    return UserResponse(
        id=str(user.id),
        email=user.email,
        username=user.username,
        role=user.role,
        is_active=user.is_active,
        rater_tier=user.rater_tier,
        tenant_id=str(user.tenant_id) if user.tenant_id else None,
        created_at=user.created_at,
        last_login=user.last_login
    )


@router.post("/login", response_model=TokenResponse)
async def login(
    credentials: UserLogin,
    db: AsyncSession = Depends(get_db)
):
    """
    Authenticate user and return JWT tokens.
    """
    # Find user by email
    result = await db.execute(
        select(User).where(User.email == credentials.email)
    )
    user = result.scalar_one_or_none()

    if not user or not verify_password(credentials.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled"
        )

    # Create tokens
    token_data = {
        "sub": str(user.id),
        "email": user.email,
        "username": user.username,
        "role": user.role
    }

    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token(token_data)

    # Store refresh token hash in session
    session = Session(
        id=uuid.uuid4(),
        user_id=user.id,
        token_hash=hash_token(refresh_token),
        expires_at=datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
        created_at=datetime.utcnow()
    )
    db.add(session)

    # Update last login
    user.last_login = datetime.utcnow()
    await db.commit()

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token
    )


@router.post("/logout")
async def logout(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Logout user and invalidate all sessions.
    """
    # Delete all sessions for this user
    result = await db.execute(
        select(Session).where(Session.user_id == user.id)
    )
    sessions = result.scalars().all()

    for session in sessions:
        await db.delete(session)

    await db.commit()

    return {"message": "Successfully logged out"}


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    request: RefreshRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Refresh access token using refresh token.
    """
    # Decode refresh token
    token_data = decode_token(request.refresh_token)
    if not token_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )

    # Verify token hash exists in sessions
    token_hash = hash_token(request.refresh_token)
    result = await db.execute(
        select(Session).where(
            Session.token_hash == token_hash,
            Session.expires_at > datetime.utcnow()
        )
    )
    session = result.scalar_one_or_none()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token expired or revoked"
        )

    # Get user
    result = await db.execute(
        select(User).where(User.id == session.user_id)
    )
    user = result.scalar_one_or_none()

    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or disabled"
        )

    # Create new tokens
    new_token_data = {
        "sub": str(user.id),
        "email": user.email,
        "username": user.username,
        "role": user.role
    }

    new_access_token = create_access_token(new_token_data)
    new_refresh_token = create_refresh_token(new_token_data)

    # Update session with new refresh token hash
    session.token_hash = hash_token(new_refresh_token)
    session.expires_at = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    await db.commit()

    return TokenResponse(
        access_token=new_access_token,
        refresh_token=new_refresh_token
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get current authenticated user information.
    """
    tenant_name = None
    if user.tenant_id:
        t = await db.execute(select(Tenant.name).where(Tenant.id == user.tenant_id))
        row = t.one_or_none()
        tenant_name = row[0] if row else None
    return UserResponse(
        id=str(user.id),
        email=user.email,
        username=user.username,
        role=user.role,
        is_active=user.is_active,
        rater_tier=user.rater_tier,
        tenant_id=str(user.tenant_id) if user.tenant_id else None,
        tenant_name=tenant_name,
        created_at=user.created_at,
        last_login=user.last_login
    )


@router.put("/password")
async def change_password(
    password_data: PasswordChange,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Change current user's password.
    """
    # Verify current password
    if not verify_password(password_data.current_password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect"
        )

    # Update password
    user.password_hash = get_password_hash(password_data.new_password)
    await db.commit()

    # Invalidate all sessions (force re-login)
    result = await db.execute(
        select(Session).where(Session.user_id == user.id)
    )
    sessions = result.scalars().all()
    for session in sessions:
        await db.delete(session)
    await db.commit()

    return {"message": "Password changed successfully. Please login again."}


@router.get("/tenants")
async def list_tenants(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    List all tenants (admin only). Used e.g. when creating a researcher to pick their tenant.
    """
    if user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    result = await db.execute(select(Tenant).where(Tenant.is_active == True).order_by(Tenant.name))
    tenants = result.scalars().all()
    return [{"id": str(t.id), "name": t.name} for t in tenants]


@router.get("/users", response_model=list[UserResponse])
async def list_users(
    skip: int = 0,
    limit: int = 100,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    List users. Admin: all users. Tenant_admin: only users in own tenant.
    """
    if user.role not in ("admin", "tenant_admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin or tenant_admin access required"
        )

    query = select(User)
    if user.role == "tenant_admin":
        if not user.tenant_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Tenant not assigned")
        query = query.where(User.tenant_id == user.tenant_id)
    result = await db.execute(query.offset(skip).limit(limit))
    users = result.scalars().all()
    tenant_ids = list({u.tenant_id for u in users if u.tenant_id})
    tenant_name_by_id = {}
    if tenant_ids:
        t_result = await db.execute(select(Tenant.id, Tenant.name).where(Tenant.id.in_(tenant_ids)))
        for row in t_result:
            tenant_name_by_id[str(row.id)] = row.name

    return [
        UserResponse(
            id=str(u.id),
            email=u.email,
            username=u.username,
            role=u.role,
            is_active=u.is_active,
            rater_tier=u.rater_tier,
            tenant_id=str(u.tenant_id) if u.tenant_id else None,
            tenant_name=tenant_name_by_id.get(str(u.tenant_id)) if u.tenant_id else None,
            created_at=u.created_at,
            last_login=u.last_login
        )
        for u in users
    ]


@router.put("/users/{user_id}/role")
async def update_user_role(
    user_id: str,
    role: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Update a user's role. Admin: any role. Tenant_admin: only users in own tenant, roles researcher/viewer.
    """
    if current_user.role not in ("admin", "tenant_admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin or tenant_admin access required"
        )

    allowed_roles = ["admin", "tenant_admin", "researcher", "viewer", "rater"]
    if current_user.role == "tenant_admin":
        allowed_roles = ["researcher", "viewer"]
    if role not in allowed_roles:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role. Allowed: {allowed_roles}"
        )

    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    if current_user.role == "tenant_admin" and user.tenant_id != current_user.tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot change role of user in another tenant"
        )

    user.role = role
    if role == "rater" and not user.rater_tier:
        user.rater_tier = "bronze"
    await db.commit()

    return {"message": f"User role updated to {role}"}


@router.put("/users/{user_id}/status")
async def update_user_status(
    user_id: str,
    is_active: bool,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Enable or disable a user account (admin only).
    """
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )

    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    # Prevent self-disable
    if str(user.id) == str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot disable your own account"
        )

    user.is_active = is_active
    await db.commit()

    status_text = "enabled" if is_active else "disabled"
    return {"message": f"User account {status_text}"}


class AdminUserCreate(BaseModel):
    """Admin user creation request - allows setting role and tenant (by name)"""
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=100)
    password: str = Field(..., min_length=8)
    role: str = Field(default="rater", pattern="^(admin|tenant_admin|researcher|viewer|rater)$")
    rater_tier: Optional[str] = Field(default=None, pattern="^(gold|silver|bronze)$")
    tenant_name: Optional[str] = Field(default=None, max_length=255, description="Tenant name for researcher/viewer/tenant_admin (e.g. Farm1, Farm2). Ignored for admin/rater.")


@router.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    user_data: AdminUserCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new user with any role (admin only).
    Allows admins to create admin, researcher, or rater accounts.
    """
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )

    # Check if email already exists
    result = await db.execute(
        select(User).where(User.email == user_data.email)
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    # Check if username already exists
    result = await db.execute(
        select(User).where(User.username == user_data.username)
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already taken"
        )

    # Set rater tier based on role
    rater_tier = None
    if user_data.role == "rater":
        rater_tier = user_data.rater_tier or "bronze"

    # Resolve tenant: for researcher, viewer, tenant_admin use provided tenant_name (get or create) or default; for admin/rater use None
    tenant_id = None
    tenant_for_response = None
    if user_data.role in ("researcher", "viewer", "tenant_admin"):
        name = (user_data.tenant_name or "").strip()
        if name:
            t_result = await db.execute(select(Tenant).where(Tenant.name == name))
            tenant_row = t_result.scalar_one_or_none()
            if tenant_row:
                tenant_id = tenant_row.id
                tenant_for_response = tenant_row.name
            else:
                new_tenant = Tenant(id=uuid.uuid4(), name=name, is_active=True)
                db.add(new_tenant)
                await db.flush()
                tenant_id = new_tenant.id
                tenant_for_response = new_tenant.name
        else:
            default_tenant = await _get_or_create_default_tenant(db)
            tenant_id = default_tenant.id
            tenant_for_response = default_tenant.name

    # Create user
    user = User(
        id=uuid.uuid4(),
        email=user_data.email,
        username=user_data.username,
        password_hash=get_password_hash(user_data.password),
        role=user_data.role,
        is_active=True,
        rater_tier=rater_tier,
        tenant_id=tenant_id,
        created_at=datetime.utcnow()
    )

    db.add(user)
    await db.commit()
    await db.refresh(user)

    return UserResponse(
        id=str(user.id),
        email=user.email,
        username=user.username,
        role=user.role,
        is_active=user.is_active,
        rater_tier=user.rater_tier,
        tenant_id=str(user.tenant_id) if user.tenant_id else None,
        tenant_name=tenant_for_response,
        created_at=user.created_at,
        last_login=user.last_login
    )


@router.put("/users/{user_id}/tier")
async def update_user_tier(
    user_id: str,
    tier: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Update a rater's tier level (admin only).
    Valid tiers: gold, silver, bronze
    """
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )

    if tier not in ["gold", "silver", "bronze"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid tier. Must be gold, silver, or bronze"
        )

    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    if user.role != "rater":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only raters can have tiers"
        )

    user.rater_tier = tier
    await db.commit()

    return {"message": f"User tier updated to {tier}"}


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a user account (admin only).
    """
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )

    # Prevent self-deletion
    if str(current_user.id) == user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own account"
        )

    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    # Delete user's sessions first
    sessions_result = await db.execute(
        select(Session).where(Session.user_id == user.id)
    )
    sessions = sessions_result.scalars().all()
    for session in sessions:
        await db.delete(session)

    # Delete user
    await db.delete(user)
    await db.commit()

    return {"message": "User deleted successfully"}


@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get a specific user's details (admin only).
    """
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )

    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    tenant_name = None
    if user.tenant_id:
        t = await db.execute(select(Tenant.name).where(Tenant.id == user.tenant_id))
        row = t.one_or_none()
        tenant_name = row[0] if row else None
    return UserResponse(
        id=str(user.id),
        email=user.email,
        username=user.username,
        role=user.role,
        is_active=user.is_active,
        rater_tier=user.rater_tier,
        tenant_id=str(user.tenant_id) if user.tenant_id else None,
        tenant_name=tenant_name,
        created_at=user.created_at,
        last_login=user.last_login
    )
