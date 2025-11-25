import enum
from sqlalchemy import Column, String, Integer, Date, Enum, DateTime, Boolean, DATETIME, ForeignKey, Text, Enum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from database import Base

class UserRole(enum.Enum):
    Admin = "Admin"
    Rescuer = "Rescuer"
    User = "User"

class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True)
    email = Column(String, unique=True, nullable=False)
    first_name = Column(String)
    middle_name = Column(String)
    last_name = Column(String)
    permanent_address = Column(String)
    age = Column(Integer)
    birth_date = Column(Date)
    emergency_contact = Column(String)
    role = Column(Enum(UserRole), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class RescuerStatus(Base):
    __tablename__ = "rescuer_status"

    id = Column(UUID(as_uuid=True), primary_key=True)
    rescuer_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    is_available = Column(Boolean, default=True)
    last_known_location = Column(Text)  # can later be changed to PostGIS POINT
    last_active = Column(DateTime(timezone=True), server_default=func.now())

class UserStatusEnum(str, enum.Enum):
    Safe = "Safe"
    Unsafe = "Unsafe"
    Unknown = "Unknown"

class UserStatusUpdate(Base):
    __tablename__ = "user_status_update"

    id = Column(UUID(as_uuid=True), primary_key=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    calamity_id = Column(UUID(as_uuid=True), nullable=False)
    status = Column(Enum(UserStatusEnum), nullable=False)
    current_situation = Column(Text)
    update_datetime = Column(DateTime(timezone=True), server_default=func.now())
    location = Column(String) 