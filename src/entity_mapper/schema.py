from typing import List, Dict, Optional, Any, Literal
from enum import Enum
from pydantic import BaseModel, Field


class EntityType(str, Enum):
    """Types of entities that can be extracted from documents."""
    COMPANY = "company"
    PERSON = "person"
    LOCATION = "location"
    CONTACT = "contact"
    PRODUCT = "product"
    SERVICE = "service"
    OTHER = "other"


class NameChange(BaseModel):
    """Represents a name change or alias for an entity."""
    previous_name: str = Field(..., description="Previous name of the entity")
    current_name: str = Field(..., description="Current name of the entity")
    change_date: Optional[str] = Field(None, description="Date when the name change occurred")
    change_reason: Optional[str] = Field(None, description="Reason for name change (acquisition, rebranding)")
    confidence: float = Field(default=1.0, description="Confidence score for this mapping")


class Address(BaseModel):
    """Structured representation of an address."""
    full_address: Optional[str] = Field(None, description="Complete address as a single string")
    street: Optional[str] = Field(None, description="Street address")
    city: Optional[str] = Field(None, description="City")
    state: Optional[str] = Field(None, description="State or province")
    postal_code: Optional[str] = Field(None, description="ZIP or postal code")
    country: Optional[str] = Field(None, description="Country")


class ContactInfo(BaseModel):
    """Contact information for an entity."""
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    website: Optional[str] = Field(None, description="Website URL")


class Entity(BaseModel):
    """Base model for extracted entities."""
    id: Optional[str] = Field(None, description="Unique identifier for the entity")
    name: str = Field(..., description="Primary name of the entity")
    type: EntityType = Field(..., description="Type of entity")
    aliases: List[str] = Field(default_factory=list, description="Alternative names or abbreviations")
    description: Optional[str] = Field(None, description="Brief description of the entity")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Additional attributes")
    confidence: float = Field(default=1.0, description="Confidence score for extraction")


class CompanyEntity(Entity):
    """Model for extracted company entities."""
    type: Literal[EntityType.COMPANY] = Field(EntityType.COMPANY, description="Entity type (always company)")
    industry: Optional[str] = Field(None, description="Industry or sector")
    founding_date: Optional[str] = Field(None, description="Company founding date")
    address: Optional[Address] = Field(None, description="Company address")
    contact: Optional[ContactInfo] = Field(None, description="Contact information")
    name_changes: List[NameChange] = Field(default_factory=list, description="History of name changes")
    parent_company: Optional[str] = Field(None, description="Parent company name, if any")
    subsidiaries: List[str] = Field(default_factory=list, description="Subsidiary companies, if any")


class PersonEntity(Entity):
    """Model for extracted person entities."""
    type: Literal[EntityType.PERSON] = Field(EntityType.PERSON, description="Entity type (always person)")
    title: Optional[str] = Field(None, description="Job title or role")
    organization: Optional[str] = Field(None, description="Affiliated organization")
    contact: Optional[ContactInfo] = Field(None, description="Contact information")


class MappingResult(BaseModel):
    """Result of entity mapping process."""
    original_entity: Entity
    mapped_entity_id: Optional[str] = Field(None, description="ID of the mapped entity in database")
    mapped_entity_name: Optional[str] = Field(None, description="Name of the mapped entity in database")
    confidence: float = Field(default=0.0, description="Confidence score for this mapping")
    name_change_detected: bool = Field(default=False, description="Whether a name change was detected")
    name_change: Optional[NameChange] = Field(None, description="Name change details if detected")
