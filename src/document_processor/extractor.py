"""
Document extraction module that processes images and extracts structured data
using Vision-enabled LLMs.
"""

import os
import logging
import base64
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from io import BytesIO

import numpy as np
from PIL import Image
from pdf2image import convert_from_path
import openai
import google.generativeai as genai
import anthropic

# Add parent directory to path to enable imports
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.entity_mapper.schema import Entity, CompanyEntity, PersonEntity, EntityType, Address, ContactInfo

logger = logging.getLogger(__name__)

class DocumentExtractor:
    """
    Extracts structured information from document images using 
    vision-enabled LLMs.
    """
    
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        detail_level: str = "high"
    ):
        """
        Initialize document extractor with model and API key.
        
        Args:
            model: Model name to use (e.g., 'gpt-4o', 'claude-3-opus-20240229', 'gemini-pro-vision', 'mock')
            api_key: API key for the selected model's provider
            detail_level: Level of detail for vision analysis ('high', 'medium', 'low')
        """
        self.model = model
        self.detail_level = detail_level
        
        # Special handling for mock mode
        if model.lower() == 'mock':
            self.client_type = "mock"
            logger.info("Using mock mode - no API call will be made")
        else:
            self._setup_client(api_key)
        
    def _setup_client(self, api_key: Optional[str]) -> None:
        """Set up the appropriate client based on the model name."""
        if 'gpt' in self.model.lower():
            openai.api_key = api_key or os.environ.get("OPENAI_API_KEY")
            self.client_type = "openai"
        elif 'claude' in self.model.lower():
            self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
            self.client_type = "anthropic"
        elif 'gemini' in self.model.lower():
            genai.configure(api_key=api_key or os.environ.get("GEMINI_API_KEY"))
            self.client = genai.GenerativeModel(self.model)
            self.client_type = "gemini"
        else:
            raise ValueError(f"Unsupported model: {self.model}")
    
    def extract_from_pdf(self, pdf_path: Union[str, Path]) -> List[Entity]:
        """
        Extract entities from a PDF document.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of extracted entities
        """
        # Convert PDF to images
        pdf_path = Path(pdf_path)
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Special handling for mock mode
        if self.client_type == "mock":
            logger.info("Using mock data for PDF processing")
            return self._generate_mock_entities(pdf_path.stem)
        
        # Normal processing for non-mock mode
        images = convert_from_path(pdf_path)
        logger.info(f"Converted PDF to {len(images)} images")
        
        # Process each image
        all_entities = []
        for i, image in enumerate(images):
            logger.info(f"Processing page {i+1}/{len(images)}")
            page_entities = self.extract_from_image(image)
            all_entities.extend(page_entities)
            
        return all_entities
    
    def extract_from_image(self, image: Union[str, Path, Image.Image]) -> List[Entity]:
        """
        Extract entities from an image.
        
        Args:
            image: Path to image file or PIL Image object
            
        Returns:
            List of extracted entities
        """
        # Special handling for mock mode
        if self.client_type == "mock":
            logger.info("Using mock data for image processing")
            # If we have a path, use it to generate appropriate mock entities
            if isinstance(image, (str, Path)):
                image_path = Path(image)
                return self._generate_mock_entities(image_path.stem)
            # Otherwise, just return generic mock entities
            return self._generate_mock_entities()
        
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        
        # Prepare the prompt
        prompt = """
        Extract all entities from this transportation/logistics document, including:
        
        1. Companies (carriers, brokers, shippers, consignees)
        2. People (drivers, contacts)
        3. Locations (addresses, origin, destination)
        4. Identifiers (order numbers, BOL numbers, tracking numbers)
        5. Financial information (rates, amounts, payment terms)
        6. Dates and times
        7. Product information
        
        Pay special attention to:
        - Company name changes, acquisitions, affiliations ("a division of", "formerly known as", etc.)
        - All contact information
        - Complete addresses
        
        Format as JSON with proper entity types and relationships.
        """
        
        # Process with the appropriate model
        if self.client_type == "openai":
            return self._extract_with_openai(image, prompt)
        elif self.client_type == "anthropic":
            return self._extract_with_anthropic(image, prompt)
        elif self.client_type == "gemini":
            return self._extract_with_gemini(image, prompt)
        else:
            raise ValueError(f"Unsupported client type: {self.client_type}")
    
    def _generate_mock_entities(self, file_stem: Optional[str] = None) -> List[Entity]:
        """Generate mock entities based on the file name or generic ones if not provided."""
        entities = []
        
        # Create entities based on the document name if provided
        if file_stem:
            if "steves" in file_stem.lower() or "freight_invoice" in file_stem.lower():
                # Steve's Trucking freight invoice
                company = CompanyEntity(
                    name="Steve's Trucking",
                    type=EntityType.COMPANY,
                    aliases=["STC"],
                    industry="trucking",
                    address=Address(
                        street="PO Box 915654",
                        city="Kansas City",
                        state="MO",
                        postal_code="64111"
                    ),
                    contact=ContactInfo(
                        phone="(888) 564-6546"
                    )
                )
                
                customer = CompanyEntity(
                    name="Customer Company Name",
                    type=EntityType.COMPANY,
                    address=Address(
                        street="7834 18th St.",
                        city="Dallas",
                        state="TX",
                        postal_code="75391"
                    )
                )
                
                driver = PersonEntity(
                    name="Driver Name",
                    type=EntityType.PERSON,
                    title="Driver",
                    organization="Steve's Trucking"
                )
                
                entities.extend([company, customer, driver])
                
            elif "bennett" in file_stem.lower() or "rate_confirmation" in file_stem.lower():
                # Bennett International Logistics rate confirmation
                company = CompanyEntity(
                    name="Bennett International Logistics, LLC",
                    type=EntityType.COMPANY,
                    aliases=["BIL"],
                    industry="logistics",
                    address=Address(
                        street="PO Box 569",
                        city="McDonough",
                        state="GA",
                        postal_code="30253"
                    ),
                    contact=ContactInfo(
                        phone="770-957-1866",
                        fax="877-251-8541"
                    )
                )
                
                parent_company = CompanyEntity(
                    name="BENNETT TRUCK TRANSPORT, LLC",
                    type=EntityType.COMPANY,
                    industry="trucking"
                )
                
                carrier = CompanyEntity(
                    name="GT XPRESS INC",
                    type=EntityType.COMPANY,
                    industry="trucking",
                    contact=ContactInfo(
                        phone="8773772720"
                    )
                )
                
                driver = PersonEntity(
                    name="MARTY ROWE",
                    type=EntityType.PERSON,
                    title="Driver",
                    organization="GT XPRESS INC",
                    contact=ContactInfo(
                        phone="5174251761"
                    )
                )
                
                origin = CompanyEntity(
                    name="AGRI EMPRESSA",
                    type=EntityType.COMPANY,
                    address=Address(
                        street="6001 W INDUSTRIAL AVE",
                        city="MIDLAND",
                        state="TX",
                        postal_code="79701"
                    )
                )
                
                destination = CompanyEntity(
                    name="IDC 301 CYCLONE EA 23H",
                    type=EntityType.COMPANY,
                    address=Address(
                        street="CR 194",
                        city="SMILEY",
                        state="TX",
                        postal_code="78159"
                    )
                )
                
                entities.extend([company, parent_company, carrier, driver, origin, destination])
                
            elif "linbis" in file_stem.lower() or "bill_of_lading" in file_stem.lower():
                # Linbis Logistics Software bill of lading
                company = CompanyEntity(
                    name="Linbis Logistics Software",
                    type=EntityType.COMPANY,
                    industry="technology",
                    address=Address(
                        street="5406 NW 72 AVE",
                        city="Miami",
                        state="FL",
                        postal_code="33166"
                    ),
                    contact=ContactInfo(
                        phone="(305) 513-8555",
                        fax="(305) 513-8555",
                        email="info@linbis.com",
                        website="www.linbis.com"
                    )
                )
                
                shipper = CompanyEntity(
                    name="Sample Company TFASCLO",
                    type=EntityType.COMPANY,
                    address=Address(
                        street="8551 EAST 88 TH STREET",
                        city="Sample City",
                        state="CA",
                        postal_code="55532"
                    )
                )
                
                consignee = CompanyEntity(
                    name="Sample Company CODE001",
                    type=EntityType.COMPANY,
                    address=Address(
                        street="88185 NW 51th St Unit 811",
                        city="Sample City",
                        state="FL", 
                        postal_code="55532"
                    ),
                    contact=ContactInfo(
                        phone="7866839976"
                    )
                )
                
                carrier = CompanyEntity(
                    name="FEDEX",
                    type=EntityType.COMPANY,
                    industry="shipping"
                )
                
                driver = PersonEntity(
                    name="John Smith",
                    type=EntityType.PERSON,
                    title="Driver",
                    organization="FEDEX"
                )
                
                entities.extend([company, shipper, consignee, carrier, driver])
        
        # If no entities created based on file name, create generic ones
        if not entities:
            # Generic logistics document entities
            company = CompanyEntity(
                name="Example Logistics Corp",
                type=EntityType.COMPANY,
                aliases=["ELC", "Example Logistics"],
                industry="logistics",
                address=Address(
                    street="123 Main St",
                    city="Anytown",
                    state="CA",
                    postal_code="90210"
                ),
                contact=ContactInfo(
                    phone="(555) 123-4567",
                    email="info@examplelogistics.com"
                )
            )
            
            driver = PersonEntity(
                name="John Driver",
                type=EntityType.PERSON,
                title="Driver",
                organization="Example Logistics Corp"
            )
            
            entities.extend([company, driver])
            
        return entities
    
    def _extract_with_openai(self, image: Image.Image, prompt: str) -> List[Entity]:
        """Extract entities using OpenAI's vision models."""
        # Convert image to base64
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # Create message with image
        messages = [
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_str}",
                            "detail": self.detail_level
                        }
                    }
                ]
            }
        ]
        
        # Call API
        response = openai.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"}
        )
        
        # Parse response
        content = response.choices[0].message.content
        return self._parse_response(content)
    
    def _extract_with_anthropic(self, image: Image.Image, prompt: str) -> List[Entity]:
        """Extract entities using Anthropic's Claude vision models."""
        # Implementation for Anthropic
        # Will need to be fleshed out based on their API
        pass
    
    def _extract_with_gemini(self, image: Image.Image, prompt: str) -> List[Entity]:
        """Extract entities using Google's Gemini vision models."""
        # Implementation for Gemini
        # Will need to be fleshed out based on their API
        pass
    
    def _parse_response(self, response_content: str) -> List[Entity]:
        """
        Parse LLM response into structured entities.
        
        Args:
            response_content: JSON string from LLM
            
        Returns:
            List of structured entities
        """
        try:
            data = json.loads(response_content)
            entities = []
            
            # Process companies
            for company_data in data.get("companies", []):
                company = CompanyEntity(
                    name=company_data.get("name"),
                    type=EntityType.COMPANY,
                    aliases=company_data.get("aliases", []),
                    industry=company_data.get("industry"),
                    # Process other fields as needed
                )
                entities.append(company)
            
            # Process people
            for person_data in data.get("people", []):
                person = PersonEntity(
                    name=person_data.get("name"),
                    type=EntityType.PERSON,
                    title=person_data.get("title"),
                    organization=person_data.get("organization"),
                    # Process other fields as needed
                )
                entities.append(person)
            
            # Process other entity types...
            
            return entities
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.debug(f"Response content: {response_content}")
            return []
