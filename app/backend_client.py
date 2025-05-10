import httpx
import json
from typing import List, Dict, Any, Optional

from .config import BACKEND_URL, BACKEND_API_PREFIX, DEFAULT_AUTH_TOKEN, REQUEST_TIMEOUT, logger

class BackendClient:
    """Client for interacting with the TrueGift Backend API"""
    
    def __init__(self, base_url: Optional[str] = None, api_prefix: Optional[str] = None):
        self.base_url = base_url or BACKEND_URL
        self.api_prefix = api_prefix or BACKEND_API_PREFIX
        self.timeout = REQUEST_TIMEOUT
        logger.info(f"BackendClient initialized with base URL: {self.base_url}{self.api_prefix}")
        
    async def check_status(self) -> bool:
        """Check if backend API is available"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}")
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Error checking backend API status: {e}")
            return False
            
    async def fetch_user_photos(self, max_photos: int = 50, auth_token: Optional[str] = None) -> Dict[str, Any]:
        try:
            # Use provided token or default
            token = auth_token or DEFAULT_AUTH_TOKEN
            
            # Log token status - simplified
            if not token or token.strip() == "":
                raise ValueError("No valid auth token provided. Authentication required to fetch photos.")
            
            # Construct API URL
            api_url = f"{self.base_url}{self.api_prefix}/photos/ai/user-content?max_photos={max_photos}"
            
            # Set up headers with auth token
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            
            # Make API request
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(api_url, headers=headers)
                
                # Check response status
                if response.status_code != 200:
                    error_msg = f"Backend API request failed: {response.status_code}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                # Parse response
                data = response.json()        
                return data
                
        except Exception as e:
            raise 