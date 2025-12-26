"""
NATS client utilities for all services
"""
import os
import json
import asyncio
import nats
from typing import Optional, Dict, Any
import yaml
from pathlib import Path


class NATSClient:
    """NATS client wrapper for easy messaging"""
    
    def __init__(self, config_path: str = "/app/shared/config/config.yaml"):
        self.nc = None
        self.config_path = Path(config_path)
        self.config = None
        self._load_config()
    
    def _load_config(self):
        """Load configuration"""
        if self.config_path.exists():
            with open(self.config_path) as f:
                self.config = yaml.safe_load(f)
        else:
            # Fallback to environment variables
            self.config = {
                "nats": {
                    "url": os.getenv("NATS_URL", "nats://localhost:4222"),
                    "subjects": {}
                }
            }
    
    async def connect(self, url: Optional[str] = None):
        """Connect to NATS server"""
        nats_url = url or self.config["nats"]["url"]
        self.nc = await nats.connect(nats_url)
        print(f"Connected to NATS at {nats_url}")
        return self.nc
    
    async def publish(self, subject: str, data: Dict[Any, Any]):
        """Publish message to NATS"""
        if not self.nc:
            await self.connect()
        
        message = json.dumps(data).encode()
        await self.nc.publish(subject, message)
        print(f"Published to {subject}")
    
    async def subscribe(self, subject: str, callback):
        """Subscribe to NATS subject"""
        if not self.nc:
            await self.connect()
        
        async def message_handler(msg):
            try:
                data = json.loads(msg.data.decode())
                await callback(data)
            except Exception as e:
                print(f"Error processing message: {e}")
        
        sub = await self.nc.subscribe(subject, cb=message_handler)
        print(f"Subscribed to {subject}")
        return sub
    
    async def close(self):
        """Close NATS connection"""
        if self.nc:
            await self.nc.close()
            self.nc = None

