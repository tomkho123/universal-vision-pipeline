"""
title: Universal Vision and Document Pipeline with Gemini
author: Kilo Code & Gemini AI Assistant
date: 2025-07-28
version: 1.2
license: MIT
description: A universal processing pipeline that uses Gemini to handle vision and documents for all non-vision models.
requirements: requests, pillow, pydantic, aiohttp
"""

import os
import requests
import base64
import json
import re
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from PIL import Image
import io
import asyncio
import aiohttp
from datetime import datetime

class Pipeline:
    class Valves(BaseModel):
        # Pipeline Configuration
        pipelines: List[str] = ["*"]
        priority: int = 0
        
        # Gemini API Configuration
        gemini_api_key: str = Field(
            default=os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE"),
            description="Gemini API key for vision processing"
        )
        gemini_model: str = Field(
            default="gemini-1.5-flash-latest",
            description="Gemini model to use for vision and document processing"
        )
        
        # Vision Processing Configuration
        enable_vision_processing: bool = Field(
            default=True,
            description="Enable automatic vision processing for images"
        )
        vision_prompt_template: str = Field(
            default="Analyze this image in detail. Describe what you see, including objects, people, text, colors, composition, and any other relevant details. Be comprehensive and accurate.",
            description="Default prompt template for vision analysis"
        )
        max_image_size: int = Field(
            default=2048, # Increased for better quality with modern models
            description="Maximum image size (width/height) for processing"
        )
        image_quality: int = Field(
            default=90,
            description="JPEG quality for image compression (1-100)"
        )

        # Document Processing Configuration
        enable_document_processing: bool = Field(
            default=True,
            description="Enable automatic document processing via Gemini File API"
        )
        document_prompt_template: str = Field(
            default="Summarize the key points of this document. Identify the main topics, conclusions, and any action items mentioned.",
            description="Default prompt template for document analysis"
        )
        supported_document_mime_types: List[str] = Field(
            default=[
                "application/pdf",
                "application/vnd.openxmlformats-officedocument.wordprocessing.document",
                "text/plain",
                "text/markdown",
                "text/csv",
                "application/msword",
            ],
            description="List of supported MIME types for document processing."
        )
        
        # Model Detection Configuration
        non_vision_models: List[str] = Field(
            default=[
                "gpt-3.5-turbo", "gpt-4", "claude-3-haiku", "claude-3-sonnet", 
                "llama", "mistral", "gemma", "qwen", "deepseek", "phi",
                "codellama", "vicuna", "alpaca", "openchat", "starling"
            ],
            description="List of model names/patterns that don't support vision/document processing natively"
        )
        
        # Response Configuration
        inject_analysis_as_text: bool = Field(
            default=True,
            description="Inject vision/document analysis as text into the conversation"
        )
        analysis_response_prefix: str = Field(
            default="[Attachment Analysis]",
            description="Prefix for vision/document analysis responses"
        )
        
        # Debug Configuration
        enable_debug_logging: bool = Field(
            default=True,
            description="Enable detailed debug logging"
        )
        log_file_processing: bool = Field(
            default=False,
            description="Log file processing details (may expose sensitive data)"
        )

    def __init__(self):
        self.type = "filter"
        self.valves = self.Valves()
        self.gemini_api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.valves.gemini_model}:generateContent"
        self.gemini_file_api_url = "https://generativelanguage.googleapis.com/v1beta/files"

    async def on_startup(self):
        print(f"🚀 Starting Universal Vision & Document Pipeline: {__name__}")
        print(f"🔧 Gemini Model: {self.valves.gemini_model}")
        print(f"👁️ Vision Processing: {'Enabled' if self.valves.enable_vision_processing else 'Disabled'}")
        print(f"📄 Document Processing: {'Enabled' if self.valves.enable_document_processing else 'Disabled'}")

    async def on_shutdown(self):
        print(f"🛑 Shutting down Universal Vision & Document Pipeline: {__name__}")

    def is_non_vision_model(self, model_name: str) -> bool:
        """Check if the model doesn't support vision natively"""
        if not model_name:
            return True
            
        model_lower = model_name.lower()
        
        # Explicitly treat gemini models as vision-capable
        if "gemini" in model_lower:
            return False

        # Check against configured non-vision models
        for pattern in self.valves.non_vision_models:
            if pattern.lower() in model_lower:
                return True
                
        # Additional heuristics for common non-vision models
        non_vision_patterns = [
            "text-", "chat-", "instruct", "base", "7b", "13b", "70b", 
            "code", "math", "reasoning"
        ]
        
        for pattern in non_vision_patterns:
            if pattern in model_lower:
                return True
                
        return False

    def is_document_mime_type(self, mime_type: str) -> bool:
        """Check if the MIME type is a supported document format."""
        return mime_type in self.valves.supported_document_mime_types

    def extract_attachments_from_message(self, content: Union[str, List[Dict]]) -> List[Dict]:
        """Extract attachments (images and documents) from message content"""
        attachments = []
        
        if isinstance(content, str):
            # Look for base64 data in markdown-like format `![filename](data:mime/type;base64,...)`
            pattern = r'!\[(.*?)\]\(data:([^;]+);base64,([^)]+)\)'
            matches = re.findall(pattern, content)
            
            for filename, mime_type, base64_data in matches:
                attachments.append({
                    "filename": filename if filename else "untitled",
                    "mime_type": mime_type.strip(),
                    "data": base64_data
                })
                
        elif isinstance(content, list):
            # OpenAI format with image_url
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    image_url = item.get("image_url", {}).get("url", "")
                    if image_url.startswith("data:"):
                        match = re.match(r'data:([^;]+);base64,(.+)', image_url)
                        if match:
                            mime_type, base64_data = match.groups()
                            attachments.append({
                                "filename": "image.png", # Default filename for image_url type
                                "mime_type": mime_type.strip(),
                                "data": base64_data
                            })
        
        return attachments

    def clean_message_content(self, content: Union[str, List[Dict]]) -> Union[str, List[Dict]]:
        """Remove attachments from message content for non-vision models"""
        if isinstance(content, str):
            # Remove attachment markdown from string content
            cleaned_content = re.sub(r'!\[.*?\]\(data:[^)]+\)', '', content)
            return cleaned_content.strip()
            
        elif isinstance(content, list):
            # Remove image_url items from list content
            cleaned_content = [item for item in content if not (isinstance(item, dict) and item.get("type") == "image_url")]
            return cleaned_content
            
        return content

    def resize_image(self, base64_data: str, mime_type: str) -> str:
        """Resize image if it's too large"""
        try:
            # Decode base64 image
            image_data = base64.b64decode(base64_data)
            image = Image.open(io.BytesIO(image_data))
            
            # Check if resize is needed
            max_size = self.valves.max_image_size
            if image.width <= max_size and image.height <= max_size:
                return base64_data
            
            # Calculate new size maintaining aspect ratio
            if image.width > image.height:
                new_width = max_size
                new_height = int((max_size * image.height) / image.width)
            else:
                new_height = max_size
                new_width = int((max_size * image.width) / image.height)
            
            # Resize image
            resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert back to base64
            output_buffer = io.BytesIO()
            format_map = {
                "image/jpeg": "JPEG",
                "image/jpg": "JPEG", 
                "image/png": "PNG",
                "image/webp": "WEBP"
            }
            
            output_format = format_map.get(mime_type, "JPEG")
            if output_format == "JPEG":
                resized_image = resized_image.convert("RGB")
                resized_image.save(output_buffer, format=output_format, quality=self.valves.image_quality)
            else:
                resized_image.save(output_buffer, format=output_format)
            
            resized_base64 = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
            
            if self.valves.enable_debug_logging:
                print(f"🖼️ Resized image from {image.width}x{image.height} to {new_width}x{new_height}")
            
            return resized_base64
            
        except Exception as e:
            if self.valves.enable_debug_logging:
                print(f"❌ Error resizing image: {e}")
            return base64_data

    async def _make_gemini_request(self, payload: dict) -> str:
        """Helper to make requests to the Gemini API."""
        try:
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": self.valves.gemini_api_key
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(self.gemini_api_url, headers=headers, json=payload, timeout=120) as response:
                    if response.status == 200:
                        result = await response.json()
                        if "candidates" in result and result["candidates"]:
                            candidate = result["candidates"][0]
                            if "content" in candidate and "parts" in candidate["content"]:
                                text_parts = [part.get("text", "") for part in candidate["content"]["parts"]]
                                return "".join(text_parts).strip()
                        
                        if self.valves.enable_debug_logging:
                            print(f"⚠️ Unexpected Gemini response format: {result}")
                        return "Unable to analyze - unexpected response format"
                    else:
                        error_text = await response.text()
                        error_msg = f"Gemini API error: {response.status} - {error_text}"
                        if self.valves.enable_debug_logging:
                            print(f"❌ {error_msg}")
                        return f"Unable to analyze: {error_msg}"
        except Exception as e:
            error_msg = f"Error calling Gemini API: {str(e)}"
            if self.valves.enable_debug_logging:
                print(f"❌ {error_msg}")
            return error_msg

    async def analyze_image_with_gemini(self, image_data: str, mime_type: str, custom_prompt: str = None) -> str:
        """Analyze image using Gemini Vision API"""
        processed_image_data = self.resize_image(image_data, mime_type)
        prompt = custom_prompt or self.valves.vision_prompt_template
        
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": mime_type, "data": processed_image_data}}
                ]
            }],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 4096}
        }
        
        if self.valves.log_file_processing:
            print(f"🔍 Analyzing image with Gemini: {mime_type}")

        analysis = await self._make_gemini_request(payload)
        
        if self.valves.enable_debug_logging:
            print(f"✅ Vision analysis completed: {len(analysis)} characters")
        return analysis
    
    async def upload_file_to_gemini(self, attachment: dict, session: aiohttp.ClientSession) -> Optional[dict]:
        """Uploads a file to the Gemini File API and returns its details."""
        if self.valves.enable_debug_logging:
            print(f"⬆️ Uploading to Gemini File API: {attachment['filename']} ({attachment['mime_type']})")
        
        try:
            # Reusing the session is more efficient
            headers = {"x-goog-api-key": self.valves.gemini_api_key}
            
            file_payload = {"file": {"display_name": attachment['filename']}}
            async with session.post(self.gemini_file_api_url, headers=headers, json=file_payload, timeout=20) as r:
                r.raise_for_status()
                file_info = await r.json()
            
            upload_uri = file_info["file"]["upload_uri"]
            file_data = base64.b64decode(attachment['data'])
            
            upload_headers = {
                "Content-Type": attachment['mime_type'],
                "x-goog-content-length": str(len(file_data))
            }
            async with session.put(upload_uri, headers=upload_headers, data=file_data, timeout=60) as r:
                r.raise_for_status()

            if self.valves.enable_debug_logging:
                print(f"✅ File uploaded successfully: {file_info['file']['name']}")
            
            # Return the necessary details for the next step
            return {
                "name": file_info["file"]["name"],
                "mime_type": attachment['mime_type']
            }

        except Exception as e:
            if self.valves.enable_debug_logging:
                print(f"❌ Gemini File API upload error for {attachment['filename']}: {e}")
            return None

    def extract_custom_prompt(self, text_content: str) -> tuple[str, str]:
        """Extract custom analysis prompt from text content"""
        # Look for patterns like [vision: ...], [analyze: ...], [summarize: ...]
        patterns = [
            r'\[vision:\s*([^\]]+)\]', r'\[analyze:\s*([^\]]+)\]',
            r'\[describe:\s*([^\]]+)\]', r'\[image:\s*([^\]]+)\]',
            r'\[document:\s*([^\]]+)\]', r'\[summarize:\s*([^\]]+)\]'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text_content, re.IGNORECASE)
            if match:
                custom_prompt = match.group(1).strip()
                cleaned_text = re.sub(pattern, '', text_content, flags=re.IGNORECASE).strip()
                return cleaned_text, custom_prompt
        
        return text_content, None

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """Process messages and add vision/document analysis for non-compatible models"""
        
        is_vision_enabled = self.valves.enable_vision_processing
        is_doc_enabled = self.valves.enable_document_processing

        if not is_vision_enabled and not is_doc_enabled:
            return body
        
        model_name = body.get("model", "")
        if self.valves.enable_debug_logging:
            print(f"🤖 Processing request for model: {model_name}")
        
        if not self.is_non_vision_model(model_name):
            if self.valves.enable_debug_logging:
                print(f"✅ Model '{model_name}' is assumed to be multi-modal, skipping pipeline.")
            return body
        
        messages = body.get("messages", [])
        if not messages:
            return body
            
        # Asynchronously process the last user message for attachments
        last_user_message_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                last_user_message_idx = i
                break
        
        if last_user_message_idx == -1:
            return body

        message = messages[last_user_message_idx]
        content = message.get("content")
        if not content:
            return body

        attachments = self.extract_attachments_from_message(content)
        if not attachments:
            return body

        images = [att for att in attachments if att["mime_type"].startswith("image/") and is_vision_enabled]
        documents = [att for att in attachments if self.is_document_mime_type(att["mime_type"]) and is_doc_enabled]

        if not images and not documents:
            return body

        if self.valves.enable_debug_logging:
            print(f"📎 Found {len(images)} image(s) and {len(documents)} document(s) for analysis.")
            
        # Extract custom prompt and clean text content
        text_content = ""
        if isinstance(content, str):
            text_content = content
        elif isinstance(content, list):
            text_parts = [item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"]
            text_content = " ".join(text_parts)
        
        cleaned_text, custom_prompt = self.extract_custom_prompt(text_content)
        
        all_analyses = []
        
        # 1. Analyze Images
        if images:
            image_tasks = [
                self.analyze_image_with_gemini(img["data"], img["mime_type"], custom_prompt) for img in images
            ]
            image_analyses = await asyncio.gather(*image_tasks)
            for i, analysis in enumerate(image_analyses):
                all_analyses.append(f"Image '{images[i]['filename']}': {analysis}")

        # 2. Analyze Documents
        if documents:
            async with aiohttp.ClientSession() as session:
                upload_tasks = [self.upload_file_to_gemini(doc, session) for doc in documents]
                uploaded_files_details = await asyncio.gather(*upload_tasks)
            
            valid_files = [f for f in uploaded_files_details if f]

            if valid_files:
                doc_prompt = custom_prompt or self.valves.document_prompt_template
                file_parts = [{"file_data": {"mime_type": f["mime_type"], "file_uri": f["name"]}} for f in valid_files]
                
                # Create a single prompt for all documents
                doc_payload = {
                    "contents": [{"parts": [{"text": doc_prompt}] + file_parts}],
                    "generationConfig": {"temperature": 0.2, "maxOutputTokens": 8192}
                }
                
                if self.valves.log_file_processing:
                    print(f"🔍 Analyzing {len(valid_files)} documents with Gemini...")

                doc_analysis = await self._make_gemini_request(doc_payload)
                all_analyses.append(f"Document Analysis: {doc_analysis}")

        # 3. Inject combined analysis into the message
        if all_analyses:
            analysis_text = f"\n\n{self.valves.analysis_response_prefix}\n" + "\n\n".join(all_analyses)
            
            cleaned_content = self.clean_message_content(content)
            
            if isinstance(cleaned_content, str):
                message["content"] = cleaned_text + analysis_text
            elif isinstance(cleaned_content, list):
                # Remove old text parts to avoid duplication
                new_content = [item for item in cleaned_content if item.get("type") != "text"]
                new_content.insert(0, {"type": "text", "text": cleaned_text})
                new_content.append({"type": "text", "text": analysis_text})
                message["content"] = new_content
            
            if self.valves.enable_debug_logging:
                print(f"✅ Injected combined analysis into message.")
            
            body["messages"][last_user_message_idx] = message
        
        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """Process response after model generation (optional post-processing)"""
        return body