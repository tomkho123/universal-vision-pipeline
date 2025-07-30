# Universal Vision and Document Pipeline

A powerful OpenWebUI pipeline that adds vision and document processing capabilities to any language model using Google's Gemini AI. This pipeline automatically detects when you're using a non-vision model and seamlessly processes images and documents through Gemini's advanced multimodal capabilities.

## 🌟 Features

### Vision Processing
- **Automatic Image Analysis**: Processes images attached to conversations
- **Smart Model Detection**: Only activates for non-vision models
- **Image Optimization**: Automatic resizing and compression for optimal processing
- **Custom Prompts**: Support for custom vision analysis prompts
- **Multiple Formats**: Supports JPEG, PNG, WebP image formats

### Document Processing
- **File Upload Support**: Processes documents via Gemini File API
- **Multiple Formats**: PDF, DOCX, TXT, Markdown, CSV, and more
- **Intelligent Summarization**: Extracts key points and insights
- **Batch Processing**: Handles multiple documents simultaneously

### Smart Integration
- **Non-Intrusive**: Only processes attachments for models that need it
- **Seamless Injection**: Analysis results are injected as text into conversations
- **Configurable**: Extensive customization options via valves
- **Debug Support**: Comprehensive logging for troubleshooting

## 🚀 Quick Start

### 1. Installation

1. Copy the pipeline file to your OpenWebUI pipelines directory:
   ```bash
   cp pipeline.py /path/to/openwebui/pipelines/universal_vision_pipeline/
   ```

2. Install required dependencies:
   ```bash
   pip install requests pillow pydantic aiohttp
   ```

### 2. Configuration

1. **Get Gemini API Key**:
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Copy the key for configuration

2. **Configure the Pipeline**:
   - Open OpenWebUI admin panel
   - Navigate to Pipelines
   - Find "Universal Vision and Document Pipeline"
   - Set your Gemini API key in the valves

### 3. Usage

Simply attach images or documents to your conversations with any non-vision model. The pipeline will automatically:

1. Detect the attachment
2. Process it through Gemini
3. Inject the analysis into your conversation
4. Continue with your original model

## ⚙️ Configuration Options

### API Configuration
- **`gemini_api_key`**: Your Gemini API key
- **`gemini_model`**: Gemini model to use (default: `gemini-2.5-flash`)

### Vision Processing
- **`enable_vision_processing`**: Enable/disable image analysis
- **`vision_prompt_template`**: Default prompt for image analysis
- **`max_image_size`**: Maximum image dimensions (default: 2048px)
- **`image_quality`**: JPEG compression quality (1-100)

### Document Processing
- **`enable_document_processing`**: Enable/disable document analysis
- **`document_prompt_template`**: Default prompt for document analysis
- **`supported_document_mime_types`**: List of supported file types

### Model Detection
- **`non_vision_models`**: List of models that need vision assistance
- **`inject_analysis_as_text`**: How to inject analysis results
- **`analysis_response_prefix`**: Prefix for analysis responses

## 💡 Usage Examples

### Basic Image Analysis
```
User: What's in this image? [attaches photo]
Pipeline: [Attachment Analysis] Image 'photo.jpg': This image shows a beautiful sunset over a mountain landscape. The sky displays vibrant orange and pink hues...
Model: Based on the image analysis, I can see you've shared a stunning sunset photograph...
```

### Custom Vision Prompts
```
User: [vision: Focus on the technical details] Analyze this circuit diagram [attaches image]
Pipeline: [Attachment Analysis] Image 'circuit.png': This circuit diagram shows a basic amplifier configuration with...
```

### Document Processing
```
User: Summarize this report [attaches PDF]
Pipeline: [Attachment Analysis] Document Analysis: The report covers quarterly sales performance with key findings including...
```

### Multiple Attachments
```
User: Compare these documents [attaches 2 PDFs and 1 image]
Pipeline: [Attachment Analysis] 
Image 'chart.png': The chart displays revenue trends...
Document Analysis: Both documents discuss market analysis with contrasting viewpoints...
```

## 🔧 Advanced Configuration

### Custom Model Detection
Add your specific models to the non-vision list:
```python
non_vision_models: [
    "your-custom-model",
    "another-model-pattern"
]
```

### Custom Prompts
Use inline prompts for specific analysis:
- `[vision: describe the colors and composition]`
- `[analyze: focus on technical aspects]`
- `[document: extract action items]`
- `[summarize: key financial metrics only]`

### Supported Document Types
- PDF (`.pdf`)
- Word Documents (`.docx`, `.doc`)
- Text Files (`.txt`)
- Markdown (`.md`)
- CSV Files (`.csv`)

## 🛠️ Troubleshooting

### Common Issues

1. **API Key Not Working**
   - Verify your Gemini API key is correct
   - Check API quotas and billing
   - Ensure the key has necessary permissions

2. **Images Not Processing**
   - Check image format (JPEG, PNG, WebP supported)
   - Verify image size (max 2048px by default)
   - Enable debug logging to see processing details

3. **Documents Not Uploading**
   - Confirm file type is in supported list
   - Check file size limits
   - Verify network connectivity to Gemini API

### Debug Mode
Enable detailed logging:
```python
enable_debug_logging: True
log_file_processing: True  # Be careful with sensitive data
```

### Performance Optimization
- Adjust `max_image_size` for faster processing
- Lower `image_quality` for smaller uploads
- Use `gemini-2.5-flash` for faster responses

## 🔒 Security Considerations

- **API Key Security**: Store your Gemini API key securely
- **Data Privacy**: Files are uploaded to Google's servers for processing
- **Logging**: Disable `log_file_processing` in production to avoid logging sensitive data
- **Access Control**: Configure pipeline access appropriately

## 📊 Performance

### Typical Processing Times
- **Images**: 2-5 seconds per image
- **Documents**: 5-15 seconds depending on size
- **Multiple Files**: Processed in parallel for efficiency

### Resource Usage
- **Memory**: Minimal overhead, images processed in chunks
- **Network**: Depends on file sizes and API response times
- **CPU**: Low impact, mostly I/O bound operations

## 🤝 Contributing

This pipeline is part of the OpenWebUI ecosystem. Contributions welcome:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

MIT License - see the pipeline header for full details.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section above
2. Enable debug logging for detailed error information
3. Review OpenWebUI pipeline documentation
4. Check Gemini API status and documentation

## 🔄 Version History

- **v1.2**: Enhanced document processing, improved error handling
- **v1.1**: Added custom prompt support, better model detection
- **v1.0**: Initial release with basic vision and document processing

---

**Note**: This pipeline requires a valid Gemini API key and processes files through Google's servers. Ensure compliance with your organization's data policies before use.
