import os
from rest_framework.response import Response
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from django.conf import settings
from .rag_model import process_with_rag

UPLOAD_DIR = os.path.join(settings.MEDIA_ROOT, 'images/')

@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def upload_image(request):
    """
    API to handle image upload and process it with RAG model.
    """
    if 'image' not in request.FILES:
        return Response({"error": "No image provided"}, status=400)

    image = request.FILES['image']
    
    # Save image to media folder
    file_path = os.path.join(UPLOAD_DIR, image.name)
    with open(file_path, 'wb+') as destination:
        for chunk in image.chunks():
            destination.write(chunk)

    # Process with RAG model (Dummy response for now)
    response = process_with_rag("pimple")  # Replace with actual CNN model output

    return Response({"message": "Image uploaded successfully", "rag_output": response})