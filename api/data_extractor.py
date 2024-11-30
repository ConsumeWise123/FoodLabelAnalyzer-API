# api/index.py
from fastapi import FastAPI, HTTPException, File, UploadFile
from motor.motor_asyncio import AsyncIOMotorClient
from openai import AsyncOpenAI
import os
import json
import re
from typing import List, Dict, Any
from pydantic import BaseModel
from tempfile import NamedTemporaryFile
import cv2
import numpy as np
from PIL import Image
from fastapi.responses import JSONResponse
import boto3
from botocore.exceptions import ClientError
import uuid
import uvicorn
import numpy as np
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Move configuration and constants to separate files
from .config import MONGODB_URL, OPENAI_API_KEY
from .schemas import label_reader_schema

# Initialize clients
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
mongodb_client = AsyncIOMotorClient(MONGODB_URL)
db = mongodb_client.consumeWise
collection = db.products


# Configure AWS credentials
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_REGION = os.getenv("AWS_REGION")
BUCKET_NAME = os.getenv("BUCKET_NAME")


# Initialize S3 client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION,
)

def upload_to_s3(file_obj, filename: str) -> str:
    """Upload a file to S3 and return its URL"""
    try:
        s3_client.upload_fileobj(file_obj, BUCKET_NAME, filename)
        url = f"https://{BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{filename}"
        return url
    except ClientError as e:
        print(f"Error uploading to S3: {e}")
        return None

def check_image_quality(image_path, blur_threshold=100):
    """
    Analyzes an image for OCR suitability by checking blurriness.

    Parameters:
        image_path (str): Path to the image file
        blur_threshold (float): Threshold for determining if image is too blurry
                              Higher values indicate stricter blur detection
                              Recommended range: 50-150

    Returns:
        dict: Dictionary containing analysis results including:
              - blur_score: Variance of Laplacian (higher = sharper)
              - is_blurry: Boolean indicating if image is too blurry
              - can_ocr: Boolean indicating if OCR is recommended
    """
    # Read image in grayscale
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not load image")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate variance of Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    blur_score = laplacian.var()

    max_variance = 1000
    normalized_score = 1 - min(1, np.log1p(blur_score) / np.log1p(max_variance))


    # Determine if image is too blurry
    is_blurry = bool(blur_score < blur_threshold)


    # Determine if OCR is likely to succeed
    can_ocr = not is_blurry

    return {
        'blur_score': normalized_score,
        'is_blurry': is_blurry,
        'can_ocr': bool(can_ocr),
    }
    
async def extract_information(image_links: List[str], blur_threshold: float = 100) -> Dict[str, Any]:
    global openai_client
    print(f"DEBUG - openai_client : {openai_client}")

    valid_image_links = []
    
    for single_image_link in image_links:
        quality_result = check_image_quality(single_image_link, blur_threshold)
        if bool(quality_result['can_ocr']):
            #image is readable
            valid_image_links.append(single_image_link)
        
    LABEL_READER_PROMPT = """
You will be provided with a set of images corresponding to a single product. These images are found printed on the packaging of the product.
Your goal will be to extract information from these images to populate the schema provided. Here is some information you will routinely encounter. Ensure that you capture complete information, especially for nutritional information and ingredients:
- Ingredients: List of ingredients in the item. They may have some percent listed in brackets. They may also have metadata or classification like Preservative (INS 211) where INS 211 forms the metadata. Structure accordingly. If ingredients have subingredients like sugar: added sugar, trans sugar, treat them as different ingredients.
- Claims: Like a mango fruit juice says contains fruit.
- Nutritional Information: This will have nutrients, serving size, and nutrients listed per serving. Extract the base value for reference.
- FSSAI License number: Extract the license number. There might be many, so store relevant ones.
- Name: Extract the name of the product.
- Brand/Manufactured By: Extract the parent company of this product.
- Serving size: This might be explicitly stated or inferred from the nutrients per serving.
"""
    try:
        image_message = [{"type": "image_url", "image_url": {"url": il}} for il in valid_image_links]
        response = await openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": LABEL_READER_PROMPT},
                        *image_message,
                    ],
                },
            ],
            response_format={"type": "json_schema", "json_schema": label_reader_schema}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting information: {str(e)}")
        
    
@app.post("/api/extract-data")
async def extract_data(image_links_json: Dict[str, List[str]]):
    if not image_links_json or "image_links" not in image_links_json:
        raise HTTPException(status_code=400, detail="Image links not found")
    
    try:
        extracted_data = await extract_information(image_links_json["image_links"])
        result = await collection.insert_one(extracted_data)
        extracted_data["_id"] = str(result.inserted_id)
        return extracted_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/find-product")
async def find_product(product_name: str):
    if not product_name:
        raise HTTPException(status_code=400, detail="Please provide a valid product name")
    
    try:
        words = product_name.split()
        search_terms = [' '.join(words[:i]) for i in range(2, len(words) + 1)] + words
        product_list = []
        temp_list = []
        
        for term in search_terms:
            query = {"productName": {"$regex": f".*{re.escape(term)}.*", "$options": "i"}}
            async for product in collection.find(query):
                brand_product_name = f"{len(product_list) + 1}. {product['productName']} by {product['brandName']}"
                
                if f"{product['productName']} by {product['brandName']}" not in temp_list:
                    product_list.append(brand_product_name)
                    temp_list.append(f"{product['productName']} by {product['brandName']}")

        product_list.append("\ntype None")
        
        if len(product_list) > 1:
            return {
                "products": "\n".join(product_list),
                "message": "Products found"
            }
        else:
            return {
                "products": "",
                "message": "No products found"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ProductRequest(BaseModel):
    product_list: str
    ind: int
    
@app.post("/api/get-product")
async def get_product(request: ProductRequest):
    product_lines = request.product_list.split('\n')  # Store split result to avoid repetition

    if len(product_lines) == 0:
        raise HTTPException(status_code=400, detail="Please provide a valid product list")
    
    if request.ind - 1 < 0 or request.ind - 1 >= len(product_lines) - 1:
        raise HTTPException(status_code=400, detail=f"Index {request.ind - 1} is out of range for product list of length {len(product_lines) - 1}")
    
    try:
        product_name = request.product_list.split("\n")[request.ind - 1].split(".")[1].strip().split(" by ")[0]
        if not product_name:
            raise HTTPException(status_code=400, detail="Product name at given index is empty")
        
        product = await collection.find_one({"productName": product_name})
        if not product:
            raise HTTPException(status_code=404, detail=f"Product not found: {product_name}")
        
        product["_id"] = str(product["_id"])
        return product
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/multiple/")
async def upload_multiple_images(files: List[UploadFile] = File(...)):
    """Upload multiple image files to S3"""
    uploaded_files = []

    for file in files:
        if not file.content_type.startswith("image/"):
            continue

        # Save the uploaded file temporarily
        with NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(file.file.read())
            temp_path = temp_file.name
         quality_result = check_image_quality(temp_path, blur_threshold)

        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{str(uuid.uuid4())}{file_extension}"

        s3_url = upload_to_s3(file.file, unique_filename)

        if s3_url:
            uploaded_files.append(
                {
                    "original_filename": file.filename,
                    "new_filename": unique_filename,
                    "url": s3_url,
                }
            )

    return {
        "message": f"Successfully uploaded {len(uploaded_files)} images",
        "files": uploaded_files,
    }


@app.post("/upload-and-analyze/")
async def upload_and_analyze_image(file: UploadFile = File(...), blur_threshold: float = 100):
    """
    Upload an image, analyze its quality, and optionally upload to S3 if it is suitable for OCR.
    """
    if not file.content_type.startswith('image/'):
        return JSONResponse(
            status_code=400, content={"message": "File must be an image"}
        )

    try:
        # Save the uploaded file temporarily
        with NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(file.file.read())
            temp_path = temp_file.name

        # Check the image quality
        quality_result = check_image_quality(temp_path, blur_threshold)

        if bool(not quality_result['can_ocr']):
            # If the image is unreadable, delete temp file and return message
            os.remove(temp_path)
            return {
                "message": "Unreadable image. Cannot upload to S3.",
                "analysis": quality_result
            }

        # Generate a unique filename and upload the image to S3
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{str(uuid.uuid4())}{file_extension}"

        with open(temp_path, "rb") as f:
            s3_url = upload_to_s3(f, unique_filename)

        # Clean up the temporary file
        os.remove(temp_path)

        if s3_url:
            return {
                "message": "Image uploaded successfully",
                "url": s3_url,
                "filename": unique_filename,
                "analysis": quality_result,
            }
        else:
            return JSONResponse(
                status_code=500, content={"message": "Failed to upload image"}
            )

    except ValueError as e:
        return JSONResponse(
            status_code=400, content={"message": str(e)}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"message": "Internal server error", "error": str(e)}
        )
        
