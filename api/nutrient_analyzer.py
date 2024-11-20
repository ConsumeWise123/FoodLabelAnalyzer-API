import logging
import traceback
import sys
from functools import wraps
from .icmr import analyze_nutrients
from .rda import find_nutrition, rda_analysis, analyze_nutrition_icmr_rda
import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
from typing import Dict, Any

# Set up logging with file name and line numbers
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

#logging.getLogger('pymongo').setLevel(logging.WARNING)

# Create an error handling decorator
def log_exceptions(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            # Get the current stack trace
            exc_info = sys.exc_info()
            # Get the full stack trace as a string
            stack_trace = ''.join(traceback.format_exception(*exc_info))
            
            logger.error(
                f"Exception in {func.__name__}: {str(e)}\n"
                f"Stack trace:\n{stack_trace}"
            )
            raise
    return wrapper

def find_product_nutrients(product_info_from_db):
    #GET Response: {'_id': '6714f0487a0e96d7aae2e839',
    #'brandName': 'Parle', 'claims': ['This product does not contain gold'],
    #'fssaiLicenseNumbers': [10013022002253],
    #'ingredients': [{'metadata': '', 'name': 'Refined Wheat Flour (Maida)', 'percent': '63%'}, {'metadata': '', 'name': 'Sugar', 'percent': ''}, {'metadata': '', 'name': 'Refined Palm Oil', 'percent': ''}, {'metadata': '(Glucose, Levulose)', 'name': 'Invert Sugar Syrup', 'percent': ''}, {'metadata': 'I', 'name': 'Sugar Citric Acid', 'percent': ''}, {'metadata': '', 'name': 'Milk Solids', 'percent': '1%'}, {'metadata': '', 'name': 'Iodised Salt', 'percent': ''}, {'metadata': '503(I), 500 (I)', 'name': 'Raising Agents', 'percent': ''}, {'metadata': '1101 (i)', 'name': 'Flour Treatment Agent', 'percent': ''}, {'metadata': 'Diacetyl Tartaric and Fatty Acid Esters of Glycerol (of Vegetable Origin)', 'name': 'Emulsifier', 'percent': ''}, {'metadata': 'Vanilla', 'name': 'Artificial Flavouring Substances', 'percent': ''}],
    
    #'nutritionalInformation': [{'name': 'Energy', 'unit': 'kcal', 'values': [{'base': 'per 100 g','value': 462}]},
    #{'name': 'Protein', 'unit': 'g', 'values': [{'base': 'per 100 g', 'value': 6.7}]},
    #{'name': 'Carbohydrate', 'unit': 'g', 'values': [{'base': 'per 100 g', 'value': 76.0}, {'base': 'of which sugars', 'value': 26.9}]},
    #{'name': 'Fat', 'unit': 'g', 'values': [{'base': 'per 100 g', 'value': 14.6}, {'base': 'Saturated Fat', 'value': 6.8}, {'base': 'Trans Fat', 'value': 0}]},
    #{'name': 'Total Sugars', 'unit': 'g', 'values': [{'base': 'per 100 g', 'value': 27.7}]},
    #{'name': 'Added Sugars', 'unit': 'g', 'values': [{'base': 'per 100 g', 'value': 26.9}]},
    #{'name': 'Cholesterol', 'unit': 'mg', 'values': [{'base': 'per 100 g', 'value': 0}]},
    #{'name': 'Sodium', 'unit': 'mg', 'values': [{'base': 'per 100 g', 'value': 281}]}],
    
    #'packagingSize': {'quantity': 82, 'unit': 'g'},
    #'productName': 'Parle-G Gold Biscuits',
    #'servingSize': {'quantity': 18.8, 'unit': 'g'},
    #'servingsPerPack': 3.98,
    #'shelfLife': '7 months from packaging'}

    product_type = None
    calories = None
    sugar = None
    total_sugar = None
    added_sugar = None
    salt = None
    serving_size = None

    if product_info_from_db["servingSize"]["unit"].lower() == "g":
        product_type = "solid"
    elif product_info_from_db["servingSize"]["unit"].lower() == "ml":
        product_type = "liquid"
    serving_size = product_info_from_db["servingSize"]["quantity"]

    for item in product_info_from_db["nutritionalInformation"]:
        if 'energy' in item['name'].lower():
            calories = item['values'][0]['value']
        if 'total sugar' in item['name'].lower():
            total_sugar = item['values'][0]['value']
        if 'added sugar' in item['name'].lower():
            added_sugar = item['values'][0]['value']
        if 'sugar' in item['name'].lower() and 'added sugar' not in item['name'].lower() and 'total sugar' not in item['name'].lower():
            sugar = item['values'][0]['value']
        if 'salt' in item['name'].lower():
            if salt is None:
                salt = 0
            salt += item['values'][0]['value']

    if salt is None:
        salt = 0
        for item in product_info_from_db["nutritionalInformation"]:
            if 'sodium' in item['name'].lower():
                salt += item['values'][0]['value']

    if added_sugar is not None and added_sugar > 0 and sugar is None:
        sugar = added_sugar
    elif total_sugar is not None and total_sugar > 0 and added_sugar is None and sugar is None:
        sugar = total_sugar

    return product_type, calories, sugar, salt, serving_size
    
app = FastAPI(debug=True)

# Apply the decorator to your endpoint
@app.post("/api/nutrient-analysis")
@log_exceptions
async def get_nutrient_analysis(product_info: Dict[str, Any]):
    try:
        if ("nutritionalInformation" not in product_info or "servingSize" not in product_info or "quantity" not in product_info["servingSize"]):
            return ""
        if (len(product_info["nutritionalInformation"]) == 0 or product_info["servingSize"]["quantity"] == 0):
            return ""
            
        nutritional_information = product_info["nutritionalInformation"]
        logger.debug(f"Processing nutritional information: {nutritional_information}")
        
        serving_size = product_info["servingSize"]["quantity"]
        logger.debug(f"Serving size: {serving_size}")
        
        nutrient_analysis_rda = ""
        nutrient_analysis = ""
        nutritional_level = ""
                
        if nutritional_information:
            try:
                product_type, calories, sugar, salt, serving_size = find_product_nutrients(product_info)
                logger.info(
                    "find_product_nutrients successful",
                    extra={
                        'product_type': product_type,
                        'calories': calories,
                        'sugar': sugar,
                        'salt': salt,
                        'serving_size': serving_size
                    }
                )
            except Exception as e:
                logger.error(f"Error in find_product_nutrients: {str(e)}", exc_info=True)
                raise
                
            if product_type is not None and serving_size is not None and serving_size > 0:
                try:
                    nutrient_analysis = await analyze_nutrients(product_type, calories, sugar, salt, serving_size)
                    logger.debug(f"Nutrient analysis completed: {nutrient_analysis}")
                except Exception as e:
                    logger.error(f"Error in analyze_nutrients: {str(e)}", exc_info=True)
                    raise
            else:
                error_msg = "Product information in the db is corrupt"
                logger.error(error_msg, extra={
                    'product_type': product_type,
                    'serving_size': serving_size
                })
                raise HTTPException(status_code=400, detail=error_msg)

            try:
                nutrient_analysis_rda_data = await rda_analysis(nutritional_information, serving_size)
                logger.debug(
                    "RDA analysis data retrieved",
                    extra={
                        'data_type': type(nutrient_analysis_rda_data),
                        'nutrition_per_serving': nutrient_analysis_rda_data['nutritionPerServing'],
                        'user_serving_size': nutrient_analysis_rda_data['userServingSize']
                    }
                )
            except Exception as e:
                logger.error(f"Error in rda_analysis: {str(e)}", exc_info=True)
                raise

            try:
                nutrient_analysis_rda = await find_nutrition(nutrient_analysis_rda_data)
                logger.debug(f"RDA nutrient analysis completed: {nutrient_analysis_rda}")
            except Exception as e:
                logger.error(f"Error in find_nutrition: {str(e)}", exc_info=True)
                raise
                    
            try:
                nutritional_level = await analyze_nutrition_icmr_rda(nutrient_analysis, nutrient_analysis_rda)
                logger.info("Analysis completed successfully")
                return nutritional_level
            except Exception as e:
                logger.error(f"Error in analyze_nutrition_icmr_rda: {str(e)}", exc_info=True)
                raise
                
        else:
            error_msg = "Nutritional information is required"
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)

    except HTTPException as http_ex:
        logger.warning(
            f"HTTP error occurred: {http_ex.detail}",
            extra={'status_code': http_ex.status_code}
        )
        raise http_ex
        
    except Exception as e:
        logger.error("Unexpected error occurred", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
