from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
import uvicorn

from disease_model import predict_disease
from chatbot import KrishiMitra
from pest_model import predict_pest

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate the chatbot
chatbot = KrishiMitra()

# Disease Detection Endpoint
@app.post("/disease-prediction/")
async def disease_prediction(file: UploadFile = File(...)):
    """Handle image upload and return disease prediction."""
    contents = await file.read()

    # Get the prediction result
    result = predict_disease(contents)

    return JSONResponse(content=result)

# Pest Detection Endpoint
@app.post("/pest-prediction/")
async def pest_prediction(file: UploadFile = File(...)):
    """Handle image upload and return pest prediction."""
    contents = await file.read()
    # Get the prediction result
    result = predict_pest(contents)
    return JSONResponse(content=result)


# Chatbot Interaction Endpoint with Weather Context
@app.get("/chatbot/")
async def chatbot_response(query: str = Query(..., title="User query")):
    """Get AI-generated chatbot response with weather context."""
    response = chatbot.generate_ai_response(query)
    return JSONResponse(content={"response": response})

@app.get("/weather/")
async def get_weather_data():
    """
    Endpoint to get weather updates.
    """
    location = chatbot.location  # Get the current location from the chatbot instance
    weather_data = chatbot.get_weather_data(location)
    
    if weather_data:
        return JSONResponse(content=weather_data)
    else:
        return JSONResponse(content={"error": "Failed to fetch weather data"}, status_code=500)

# Crop Advice Endpoint
@app.get("/crop-advice/")
async def crop_advice(crop_name: str = Query(..., title="Crop name")):
    """Get crop-specific advice."""
    advice = chatbot.get_crop_specific_advice(crop_name)
    return JSONResponse(content={"advice": advice})

# Set Location Endpoint (PUT)
@app.put("/set-location/")
async def set_location(location: str = Query(..., description="Location to set for chatbot")):
    """
    Update the chatbot's location dynamically.
    """
    chatbot.location = location  # Set the new location dynamically
    return JSONResponse(content={"message": f"Location set to {location}"})

# Run FastAPI server
if __name__ == "__main__":
    # uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
