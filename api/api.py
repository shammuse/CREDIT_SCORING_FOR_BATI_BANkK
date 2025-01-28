from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import pandas as pd
from data_preprocess import preprocess_transactions, load_model
from pydantic import BaseModel
from typing import Optional
import uvicorn

class InputData(BaseModel):
    TransactionId: int
    BatchId: int
    AccountId: int
    SubscriptionId: int
    CustomerId: int
    CurrencyCode: str
    CountryCode: str
    ProviderId: int
    ProductId: int
    ProductCategory: str
    ChannelId: str
    Amount: float
    Value: float
    TransactionStartTime: str
    PricingStrategy: str

app = FastAPI()
model = load_model()
# Define the root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Credit Scoring API"}

@app.post("/predict")
async def predict(input_data: InputData):
    """Predicts using the loaded model."""
    try:
        # Prepare input data
        input_dict = input_data.dict()
        input_features = pd.DataFrame([input_dict])
         # Drop unnecessay features
        df = input_features.drop(['BatchId', 'AccountId', 'SubscriptionId', 'CurrencyCode', 'CountryCode', 'ChannelId', 'ProviderId',
                                    'ProductId', 'ProductCategory'], axis=1)
        # Preprocess data and get model input
        X = preprocess_transactions(df)

        # Make prediction
        prediction = model.predict(X)[0]

        return JSONResponse(content={"prediction": int(prediction)})

    except HTTPException as http_exc:
         raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the application
if __name__ == "__main__":
    pass
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)


