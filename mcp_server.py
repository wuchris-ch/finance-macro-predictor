#!/usr/bin/env python3
"""
M2 Money Supply Prediction MCP Server

This MCP server exposes tools for predicting M2 money supply values
using a trained Random Forest model.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any

import pandas as pd
import numpy as np
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from ml_model import M2Predictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize the MCP server
app = Server("m2-predictor")

# Global variables for model and data
predictor: M2Predictor = None
df: pd.DataFrame = None


def load_model_and_data():
    """Load the trained model and historical data on server startup"""
    global predictor, df
    
    try:
        logger.info("Loading trained M2 prediction model...")
        predictor = M2Predictor()
        predictor.load_model('m2_model.pkl')
        logger.info("Model loaded successfully")
        
        logger.info("Loading historical M2 data...")
        df = pd.read_csv('M2SL.csv')
        df['observation_date'] = pd.to_datetime(df['observation_date'])
        logger.info(f"Loaded {len(df)} historical observations")
        
    except FileNotFoundError as e:
        logger.error(f"Required file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading model or data: {e}")
        raise


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools for M2 prediction"""
    return [
        Tool(
            name="predict_m2_future",
            description=(
                "Predict future M2 money supply values for a specified number of months. "
                "Uses a trained Random Forest model with R² = 0.7958 and MAPE = 0.69%. "
                "Returns predictions with dates and values in billions of dollars."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "months": {
                        "type": "integer",
                        "description": "Number of months to predict into the future",
                        "default": 12,
                        "minimum": 1,
                        "maximum": 24
                    }
                }
            }
        ),
        Tool(
            name="get_m2_current",
            description=(
                "Get the most recent M2 money supply value from the historical dataset. "
                "Returns the latest observation date and M2 value in billions of dollars."
            ),
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="get_m2_statistics",
            description=(
                "Get statistical summary of M2 money supply for a specified time period. "
                "Returns mean, minimum, maximum, and growth rate statistics. "
                "Useful for understanding historical trends and volatility."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "period": {
                        "type": "string",
                        "description": "Time period for statistics",
                        "enum": ["1year", "5year", "10year", "all"],
                        "default": "all"
                    }
                }
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Handle tool calls for M2 predictions and statistics"""
    
    try:
        if name == "predict_m2_future":
            return await predict_m2_future(arguments)
        elif name == "get_m2_current":
            return await get_m2_current(arguments)
        elif name == "get_m2_statistics":
            return await get_m2_statistics(arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")
            
    except Exception as e:
        logger.error(f"Error executing tool {name}: {e}")
        return [TextContent(
            type="text",
            text=f"Error: {str(e)}"
        )]


async def predict_m2_future(arguments: dict) -> list[TextContent]:
    """Predict future M2 values"""
    months = arguments.get("months", 12)
    
    # Validate input
    if not isinstance(months, int) or months < 1 or months > 24:
        return [TextContent(
            type="text",
            text="Error: months must be an integer between 1 and 24"
        )]
    
    logger.info(f"Predicting M2 for next {months} months...")
    
    # Generate predictions
    predictions = predictor.predict_future(df, n_months=months)
    
    # Format results
    results = {
        "prediction_date": datetime.now().isoformat(),
        "months_predicted": months,
        "model_performance": {
            "r2_score": 0.7958,
            "mape_percent": 0.69
        },
        "predictions": []
    }
    
    for _, row in predictions.iterrows():
        results["predictions"].append({
            "date": row['observation_date'].strftime('%Y-%m-%d'),
            "predicted_m2_billions": round(row['predicted_M2SL'], 2)
        })
    
    # Calculate summary statistics
    pred_values = [p["predicted_m2_billions"] for p in results["predictions"]]
    results["summary"] = {
        "mean_predicted_m2": round(np.mean(pred_values), 2),
        "min_predicted_m2": round(np.min(pred_values), 2),
        "max_predicted_m2": round(np.max(pred_values), 2),
        "total_growth_percent": round(
            ((pred_values[-1] - pred_values[0]) / pred_values[0]) * 100, 2
        ) if len(pred_values) > 1 else 0
    }
    
    return [TextContent(
        type="text",
        text=json.dumps(results, indent=2)
    )]


async def get_m2_current(arguments: dict) -> list[TextContent]:
    """Get the most recent M2 value"""
    logger.info("Retrieving current M2 value...")
    
    # Get the last row
    last_row = df.iloc[-1]
    
    result = {
        "observation_date": last_row['observation_date'].strftime('%Y-%m-%d'),
        "m2_billions": round(last_row['M2SL'], 2),
        "data_source": "Federal Reserve Economic Data (FRED)",
        "series_id": "M2SL",
        "frequency": "Monthly",
        "units": "Billions of Dollars",
        "seasonal_adjustment": "Seasonally Adjusted"
    }
    
    return [TextContent(
        type="text",
        text=json.dumps(result, indent=2)
    )]


async def get_m2_statistics(arguments: dict) -> list[TextContent]:
    """Get statistical summary for a time period"""
    period = arguments.get("period", "all")
    
    logger.info(f"Calculating M2 statistics for period: {period}")
    
    # Filter data based on period
    if period == "1year":
        cutoff_date = df['observation_date'].max() - pd.DateOffset(years=1)
        period_df = df[df['observation_date'] >= cutoff_date]
        period_label = "Last 12 Months"
    elif period == "5year":
        cutoff_date = df['observation_date'].max() - pd.DateOffset(years=5)
        period_df = df[df['observation_date'] >= cutoff_date]
        period_label = "Last 5 Years"
    elif period == "10year":
        cutoff_date = df['observation_date'].max() - pd.DateOffset(years=10)
        period_df = df[df['observation_date'] >= cutoff_date]
        period_label = "Last 10 Years"
    else:  # all
        period_df = df
        period_label = "All Time"
    
    # Calculate statistics
    m2_values = period_df['M2SL'].values
    
    # Calculate growth rate
    if len(period_df) > 1:
        start_value = period_df.iloc[0]['M2SL']
        end_value = period_df.iloc[-1]['M2SL']
        total_growth = ((end_value - start_value) / start_value) * 100
        
        # Annualized growth rate
        years = (period_df.iloc[-1]['observation_date'] - 
                period_df.iloc[0]['observation_date']).days / 365.25
        annualized_growth = ((end_value / start_value) ** (1 / years) - 1) * 100 if years > 0 else 0
    else:
        total_growth = 0
        annualized_growth = 0
    
    result = {
        "period": period_label,
        "date_range": {
            "start": period_df.iloc[0]['observation_date'].strftime('%Y-%m-%d'),
            "end": period_df.iloc[-1]['observation_date'].strftime('%Y-%m-%d')
        },
        "observations": len(period_df),
        "statistics": {
            "mean_m2_billions": round(np.mean(m2_values), 2),
            "median_m2_billions": round(np.median(m2_values), 2),
            "min_m2_billions": round(np.min(m2_values), 2),
            "max_m2_billions": round(np.max(m2_values), 2),
            "std_dev_billions": round(np.std(m2_values), 2)
        },
        "growth": {
            "total_growth_percent": round(total_growth, 2),
            "annualized_growth_percent": round(annualized_growth, 2),
            "start_value_billions": round(period_df.iloc[0]['M2SL'], 2),
            "end_value_billions": round(period_df.iloc[-1]['M2SL'], 2)
        }
    }
    
    return [TextContent(
        type="text",
        text=json.dumps(result, indent=2)
    )]


async def main():
    """Main entry point for the MCP server"""
    logger.info("Starting M2 Prediction MCP Server...")
    
    # Load model and data
    load_model_and_data()
    
    logger.info("Server ready to accept connections")
    
    # Run the server using stdio transport
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())