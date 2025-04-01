# Home Energy Consumption Optimizer

A Streamlit application that helps users analyze and optimize their home energy usage with AI-powered insights and recommendations.

## Features

- **Dashboard Overview**: Key metrics and quick insights about your energy usage
- **Usage Analysis**: Detailed analysis of consumption patterns by time period and appliance
- **Predictions & Forecasting**: ML-powered predictions of future energy usage and costs
- **Energy Saving Recommendations**: Personalized suggestions to reduce consumption
- **Appliance Insights**: Detailed breakdown of how different appliances contribute to your bill
- **Solar Potential**: Analysis of solar energy potential for your home
- **Carbon Footprint**: Environmental impact analysis and reduction opportunities

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the app:
   ```
   streamlit run app.py
   ```

## Usage

- Use the sidebar to navigate between different sections
- Upload your own energy data (CSV format) or use sample data
- Explore insights and recommendations based on your energy consumption patterns

## Data Format

If uploading your own data, ensure it includes at least these columns:
- `date`: Date of energy reading (YYYY-MM-DD format)
- `total_kwh`: Total energy consumption in kilowatt-hours

For enhanced insights, consider including:
- Appliance-level breakdowns (`hvac_kwh`, `water_heater_kwh`, etc.)
- Temperature data
- Price information

## Deployment

This app can be deployed to Streamlit Cloud:
1. Push your code to GitHub
2. Connect to Streamlit Cloud
3. Deploy from your GitHub repository

## License

MIT