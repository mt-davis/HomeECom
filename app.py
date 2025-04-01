import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Page config
st.set_page_config(page_title="Home Energy Optimizer", page_icon="⚡", layout="wide")

# CSS styling (simplified)
st.markdown("""
<style>
    .success-metric { color: #27ae60; font-weight: bold; }
    .warning-metric { color: #e67e22; font-weight: bold; }
    .danger-metric { color: #e74c3c; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Generate sample data
@st.cache_data
def generate_sample_data(num_days=365):
    np.random.seed(42)
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=num_days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    seasonal_factor = np.sin(np.linspace(0, 2*np.pi, len(dates))) * 10 + 30
    
    df = pd.DataFrame({
        'date': dates,
        'total_kwh': seasonal_factor + np.random.normal(0, 5, len(dates)),
        'temperature': 70 + seasonal_factor/2 + np.random.normal(0, 5, len(dates))
    })
    
    df['dayofweek'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['day'] = df['date'].dt.day
    
    # Appliance breakdown
    df['hvac_kwh'] = df['total_kwh'] * (0.45 + np.random.normal(0, 0.05, len(df)))
    df['water_heater_kwh'] = df['total_kwh'] * (0.15 + np.random.normal(0, 0.03, len(df)))
    df['refrigerator_kwh'] = df['total_kwh'] * (0.08 + np.random.normal(0, 0.01, len(df)))
    df['lighting_kwh'] = df['total_kwh'] * (0.12 + np.random.normal(0, 0.03, len(df)))
    df['other_kwh'] = df['total_kwh'] - (df['hvac_kwh'] + df['water_heater_kwh'] + df['refrigerator_kwh'] + df['lighting_kwh'])
    
    # Pricing data
    df['price_per_kwh'] = 0.13 + np.random.normal(0, 0.01, len(df))
    df['daily_cost'] = df['total_kwh'] * df['price_per_kwh']
    df['peak_consumption'] = np.where(df['total_kwh'] > df['total_kwh'].quantile(0.7), 1, 0)
    
    return df

# ML model functions
def prepare_features(df):
    features = df[['temperature', 'dayofweek', 'is_weekend', 'month', 'day']]
    return pd.get_dummies(features, columns=['month', 'dayofweek'])

def train_model(df):
    try:
        features = prepare_features(df)
        target = df['total_kwh']
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return model, features.columns, mae, r2
    except Exception as e:
        st.error(f"Error training model: {e}")
        return None, None, 0, 0

def predict_future(model, df, feature_names, days=30):
    if model is None:
        return pd.DataFrame()
    
    last_date = df['date'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days, freq='D')
    
    future_df = pd.DataFrame({'date': future_dates})
    future_df['temperature'] = 70 + np.sin(np.linspace(0, 2*np.pi, len(future_dates))) * 10 + np.random.normal(0, 5, len(future_dates))
    future_df['dayofweek'] = future_df['date'].dt.dayofweek
    future_df['is_weekend'] = future_df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
    future_df['month'] = future_df['date'].dt.month
    future_df['day'] = future_df['date'].dt.day
    
    try:
        future_features = pd.get_dummies(future_df[['temperature', 'dayofweek', 'is_weekend', 'month', 'day']], 
                                        columns=['month', 'dayofweek'])
        
        for col in feature_names:
            if col not in future_features.columns:
                future_features[col] = 0
        future_features = future_features[feature_names]
        
        future_df['predicted_kwh'] = model.predict(future_features)
        last_price = df['price_per_kwh'].iloc[-1]
        future_df['price_per_kwh'] = last_price
        future_df['predicted_cost'] = future_df['predicted_kwh'] * future_df['price_per_kwh']
        
        return future_df
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return pd.DataFrame()

# Energy recommendations
def get_recommendations(df):
    avg_daily = df['total_kwh'].mean()
    recommendations = [
        {
            "category": "HVAC",
            "title": "Install a smart thermostat",
            "description": "Smart thermostats can optimize heating and cooling schedules automatically.",
            "impact": "High",
            "savings_kwh": round(avg_daily * 0.15 * 30, 1),
            "savings_dollars": round(avg_daily * 0.15 * 30 * df['price_per_kwh'].mean(), 2),
            "implementation_cost": "$100-$300",
            "payback_period": "4-8 months"
        },
        {
            "category": "Lighting",
            "title": "Switch to LED bulbs",
            "description": "Replace incandescent and CFL bulbs with energy-efficient LEDs.",
            "impact": "Medium",
            "savings_kwh": round(df['lighting_kwh'].mean() * 0.7 * 30, 1),
            "savings_dollars": round(df['lighting_kwh'].mean() * 0.7 * 30 * df['price_per_kwh'].mean(), 2),
            "implementation_cost": "$20-$100",
            "payback_period": "2-3 months"
        },
        {
            "category": "Water Heating",
            "title": "Lower water heater temperature",
            "description": "Reduce your water heater temperature to 120°F (49°C).",
            "impact": "Medium",
            "savings_kwh": round(df['water_heater_kwh'].mean() * 0.1 * 30, 1),
            "savings_dollars": round(df['water_heater_kwh'].mean() * 0.1 * 30 * df['price_per_kwh'].mean(), 2),
            "implementation_cost": "Free",
            "payback_period": "Immediate"
        }
    ]
    return recommendations

# Calculate savings potential
def calc_savings_potential(df, recommendations):
    total_kwh_savings = sum(rec['savings_kwh'] for rec in recommendations)
    total_dollar_savings = sum(rec['savings_dollars'] for rec in recommendations)
    monthly_avg_kwh = df['total_kwh'].mean() * 30
    monthly_avg_cost = df['daily_cost'].mean() * 30
    
    return {
        'total_kwh_savings': total_kwh_savings,
        'total_dollar_savings': total_dollar_savings,
        'percent_kwh_savings': (total_kwh_savings / monthly_avg_kwh) * 100,
        'percent_dollar_savings': (total_dollar_savings / monthly_avg_cost) * 100,
        'yearly_dollar_savings': total_dollar_savings * 12
    }

# Find anomalies
def find_anomalies(df, window=7, threshold=2.0):
    try:
        rolling_mean = df['total_kwh'].rolling(window=window).mean()
        rolling_std = df['total_kwh'].rolling(window=window).std()
        z_scores = (df['total_kwh'] - rolling_mean) / rolling_std
        df['is_anomaly'] = (z_scores.abs() > threshold) & (~z_scores.isna())
    except:
        df['is_anomaly'] = False
    return df

# Solar potential estimation
def solar_potential(df):
    avg_daily_kwh = df['total_kwh'].mean()
    system_size_kw = avg_daily_kwh / 5  # Assuming 5 sun hours per day
    system_cost = system_size_kw * 1000 * 2.50  # $2.50 per watt
    tax_credit = system_cost * 0.30
    net_cost = system_cost - tax_credit
    annual_production = avg_daily_kwh * 365
    annual_savings = annual_production * df['price_per_kwh'].mean()
    
    return {
        'system_size_kw': system_size_kw,
        'annual_production_kwh': annual_production,
        'system_cost': system_cost,
        'net_cost': net_cost,
        'annual_savings': annual_savings,
        'payback_years': net_cost / annual_savings
    }

# Carbon footprint
def carbon_footprint(df):
    co2_per_kwh = 0.92  # lbs CO2 per kWh
    total_kwh = df['total_kwh'].sum()
    total_co2_lbs = total_kwh * co2_per_kwh
    total_co2_tons = total_co2_lbs / 2000
    
    return {
        'total_co2_tons': total_co2_tons,
        'daily_avg_co2_lbs': (df['total_kwh'].mean() * co2_per_kwh),
        'trees_needed': int(total_co2_tons * 45)
    }

# SIDEBAR
st.sidebar.image("https://images.unsplash.com/photo-1473341304170-971dccb5ac1e?auto=format&fit=crop&w=600", use_container_width=True)
st.sidebar.title("⚡ Energy Optimizer")

# Data options
data_option = st.sidebar.radio("Data source:", ["Sample data", "Upload your own"])
df = None

if data_option == "Sample data":
    num_days = st.sidebar.slider("Data period (days)", 180, 730, 365)
    df = generate_sample_data(num_days)
    st.sidebar.success(f"Generated {num_days} days of sample data")
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            required_columns = ['date', 'total_kwh']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.sidebar.error(f"Missing columns: {', '.join(missing_columns)}")
                df = generate_sample_data()
            else:
                df['date'] = pd.to_datetime(df['date'])
                st.sidebar.success("Data loaded successfully")
        except:
            st.sidebar.error("Error reading file")
            df = generate_sample_data()
    else:
        df = generate_sample_data()

# Navigation
page = st.sidebar.selectbox("Navigation", [
    "Dashboard Overview", 
    "Usage Analysis", 
    "Predictions & Forecasting", 
    "Energy Saving Recommendations", 
    "Solar Potential"
])

st.sidebar.info("This app helps you analyze energy usage and find ways to save.")

# MAIN CONTENT
if df is not None:
    # Process data once
    df = find_anomalies(df)
    model, feature_names, model_mae, model_r2 = train_model(df)
    recommendations = get_recommendations(df)
    savings = calc_savings_potential(df, recommendations)
    solar = solar_potential(df)
    carbon = carbon_footprint(df)
    future_df = predict_future(model, df, feature_names)
    
    if page == "Dashboard Overview":
        st.title("Home Energy Dashboard")
        st.write("Analyze your energy usage, discover insights, and find ways to save.")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Avg. Daily Usage", f"{df['total_kwh'].mean():.1f} kWh")
        with col2: st.metric("Avg. Daily Cost", f"${df['daily_cost'].mean():.2f}")
        with col3: st.metric("Monthly Savings Potential", f"${savings['total_dollar_savings']:.2f}")
        with col4: st.metric("Anomalies Detected", f"{df['is_anomaly'].sum()}")
        
        # Recent usage
        st.subheader("Recent Energy Consumption")
        recent_df = df.sort_values('date').tail(30)
        fig = px.line(recent_df, x='date', y='total_kwh', title="Last 30 Days")
        
        # Mark anomalies
        anomalies = recent_df[recent_df['is_anomaly']]
        if not anomalies.empty:
            fig.add_scatter(x=anomalies['date'], y=anomalies['total_kwh'], 
                          mode='markers', marker=dict(size=10, color='red', symbol='x'),
                          name='Anomaly')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Two columns for insights
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Energy by Appliance")
            appliance_data = df[['hvac_kwh', 'water_heater_kwh', 'refrigerator_kwh', 'lighting_kwh', 'other_kwh']].mean()
            appliance_labels = ['HVAC', 'Water Heater', 'Refrigerator', 'Lighting', 'Other']
            fig = px.pie(values=appliance_data.values, names=appliance_labels)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("30-Day Cost Forecast")
            current_month_cost = df[df['date'] >= (df['date'].max() - timedelta(days=30))]['daily_cost'].sum()
            next_month_cost = future_df['predicted_cost'].sum() if not future_df.empty else 0
            
            fig = go.Figure()
            fig.add_trace(go.Indicator(
                mode="number+delta",
                value=next_month_cost,
                title={"text": "Next 30 Days Cost"},
                delta={'reference': current_month_cost, 'relative': True},
                number={'prefix': "$", 'valueformat': '.2f'}
            ))
            st.plotly_chart(fig, use_container_width=True)
        
        # Top recommendations
        st.subheader("Top Savings Opportunities")
        sorted_recs = sorted(recommendations, key=lambda x: x['savings_dollars'], reverse=True)
        for rec in sorted_recs[:2]:
            with st.expander(f"{rec['title']} - Save ${rec['savings_dollars']:.2f}/month"):
                st.write(f"**Category**: {rec['category']}")
                st.write(f"**Description**: {rec['description']}")
                st.write(f"**Impact**: {rec['impact']}")
                st.write(f"**Cost**: {rec['implementation_cost']}")
    
    elif page == "Usage Analysis":
        st.title("Energy Usage Analysis")
        
        # Date filter
        st.subheader("Select Time Period")
        min_date, max_date = df['date'].min().date(), df['date'].max().date()
        col1, col2 = st.columns(2)
        with col1: start_date = st.date_input("Start", min_date, min_value=min_date, max_value=max_date)
        with col2: end_date = st.date_input("End", max_date, min_value=min_date, max_value=max_date)
        
        filtered_df = df[(df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)]
        
        # Stats
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Total Usage", f"{filtered_df['total_kwh'].sum():.1f} kWh")
        with col2: st.metric("Average Daily", f"{filtered_df['total_kwh'].mean():.1f} kWh")
        with col3: st.metric("Total Cost", f"${filtered_df['daily_cost'].sum():.2f}")
        
        # Time series
        st.subheader("Energy Over Time")
        viz_type = st.radio("View", ["Daily", "Weekly", "Monthly"])
        
        if viz_type == "Daily":
            plot_df = filtered_df
            x_col = 'date'
        elif viz_type == "Weekly":
            plot_df = filtered_df.copy()
            plot_df['week'] = plot_df['date'].dt.isocalendar().week
            plot_df = plot_df.groupby(['year', 'week'])['total_kwh'].mean().reset_index()
            plot_df['date_label'] = plot_df.apply(lambda x: f"{x['year']}-W{x['week']}", axis=1)
            x_col = 'date_label'
        else:
            plot_df = filtered_df.groupby(['year', 'month'])['total_kwh'].mean().reset_index()
            plot_df['date_label'] = plot_df.apply(lambda x: f"{x['year']}-{x['month']}", axis=1)
            x_col = 'date_label'
        
        fig = px.line(plot_df, x=x_col, y='total_kwh', title=f"Energy Consumption ({viz_type})")
        st.plotly_chart(fig, use_container_width=True)
        
        # Patterns
        st.subheader("Usage Patterns")
        pattern = st.selectbox("Analyze by", ["Day of Week", "Temperature"])
        
        if pattern == "Day of Week":
            dow_avg = filtered_df.groupby('dayofweek')['total_kwh'].mean().reset_index()
            dow_avg['day_name'] = dow_avg['dayofweek'].apply(
                lambda x: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][x])
            fig = px.bar(dow_avg, x='day_name', y='total_kwh', title="By Day of Week")
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.scatter(filtered_df, x='temperature', y='total_kwh', 
                           trendline="ols", title="Energy vs Temperature")
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Predictions & Forecasting":
        st.title("Energy Predictions & Forecasting")
        
        if model is not None:
            st.info(f"Model quality - MAE: {model_mae:.2f} kWh, R²: {model_r2:.2f}")
            
            forecast_days = st.slider("Forecast days", 7, 90, 30)
            future_df = predict_future(model, df, feature_names, days=forecast_days)
            
            st.subheader("Energy Usage Forecast")
            hist_plot = df[['date', 'total_kwh']].rename(columns={'total_kwh': 'value'})
            hist_plot['type'] = 'Historical'
            
            future_plot = future_df[['date', 'predicted_kwh']].rename(columns={'predicted_kwh': 'value'})
            future_plot['type'] = 'Predicted'
            
            plot_df = pd.concat([hist_plot.tail(30), future_plot])
            fig = px.line(plot_df, x='date', y='value', color='type',
                         title="Energy Usage: Historical and Forecasted")
            st.plotly_chart(fig, use_container_width=True)
            
            # Cost forecast
            st.subheader("Cost Forecast")
            last_month = df[df['date'] >= (df['date'].max() - timedelta(days=30))]['daily_cost'].sum()
            forecast = future_df['predicted_cost'].sum()
            
            col1, col2 = st.columns(2)
            with col1: st.metric("Last 30 Days", f"${last_month:.2f}")
            with col2: st.metric("Next 30 Days", f"${forecast:.2f}", 
                              delta=f"{((forecast/last_month)-1)*100:.1f}%")
            
            # What-if scenario
            st.subheader("What-If Scenarios")
            scenario = st.selectbox("Choose scenario", [
                "Implement all recommendations",
                "Install a smart thermostat",
                "Switch to LED lighting"
            ])
            
            if scenario == "Implement all recommendations":
                savings_pct = savings['percent_kwh_savings'] / 100
            elif scenario == "Install a smart thermostat":
                savings_pct = 0.15 * (df['hvac_kwh'].sum() / df['total_kwh'].sum())
            else:
                savings_pct = 0.7 * (df['lighting_kwh'].sum() / df['total_kwh'].sum())
            
            baseline = future_df['predicted_cost'].sum()
            scenario_cost = baseline * (1 - savings_pct)
            
            col1, col2 = st.columns(2)
            with col1: st.metric("Baseline Cost", f"${baseline:.2f}")
            with col2: st.metric("Scenario Cost", f"${scenario_cost:.2f}", 
                              delta=f"-${baseline - scenario_cost:.2f}")
    
    elif page == "Energy Saving Recommendations":
        st.title("Energy Saving Recommendations")
        
        # Overall savings
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Monthly Savings", f"${savings['total_dollar_savings']:.2f}")
        with col2: st.metric("Annual Savings", f"${savings['yearly_dollar_savings']:.2f}")
        with col3: st.metric("Reduction", f"{savings['percent_kwh_savings']:.1f}%")
        
        # All recommendations
        categories = sorted(list(set(rec['category'] for rec in recommendations)))
        selected = st.radio("Filter by", ["All Categories"] + categories)
        
        filtered_recs = recommendations if selected == "All Categories" else [r for r in recommendations if r['category'] == selected]
        sorted_recs = sorted(filtered_recs, key=lambda x: x['savings_dollars'], reverse=True)
        
        for i, rec in enumerate(sorted_recs):
            with st.expander(f"{i+1}. {rec['title']} - Save ${rec['savings_dollars']:.2f}/month"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Category**: {rec['category']}")
                    st.write(f"**Impact**: {rec['impact']}")
                    st.write(f"**Cost**: {rec['implementation_cost']}")
                with col2:
                    st.write(f"**Energy Savings**: {rec['savings_kwh']:.1f} kWh/month")
                    st.write(f"**Annual Savings**: ${rec['savings_dollars']*12:.2f}/year")
                    st.write(f"**Payback**: {rec['payback_period']}")
                st.write(f"**Description**: {rec['description']}")
    
    elif page == "Solar Potential":
        st.title("Solar Energy Potential")
        
        # System details
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("System Size", f"{solar['system_size_kw']:.1f} kW")
        with col2: st.metric("Annual Production", f"{solar['annual_production_kwh']:.0f} kWh")
        with col3: st.metric("Roof Area Needed", f"{solar['system_size_kw'] * 100:.0f} sq ft")
        
        # Financial analysis
        st.subheader("Solar Investment Analysis")
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("System Cost", f"${solar['system_cost']:.0f}")
        with col2: st.metric("After Tax Credit", f"${solar['net_cost']:.0f}")
        with col3: st.metric("Payback Period", f"{solar['payback_years']:.1f} years")
        
        # Monthly comparison
        st.subheader("Production vs. Consumption")
        monthly_consumption = df.groupby('month')['total_kwh'].mean() * 30
        monthly_consumption = monthly_consumption.reset_index()
        
        # Production factors by month (northern hemisphere)
        production_factors = {
            1: 0.7, 2: 0.8, 3: 0.9, 4: 1.0, 5: 1.1, 6: 1.2,
            7: 1.2, 8: 1.1, 9: 1.0, 10: 0.9, 11: 0.7, 12: 0.6
        }
        
        monthly_consumption['solar_production'] = monthly_consumption['month'].apply(
            lambda m: solar['system_size_kw'] * 30 * 5 * production_factors[m])
        
        monthly_consumption['month_name'] = monthly_consumption['month'].apply(
            lambda x: datetime(2020, x, 1).strftime('%b'))
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=monthly_consumption['month_name'],
            y=monthly_consumption['total_kwh'],
            name='Consumption',
            marker_color='#e74c3c'
        ))
        fig.add_trace(go.Bar(
            x=monthly_consumption['month_name'],
            y=monthly_consumption['solar_production'],
            name='Solar Production',
            marker_color='#2ecc71'
        ))
        fig.update_layout(title='Monthly Comparison', barmode='group')
        st.plotly_chart(fig, use_container_width=True)
        
        # ROI chart
        st.subheader("25-Year Return on Investment")
        years = list(range(26))
        initial_investment = -solar['net_cost']
        annual_return = solar['annual_savings']
        cumulative_returns = [initial_investment + annual_return * year for year in years]
        
        roi_df = pd.DataFrame({
            'Year': years,
            'Return': cumulative_returns
        })
        
        fig = px.line(roi_df, x='Year', y='Return', title="25-Year Solar ROI")
        fig.add_shape(type="line", x0=0, y0=0, x1=25, y1=0,
                     line=dict(color="black", width=2, dash="dash"))
        st.plotly_chart(fig, use_container_width=True)
        
        # Environmental impact
        st.subheader("Environmental Impact")
        co2_per_kwh = 0.92 / 2000  # tons of CO2 per kWh
        annual_co2_offset = solar['annual_production_kwh'] * co2_per_kwh
        lifetime_co2_offset = annual_co2_offset * 25
        
        st.metric("25-Year CO₂ Offset", f"{lifetime_co2_offset:.1f} tons")
        st.write(f"Equivalent to planting {int(lifetime_co2_offset * 45)} trees")

else:
    st.error("Error loading data. Please try again.")