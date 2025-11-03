import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

# Import prediction modules
from funding_prediction import predict_funding, get_funding_range_description, get_startup_recommendations

# Try to import plotly with error handling
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.error("Plotly is not installed. Please install it using: pip install plotly")
    st.stop()

# Page configs
st.set_page_config(
    page_title="Indian Startup Ecosystem Analysis & Funding Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 2rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
            
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .insight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #1f77b4;
        margin: 1rem 0;
    }
    .prediction-card {
        background-color: #f0f8ff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .recommendation-card {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
        margin-bottom: 10px;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    .section-divider {
        margin-top: 2rem;
        margin-bottom: 2rem;
        border-top: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the processed data"""
    try:
        # Try to load processed data first
        df = pd.read_csv('startup_data_processed.csv')
        df['date'] = pd.to_datetime(df['date'])
        return df
    except FileNotFoundError:
        try:
            # If processed file doesn't exist, load and clean the original
            df = pd.read_csv('startup_cleaned.csv')
            
            # Basic cleaning
            df.columns = df.columns.str.strip().str.lower()
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
            df['amount'] = df['amount'].fillna(0)
            
            # Extract year and month
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            
            # Clean text columns
            df['startup'] = df['startup'].astype(str).str.strip()
            df['city'] = df['city'].astype(str).str.strip()
            df['vertical'] = df['vertical'].astype(str).str.strip()
            df['round'] = df['round'].astype(str).str.strip()
            
            # Remove rows with missing critical data
            df = df.dropna(subset=['date', 'startup'])
            
            # Save processed data
            df.to_csv('startup_data_processed.csv', index=False)
            return df
            
        except FileNotFoundError:
            st.error("Data file 'startup_cleaned.csv' not found. Please make sure the file exists in the project directory.")
            st.stop()
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.stop()

def create_funding_timeline(df):
    """Create interactive funding timeline"""
    monthly_funding = df.groupby(['year', 'month']).agg({
        'amount': 'sum',
        'startup': 'count'
    }).reset_index()
    
    monthly_funding['date'] = pd.to_datetime(monthly_funding[['year', 'month']].assign(day=1))
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=monthly_funding['date'], y=monthly_funding['amount'],
                  mode='lines+markers', name='Funding Amount (Cr)',
                  line=dict(color='#1f77b4', width=3),
                  hovertemplate='<b>%{x}</b><br>Funding: ₹%{y:.1f} Cr<extra></extra>'),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(x=monthly_funding['date'], y=monthly_funding['startup'],
                  mode='lines+markers', name='Number of Deals',
                  line=dict(color='#ff7f0e', width=2),
                  hovertemplate='<b>%{x}</b><br>Deals: %{y}<extra></extra>'),
        secondary_y=True,
    )
    
    fig.update_xaxes(title_text="Timeline")
    fig.update_yaxes(title_text="Funding Amount (₹ Crores)", secondary_y=False)
    fig.update_yaxes(title_text="Number of Deals", secondary_y=True)
    
    fig.update_layout(
        title="Indian Startup Funding Timeline",
        hovermode='x unified',
        height=500
    )
    
    return fig

def create_heatmap(df):
    """Create funding heatmap by year and round"""
    heatmap_data = df.groupby(['year', 'round'])['amount'].sum().reset_index()
    heatmap_pivot = heatmap_data.pivot(index='round', columns='year', values='amount').fillna(0)
    
    fig = px.imshow(
        heatmap_pivot,
        labels=dict(x="Year", y="Funding Round", color="Amount (₹ Cr)"),
        title="Funding Heatmap: Amount by Round and Year",
        aspect="auto",
        color_continuous_scale="Blues"
    )
    
    fig.update_layout(height=400)
    return fig

def create_city_map(df):
    """Create geographical distribution of startups"""
    city_data = df.groupby('city').agg({
        'amount': 'sum',
        'startup': 'count'
    }).reset_index()
    
    city_data = city_data.sort_values('amount', ascending=False).head(15)
    
    fig = px.scatter(
        city_data,
        x='startup',
        y='amount',
        size='amount',
        hover_name='city',
        title="City-wise Startup Distribution",
        labels={'startup': 'Number of Startups', 'amount': 'Total Funding (₹ Cr)'},
        size_max=60
    )
    
    fig.update_layout(height=500)
    return fig

def create_sector_analysis(df):
    """Create sector-wise analysis"""
    sector_data = df.groupby('vertical').agg({
        'amount': ['sum', 'mean', 'count']
    }).round(2)
    
    sector_data.columns = ['Total_Funding', 'Avg_Funding', 'Deal_Count']
    sector_data = sector_data.reset_index().sort_values('Total_Funding', ascending=False).head(12)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Total Funding by Sector', 'Average Deal Size by Sector',
                       'Number of Deals by Sector', 'Funding vs Deal Count'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "scatter"}]]
    )
    
    # Total funding
    fig.add_trace(
        go.Bar(x=sector_data['vertical'], y=sector_data['Total_Funding'],
               name='Total Funding', marker_color='skyblue'),
        row=1, col=1
    )
    
    # Average deal size
    fig.add_trace(
        go.Bar(x=sector_data['vertical'], y=sector_data['Avg_Funding'],
               name='Avg Deal Size', marker_color='lightcoral'),
        row=1, col=2
    )
    
    # Deal count
    fig.add_trace(
        go.Bar(x=sector_data['vertical'], y=sector_data['Deal_Count'],
               name='Deal Count', marker_color='lightgreen'),
        row=2, col=1
    )
    
    # Scatter plot
    fig.add_trace(
        go.Scatter(x=sector_data['Deal_Count'], y=sector_data['Total_Funding'],
                  mode='markers+text', text=sector_data['vertical'],
                  textposition='top center', name='Funding vs Deals',
                  marker=dict(size=10, color='purple')),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=False)
    fig.update_xaxes(tickangle=45)
    
    return fig

def create_round_analysis(df):
    """Create analysis by funding round"""
    round_data = df.groupby('round').agg({
        'amount': ['sum', 'mean', 'count'],
        'startup': 'nunique'
    })
    
    round_data.columns = ['Total_Funding', 'Avg_Funding', 'Deal_Count', 'Unique_Startups']
    round_data = round_data.reset_index().sort_values('Total_Funding', ascending=False)
    
    # Calculate percentage of total funding
    total_funding = round_data['Total_Funding'].sum()
    round_data['Funding_Percentage'] = (round_data['Total_Funding'] / total_funding * 100).round(1)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Funding Distribution by Round", "Average Deal Size by Round",
                        "Number of Deals by Round", "Funding Percentage by Round"),
        specs=[[{"type": "pie"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Add traces
    fig.add_trace(
        go.Pie(
            labels=round_data['round'],
            values=round_data['Total_Funding'],
            hole=0.4,
            name="Total Funding"
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=round_data['round'],
            y=round_data['Avg_Funding'],
            name="Average Deal Size",
            marker_color='coral'
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(
            x=round_data['round'],
            y=round_data['Deal_Count'],
            name="Number of Deals",
            marker_color='lightblue'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=round_data['round'],
            y=round_data['Funding_Percentage'],
            name="Funding Percentage",
            marker_color='lightgreen',
            text=round_data['Funding_Percentage'].apply(lambda x: f"{x}%"),
            textposition='auto'
        ),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=False)
    fig.update_xaxes(tickangle=45)
    
    return fig

def create_investor_analysis(df):
    """Create analysis of top investors"""
    # Extract individual investors from the investors column
    investor_list = []
    for investors_str in df['investors'].dropna():
        investors = str(investors_str).split(',')
        for investor in investors:
            investor = investor.strip()
            if investor and investor.lower() != 'nan' and len(investor) > 2:
                investor_list.append((investor, 1))
    
    # Create a DataFrame of investors and count their occurrences
    investor_df = pd.DataFrame(investor_list, columns=['investor', 'count'])
    investor_counts = investor_df.groupby('investor')['count'].sum().reset_index()
    top_investors = investor_counts.sort_values('count', ascending=False).head(15)
    
    # Create a horizontal bar chart of top investors
    fig = px.bar(
        top_investors,
        y='investor',
        x='count',
        orientation='h',
        title="Top 15 Investors by Number of Investments",
        labels={'count': 'Number of Investments', 'investor': 'Investor'},
        color='count',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(height=600)
    
    return fig

def create_funding_predictor_ui():
    """Create UI for the funding predictor"""
    st.markdown('<h2 class="sub-header">💰 Startup Funding Predictor</h2>', unsafe_allow_html=True)
    
    # Check if model exists
    model_exists = os.path.exists('funding_prediction_model.pkl')
    
    if not model_exists:
        st.warning("⚠️ Prediction model not found. Please run the model training script first.")
        
        if st.button("Train Model Now"):
            st.info("Starting model training... This may take a few minutes.")
            try:
                from model_training import train_and_save_model
                with st.spinner("Training in progress..."):
                    success = train_and_save_model()
                
                if success:
                    st.success("✅ Model training completed successfully!")
                    st.experimental_rerun()
                else:
                    st.error("❌ Model training failed. Please check the logs for details.")
            except Exception as e:
                st.error(f"❌ Error during model training: {str(e)}")
        return
    
    st.write("Enter your startup details to predict potential funding:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Collect user inputs
        city = st.selectbox(
            "City",
            options=[
                "Bengaluru", "Mumbai", "Delhi", "Gurugram", "Hyderabad", 
                "Chennai", "Pune", "Noida", "Ahmedabad", "Kolkata", "Other"
            ]
        )
        
        vertical = st.selectbox(
            "Sector/Vertical",
            options=[
                "eCommerce", "Fintech", "EdTech", "Health", "Consumer Tech",
                "Food & Beverage", "Tech", "SaaS", "Logistics", "Real Estate",
                "Transportation", "Hospitality", "Media", "Gaming", "Other"
            ]
        )
    
    with col2:
        round = st.selectbox(
            "Funding Round",
            options=[
                "Seed", "Angel", "Pre-Series A", "Series A", "Series B",
                "Series C", "Series D", "Series E", "Bridge", "Private Equity", "Debt"
            ]
        )
        
        investor_count = st.number_input(
            "Number of Investors",
            min_value=1,
            max_value=50,
            value=2,
            step=1
        )
    
    funding_date = st.date_input(
        "Expected Funding Date",
        value=datetime.now()
    )
    
    # Create prediction button
    predict_button = st.button("Predict Funding Amount", key="predict_funding_button")
    
    # Display prediction results when button is clicked
    if predict_button:
        input_data = {
            'city': city,
            'vertical': vertical,
            'round': round,
            'investor_count': investor_count,
            'date': funding_date
        }
        
        with st.spinner("Calculating prediction..."):
            prediction_result = predict_funding(input_data)
        
        if prediction_result['success']:
            predicted_amount = prediction_result['predicted_amount']
            funding_range = get_funding_range_description(predicted_amount)
            recommendations = get_startup_recommendations(input_data, predicted_amount)
            
            # Display prediction in a nice card
            st.markdown(f"""
            <div class="prediction-card">
                <h3>Predicted Funding Amount</h3>
                <h1 style="color: #1f77b4;">₹{predicted_amount:,.2f} Cr</h1>
                <p><strong>Funding Range:</strong> {funding_range}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display recommendations
            st.markdown("<h3>Strategic Recommendations</h3>", unsafe_allow_html=True)
            for rec in recommendations:
                st.markdown(f"""
                <div class="recommendation-card">
                    {rec}
                </div>
                """, unsafe_allow_html=True)
            
            # Add visualization of prediction context
            st.markdown("<h3>How This Compares</h3>", unsafe_allow_html=True)
            
            # Load data for comparison
            df = load_data()
            
            # Filter data for the selected round and vertical for comparison
            comparable_data = df[(df['round'] == round) & (df['vertical'] == vertical)]
            
            if not comparable_data.empty:
                avg_amount = comparable_data['amount'].mean()
                median_amount = comparable_data['amount'].median()
                max_amount = comparable_data['amount'].max()
                
                # Create comparison metrics
                comparison_col1, comparison_col2, comparison_col3 = st.columns(3)
                
                with comparison_col1:
                    st.metric("Your Prediction", f"₹{predicted_amount:.2f} Cr")
                
                with comparison_col2:
                    difference = ((predicted_amount - avg_amount) / avg_amount * 100)
                    st.metric("Sector Average", f"₹{avg_amount:.2f} Cr", f"{difference:.1f}%" if abs(difference) > 0.1 else "")
                
                with comparison_col3:
                    st.metric("Sector Maximum", f"₹{max_amount:.2f} Cr")
                
                # Create histogram of comparable fundings
                fig = px.histogram(
                    comparable_data, 
                    x='amount',
                    nbins=20,
                    title=f"Distribution of {vertical} Funding for {round} Round",
                    labels={'amount': 'Funding Amount (₹ Cr)'}
                )
                
                # Add a vertical line for the predicted value
                fig.add_vline(x=predicted_amount, line_width=2, line_dash="dash", line_color="red")
                fig.add_annotation(x=predicted_amount, y=0.85, yref="paper", text="Your Prediction", showarrow=True, arrowhead=1)
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough data available for this specific sector and round combination to show comparison.")
            
        else:
            st.error(f"Prediction failed: {prediction_result['error']}")

# Main app
def main():
    if not PLOTLY_AVAILABLE:
        st.error("Plotly is required for this dashboard. Please install it and restart the app.")
        return
        
    st.markdown('<h1 class="main-header">🚀 Indian Startup Ecosystem Analysis & Funding Predictor</h1>', unsafe_allow_html=True)
    
    # Load data
    try:
        df = load_data()
        if df.empty:
            st.warning("No data available. Please check your data file.")
            return
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        return
    
    # App navigation
    app_mode = st.sidebar.selectbox(
        "Choose Mode",
        ["📊 Startup Ecosystem Analysis", "💰 Funding Predictor", "📈 Deep Dive Analytics", "📋 Data Explorer"]
    )
    
    # Sidebar filters for analysis mode
    if app_mode in ["📊 Startup Ecosystem Analysis", "📈 Deep Dive Analytics", "📋 Data Explorer"]:
        st.sidebar.header("🔍 Filters")
        
        # Date range filter
        min_date = df['date'].min().date()
        max_date = df['date'].max().date()
        
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        # City filter
        cities = ['All'] + sorted(df['city'].unique().tolist())
        selected_cities = st.sidebar.multiselect(
            "Select Cities",
            cities,
            default=['All']
        )
        
        # Sector filter
        sectors = ['All'] + sorted(df['vertical'].unique().tolist())
        selected_sectors = st.sidebar.multiselect(
            "Select Sectors",
            sectors,
            default=['All']
        )
        
        # Round filter
        rounds = ['All'] + sorted(df['round'].unique().tolist())
        selected_rounds = st.sidebar.multiselect(
            "Select Funding Rounds",
            rounds,
            default=['All']
        )
        
        # Apply filters
        filtered_df = df.copy()
        
        if len(date_range) == 2:
            filtered_df = filtered_df[
                (filtered_df['date'].dt.date >= date_range[0]) &
                (filtered_df['date'].dt.date <= date_range[1])
            ]
        
        if 'All' not in selected_cities:
            filtered_df = filtered_df[filtered_df['city'].isin(selected_cities)]
        
        if 'All' not in selected_sectors:
            filtered_df = filtered_df[filtered_df['vertical'].isin(selected_sectors)]
        
        if 'All' not in selected_rounds:
            filtered_df = filtered_df[filtered_df['round'].isin(selected_rounds)]
    
    # Main content based on selected mode
    if app_mode == "📊 Startup Ecosystem Analysis":
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_funding = filtered_df['amount'].sum()
            st.metric("Total Funding", f"₹{total_funding:,.0f} Cr")
        
        with col2:
            total_deals = len(filtered_df)
            st.metric("Total Deals", f"{total_deals:,}")
        
        with col3:
            avg_deal_size = filtered_df['amount'].mean()
            st.metric("Avg Deal Size", f"₹{avg_deal_size:.1f} Cr")
        
        with col4:
            unique_startups = filtered_df['startup'].nunique()
            st.metric("Unique Startups", f"{unique_startups:,}")
        
        st.markdown("---")
        
        # Timeline chart
        st.plotly_chart(create_funding_timeline(filtered_df), use_container_width=True)
        
        # Two column layout for additional charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_heatmap(filtered_df), use_container_width=True)
        
        with col2:
            # Top rounds pie chart
            round_funding = filtered_df.groupby('round')['amount'].sum().sort_values(ascending=False).head(8)
            fig_pie = px.pie(
                values=round_funding.values,
                names=round_funding.index,
                title="Funding Distribution by Round",
                hole=0.4
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # City analysis
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown('<h2 class="sub-header">🏙️ Top Startup Hubs</h2>', unsafe_allow_html=True)
        
        # City stats
        city_stats = filtered_df.groupby('city').agg({
            'amount': 'sum',
            'startup': 'nunique'
        }).sort_values('amount', ascending=False).head(10).reset_index()
        
        # City bar chart
        fig_city = px.bar(
            city_stats,
            x='city',
            y='amount',
            title="Top 10 Cities by Funding Amount",
            color='amount',
            labels={'city': 'City', 'amount': 'Total Funding (₹ Cr)'},
            color_continuous_scale='Blues'
        )
        fig_city.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig_city, use_container_width=True)
        
        # Sector analysis
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown('<h2 class="sub-header">📊 Sector Analysis</h2>', unsafe_allow_html=True)
        
        # Sector stats
        sector_stats = filtered_df.groupby('vertical').agg({
            'amount': 'sum',
            'startup': 'nunique'
        }).sort_values('amount', ascending=False).head(10).reset_index()
        
        # Sector bar chart
        fig_sector = px.bar(
            sector_stats,
            x='vertical',
            y='amount',
            title="Top 10 Sectors by Funding Amount",
            color='amount',
            labels={'vertical': 'Sector', 'amount': 'Total Funding (₹ Cr)'},
            color_continuous_scale='Greens'
        )
        fig_sector.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig_sector, use_container_width=True)
        
        # Key insights
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown('<h2 class="sub-header">🔍 Key Insights</h2>', unsafe_allow_html=True)
        
        # Calculate insights
        top_city = city_stats['city'].iloc[0] if not city_stats.empty else "N/A"
        top_sector = sector_stats['vertical'].iloc[0] if not sector_stats.empty else "N/A"
        top_round = filtered_df.groupby('round')['amount'].sum().sort_values(ascending=False).index[0] if len(filtered_df) > 0 else "N/A"
        
        st.markdown(f"""
        <div class="insight-box">
            <h3>Summary of Indian Startup Ecosystem</h3>
            <ul>
                <li>Total funding of <b>₹{total_funding:,.0f} Cr</b> across <b>{total_deals}</b> deals</li>
                <li><b>{top_city}</b> is the leading startup hub with the highest funding</li>
                <li><b>{top_sector}</b> is the most funded sector</li>
                <li><b>{top_round}</b> funding rounds attracted the largest portion of funding</li>
                <li>The average deal size is <b>₹{avg_deal_size:.1f} Cr</b></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    elif app_mode == "💰 Funding Predictor":
        create_funding_predictor_ui()
    
    elif app_mode == "📈 Deep Dive Analytics":
        st.markdown('<h2 class="sub-header">📈 Deep Dive Analytics</h2>', unsafe_allow_html=True)
        
        # Advanced Analytics Tabs
        analytics_tab1, analytics_tab2, analytics_tab3, analytics_tab4 = st.tabs([
            "Funding Rounds", "Sector Analysis", "Investor Analysis", "Startup Growth"
        ])
        
        with analytics_tab1:
            st.markdown("### Funding Round Analysis")
            st.plotly_chart(create_round_analysis(filtered_df), use_container_width=True)
            
            # Additional round insights
            st.markdown("#### 🔍 Round Conversion Analysis")
            
            # Calculate round progression
            round_progression = []
            for startup in filtered_df['startup'].unique():
                startup_rounds = filtered_df[filtered_df['startup'] == startup].sort_values('date')
                if len(startup_rounds) > 1:
                    for i in range(len(startup_rounds) - 1):
                        round_progression.append({
                            'startup': startup,
                            'from_round': startup_rounds.iloc[i]['round'],
                            'to_round': startup_rounds.iloc[i+1]['round'],
                            'time_days': (startup_rounds.iloc[i+1]['date'] - startup_rounds.iloc[i]['date']).days
                        })
            
            if round_progression:
                round_progression_df = pd.DataFrame(round_progression)
                
                # Calculate average time between rounds
                avg_time_between_rounds = round_progression_df.groupby(['from_round', 'to_round'])['time_days'].mean().reset_index()
                avg_time_between_rounds = avg_time_between_rounds.sort_values('time_days')
                
                if not avg_time_between_rounds.empty:
                    st.write("Average Time Between Funding Rounds (in days):")
                    
                    # Create chart
                    fig_time = px.bar(
                        avg_time_between_rounds,
                        x='from_round',
                        y='time_days',
                        color='to_round',
                        labels={'from_round': 'From Round', 'to_round': 'To Round', 'time_days': 'Avg. Days'},
                        title="Average Time Between Funding Rounds"
                    )
                    fig_time.update_layout(xaxis_tickangle=45)
                    st.plotly_chart(fig_time, use_container_width=True)
            else:
                st.info("Not enough data to analyze round progression.")
        
        with analytics_tab2:
            st.markdown("### Sector Deep Dive")
            st.plotly_chart(create_sector_analysis(filtered_df), use_container_width=True)
            
            # Sector growth over time
            st.markdown("#### 🚀 Sector Growth Over Time")
            
            # Get top sectors
            top_sectors = filtered_df.groupby('vertical')['amount'].sum().nlargest(6).index.tolist()
            
            # Create time series for each top sector
            sector_time_df = filtered_df[filtered_df['vertical'].isin(top_sectors)].copy()
            sector_time_df['year_quarter'] = sector_time_df['date'].dt.to_period('Q').astype(str)
            
            sector_time_pivot = sector_time_df.groupby(['year_quarter', 'vertical'])['amount'].sum().reset_index()
            
            # Plot sector growth over time
            fig_sector_time = px.line(
                sector_time_pivot,
                x='year_quarter',
                y='amount',
                color='vertical',
                markers=True,
                title="Funding Trends for Top Sectors Over Time",
                labels={'year_quarter': 'Year-Quarter', 'amount': 'Funding Amount (₹ Cr)', 'vertical': 'Sector'}
            )
            
            fig_sector_time.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig_sector_time, use_container_width=True)
        
        with analytics_tab3:
            st.markdown("### Investor Analysis")
            st.plotly_chart(create_investor_analysis(filtered_df), use_container_width=True)
            
            # Additional investor insights
            st.markdown("#### 🔍 Investor Preferences")
            
            # Analyze investor preferences by sector and round
            investor_list = []
            for idx, row in filtered_df.iterrows():
                investors = str(row['investors']).split(',')
                for investor in investors:
                    investor = investor.strip()
                    if investor and investor.lower() != 'nan' and len(investor) > 2:
                        investor_list.append({
                            'investor': investor,
                            'vertical': row['vertical'],
                            'round': row['round'],
                            'amount': row['amount']
                        })
            
            if investor_list:
                investor_df = pd.DataFrame(investor_list)
                
                # Top investors by total funding
                top_investors_funding = investor_df.groupby('investor')['amount'].sum().nlargest(10).reset_index()
                
                fig_investor_funding = px.bar(
                    top_investors_funding,
                    y='investor',
                    x='amount',
                    orientation='h',
                    title="Top 10 Investors by Total Funding",
                    labels={'amount': 'Total Funding (₹ Cr)', 'investor': 'Investor'},
                    color='amount',
                    color_continuous_scale='Oranges'
                )
                
                st.plotly_chart(fig_investor_funding, use_container_width=True)
                
                # Investor sector preferences
                st.markdown("#### Sector Preferences of Top Investors")
                
                # Get top 5 investors
                top_5_investors = top_investors_funding['investor'].head(5).tolist()
                
                # Filter for top investors
                top_investor_sectors = investor_df[investor_df['investor'].isin(top_5_investors)]
                
                # Create sector preference chart
                fig_investor_sectors = px.bar(
                    top_investor_sectors.groupby(['investor', 'vertical'])['amount'].sum().reset_index(),
                    x='investor',
                    y='amount',
                    color='vertical',
                    title="Sector Preferences of Top Investors",
                    labels={'amount': 'Funding Amount (₹ Cr)', 'investor': 'Investor', 'vertical': 'Sector'}
                )
                
                st.plotly_chart(fig_investor_sectors, use_container_width=True)
            else:
                st.info("Not enough investor data for analysis.")
        
        with analytics_tab4:
            st.markdown("### Startup Growth Analysis")
            
            # Top startups by total funding
            top_startups = filtered_df.groupby('startup')['amount'].sum().nlargest(15).reset_index()
            
            fig_top_startups = px.bar(
                top_startups,
                x='startup',
                y='amount',
                title="Top 15 Startups by Total Funding",
                labels={'amount': 'Total Funding (₹ Cr)', 'startup': 'Startup'},
                color='amount',
                color_continuous_scale='Viridis'
            )
            
            fig_top_startups.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig_top_startups, use_container_width=True)
            
            # Startup funding journey
            st.markdown("#### 🚀 Startup Funding Journey")
            
            # Select a startup to visualize its funding journey
            selected_startup_for_journey = st.selectbox(
                "Select a startup to visualize its funding journey",
                options=top_startups['startup'].tolist()
            )
            
            startup_journey = filtered_df[filtered_df['startup'] == selected_startup_for_journey].sort_values('date')
            
            if not startup_journey.empty:
                # Create a timeline visualization
                fig_journey = px.line(
                    startup_journey,
                    x='date',
                    y='amount',
                    markers=True,
                    title=f"Funding Journey of {selected_startup_for_journey}",
                    labels={'amount': 'Funding Amount (₹ Cr)', 'date': 'Date'}
                )
                
                # Add round information as annotations
                for i, row in startup_journey.iterrows():
                    fig_journey.add_annotation(
                        x=row['date'],
                        y=row['amount'],
                        text=row['round'],
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor="#636363",
                        ax=0,
                        ay=-40
                    )
                
                st.plotly_chart(fig_journey, use_container_width=True)
                
                # Show startup details in a table
                st.markdown(f"#### {selected_startup_for_journey} Funding Details")
                
                journey_table = startup_journey[['date', 'round', 'amount', 'investors', 'city']].copy()
                journey_table['date'] = journey_table['date'].dt.strftime('%Y-%m-%d')
                
                st.table(journey_table)
            else:
                st.info(f"No funding data available for {selected_startup_for_journey}.")
    
    elif app_mode == "📋 Data Explorer":
        st.markdown('<h2 class="sub-header">📋 Data Explorer</h2>', unsafe_allow_html=True)
        
        # Search functionality
        search_term = st.text_input("🔍 Search startups, investors, sectors, or cities")
        
        if search_term:
            search_results = filtered_df[
                filtered_df['startup'].str.contains(search_term, case=False) |
                filtered_df['investors'].str.contains(search_term, case=False) |
                filtered_df['vertical'].str.contains(search_term, case=False) |
                filtered_df['city'].str.contains(search_term, case=False)
            ]
            display_df = search_results
        else:
            display_df = filtered_df
        
        # Sort options
        sort_options = {
            "Date (newest first)": ("date", False),
            "Date (oldest first)": ("date", True),
            "Funding Amount (highest first)": ("amount", False),
            "Funding Amount (lowest first)": ("amount", True),
            "Startup Name (A-Z)": ("startup", True),
            "Startup Name (Z-A)": ("startup", False)
        }
        
        sort_by = st.selectbox("Sort by:", list(sort_options.keys()))
        sort_col, sort_asc = sort_options[sort_by]
        
        # Apply sorting
        display_df = display_df.sort_values(sort_col, ascending=sort_asc)
        
        # Display data
        st.markdown(f"### Showing {len(display_df)} results")
        
        # Format the dataframe for display
        display_cols = ['date', 'startup', 'vertical', 'city', 'round', 'amount', 'investors']
        formatted_df = display_df[display_cols].copy()
        formatted_df['date'] = formatted_df['date'].dt.strftime('%Y-%m-%d')
        
        # Display the data
        st.dataframe(formatted_df, use_container_width=True)
        
        # Download option
        st.download_button(
            label="Download Data as CSV",
            data=formatted_df.to_csv(index=False).encode('utf-8'),
            file_name="startup_funding_data.csv",
            mime="text/csv"
        )
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>📊 Indian Startup Ecosystem Analysis & Funding Predictor | Data covers 2016-2020 | Built By Ayush Singh </p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

