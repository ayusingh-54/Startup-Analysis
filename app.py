import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

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

# Page config
st.set_page_config(
    page_title="Indian Startup Ecosystem Dashboard",
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

# Main app
def main():
    if not PLOTLY_AVAILABLE:
        st.error("Plotly is required for this dashboard. Please install it and restart the app.")
        return
        
    st.markdown('<h1 class="main-header">🚀 Indian Startup Ecosystem Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    try:
        df = load_data()
        if df.empty:
            st.warning("No data available. Please check your data file.")
            return
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        return
    
    # Sidebar filters
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
    
    # Main dashboard
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🎯 Startup Analysis", "🏙️ City Analysis", "📈 Sector Analysis"])
    
    with tab1:
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
                title="Funding Distribution by Round"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with tab2:
        st.header("🎯 Startup Analysis")
        
        # Startup selector
        selected_startup = st.selectbox(
            'Select Startup',
            sorted(filtered_df['startup'].unique().tolist())
        )
        
        if selected_startup:
            startup_data = filtered_df[filtered_df['startup'] == selected_startup]
            
            if not startup_data.empty:
                # Startup metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    total_raised = startup_data['amount'].sum()
                    st.metric("Total Funding Raised", f"₹{total_raised:.1f} Cr")
                
                with col2:
                    funding_rounds = len(startup_data)
                    st.metric("Funding Rounds", funding_rounds)
                
                with col3:
                    latest_round = startup_data['round'].iloc[-1]
                    st.metric("Latest Round", latest_round)
                
                # Startup details
                st.subheader("📋 Startup Details")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Sector:** {startup_data['vertical'].iloc[0]}")
                    st.write(f"**Sub-sector:** {startup_data['subvertical'].iloc[0]}")
                    st.write(f"**City:** {startup_data['city'].iloc[0]}")
                
                with col2:
                    st.write(f"**First Funding:** {startup_data['date'].min().strftime('%Y-%m-%d')}")
                    st.write(f"**Latest Funding:** {startup_data['date'].max().strftime('%Y-%m-%d')}")
                
                # Funding timeline for startup
                fig_startup = px.line(
                    startup_data.sort_values('date'),
                    x='date',
                    y='amount',
                    markers=True,
                    title=f"Funding Timeline for {selected_startup}",
                    labels={'amount': 'Funding Amount (₹ Cr)', 'date': 'Date'}
                )
                fig_startup.update_layout(height=400)
                st.plotly_chart(fig_startup, use_container_width=True)
                
                # Detailed funding table
                st.subheader("💰 Funding History")
                funding_history = startup_data[['date', 'round', 'amount', 'investors']].sort_values('date', ascending=False)
                funding_history['date'] = funding_history['date'].dt.strftime('%Y-%m-%d')
                st.dataframe(funding_history, use_container_width=True)
            else:
                st.warning("No data available for the selected startup.")
    
    with tab3:
        st.header("🏙️ City Analysis")
        
        # City metrics
        try:
            city_stats = filtered_df.groupby('city').agg({
                'amount': 'sum',
                'startup': 'nunique'
            }).sort_values('amount', ascending=False).head(15)
            
            if city_stats.empty:
                st.warning("No city data available for the selected filters.")
                return
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Reset index to make city a column
                city_funding_data = city_stats.reset_index()
                fig_city_funding = px.bar(
                    city_funding_data,
                    x='city',
                    y='amount',
                    title="Top Cities by Total Funding",
                    labels={'city': 'City', 'amount': 'Total Funding (₹ Cr)'},
                    color='amount',
                    color_continuous_scale='Blues'
                )
                fig_city_funding.update_layout(xaxis_tickangle=45, height=400)
                st.plotly_chart(fig_city_funding, use_container_width=True)
            
            with col2:
                fig_city_startups = px.bar(
                    city_funding_data,
                    x='city',
                    y='startup',
                    title="Top Cities by Number of Startups",
                    labels={'city': 'City', 'startup': 'Number of Startups'},
                    color='startup',
                    color_continuous_scale='viridis'
                )
                fig_city_startups.update_layout(xaxis_tickangle=45, height=400)
                st.plotly_chart(fig_city_startups, use_container_width=True)
            
            # Geographical scatter plot
            st.plotly_chart(create_city_map(filtered_df), use_container_width=True)
            
        except Exception as e:
            st.error(f"Error creating city analysis: {str(e)}")
    
    with tab4:
        st.header("📈 Sector Analysis")
        
        # Sector overview
        st.plotly_chart(create_sector_analysis(filtered_df), use_container_width=True)
        
        # Sector comparison table
        st.subheader("📊 Sector Comparison Table")
        sector_comparison = filtered_df.groupby('vertical').agg({
            'amount': ['sum', 'mean', 'count'],
            'startup': 'nunique'
        }).round(2)
        
        sector_comparison.columns = ['Total Funding (₹ Cr)', 'Avg Deal Size (₹ Cr)', 'Total Deals', 'Unique Startups']
        sector_comparison = sector_comparison.sort_values('Total Funding (₹ Cr)', ascending=False)
        
        st.dataframe(sector_comparison, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>📊 Indian Startup Ecosystem Dashboard | Data covers 2016-2020 | Built By Ayush Singh </p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

