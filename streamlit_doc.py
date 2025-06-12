import streamlit as st
import pandas as pd
import plotly.express as px
import time

st.title('Startup Dashboard')
st.header('Advanced Interactive Dashboard')
st.subheader('Welcome to the Enhanced Dashboard!')

st.write('Explore the features below:')

st.markdown("""
### Features
- Dynamic charts
- File validation
- Data cleaning
- Interactive visualizations
""")

file = st.file_uploader('Upload a CSV file', type=['csv'])

if file is not None:
    try:
        df = pd.read_csv(file)
        st.success('File uploaded successfully!')
        
        # Clean the data
        st.markdown("### Cleaning Data")
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)  # Convert 'amount' to numeric
        df['date'] = pd.to_datetime(df['date'], errors='coerce')  # Convert 'date' to datetime
        df.dropna(subset=['date'], inplace=True)  # Remove rows with invalid dates
        df.drop_duplicates(inplace=True)  # Remove duplicate rows
        
        # Trim whitespace from string columns
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.strip()

        st.markdown("### Cleaned Data")
        st.dataframe(df)

        # Display summary statistics
        st.markdown("### Data Summary")
        st.write(df.describe())

        # Add interactive graphs
        st.markdown("### Funding by Round")
        funding_by_round = df.groupby('round')['amount'].sum().sort_values(ascending=False)
        fig1 = px.bar(funding_by_round, x=funding_by_round.index, y=funding_by_round.values, title='Funding by Round')
        st.plotly_chart(fig1)

        st.markdown("### Funding by City")
        funding_by_city = df.groupby('city')['amount'].sum().sort_values(ascending=False).head(10)
        fig2 = px.bar(funding_by_city, x=funding_by_city.index, y=funding_by_city.values, title='Top 10 Cities by Funding')
        st.plotly_chart(fig2)

        st.markdown("### Funding Over Time")
        funding_over_time = df.groupby('date')['amount'].sum()
        fig3 = px.line(funding_over_time, x=funding_over_time.index, y=funding_over_time.values, title='Funding Over Time')
        st.plotly_chart(fig3)

        st.markdown("### Top Investors")
        top_investors = df['investors'].value_counts().head(10)
        fig4 = px.bar(top_investors, x=top_investors.index, y=top_investors.values, title='Top 10 Investors')
        st.plotly_chart(fig4)

        st.markdown("### Funding by Vertical")
        funding_by_vertical = df.groupby('vertical')['amount'].sum().sort_values(ascending=False).head(10)
        fig5 = px.bar(funding_by_vertical, x=funding_by_vertical.index, y=funding_by_vertical.values, title='Top 10 Verticals by Funding')
        st.plotly_chart(fig5)

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info('Please upload a CSV file to proceed.')

# Sidebar interactivity
st.sidebar.title('User Inputs')
email = st.sidebar.text_input('Enter email', key='email_input')
password = st.sidebar.text_input('Enter password', type='password', key='password_input')
gender = st.sidebar.selectbox('Select gender', ['male', 'female', 'others'], key='gender_select')
btn = st.sidebar.button('Login')

# Conditional rendering based on login
if btn:
    if email == 'nitish@gmail.com' and password == '1234':
        st.balloons()
        st.success(f'Welcome, {gender.capitalize()}!')
    else:
        st.error('Login Failed')

# Progress bar with dynamic updates
st.markdown("### Progress Tracker")
bar = st.progress(0)
for i in range(1, 101):
    time.sleep(0.01)
    bar.progress(i)

st.markdown("### Feedback")
feedback = st.text_area('Share your feedback:')
if feedback:
    st.success('Thank you for your feedback!')








