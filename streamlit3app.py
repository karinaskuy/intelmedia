import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import json
import os

# Set page config for a modern look
st.set_page_config(
    page_title="Interactive Media Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Header ---
st.title("Interactive Media Intelligence Dashboard")
st.markdown("Analyze your media data with powerful visualizations and AI-driven insights.")
st.markdown("---") # Horizontal line for separation

# --- 1. Upload Your CSV File ---
st.header("1. Upload Your CSV File")
st.markdown("""
    Please upload a CSV file containing the following columns:
    `Date`, `plateform`, `Sentiment`, `location`, `engagements`, `Media Type`.
""")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# Initialize session state variables if they don't exist
if 'csv_data' not in st.session_state:
    st.session_state.csv_data = None
if 'cleaned_df' not in st.session_state:
    st.session_state.cleaned_df = None
if 'sentiment_insights' not in st.session_state:
    st.session_state.sentiment_insights = []
if 'engagement_insights' not in st.session_state:
    st.session_state.engagement_insights = []
if 'platform_insights' not in st.session_state:
    st.session_state.platform_insights = []
if 'media_type_insights' not in st.session_state:
    st.session_state.media_type_insights = []
if 'location_insights' not in st.session_state:
    st.session_state.location_insights = []

# --- Data Cleaning (on file upload) ---
if uploaded_file is not None:
    # Read the CSV file
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state.csv_data = df.copy() # Store original for reference if needed
        st.success("CSV file loaded successfully!")

        st.header("2. Data Cleaning Summary")
        st.markdown("Your data has been automatically cleaned:")
        st.markdown("""
            * 'Date' column converted to datetime format (YYYY-MM-DD).
            * Missing 'engagements' values filled with 0.
            * Column names normalized to camelCase (e.g., 'Media Type' to 'mediaType', 'plateform' to 'platform').
        """)

        # Clean the data
        # Normalize Column Names
        df.columns = [col.strip().lower().replace(' ', '_').replace('plateform', 'platform').replace('media_type', 'mediaType') for col in df.columns]

        # Convert 'Date' to datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date']) # Remove rows with invalid dates
            df['date'] = df['date'].dt.strftime('%Y-%m-%d') # Format for consistency

        # Fill missing 'engagements' with 0
        if 'engagements' in df.columns:
            df['engagements'] = pd.to_numeric(df['engagements'], errors='coerce').fillna(0).astype(int)
        else:
            df['engagements'] = 0 # Add engagements column if not present

        st.session_state.cleaned_df = df
        st.success("Data cleaning complete. Ready for visualization!")

    except Exception as e:
        st.error(f"Error processing file: {e}. Please check your CSV format and column names.")
        st.session_state.csv_data = None
        st.session_state.cleaned_df = None

# --- Gemini API Call Function ---
def generate_insights_from_gemini(chart_name, data_to_analyze):
    # Retrieve API key from environment variables
    # This is critical for the Gemini API call to work.
    # Ensure GEMINI_API_KEY is set in your Canvas environment settings.
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    if not GEMINI_API_KEY:
        st.error("Gemini API Key is not configured. Please set the 'GEMINI_API_KEY' environment variable in your Canvas environment settings.")
        return [f"Failed to generate insights: Gemini API Key is missing. Check your Canvas environment settings."]

    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

    prompt = f"""Based on the following data for the {chart_name} chart, provide 3 concise and insightful bullet points. Focus on key trends, patterns, and implications.
    Data: {json.dumps(data_to_analyze)}"""

    chat_history = [{"role": "user", "parts": [{"text": prompt}]}]
    payload = {"contents": chat_history}

    try:
        response = requests.post(api_url, headers={'Content-Type': 'application/json'}, json=payload)

        if not response.ok:
            error_text = response.text
            st.error(f"API Error for {chart_name}: {response.status_code} {response.reason} - {error_text}")
            if response.status_code == 401:
                return [
                    f"Failed to generate insights: Authorization error (Status 401).",
                    "Please ensure your API Key for the Gemini API is correctly configured in your Canvas environment settings."
                ]
            return [f"Failed to generate insights: API returned an error ({response.status_code})."]

        result = response.json()

        if result.get('candidates') and len(result['candidates']) > 0 and \
           result['candidates'][0].get('content') and result['candidates'][0]['content'].get('parts') and \
           len(result['candidates'][0]['content']['parts']) > 0:
            text = result['candidates'][0]['content']['parts'][0]['text']
            # Split insights by bullet points or new lines
            insights = [line.strip() for line in text.split('\n') if line.strip().startswith(('-', '*', '•'))]
            if not insights: # Fallback if bullet points are not parsed as expected
                insights = [line.strip() for line in text.split('\n') if line.strip()]
            return insights[:3] # Ensure max 3 insights
        else:
            return ['Failed to generate insights: Unexpected API response structure.']

    except requests.exceptions.RequestException as e:
        st.error(f"Network or API request error for {chart_name}: {e}")
        return [f"Error generating insights: Network or API request failed. Please check your internet connection or API endpoint."]
    except json.JSONDecodeError as e:
        st.error(f"JSON decoding error for {chart_name}: {e}")
        return [f"Error generating insights: Could not parse API response."]
    except Exception as e:
        st.error(f"An unexpected error occurred for {chart_name}: {e}")
        return [f"An unexpected error occurred while generating insights."]


# --- Visualization Section ---
if st.session_state.cleaned_df is not None and not st.session_state.cleaned_df.empty:
    df = st.session_state.cleaned_df
    st.header("3. Data Visualizations")

    # --- Sentiment Breakdown Pie Chart ---
    st.subheader("Sentiment Breakdown")
    sentiment_data = df['sentiment'].value_counts().reset_index()
    sentiment_data.columns = ['sentiment', 'count']
    fig_sentiment = px.pie(
        sentiment_data,
        values='count',
        names='sentiment',
        title='Sentiment Distribution',
        color_discrete_sequence=px.colors.qualitative.Pastel # Using a pastel color sequence
    )
    # Applying specific colors if they match the sentiment names
    sentiment_colors = {
        'Positive': '#22C55E',  # Green
        'Negative': '#EF4444',  # Red
        'Neutral': '#FBBF24',   # Yellow
        'Unknown': '#8B5CF6'    # Purple (for fallback)
    }
    fig_sentiment.update_traces(
        textposition='inside',
        textinfo='percent+label',
        marker=dict(colors=[sentiment_colors.get(s, '#3B82F6') for s in sentiment_data['sentiment']]) # Default to blue if not found
    )
    st.plotly_chart(fig_sentiment, use_container_width=True)

    # Generate and display insights for Sentiment
    if st.button("Generate Insights for Sentiment", key="sentiment_btn"):
        with st.spinner("Generating insights for Sentiment..."):
            st.session_state.sentiment_insights = generate_insights_from_gemini("Sentiment Breakdown", sentiment_data.to_dict('records'))
    if st.session_state.sentiment_insights:
        st.markdown("#### Insights:")
        for insight in st.session_state.sentiment_insights:
            st.markdown(f"- {insight}")
    st.markdown("---")


    # --- Engagement Trend over Time Line Chart ---
    st.subheader("Engagement Trend over Time")
    engagement_trend_data = df.groupby('date')['engagements'].sum().reset_index()
    fig_engagement = px.line(
        engagement_trend_data,
        x='date',
        y='engagements',
        title='Total Engagements Over Time',
        markers=True,
        line_shape='spline',
        color_discrete_sequence=['#3B82F6']
    )
    st.plotly_chart(fig_engagement, use_container_width=True)

    # Generate and display insights for Engagement Trend
    if st.button("Generate Insights for Engagement Trend", key="engagement_btn"):
        with st.spinner("Generating insights for Engagement Trend..."):
            st.session_state.engagement_insights = generate_insights_from_gemini("Engagement Trend over Time", engagement_trend_data.to_dict('records'))
    if st.session_state.engagement_insights:
        st.markdown("#### Insights:")
        for insight in st.session_state.engagement_insights:
            st.markdown(f"- {insight}")
    st.markdown("---")

    # --- Platform Engagements Bar Chart ---
    st.subheader("Platform Engagements")
    platform_engagement_data = df.groupby('platform')['engagements'].sum().reset_index()
    platform_engagement_data = platform_engagement_data.sort_values(by='engagements', ascending=True) # Sort ascending for horizontal bar chart
    fig_platform = px.bar(
        platform_engagement_data,
        x='engagements',
        y='platform',
        orientation='h',
        title='Total Engagements by Platform',
        color='platform', # Color by platform
        color_discrete_sequence=px.colors.qualitative.Pastel # Using a pastel color sequence
    )
    fig_platform.update_layout(yaxis={'categoryorder':'total ascending'}) # Ensure consistent sorting
    st.plotly_chart(fig_platform, use_container_width=True)

    # Generate and display insights for Platform Engagements
    if st.button("Generate Insights for Platform Engagements", key="platform_btn"):
        with st.spinner("Generating insights for Platform Engagements..."):
            st.session_state.platform_insights = generate_insights_from_gemini("Platform Engagements", platform_engagement_data.to_dict('records'))
    if st.session_state.platform_insights:
        st.markdown("#### Insights:")
        for insight in st.session_state.platform_insights:
            st.markdown(f"- {insight}")
    st.markdown("---")

    # --- Media Type Mix Pie Chart ---
    st.subheader("Media Type Mix")
    media_type_data = df['mediaType'].value_counts().reset_index()
    media_type_data.columns = ['mediaType', 'count']
    fig_media_type = px.pie(
        media_type_data,
        values='count',
        names='mediaType',
        title='Media Type Distribution',
        color_discrete_sequence=px.colors.qualitative.Pastel # Using a pastel color sequence
    )
    # Applying specific colors if they match the media type names
    media_type_colors = {
        'Video': '#3B82F6',
        'Image': '#22C55E',
        'Text': '#FBBF24',
        'Audio': '#EF4444',
        'Unknown': '#8B5CF6'
    }
    fig_media_type.update_traces(
        textposition='inside',
        textinfo='percent+label',
        marker=dict(colors=[media_type_colors.get(mt, '#3B82F6') for mt in media_type_data['mediaType']])
    )
    st.plotly_chart(fig_media_type, use_container_width=True)

    # Generate and display insights for Media Type
    if st.button("Generate Insights for Media Type Mix", key="media_type_btn"):
        with st.spinner("Generating insights for Media Type Mix..."):
            st.session_state.media_type_insights = generate_insights_from_gemini("Media Type Mix", media_type_data.to_dict('records'))
    if st.session_state.media_type_insights:
        st.markdown("#### Insights:")
        for insight in st.session_state.media_type_insights:
            st.markdown(f"- {insight}")
    st.markdown("---")

    # --- Top 5 Locations Bar Chart ---
    st.subheader("Top 5 Locations by Engagement")
    location_engagement_data = df.groupby('location')['engagements'].sum().reset_index()
    top_5_locations = location_engagement_data.nlargest(5, 'engagements')
    fig_location = px.bar(
        top_5_locations,
        x='engagements',
        y='location',
        orientation='h',
        title='Top 5 Locations by Total Engagements',
        color='location', # Color by location
        color_discrete_sequence=px.colors.qualitative.Pastel # Using a pastel color sequence
    )
    fig_location.update_layout(yaxis={'categoryorder':'total ascending'}) # Ensure consistent sorting
    st.plotly_chart(fig_location, use_container_width=True)

    # Generate and display insights for Top 5 Locations
    if st.button("Generate Insights for Top 5 Locations", key="location_btn"):
        with st.spinner("Generating insights for Top 5 Locations..."):
            st.session_state.location_insights = generate_insights_from_gemini("Top 5 Locations", top_5_locations.to_dict('records'))
    if st.session_state.location_insights:
        st.markdown("#### Insights:")
        for insight in st.session_state.location_insights:
            st.markdown(f"- {insight}")
    st.markdown("---")

# --- Footer ---
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>&copy; 2023 Media Intelligence Dashboard. Powered by Gemini.</p>", unsafe_allow_html=True)
