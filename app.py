import streamlit as st
import pandas as pd

# Load dataset from Google Drive
@st.cache_data
def load_data():
    # Replace FILE_ID with your actual Google Drive file ID
    url = "https://drive.google.com/uc?id=YOUR_FILE_ID"
    return pd.read_csv(url)

df = load_data()

st.title("🏠 Indian Housing Price Explorer")
st.write("Filter properties and view details from the dataset.")

# Sidebar filters
st.sidebar.header("Filters")
state = st.sidebar.selectbox("Select State", ["All"] + sorted(df["State"].dropna().unique().tolist()))
city = st.sidebar.selectbox("Select City", ["All"] + sorted(df["City"].dropna().unique().tolist()))
bhk = st.sidebar.selectbox("Select BHK", ["All"] + sorted(df["BHK"].dropna().unique().tolist()))
furnished = st.sidebar.selectbox("Furnished Status", ["All"] + sorted(df["Furnished_Status"].dropna().unique().tolist()))

# Apply filters
filtered_df = df.copy()
if state != "All":
    filtered_df = filtered_df[filtered_df["State"] == state]
if city != "All":
    filtered_df = filtered_df[filtered_df["City"] == city]
if bhk != "All":
    filtered_df = filtered_df[filtered_df["BHK"] == bhk]
if furnished != "All":
    filtered_df = filtered_df[filtered_df["Furnished_Status"] == furnished]

# Show filtered results
st.subheader("Filtered Properties")
st.dataframe(filtered_df)

# Show average price
if not filtered_df.empty:
    avg_price = filtered_df["Price_in_Lakhs"].mean()
    st.metric(label="Average Price (Lakhs)", value=f"{avg_price:,.2f}")
else:
    st.warning("No properties match your filters.")


##for running -- streamlit run "C:\Users\mrish\Downloads\app.py"

