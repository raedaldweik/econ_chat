import os
from dotenv import load_dotenv

import pandas as pd
import streamlit as st

from langchain_experimental.agents import create_csv_agent
from langchain_openai import ChatOpenAI

# 1) Load .env and set API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("API key not found. Please check your .env file.")
    st.stop()
os.environ["OPENAI_API_KEY"] = api_key

# 2) Convert each sheet in Excel to a CSV (utf-8 with BOM) if needed
EXCEL_PATH = "full_data.xlsx"
CSV_DIR = "csv_data"
if not os.path.exists(CSV_DIR):
    os.makedirs(CSV_DIR)

excel = pd.ExcelFile(EXCEL_PATH)
csv_paths = []
for sheet in excel.sheet_names:
    csv_path = os.path.join(CSV_DIR, f"{sheet}.csv")
    if not os.path.exists(csv_path):
        df = pd.read_excel(EXCEL_PATH, sheet_name=sheet)
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    csv_paths.append(csv_path)

# 3) Initialize the CSV agent (with dangerous code allowed)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent_executor = create_csv_agent(
    llm,
    csv_paths,
    pandas_kwargs={"encoding": "utf-8-sig"},
    agent_type="openai-tools",
    verbose=True,
    allow_dangerous_code=True,
)

# 4) Data dictionary for context
data_dictionary = """
| Column Name                           | Description                                                                       |
|---------------------------------------|-----------------------------------------------------------------------------------|
| **media sheet**                       |                                                                                   |
| Article_ID                            | Unique identifier for each news article                                           |
| Date                                  | Publication date of the article (YYYY-MM-DD)                                      |
| Source                                | Source publication name (e.g., Khaleej Times)                                      |
| Title                                 | Title of the article                                                              |
| Content_Snippet                       | Excerpt or snippet from the article                                               |
| Entity_Mentioned                      | Named entity mentioned in the article                                             |
| Sector                                | Industry sector (e.g., Real Estate)                                               |
| Sentiment_Score                       | Sentiment polarity score (-1 to 1)                                                 |
| Sentiment_Label                       | Sentiment category (Positive, Neutral, Negative)                                  |
| Language                              | Language of the article (Arabic, English)                                         |
| **hotels sheet**                      |                                                                                   |
| Hotel_ID                              | Unique hotel identifier                                                           |
| Hotel_Name                            | Name of the hotel                                                                 |
| Hotel_Group                           | Hotel group affiliation                                                           |
| Class                                 | Star rating classification (e.g., 5 star)                                         |
| Total_Rooms                           | Total number of rooms in the hotel                                                |
| Year                                  | Year of the record                                                                |
| Month                                 | Month number (1–12)                                                               |
| Month_Name                            | Month name (e.g., January)                                                        |
| Occupied_Rooms                        | Number of rooms occupied                                                          |
| Occupancy_Rate                        | Occupancy rate (0–1)                                                              |
| Avg_Room_Rate_AED                     | Average room rate in AED                                                          |
| Room_Revenue_AED                      | Total room revenue in AED                                                         |
| Guest_Arrivals                        | Number of guest arrivals                                                          |
| Avg_Length_of_Stay                    | Average length of stay (days)                                                     |
| Source_Region                         | Region of guest origin                                                            |
| Source_Country                        | Country of guest origin                                                           |
| RevPAR                                | Revenue per available room (AED)                                                  |
| ADR                                   | Average daily rate (AED)                                                          |
| Total_Available_Rooms                 | Total rooms available                                                             |
| **license sheet**                     |                                                                                   |
| License No.                           | License number                                                                    |
| Trade Name A                          | Trade name in Arabic                                                              |
| Trade Name E                          | Trade name in English                                                             |
| License Status A                      | License status in Arabic (ساري, etc.)                                             |
| Issue Date                            | License issue date (DD/MM/YYYY)                                                   |
| Expire Date                           | License expiry date (DD/MM/YYYY)                                                  |
| Address                               | Registered address                                                                |
| Person Legal Type A                   | Person legal type in Arabic                                                       |
| Person Serial No.                     | Person serial number                                                              |
| Person Name A                         | Person name in Arabic                                                             |
| Partner Status A                      | Partner status in Arabic                                                          |
| Partner Add Date                      | Partner addition date (DD/MM/YYYY)                                                |
| Partner Cancel Date                   | Partner cancellation date (DD/MM/YYYY)                                            |
| Emirates ID                           | Emirates ID number                                                                |
| Nationality A                         | Nationality in Arabic                                                             |
| Mobile                                | Mobile phone number                                                               |
| Activity Master Group A               | Master activity group in Arabic                                                   |
| Activity Category A                   | Activity category in Arabic                                                       |
| Activity Consist Group A              | Consist activity group in Arabic                                                  |
| Activity Code ISIC4                   | ISIC4 classification code                                                         |
| Activity A                            | Activity name in Arabic                                                           |
| **class_summary sheet**               |                                                                                   |
| Year (class_summary)                  | Year of the summary record                                                        |
| Class (class_summary)                 | Hotel star rating category (e.g., 1-2 star, 3 star)                               |
| Occupancy_Rate (class_summary)        | Aggregated occupancy rate for that class                                          |
| Avg_Room_Rate_AED (class_summary)     | Aggregated average room rate in AED                                               |
| Room_Revenue_AED (class_summary)      | Aggregated room revenue in AED                                                    |
| Guest_Arrivals (class_summary)        | Aggregated number of guest arrivals                                               |
| **real_estate sheet**                 |                                                                                   |
| Transaction_ID                        | Unique transaction identifier                                                     |
| Property_ID                           | Unique property identifier                                                        |
| Investor_ID                           | Unique investor identifier                                                        |
| Buyer_Type                            | Buyer category (Resident, Investor)                                               |
| Party_Type                            | Party type (شخص, جهة)                                                              |
| Gender                                | Buyer gender (ذكر, أنثى)                                                           |
| Nationality                           | Buyer nationality                                                                 |
| Transaction_Type                      | Transaction type (Buy, Sell)                                                      |
| Property_Type                         | Main property type (Apartment, Villa, etc.)                                       |
| Property_Subtype                      | Specific property subtype (Studio, 2BR, etc.)                                     |
| Area                                  | Community or area name                                                            |
| Size_sqm                              | Property size in square meters                                                    |
| Price_AED                             | Transaction price in AED                                                          |
| Transaction_Date                      | Date of transaction (DD/MM/YYYY)                                                  |
| Payment_Mode                          | Payment method (Cash, etc.)                                                       |
| Agent_Involved                        | Whether an agent was involved (Yes/No)                                            |
| Registration_Type                     | Registration type in Arabic                                                       |
| Registration_Type_English             | Registration type in English                                                      |
| Acquisition_Type                      | Acquisition type in Arabic                                                        |
| Acquisition_Type_English              | Acquisition type in English                                                       |
| Emirates_ID                           | Buyer’s Emirates ID                                                               |
| License_No                            | Buyer’s license number                                                            |
| Phone_Number                          | Buyer’s phone number                                                              |
| Transaction_Quarter                   | Quarter of transaction (1–4)                                                      |
| Transaction_Year                      | Year of transaction                                                               |
"""

# 5) Streamlit UI using chat primitives
st.title("Digital Assistant")
st.write("Ask me anything about your media, hotels, license, class summary, or real estate data!")

# initialize history if not present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# render existing messages
for role, msg in st.session_state.chat_history:
    st.chat_message(role).write(msg)

# chat input pinned at bottom
if prompt := st.chat_input("You:"):
    full_prompt = (
        f"Refer to the following data dictionary for context:\n\n"
        f"{data_dictionary}\n\n"
        f"{prompt}"
    )
    result = agent_executor.invoke({"input": full_prompt})
    response = result["output"]

    # append to history
    st.session_state.chat_history.append(("user", prompt))
    st.session_state.chat_history.append(("assistant", response))

    # display immediately
    st.chat_message("user").write(prompt)
    st.chat_message("assistant").write(response)
