# Import required libraries
import os
from dotenv import load_dotenv, find_dotenv
import streamlit as st
import pandas as pd
from langchain_groq import ChatGroq
from langchain_experimental.agents import create_pandas_dataframe_agent
import matplotlib.pyplot as plt
import seaborn as sns

import speech_recognition as sr
from audio_recorder_streamlit import audio_recorder

# Load environment variables
load_dotenv(find_dotenv())
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

# Title
st.title('AI Assistant for Data Science 🤖')

# Welcoming message
st.write("Hello, 👋 I am your AI Assistant and I am here to help you with your data science projects.")

# Sidebar explanation
with st.sidebar:
    st.write('*Your Data Science Adventure Begins with a Dataset.*')
    st.caption('''**Upload your dataset in CSV or Excel format. Once uploaded, I will help you explore, clean, and analyze it.**''')
    st.divider()

# Initialize session state
if 'clicked' not in st.session_state:
    st.session_state.clicked = {1: False}
if 'cleaned' not in st.session_state:
    st.session_state.cleaned = False
if 'cleaned_df' not in st.session_state:
    st.session_state.cleaned_df = None
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'current_df' not in st.session_state:
    st.session_state.current_df = None
if 'original_df' not in st.session_state:
    st.session_state.original_df = None

def clicked(button):
    st.session_state.clicked[button] = True

# Cache expensive agent operations
@st.cache_resource(show_spinner=False)
def get_agent(df):
    llm = ChatGroq(model_name='gemma2-9b-it', temperature=0)
    return create_pandas_dataframe_agent(llm=llm, df=df, verbose=True, allow_dangerous_code=True)

def clean_data(df):
    """Perform data cleaning and force recomputation of analyses"""
    df_cleaned = df.drop_duplicates()
    df_cleaned = df_cleaned.fillna(method='ffill').fillna(method='bfill')
    
    # Store cleaned data and mark as cleaned
    st.session_state.cleaned = True
    st.session_state.cleaned_df = df_cleaned
    
    # Clear cached analyses so they'll recompute with cleaned data
    st.session_state.analysis_results = {}
    st.session_state.analysis_done = False
    
    return df_cleaned

def perform_initial_analysis(_agent, df_to_analyze, force_compute=False):
    """Perform initial analysis, with option to force recomputation"""
    if force_compute or 'initial_analysis' not in st.session_state.analysis_results:
        results = {}
        
        with st.spinner("Performing initial analysis..."):
            results['head'] = df_to_analyze.head()
            results['columns'] = _agent.run("Explain the meanings of the columns?")
            results['col_types'] = _agent.run("List down column type with each column in the dataframe in a table.")
            results['missing_values'] = _agent.run("How many missing values does this dataframe have? Start the answer with 'There are'")
            results['duplicates'] = _agent.run("Are there any duplicate values and if so, where?")
            results['describe'] = df_to_analyze.describe()
            results['correlation'] = _agent.run("Calculate correlations between numerical variables to identify potential relationships.")
            results['outliers'] = _agent.run("Identify outliers in the data that may be erroneous or that may have a significant impact on the analysis.")
            results['fixes'] = _agent.run("What fixes can be made (if any)?")
            results['anomaly'] = _agent.run("Identify anomaly in the dataset")
            results['patterns'] = _agent.run("Identify patterns or trends in the dataset")
        
        st.session_state.analysis_results['initial_analysis'] = results
        st.session_state.analysis_done = True
    
    return st.session_state.analysis_results['initial_analysis']

def display_initial_analysis(results):
    """Display cached initial analysis results"""
    st.write("**Data Overview**")
    st.write("The first rows of your dataset look like this:")
    st.write(results['head'])

    st.write(results['columns'])
    st.write("Column Types")
    st.write(results['col_types'])
    st.write(results['missing_values'])
    st.write(results['duplicates'])
    
    st.write("**Data Summarisation**")
    st.write(results['describe'])
    st.write(results['correlation'])
    st.write(results['outliers'])
    
    st.write("**Quick Fixes for you!**")
    st.write(results['fixes'])
    st.write("**Anomalies Found!**")
    st.write(results['anomaly'])
    st.write("**Patterns Found!**")
    st.write(results['patterns'])

st.button("Let's get started", on_click=clicked, args=[1])
if st.session_state.clicked[1]:
    tab1, tab2 = st.tabs(["Data Analysis","ChatBox"])
    
    with tab1:
        uploaded_file = st.file_uploader("Upload your file here", type=["csv", "xlsx"])

        if uploaded_file is not None:
            # Load data only once
            if st.session_state.original_df is None:
                if uploaded_file.name.endswith(".csv"):
                    st.session_state.original_df = pd.read_csv(uploaded_file, low_memory=False)
                else:
                    st.session_state.original_df = pd.read_excel(uploaded_file)
                st.session_state.current_df = st.session_state.original_df.copy()

            df = st.session_state.current_df
            pandas_agent = get_agent(df)

            # Initial Analysis Section
            st.header('Exploratory Data Analysis')
            st.subheader('General Information about the Dataset')
            
            analysis_results = perform_initial_analysis(pandas_agent, df)
            display_initial_analysis(analysis_results)

            # Clean Data Button - NOW WITH PROPER RECOMPUTATION
            if st.button("Clean Data"):
                df_cleaned = clean_data(df)
                st.success("Data cleaned successfully!")
                st.subheader("Cleaned Dataset Summary")
                
                # Update current df and agent
                st.session_state.current_df = df_cleaned
                pandas_agent = get_agent(df_cleaned)
                
                # Force fresh analysis
                cleaned_analysis = perform_initial_analysis(pandas_agent, df_cleaned, force_compute=True)
                display_initial_analysis(cleaned_analysis)

            # Column Analysis Section
            st.subheader('🔎 Smart Visual & Narrative Summaries ')
            columns = df.columns.tolist()
            selected_column = st.selectbox("Select a column to analyze:", options=columns)

            if selected_column:
                st.markdown(f"### Analysis for: `{selected_column}`")
                
                # Cache column-specific analysis
                if f'col_{selected_column}' not in st.session_state.analysis_results:
                    with st.spinner(f"Analyzing {selected_column}..."):
                        col_results = {}
                        col_results['chart_type'] = pandas_agent.run(
                            f"What is the most appropriate chart type to visualize the data in the column '{selected_column}'?"
                        ).lower()
                        
                        # Generate chart
                        try:
                            if "bar" in col_results['chart_type']:
                                st.bar_chart(df[selected_column].value_counts())
                            elif "line" in col_results['chart_type']:
                                st.line_chart(df[selected_column])
                            elif "hist" in col_results['chart_type'] or "distribution" in col_results['chart_type']:
                                st.write("Histogram:")
                                st.bar_chart(df[selected_column].value_counts().sort_index())
                            elif "pie" in col_results['chart_type']:
                                fig, ax = plt.subplots()
                                df[selected_column].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
                                st.pyplot(fig)
                        except Exception as e:
                            st.warning(f"Could not generate chart: {e}")
                        
                        # Cache other analyses
                        col_results['stats'] = pandas_agent.run(f"Give me a summary of the statistics of {selected_column}")
                        col_results['distribution'] = pandas_agent.run(f"Check for normality or specific distribution shapes of {selected_column}")
                        col_results['outliers'] = pandas_agent.run(f"Assess the presence of outliers of {selected_column}")
                        col_results['trends'] = pandas_agent.run(f"Analyse trends, seasonality, and cyclic patterns of {selected_column}")
                        col_results['missing'] = pandas_agent.run(f"Determine the extent of missing values of {selected_column}")
                        
                        st.session_state.analysis_results[f'col_{selected_column}'] = col_results
                
                # Display cached results
                col_results = st.session_state.analysis_results[f'col_{selected_column}']
                st.write(col_results['stats'])
                st.write(col_results['distribution'])
                st.write(col_results['outliers'])
                st.write(col_results['trends'])
                st.write(col_results['missing'])

            # Variable Comparison Section
            st.subheader('Engages the user: Would you like to compare Variables Side-by-Side?')
            col1, col2 = st.columns(2)
            with col1:
                var1 = st.selectbox("Select Variable 1", df.columns, key="var1_compare")
            with col2:
                var2 = st.selectbox("Select Variable 2", df.columns, key="var2_compare")

            if var1 and var2 and var1 != var2:
                st.markdown(f"### 📊 Best Chart for `{var1}` vs `{var2}`")
                
                # Cache comparison results
                comparison_key = f'compare_{var1}_{var2}'
                if comparison_key not in st.session_state.analysis_results:
                    with st.spinner(f"Comparing {var1} and {var2}..."):
                        comp_results = {}
                        comp_results['chart_type'] = pandas_agent.run(
                            f"What is the best chart type to compare the variables '{var1}' and '{var2}' together? "
                            f"Just return the chart type like 'scatter', 'line', 'bar', or 'histogram'."
                        ).lower()
                        
                        try:
                            if comp_results['chart_type'].startswith("scatter"):
                                st.write("Scatter Plot:")
                                st.scatter_chart(df[[var1, var2]])
                            elif comp_results['chart_type'].startswith("line"):
                                st.write("Line Chart:")
                                st.line_chart(df[[var1, var2]])
                            elif comp_results['chart_type'].startswith("bar"):
                                st.write("Bar Chart:")
                                st.bar_chart(df[[var1, var2]])
                            elif comp_results['chart_type'].startswith("histogram"):
                                st.write("Histograms:")
                                fig, (ax1, ax2) = plt.subplots(1, 2)
                                df[var1].hist(bins=20, ax=ax1)
                                df[var2].hist(bins=20, ax=ax2)
                                st.pyplot(fig)
                        except Exception as e:
                            st.warning(f"Chart rendering failed: {e}")
                        
                        comp_results['var1_insights'] = pandas_agent.run(f"Give me insights of {var1}")
                        comp_results['var2_insights'] = pandas_agent.run(f"Give me insights of {var2}")
                        
                        st.session_state.analysis_results[comparison_key] = comp_results
                
                    # Display cached comparison
                    comp_results = st.session_state.analysis_results[comparison_key]
                    st.markdown(f"### 🧠 LLM Insights")
                    insight1, insight2 = st.columns(2)
                    with insight1:
                        st.markdown(f"**Insights on `{var1}`**")
                        st.write(comp_results['var1_insights'])
                    with insight2:
                        st.markdown(f"**Insights on `{var2}`**")
                        st.write(comp_results['var2_insights'])

    
    with tab2:
        st.header('Ask Anything About Your Data')
        st.write("🤖 Welcome to the AI Assistant ChatBox!") 
        st.write("Got burning questions about your data? You're in the right place! Speak or type your queries below! 🔍💻")

        # Initialize chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        # Display chat history
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.write(f"**You**: {message['message']}")
            else:
                st.write(f"**AI**: {message['message']}")

        # Voice input section
        st.write("### Speak your question:")
        audio_bytes = audio_recorder(
            pause_threshold=2.0,
            text="🎤 Click to speak",
            recording_color="#e8b62c",
            neutral_color="#6aa36f",
            key="recorder"
        )

        # Process audio if recorded
        if audio_bytes:
            try:
                # Save audio to temporary file
                with open("temp_audio.wav", "wb") as f:
                    f.write(audio_bytes)
                
                # Initialize recognizer
                r = sr.Recognizer()
                
                # Recognize speech
                with sr.AudioFile("temp_audio.wav") as source:
                    audio = r.record(source)
                    user_question = r.recognize_google(audio)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({'role': 'user', 'message': user_question})
                    st.success(f"🎤 You said: {user_question}")
                    
                    # Get AI response
                    if st.session_state.current_df is not None:
                        pandas_agent = get_agent(st.session_state.current_df)
                        response = pandas_agent.run(user_question)
                        st.session_state.chat_history.append({'role': 'AI', 'message': response})
                        st.write(f"**AI**: {response}")

            except sr.UnknownValueError:
                st.error("Sorry, I couldn't understand the audio.")
            except sr.RequestError as e:
                st.error(f"Could not request results from Google Speech Recognition service; {e}")
            except Exception as e:
                st.error(f"Error processing voice input: {e}")

        # Text input section
        st.write("### Or type your question:")
        user_question = st.text_input("Ask a question about the data:", key="text_input")
        
        if user_question and st.session_state.current_df is not None:
            # Add to chat history
            st.session_state.chat_history.append({'role': 'user', 'message': user_question})
            
            # Get response
            pandas_agent = get_agent(st.session_state.current_df)
            response = pandas_agent.run(user_question)
            
            # Add response to history
            st.session_state.chat_history.append({'role': 'AI', 'message': response})
            st.write(f"**AI**: {response}")

            # Visualizations for visual questions
            if any(keyword in user_question.lower() for keyword in ["plot", "chart", "visualize", "graph", "show"]):
                try:
                    code_prompt = f"Write a Python matplotlib or seaborn code snippet to visualize this: {user_question}. Use only columns from the dataset."
                    code_response = pandas_agent.run(code_prompt)

                    st.markdown("**🔍 Visual Interpretation:**")
                    with st.expander("See the generated code"):
                        st.code(code_response, language='python')

                    # Execute the code safely
                    exec_globals = {"df": st.session_state.current_df, "plt": plt, "sns": sns}
                    exec_locals = {}
                    exec(code_response, exec_globals, exec_locals)
                    
                    # Display the plot
                    st.pyplot(plt.gcf())
                    plt.clf()
                except Exception as e:
                    st.warning(f"❌ Couldn't generate a chart: {e}")