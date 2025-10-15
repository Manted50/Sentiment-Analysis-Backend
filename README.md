# French sentiment analyser

This project was made in a studying context to show all the pipelines that we can apply for an ML project. The idea is to create a model to classify sentences and know if it's positive or negative using sklearn and MLflow. Then, we can use it for a website project using FastAPI and Streamlit libraries. At the end, we tried to automate that using the deployment of render and Streamlit Cloud. It shows how a Machine Learning project can evolve properly in companies. 

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Manted50/Sentiment-Analysis-Backend
    ```
2. Navigate to the project directory:
    ```bash
    cd Sentiment-Analysis-Backend
    ```
3. Run the backend:
    ```bash
    uvicorn main:app --reload
    ```
4. Run the frontend:
    ```bash
    streamlit run ui.py
    ```

## Usage

Before the run of the application, please modify the ui.py to use the good localhost URL.

Here is the result Website : https://mlops-sentiment-analysis-frontend.streamlit.app/
