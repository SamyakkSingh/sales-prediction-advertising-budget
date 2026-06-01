# Sales Prediction Project (Python + VS Code + Streamlit)

This project predicts product sales based on advertising budgets (TV, Radio, Newspaper).

## How to Run (VS Code)
1. Create virtual environment:
   python -m venv venv

2. Activate it:
   - Windows: venv\Scripts\activate
   - Mac/Linux: source venv/bin/activate

3. Install dependencies:
   pip install -r requirements.txt

4. Train the models:
   python train.py

5. Run prediction using CLI:
   python predict.py --tv 150 --radio 20 --newspaper 10

6. Run the Streamlit app:
   streamlit run streamlit_app.py

## Project Structure
- Advertising.csv → dataset
- train.py → trains models, saves into models/
- predict.py → make predictions from command line
- utils.py → helper functions
- streamlit_app.py → interactive UI
- models/ → saved models
- outputs/ → plots + reports
