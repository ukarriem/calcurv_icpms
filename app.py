
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import streamlit as st

st.title("ICP-MS Calibration Curve Builder & Validator")

st.markdown("""
Upload calibration data for selected analytes in CSV format.
The CSV must have two columns: `Intensity` and `Concentration`, and a column `Analyte`.
""")

uploaded_file = st.file_uploader("Upload your calibration CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    analytes = df['Analyte'].unique()
    summary_data = []

    for analyte in analytes:
        subset = df[df['Analyte'] == analyte]
        X = subset['Intensity'].values.reshape(-1, 1)
        y = subset['Concentration'].values

        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        slope = model.coef_[0]
        intercept = model.intercept_
        r2 = r2_score(y, y_pred)

        # Add to summary
        summary_data.append({
            "Analyte": analyte,
            "Slope": slope,
            "Intercept": intercept,
            "R²": r2,
            "Valid": "✅" if r2 >= 0.99 else "❌"
        })

        # Plot
        fig, ax = plt.subplots()
        ax.scatter(X, y, label='Data', color='blue')
        ax.plot(X, y_pred, label=f"Fit: y={slope:.4f}x + {intercept:.2f}\nR²={r2:.4f}", color='red')
        ax.set_title(f"Calibration Curve for {analyte}")
        ax.set_xlabel("Intensity")
        ax.set_ylabel("Concentration (ppb)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    summary_df = pd.DataFrame(summary_data)
    st.subheader("Calibration Summary")
    st.dataframe(summary_df)

    csv = summary_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Summary as CSV", csv, "calibration_summary.csv", "text/csv")
