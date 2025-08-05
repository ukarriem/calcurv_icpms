
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import streamlit as st
from fpdf import FPDF
import io
import base64

st.title("ICP-MS Calibration Curve Builder, Validator & Reporter")

st.markdown("""
Upload calibration and QC data in CSV format.
- Calibration CSV must have: `Analyte`, `Intensity`, `Concentration`
- QC CSV must have: `Analyte`, `Type` (ICV or CCV), `Measured`, `Expected`
""")

cal_file = st.file_uploader("Upload Calibration CSV", type=["csv"], key="cal")
qc_file = st.file_uploader("Upload ICV/CCV CSV", type=["csv"], key="qc")

summary_data = []
images = []

if cal_file:
    df = pd.read_csv(cal_file)
    analytes = df['Analyte'].unique()

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

        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        images.append((analyte, buf))

    summary_df = pd.DataFrame(summary_data)
    st.subheader("Calibration Summary")
    st.dataframe(summary_df)

    csv = summary_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Summary CSV", csv, "calibration_summary.csv", "text/csv")

    # QC Validation Section
    if qc_file:
        qc_df = pd.read_csv(qc_file)
        st.subheader("ICV/CCV Validation")
        thresholds = {"min": 90, "max": 110}
        qc_df['Recovery (%)'] = (qc_df['Measured'] / qc_df['Expected']) * 100
        qc_df['Pass'] = qc_df['Recovery (%)'].apply(lambda x: "✅" if thresholds['min'] <= x <= thresholds['max'] else "❌")
        st.dataframe(qc_df)

    # PDF Export
    if st.button("Generate PDF Report"):
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)

        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(200, 10, "ICP-MS Calibration Report", ln=True, align="C")

        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, "Calibration Summary", ln=True)

        for row in summary_data:
            pdf.cell(200, 10, f"{row['Analyte']}: y = {row['Slope']:.4f}x + {row['Intercept']:.2f}, R² = {row['R²']:.4f}, Valid: {row['Valid']}", ln=True)

        if qc_file:
            pdf.add_page()
            pdf.set_font("Arial", "B", 14)
            pdf.cell(200, 10, "ICV/CCV Validation", ln=True)
            pdf.set_font("Arial", size=12)
            for _, row in qc_df.iterrows():
                pdf.cell(200, 10, f"{row['Analyte']} ({row['Type']}): {row['Measured']:.2f}/{row['Expected']:.2f} = {row['Recovery (%)']:.2f}%, Result: {row['Pass']}", ln=True)

        for analyte, img_buf in images:
            pdf.add_page()
            pdf.set_font("Arial", "B", 12)
            pdf.cell(200, 10, f"Calibration Plot - {analyte}", ln=True)
            pdf.image(img_buf, x=10, y=30, w=180)

        pdf_output = f"calibration_report.pdf"
        pdf.output(pdf_output)

        with open(pdf_output, "rb") as f:
            b64_pdf = base64.b64encode(f.read()).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64_pdf}" download="{pdf_output}">Download PDF Report</a>'
        st.markdown(href, unsafe_allow_html=True)
