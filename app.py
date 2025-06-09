import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# —————————————————————————————
# 1. Load pipeline ter-fitted & data
# —————————————————————————————
pipeline = joblib.load("model_dropout_predictor.joblib")  # Preprocessing + Model
df = pd.read_csv("data.csv", delimiter=";")  # Data asli untuk opsi

# —————————————————————————————
# 2. UI Streamlit: Desain yang ditingkatkan
# —————————————————————————————
st.set_page_config(
    page_title="Deteksi Dropout Siswa",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header dengan deskripsi
st.markdown(
    """
    # 📘 Sistem Deteksi Risiko Dropout
    Aplikasi ini memprediksi risiko *dropout* siswa Jaya Jaya Institut.
    Masukkan data siswa di sidebar untuk melihat hasil prediksi.
    """
)

# —————————————————————————————
# 3. Sidebar untuk Input Data
# —————————————————————————————
with st.sidebar:
    st.header("Masukkan Data Siswa")
    
    # Mapping untuk Gender
    gender_map = {0: "Perempuan", 1: "Laki-laki"}
    gender_label = st.selectbox("Jenis Kelamin", list(gender_map.values()))
    gender_code = [k for k, v in gender_map.items() if v == gender_label][0]

    age = st.slider(
        "Usia Saat Pendaftaran",
        min_value=int(df["Age_at_enrollment"].min()),
        max_value=int(df["Age_at_enrollment"].max()),
        value=int(df["Age_at_enrollment"].median())
    )

    # Dynamic Course mapping
    course_codes = df["Course"].unique().tolist()
    course_codes.sort()
    course_map = {c: f"Course {c}" for c in course_codes}
    course_label = st.selectbox("Course", list(course_map.values()))
    course_code = [k for k, v in course_map.items() if v == course_label][0]

    # Pilihan Kualifikasi Sebelumnya
    prev_codes = df["Previous_qualification"].unique().tolist()
    prev_codes.sort()
    prev_map = {code: f"Kode {code}" for code in prev_codes}
    prev_label = st.selectbox("Kualifikasi Sebelumnya", list(prev_map.values()))
    prev_code = [k for k, v in prev_map.items() if v == prev_label][0]

    # Pilihan Negara - Menambahkan opsi "General"
    nationality_map = {1: "Portuguese", 2: "German", 6: "Spanish", 11: "Italian", 13: "Dutch", 14: "English", 
                       17: "Lithuanian", 21: "Angolan", 22: "Cape Verdean", 24: "Guinean", 25: "Mozambican", 
                       26: "Santomean", 32: "Turkish", 41: "Brazilian", 62: "Romanian", 100: "Moldova", 
                       101: "Mexican", 103: "Ukrainian", 105: "Russian", 108: "Cuban", 109: "Colombian", 
                       0: "General (No specific country)"}  # Tambah opsi "General"
    nationality_label = st.selectbox("Negara Asal", list(nationality_map.values()))
    nationality_code = [k for k, v in nationality_map.items() if v == nationality_label][0]

    admission_grade = st.slider(
        "Admission Grade",
        float(df["Admission_grade"].min()), 
        float(df["Admission_grade"].max()), 
        float(df["Admission_grade"].median())
    )

    st.markdown("---")
    pred_button = st.button("Prediksi Dropout")

# —————————————————————————————
# 4. Logic Prediksi & Hasil
# —————————————————————————————
pred_result = None  # Variabel untuk menyimpan hasil prediksi

if pred_button:
    # Input manual
    input_dict = {
        "Gender": [gender_code],
        "Age_at_enrollment": [age],
        "Course": [course_code],
        "Previous_qualification": [prev_code],
        "Admission_grade": [admission_grade],
        "Nationality": [nationality_code]  # Menambahkan Negara
    }
    input_df = pd.DataFrame(input_dict)

    # Lengkapi kolom lain (fitur lain) dengan nilai default
    all_cols = pipeline.named_steps['preprocessor'].feature_names_in_
    num_cols = pipeline.named_steps['preprocessor'].transformers_[0][2]
    cat_cols = pipeline.named_steps['preprocessor'].transformers_[1][2]

    for col in all_cols:
        if col not in input_df.columns:
            input_df[col] = df[col].median() if col in num_cols else df[col].mode()[0]
    input_df = input_df[all_cols]

    # Prediksi
    y_pred = pipeline.predict(input_df)[0]
    y_proba = pipeline.predict_proba(input_df)[0, 1]

    # Simpan hasil prediksi untuk ditampilkan
    pred_result = (y_pred, y_proba)

# —————————————————————————————
# 5. Menampilkan Hasil Prediksi di Atas
# —————————————————————————————
if pred_result:
    y_pred, y_proba = pred_result
    col1, col2 = st.columns([1, 3])

    with col1:
        # Gunakan Plotly untuk gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=y_proba * 100,
            title={"text": "Probabilitas Dropout (%)"},
            gauge={
                "axis": {"range": [None, 100]},
                "bar": {"color": "red"},
                "steps": [
                    {"range": [0, 40], "color": "green"},
                    {"range": [40, 70], "color": "yellow"},
                    {"range": [70, 100], "color": "red"}
                ]
            }
        ))
        fig.update_layout(
            height=350, 
            margin={"t": 0, "b": 20, "l": 10, "r": 10},
            title_font=dict(size=18)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Add custom CSS for padding
        st.markdown("""
            <style>
                .pred-result {
                    padding-top: 60px;
                }
            </style>
        """, unsafe_allow_html=True)

        # Add condition for dropout risk
        with st.container():
            if y_pred == 1:
                st.markdown('<div class="pred-result">', unsafe_allow_html=True)
                st.error("⚠ Siswa berisiko dropout!")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="pred-result">', unsafe_allow_html=True)
                st.success("✅ Siswa tidak berisiko tinggi.")
                st.markdown('</div>', unsafe_allow_html=True)

        # Deskripsi tambahan
        risk_desc = (
            "Intervensi segera diperlukan." if y_proba > 0.7
            else "Monitoring lanjutan direkomendasikan." if y_proba > 0.4
            else "Risiko rendah."
        )
        st.info(risk_desc)


# —————————————————————————————
# 6. Tombol Show/Hide untuk Tabel
# —————————————————————————————
if 'show_tables' not in st.session_state:
    st.session_state.show_tables = False

show_tables_button = st.button("Tampilkan/Sembunyikan Tabel Informasi")
if show_tables_button:
    st.session_state.show_tables = not st.session_state.show_tables

if st.session_state.show_tables:
    # Tabel untuk informasi
    st.subheader("Informasi Pilihan untuk Siswa")
    
    # Perbaikan pada tampilan tabel (penataan kolom)
    course_df = pd.DataFrame({
        "Kode Kursus": [33, 171, 8014, 9003, 9070, 9085, 9119, 9130, 9147, 9238, 9254, 9500, 9556, 9670, 9773, 9853, 9991],
        "Nama Kursus": [
            "Biofuel Production Technologies", "Animation and Multimedia Design", "Social Service (evening attendance)",
            "Agronomy", "Communication Design", "Veterinary Nursing", "Informatics Engineering", "Equinculture",
            "Management", "Social Service", "Tourism", "Nursing", "Oral Hygiene", "Advertising and Marketing Management",
            "Journalism and Communication", "Basic Education", "Management (evening attendance)"
        ]
    })

    prev_qualification_df = pd.DataFrame({
    "Kode Kualifikasi": [1, 2, 3, 4, 5, 6, 9, 10, 12, 14, 15, 19, 38, 39, 40, 42, 43],
    "Nama Kualifikasi": [
        "Secondary education", "Higher education - bachelor's degree", "Higher education - degree",
        "Higher education - master's", "Higher education - doctorate", "Frequency of higher education",
        "12th year of schooling - not completed", "11th year of schooling - not completed", "Other", "Basic education",
        "Second cycle (5th/6th/7th year)", "Basic education 1st cycle (2nd/3rd/4th year)", "Basic education (1st year)",
        "School attendance (children under 12 years)", "Other", "High school diploma", "Vocational education"
    ]  # Ensure both lists have 17 items
})


    nationality_df = pd.DataFrame({
        "Kode Negara": [1, 2, 6, 11, 13, 14, 17, 21, 22, 24, 25, 26, 32, 41, 62, 100, 101, 103, 105, 108, 109],
        "Nama Negara": [
            "Portuguese", "German", "Spanish", "Italian", "Dutch", "English", "Lithuanian", "Angolan", "Cape Verdean",
            "Guinean", "Mozambican", "Santomean", "Turkish", "Brazilian", "Romanian", "Moldova (Republic of)", "Mexican",
            "Ukrainian", "Russian", "Cuban", "Colombian"
        ]
    })

    st.dataframe(course_df.style.set_properties(**{'width': '100px'}))  # Kolom yang lebih rapi
    st.dataframe(prev_qualification_df)
    st.dataframe(nationality_df)

# —————————————————————————————
# 7. Footer
# —————————————————————————————
st.markdown("---")
st.markdown(
    "Dicoding | Project by Afif Hamzah"
)
