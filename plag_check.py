import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image

def open_files(uploaded_files):
    if uploaded_files:
        student_files = uploaded_files
        student_notes = [file.getvalue().decode("utf-8") for file in student_files]

        vectorize = lambda Text: TfidfVectorizer().fit_transform(Text).toarray()
        similarity = lambda doc1, doc2: cosine_similarity([doc1, doc2])

        vectors = vectorize(student_notes)
        s_vectors = list(zip(student_files, vectors))

        def check_plagiarism():
            plagiarism_results = set()
            nonlocal s_vectors
            for student_a, text_vector_a in s_vectors:
                new_vectors = s_vectors.copy()
                current_index = new_vectors.index((student_a, text_vector_a))
                del new_vectors[current_index]
                for student_b, text_vector_b in new_vectors:
                    sim_score = similarity(text_vector_a, text_vector_b)[0][1]
                    student_pair = sorted((os.path.basename(student_a.name), os.path.basename(student_b.name)))
                    score = (student_pair[0], student_pair[1], sim_score)
                    plagiarism_results.add(score)
            return plagiarism_results

        return check_plagiarism()

plagiarism_img = Image.open('plagiarism_img.png')
st.set_page_config(
    page_title="Plag Check",
    page_icon=":lock:",
)

page_bg = """
<style>
div.stButton > button:first-child {
    background-color: #b3306b;
    color:#ffffff;
}
div.stButton > button:hover {
    background-color: #b3306b;
    color:#ffffff;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
[data-testid="stAppViewContainer"] {
    background-image: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)), url("https://images.unsplash.com/photo-1544396821-4dd40b938ad3?q=80&w=2073&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
    background-size: cover;
}

.footer {
    position: fixed;
    bottom: 10px;
    right: 10px;
    font-size: 14px;
    color: #808080;
}
</style>
<div class="footer">
    Developed by Dhruvi Vaghela
</div>
"""

st.markdown(page_bg, unsafe_allow_html=True)

uploaded_files = st.file_uploader("Select Files", accept_multiple_files=True, type=["txt"])
if st.button("Check Plagiarism", key="check_button"):
    if not uploaded_files:
        st.write("No files selected")
    else:
        plagiarism_results = open_files(uploaded_files)
        if plagiarism_results:
            st.write("Plagiarism check results:")
            for data in plagiarism_results:
                st.write(
                    f"""
                    **{data[0]}** is {data[2]*100:.2f}% similar to **{data[1]}**
                    """
                )

