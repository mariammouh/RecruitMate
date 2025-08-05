import streamlit as st
import pdfplumber
import re

# --- Your existing PDF extraction function ---
def extract_text_from_pdf(pdf_file):
    """
    Extracts text from an uploaded PDF file-like object.
    """
    text = ""
    try:
        # pdf_file is a BytesIO object when uploaded via st.file_uploader
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return None
    return text

# --- Your existing text cleaning function ---
def clean_text(text):
    if not text:
        return ""

    # 1. Convert to lowercase
    cleaned_text = text.lower()

    # 2. Replace multiple newlines/whitespace with a single space
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)

    # 3. Strip leading/trailing whitespace
    cleaned_text = cleaned_text.strip()

    return cleaned_text

# --- Streamlit Application ---
def main():
    st.set_page_config(page_title="Resume Parser", layout="centered")
    st.title("📄 AI-Powered Resume Parser")
    st.markdown("Upload a PDF resume to extract its content.")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        st.success("File uploaded successfully!")
        
        # Display file details
        # file_details = {"filename": uploaded_file.name, "filetype": uploaded_file.type, "filesize": uploaded_file.size}
        # st.write(file_details)

        # Extract and Clean Text
        st.subheader("Extracted & Cleaned Text:")
        
        with st.spinner("Extracting and cleaning text..."):
            raw_text = extract_text_from_pdf(uploaded_file)
            if raw_text:
                cleaned_text = clean_text(raw_text)

                # Display raw text (optional, for comparison)
                with st.expander("Show Raw Extracted Text"):
                    st.text(raw_text)

                # Display cleaned text
                st.text_area("Cleaned Resume Content", cleaned_text, height=300)
            else:
                st.warning("Could not extract text from the PDF.")
    else:
        st.info("Please upload a PDF file to proceed.")

if __name__ == "__main__":
    main()