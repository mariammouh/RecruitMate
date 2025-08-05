import streamlit as st
import pdfplumber
import re
import spacy
from io import BytesIO
from datetime import datetime
from dateutil.relativedelta import relativedelta
from spacy.matcher import PhraseMatcher

# --- Load pre-trained spaCy model ---
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("SpaCy 'en_core_web_sm' model not found. Please run: "
             "`pip install spacy` and `python -m spacy download en_core_web_sm`")
    st.stop()

# --- 1. PDF Text Extraction Function ---
def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return None
    return text

# --- 2. Text Cleaning Function ---
def clean_text(text):
    if not text:
        return ""
    cleaned_text = re.sub(r'\s+', ' ', text)
    cleaned_text = cleaned_text.strip()
    return cleaned_text

# --- UTILITY FUNCTION: Parse various date formats ---
def parse_date_string(date_str):
    """
    Parses a date string (e.g., "Oct 2013", "10/2013", "2013") into a datetime object.
    Returns datetime.min if parsing fails or datetime.now() for 'present'/'current'.
    """
    date_str = date_str.lower().strip()
    
    if "present" in date_str or "current" in date_str:
        return datetime.now()

    formats = [
        "%b %Y",  # Oct 2013 (e.g., Nov 2023)
        "%B %Y",  # October 2013
        "%m/%Y",  # 10/2013
        "%Y",     # 2013 (assume Jan 1st if only year)
        "%Y-%m", # 2013-10
    ]
    for fmt in formats:
        try:
            # If only year, set month/day to 1 for consistent datetime object
            if fmt == "%Y":
                return datetime.strptime(date_str, fmt).replace(month=1, day=1)
            # If only year-month, set day to 1
            if fmt == "%Y-%m":
                return datetime.strptime(date_str, fmt).replace(day=1)
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return datetime.min # Return a default minimum date if unable to parse

# --- UTILITY FUNCTION: Format datetime to string for display ---
def format_datetime_for_display(dt_obj):
    if not isinstance(dt_obj, datetime) or dt_obj == datetime.min:
        return "N/A"
    # To handle 'present' consistently, check if it's close to current date
    current_time_threshold = datetime.now() - relativedelta(months=3) # Within last 3 months
    if dt_obj >= current_time_threshold: # If the end date is very recent
         return "Present"
    return dt_obj.strftime("%b %Y") # e.g., "Oct 2013"


# --- NEW: Section Segmentation Function (REVAMPED) ---
def segment_resume_sections(raw_text):
    sections = {
        "header": "", # Top part before any major section
        "profile": "",
        "employment_history": "",
        "education": "",
        "skills": "",
        "languages": "",
        "certificates": "",
        "hobbies": "",
        "accomplishments": "",
    }

    # Define common section headers and the order they might appear
    # The regex needs to be very robust: starts of line, potential whitespace, ends of line or new section.
    # Pattern: '^\s*(HEADER_TEXT)\s*$' or similar.
    # The order here helps in sequential slicing.
    section_markers = [
        ("PROFILE", r'^\s*PROFILE\s*$', "profile"),
        ("EMPLOYMENT HISTORY", r'^\s*EMPLOYMENT HISTORY\s*$', "employment_history"),
        ("EDUCATION", r'^\s*EDUCATION\s*$', "education"),
        ("SKILLS", r'^\s*SKILLS\s*$', "skills"),
        ("LANGUAGES", r'^\s*LANGUAGES\s*$', "languages"),
        ("CERTIFICATES", r'^\s*CERTIFICATES?\s*$', "certificates"), # Certificates or Certificate
        ("HOBBIES", r'^\s*HOBBIES\s*$', "hobbies"),
        ("ACCOMPLISHMENTS", r'^\s*ACCOMPLISHMENTS\s*$', "accomplishments"),
        # Add other potential top-level headers for your specific CV format:
        ("LINKS", r'^\s*LINKS\s*$', "links") # Example from your CV
        # You might need to add more variations based on real CVs
        # e.g., "WORK EXPERIENCE", "PROFESSIONAL BACKGROUND", "FORMATIONS", etc.
    ]
    
    # Compile regex patterns for efficiency
    compiled_markers = []
    for display_name, pattern, section_key in section_markers:
        compiled_markers.append((section_key, re.compile(pattern, re.MULTILINE | re.IGNORECASE)))
    
    # Find all section boundaries and their positions
    section_boundaries = [] # List of (index, section_key)
    for section_key, pattern_compiled in compiled_markers:
        for match in pattern_compiled.finditer(raw_text):
            section_boundaries.append((match.start(), section_key, match.end())) # (start_of_header, key, end_of_header_line)
    
    # Sort boundaries by their start index
    section_boundaries.sort()

    # If no sections found, put everything in header
    if not section_boundaries:
        sections["header"] = raw_text.strip()
        return sections

    # Extract header content: everything before the first section
    if section_boundaries[0][0] > 0:
        sections["header"] = raw_text[0 : section_boundaries[0][0]].strip()

    # Extract content for each main section
    for i, (start_header_idx, section_key, end_header_line_idx) in enumerate(section_boundaries):
        content_start_idx = end_header_line_idx # Content starts right after the header line
        content_end_idx = len(raw_text) # Default to end of document

        if i + 1 < len(section_boundaries):
            # The current section's content ends at the start of the next section's header
            content_end_idx = section_boundaries[i+1][0]
        
        # Assign content to the respective section key
        sections[section_key] = raw_text[content_start_idx : content_end_idx].strip()
    
    return sections

# --- NEW UTILITY: Extract Organization (Company/Institution) ---
# --- NEW UTILITY: Extract Organization (Company/Institution) ---
def _extract_organization(doc_block, positive_keywords=None, negative_keywords=None):
    """
    Extracts an organization name (Company or Institution) from a spaCy Doc block.
    Applies positive and negative keyword filtering.
    """
    if positive_keywords is None:
        positive_keywords = []
    if negative_keywords is None:
        negative_keywords = []

    # Filter out very common words/entities that spaCy might misclassify as ORG
    common_false_orgs = [
        "city", "state", "inc", "ltd", "corp", "llc", "group", "team", "client", "platform", 
        "studio", "health", "safety", "management", "project", "program", "class", "classes",
        "exercise", "fitness", "training", "nutrition", "system", "systems", "solutions",
        "leader", "manager", "director", "specialist", "analyst", "developer", "architect",
        "coach", "instructor", # These are often job titles
        "bachelor", "master", "diploma", "degree", "ph.d", # Education keywords
        "past", "current", "present", "online", "events", "us", "uk", "usa", "canada", # Location/time/generic words
        # Specific locations/noise from YOUR CV that might be misclassified as ORG:
        "los angeles", "tennessee", "miami", "datteln", "ponta porã", # From your CV
        "surpassing", "crossfit", "level", "varsity", "track", "athlete", "command", # Specific noise/roles from CV
        "pinterest", "linkedin", "template", "surpassing" # From links or other sections
    ]

    best_org = None # To store the best candidate found
    
    # Debug: Print all ORG entities found by spaCy in this block before filtering
    print(f"\n--- _extract_organization: Analyzing Doc Block for ORGs (from text: '{doc_block.text[:50]}...') ---")
    for ent in doc_block.ents:
        if ent.label_ == "ORG":
            print(f"  Candidate ORG (spaCy): '{ent.text}' (Length: {len(ent.text)})")

    for ent in doc_block.ents:
        if ent.label_ == "ORG":
            org_text = ent.text.strip()
            org_text_lower = org_text.lower()
            
            # Basic length/word count filter
            if len(org_text.split()) < 1 or len(org_text) < 2:
                print(f"  Rejected '{org_text}' (Filter: Too short)")
                continue

            # Apply positive filtering (must contain at least one positive keyword if provided)
            if positive_keywords and not any(kw in org_text_lower for kw in positive_keywords):
                print(f"  Rejected '{org_text}' (Filter: No positive keyword match)")
                continue
            
            # Apply negative filtering (must NOT contain any negative keywords or common false positives)
            # Use regex for word boundaries to ensure full word match for common_false_orgs
            is_negative = False
            for word in common_false_orgs:
                if re.search(r'\b' + re.escape(word) + r'\b', org_text_lower):
                    print(f"  Rejected '{org_text}' (Filter: Found common false ORG '{word}')")
                    is_negative = True
                    break
            if is_negative:
                continue
            
            for word in negative_keywords:
                 if re.search(r'\b' + re.escape(word) + r'\b', org_text_lower):
                    print(f"  Rejected '{org_text}' (Filter: Found specific negative keyword '{word}')")
                    is_negative = True
                    break
            if is_negative:
                continue

            # Prioritize longer, more specific ORG names if multiple are found
            if best_org is None or len(org_text) > len(best_org):
                best_org = org_text
                print(f"  Accepted '{org_text}' (Current best candidate)")
            else:
                print(f"  Accepted '{org_text}' (but not better than current best '{best_org}')")

    if best_org:
        # Specific cleaning for entities that include location/noise from your CV
        best_org = re.sub(r'•\s*Surpassing expectations in Personal Training coursework\s*\.?\s*', '', best_org, flags=re.IGNORECASE).strip()
        best_org = re.sub(r'Concordia University\s*Concordia\s*', 'Concordia University', best_org, flags=re.IGNORECASE).strip()
        best_org = re.sub(r'National CPR Foundation.*', 'National CPR Foundation', best_org, flags=re.IGNORECASE).strip()
        best_org = re.sub(r'Barlow High School.*', 'Barlow High School', best_org, flags=re.IGNORECASE).strip()
        best_org = re.sub(r'Dominist Fitness.*', 'Dominist Fitness', best_org, flags=re.IGNORECASE).strip() 
        best_org = re.sub(r'Curves Gym.*', 'Curves Gym', best_org, flags=re.IGNORECASE).strip() 
        # General location cleanup if still present (be careful not to remove valid parts of org name)
        best_org = re.sub(r'\s*(?:Ponta Porã|Datteln|Tennessee|Miami|California|Los Angeles)\b', '', best_org, flags=re.IGNORECASE).strip() 

        # Final check after aggressive cleaning
        if len(best_org.split()) < 1 or len(best_org) < 2:
            print(f"  Final Cleaned '{best_org}' too short. Returning N/A.")
            return "N/A"

        print(f"  Final ORG Chosen: '{best_org}'")
        return best_org.title() if best_org.isupper() else best_org # Standardize casing
    print(f"  No suitable ORG found. Returning N/A.")
    return "N/A"

# --- NEW UTILITY: Extract and Parse Date Range ---
def _extract_date_range_and_parse(text_block):
    """
    Extracts a date range string and parses it into formatted start/end dates.
    Returns (dates_str, formatted_start_date, formatted_end_date, start_dt_obj, end_dt_obj)
    """
    # Refined date pattern to capture full dates (Month YYYY, MM/YYYY, YYYY) and ranges
    date_range_pattern = re.compile(
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\s*[-–—to\s]+(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}|present|current)\b' # Month YYYY - Month YYYY/present
        r'|\b\d{2}/\d{4}\s*[-–—to\s]+(?:\d{2}/\d{4}|present|current)\b' # MM/YYYY - MM/YYYY/present
        r'|\b\d{4}\s*[-–—to\s]+(?:\d{4}|present|current)\b' # YYYY - YYYY/present
        , re.IGNORECASE
    )
    
    dates_found = date_range_pattern.search(text_block)
    
    dates_str = "N/A"
    start_date_display = "N/A"
    end_date_display = "N/A"
    start_dt_obj = datetime.min
    end_dt_obj = datetime.min

    if dates_found:
        dates_str = dates_found.group(0)
        date_parts = re.split(r'\s*[-–—to]\s*', dates_str, flags=re.IGNORECASE)
        start_date_part = date_parts[0]
        end_date_part = date_parts[1] if len(date_parts) > 1 else start_date_part # If single date, end=start
        
        start_dt_obj = parse_date_string(start_date_part)
        end_dt_obj = parse_date_string(end_date_part)
        
        start_date_display = format_datetime_for_display(start_dt_obj)
        end_date_display = format_datetime_for_display(end_dt_obj)

    return dates_str, start_date_display, end_date_display, start_dt_obj, end_dt_obj


# --- 3. Information Extraction Functions ---

# --- UPDATED extract_name Function ---
def extract_name(header_text): 
    if not header_text:
        return "N/A"
    lines = header_text.strip().split('\n')
    
    name_candidate = "N/A"

    # Strategy 1: Look at the very first few lines for a strong candidate
    for line in lines[:5]:
        cleaned_line = re.sub(r'\s+', ' ', line).strip()
        words_in_line = cleaned_line.split()

        if 2 <= len(words_in_line) <= 4 and (cleaned_line.isupper() or cleaned_line.istitle()):
            name_candidate = cleaned_line
            break

    # Strategy 2: Fallback to spaCy's PERSON entity on the header
    if name_candidate == "N/A":
        doc = nlp(re.sub(r'\s+', ' ', header_text).strip())
        potential_names = []
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                if (ent.text.istitle() or ent.text.isupper()) and len(ent.text.split()) >= 2:
                    potential_names.append((ent.start_char, ent.text))
        
        if potential_names:
            potential_names.sort(key=lambda x: x[0])
            name_candidate = potential_names[0][1]
    
    # --- POST-PROCESSING: Clean up name if it contains common job titles ---
    if name_candidate != "N/A":
        job_titles_to_remove = [
            "bootcamp instructor", "instructor", "leader", "coach", "manager", "engineer", 
            "specialist", "developer", "analyst", "director", "architect", "lead", "consultant", 
            "associate", "intern", "ceo", "cto", "cfo", "head", "senior", "junior", 
            "certified bootcamp instructor"
        ]
        job_titles_to_remove.sort(key=len, reverse=True)

        for title_kw in job_titles_to_remove:
            pattern = r'[,.\s-]*\b' + re.escape(title_kw) + r'\b$'
            if re.search(pattern, name_candidate, re.IGNORECASE):
                cleaned_name = re.sub(pattern, '', name_candidate, flags=re.IGNORECASE).strip()
                if len(cleaned_name.split()) >= 2:
                    name_candidate = cleaned_name
                else: 
                    name_candidate = "N/A" # If cleaning makes it too short, declare N/A
                break

    if name_candidate != "N/A":
        name_candidate = name_candidate.title() if name_candidate.isupper() else name_candidate
        
    return name_candidate


# --- UPDATED extract_contact_info Function ---
def extract_contact_info(header_text): # Takes header_text (already cleaned and lowercased in parse_resume)
    email = "N/A"
    phone = "N/A"

    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, header_text)
    if emails:
        email = emails[0]

    phone_pattern = r'\b(?:\+?\d{1,3}[-.\s]?)?(\d{10})\b|\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
    phones_found = re.findall(phone_pattern, header_text)
    clean_matched_phones = []
    for match_tuple in phones_found:
        clean_matched_phones.append(''.join(filter(None, match_tuple)))

    if clean_matched_phones:
        matched_phone = clean_matched_phones[0]

        phone = re.sub(r'[\s\.\-\(\)]+', '', matched_phone)
        
        if len(phone) >= 10:
            if phone.startswith(('06', '07')) and len(phone) == 10:
                 phone = f"{phone[:3]}-{phone[3:6]}-{phone[6:]}"
            elif phone.startswith('1') and len(phone) == 11:
                phone = f"+1-{phone[1:4]}-{phone[4:7]}-{phone[7:]}"
            elif len(phone) == 10:
                phone = f"{phone[:3]}-{phone[3:6]}-{phone[6:]}"
            else:
                 phone = phone
        else:
             phone = "N/A"
    return email, phone

def extract_skills(text):
    common_skills = [
        "python", "java", "c++", "javascript", "html", "css", "sql", "git",
        "excel", "word", "powerpoint", "project management", "customer service",
        "data analysis", "machine learning", "deep learning", "nlp", "cloud computing",
        "aws", "azure", "gcp", "docker", "kubernetes", "agile", "scrum", "leadership",
        "communication", "teamwork", "problem solving", "autocad", "solid edge", "moldflow",
        "mechanical engineering", "manufacturing", "product development", "etl", "intertek",
        "ul standards", "cad", "3d modeling", "design engineer", "qa",
        "microsoft office", "pro/engineer", "pro/mechanical", "microsoft navision",
        "microsoft project", "customer interface", "project coordination", "lean manufacturing",
        "sheet metal design", "optic design", "bills of material", "assembly process",
        "cross functional project team", "product concept", "submittal drawings",
        "lighting solutions", "fire safety codes", "field surveys", "order processing",
        "production orders", "technical solutions",
        "big data", "ai", "cloud", "data science", "machine learning", "deep learning",
        "software development", "web development", "mobile development", "database management",
        "network administration", "cyber security", "system analysis", "data mining",
        "business intelligence", "devops", "api development", "testing", "quality assurance",
        # Adding skills from the provided CV example:
        "cardio training", "fitness routines", "hiit", "client assessments", "health & safety",
        "personal training", "crossfit", "coaching", "cycling", "nutrition", "staffing", "marketing",
        "group fitness", "program budgeting", "program statistics", "fitness assessments",
        "active listening", "strength and conditioning", "group cycling", "skiing", "hockey", "knitting"
    ]
    extracted_skills = set()
    doc = nlp(text)
    matcher = PhraseMatcher(nlp.vocab)
    patterns = [nlp.make_doc(skill) for skill in common_skills]
    matcher.add("SkillList", patterns)
    matches = matcher(doc)
    for match_id, start, end in matches:
        span = doc[start:end]
        extracted_skills.add(span.text.lower())
    return sorted(list(extracted_skills))

# --- REWORKED extract_languages Function (with proficiency and direct language parsing) ---
# --- REWORKED extract_languages Function (More robust for varied formats) ---
def extract_languages(text):
    extracted_languages_with_proficiency = []
    
    languages = [
        "english", "french", "spanish", "german", "arabic", "chinese", "japanese",
        "portuguese", "russian", "italian", "korean", "dutch", "swedish"
    ]
    
    proficiency_keywords_map = { 
        "native speaker": "Native",
        "fluent": "Fluent",
        "proficient": "Proficient",
        "basic": "Basic",
        "conversational": "Conversational",
        "very good command": "Very Good", # From your CV
        "good command": "Good",
        "intermediate": "Intermediate",
        "advanced": "Advanced",
        "beginner": "Beginner",
    }
    
    # Debug: Print incoming language section text
    print(f"\n--- extract_languages: Input text (len: {len(text)}) ---\n'{text}'\n-------------------------------------------------\n")

    # Combine languages and proficiencies into a single pattern for initial search
    # This pattern tries to capture "Language [Proficiency]" or "Language"
    # and "Proficiency" separately.
    language_pattern_parts = [re.escape(lang) for lang in languages]
    proficiency_pattern_parts = [re.escape(prof) for prof in proficiency_keywords_map.keys()]
    
    # Regex to find: (Language) [optional whitespace/punctuation] (optional Proficiency)
    # OR (Proficiency) alone.
    # This will be applied to the *entire text block* (not line by line initially)
    combined_pattern = re.compile(
        r'\b(' + '|'.join(language_pattern_parts) + r')\b' # Captures Language (Group 1)
        r'(?:[\s,.-]*\b(' + '|'.join(proficiency_pattern_parts) + r')\b)?' # Optional Proficiency (Group 2)
        r'|\b(' + '|'.join(proficiency_pattern_parts) + r')\b' # OR just Proficiency (Group 3)
        , re.IGNORECASE
    )

    # Use a dictionary to store languages, keyed by their name (lowercased) for easy updates
    # Value: {'language': 'English', 'proficiency': 'Native'}
    language_results_dict = {}

    # Find all matches of the combined pattern in the entire text block
    matches = combined_pattern.finditer(text) # Use finditer to get match objects with spans

    # Process matches:
    for match in matches:
        lang_match = match.group(1) # Language from Group 1
        prof_match_group2 = match.group(2) # Proficiency from Group 2 (if with lang)
        prof_match_group3 = match.group(3) # Proficiency from Group 3 (if standalone)
        
        found_lang_name = lang_match.lower() if lang_match else None
        found_proficiency_norm = None
        if prof_match_group2:
            found_proficiency_norm = proficiency_keywords_map.get(prof_match_group2.lower())
        elif prof_match_group3:
            found_proficiency_norm = proficiency_keywords_map.get(prof_match_group3.lower())

        # --- Logic for linking language and proficiency ---
        if found_lang_name:
            if found_lang_name not in language_results_dict:
                language_results_dict[found_lang_name] = {'language': found_lang_name.title(), 'proficiency': 'N/A'}
            
            # If proficiency found directly with the language, or current proficiency is N/A, update it
            if found_proficiency_norm and language_results_dict[found_lang_name]['proficiency'] == 'N/A':
                language_results_dict[found_lang_name]['proficiency'] = found_proficiency_norm
        
        elif found_proficiency_norm: # Standalone proficiency, try to link to previous language
            # This is trickier without line context. We'll rely on the sequence of matches.
            # If the last language found is 'N/A', update it.
            if extracted_languages_with_proficiency and extracted_languages_with_proficiency[-1]['proficiency'] == 'N/A':
                extracted_languages_with_proficiency[-1]['proficiency'] = found_proficiency_norm
            # Otherwise, this proficiency is currently unassigned or belongs to a language already processed.
            # For simplicity, we'll assume it belongs to the immediately preceding N/A language in the output.

    # Convert dictionary values to final list. This automatically handles unique languages.
    extracted_languages_with_proficiency = list(language_results_dict.values())

    # Final pass to handle cases like "English \n Native speaker \n Spanish \n Very good command"
    # where the initial regex might have missed explicit linking for separated terms.
    # This loop will re-process with line-by-line context after initial matches.
    current_lang_entry_waiting_for_prof_ref = None # Reference to the dict in the list
    for i, line in enumerate(text.split('\n')):
        line_clean = line.strip().lower()
        if not line_clean: continue
        
        lang_found_on_line = next((lang for lang in languages if re.search(r'\b' + re.escape(lang) + r'\b', line_clean)), None)
        prof_found_on_line = next((prof_term for prof_phrase, prof_term in proficiency_keywords_map.items() if re.search(r'\b' + re.escape(prof_phrase) + r'\b', line_clean)), None)
        
        if lang_found_on_line:
            # Get the actual entry from our results list (it should be there now)
            entry = next((e for e in extracted_languages_with_proficiency if e['language'].lower() == lang_found_on_line), None)
            if entry:
                if entry['proficiency'] == 'N/A' and not prof_found_on_line:
                    current_lang_entry_waiting_for_prof_ref = entry
                elif prof_found_on_line:
                    current_lang_entry_waiting_for_prof_ref = None # No longer waiting if prof found

        elif prof_found_on_line and current_lang_entry_waiting_for_prof_ref and current_lang_entry_waiting_for_prof_ref['proficiency'] == 'N/A':
            current_lang_entry_waiting_for_prof_ref['proficiency'] = prof_found_on_line
            current_lang_entry_waiting_for_prof_ref = None # Consumed

    print(f"--- Languages Extracted: {extracted_languages_with_proficiency} ---\n")
    return extracted_languages_with_proficiency

def extract_certificates(text):
    common_certificates = [
        "pmp", "cism", "cissp", "ccna", "aws certified solutions architect",
        "microsoft certified", "google cloud professional", "scrum master",
        "itil", "comptia a+", "comptia network+", "comptia security+",
        "certified ethical hacker", "prince2", "six sigma", "cfa", "cpa",
        "dut", "diplôme universitaire de technologie",
        # Adding certificates from the provided CV:
        "crossfit level 1 instructor", "coach's prep certified", "advanced first aid diploma",
        "national cpr foundation", 
        "cpr certified"
    ]
    extracted_certificates = set()
    doc = nlp(text)
    matcher = PhraseMatcher(nlp.vocab)
    patterns = [nlp.make_doc(cert) for cert in common_certificates]
    matcher.add("CertificateList", patterns)
    matches = matcher(doc)
    for match_id, start, end in matches:
        span = doc[start:end]
        extracted_certificates.add(span.text.lower())
    
    # Broader regex for certificate-like entries within the dedicated section (if exists)
    # This pattern looks for lines that are largely capitalized or look like certificate names
    certificate_line_pattern = r'^\s*(?:[A-Z][a-z\s]*){2,}\s*(?:diploma|certificate|certified|instructor)\b.*$'
    for line in text.split('\n'):
        line_clean = line.strip()
        if re.search(certificate_line_pattern, line_clean, re.IGNORECASE):
            if len(line_clean.split()) >=2 and (line_clean.istitle() or line_clean.isupper()):
                extracted_certificates.add(line_clean.lower())
    
    # Clean up common education terms that might be mistaken as certificates
    certificates_to_filter = ["high school diploma", "diploma", "degree", "bachelor", "master", "doctorate", "associate"]
    extracted_certificates = {cert for cert in extracted_certificates if cert not in certificates_to_filter}
    
    return sorted(list(extracted_certificates))


# --- UPDATED extract_education Function ---
def extract_education(text):
    education_entries = []
    
    # Segment education text into potential blocks (by date ranges)
    # Re-use date pattern from experience for consistency
    date_range_pattern_edu = re.compile(
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\s*[-–—to\s]+(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}|present|current)\b' # Month YYYY - Month YYYY/present
        r'|\b\d{2}/\d{4}\s*[-–—to\s]+(?:\d{2}/\d{4}|present|current)\b' # MM/YYYY - MM/YYYY/present
        r'|\b\d{4}(?:\s*[-–—to\s]+(?:\d{4}|present|current))?\b' # YYYY - YYYY/present OR single YYYY
        , re.IGNORECASE
    )
    
    degree_keywords = [
        "bachelor", "master", "ph.d", "doctorate", "diploma", "associate",
        "b.s", "m.s", "mba", "b.a", "m.a", "engineer", "ingenieur", "dut",
        "licence", "certificat", "licentiate", "degree", "diplome", "baccalauréat",
        "high school diploma", "ged", "science", "arts", "technology", "physics", "computer", "math", "statistics", "data", "software", "informatique", "exercise"
    ]
    
    edu_org_positive_keywords = ["university", "college", "institute", "école", "ecole", "supérieure", "academy", "lycée", "lycee", "school", "foundation", "est", "ensias", "barlow", "concordia", "cpr foundation", "foundation"]
    edu_org_negative_keywords = ["fitness", "gym", "inc", "ltd", "corp", "llc", "group", "company"] # Add negative keywords to filter out job orgs

    # Find all date range matches to segment education blocks
    edu_date_matches = list(date_range_pattern_edu.finditer(text))
    
    edu_blocks_raw_text = []
    if edu_date_matches:
        for i, current_date_match in enumerate(edu_date_matches):
            block_start = current_date_match.start()
            block_end = len(text)
            if i + 1 < len(edu_date_matches):
                block_end = edu_date_matches[i+1].start()
            edu_blocks_raw_text.append(text[block_start:block_end].strip())
    else: # Fallback if no date ranges, assume paragraphs are entries
        edu_blocks_raw_text = [b.strip() for b in text.split('\n\n') if b.strip()]
        

    for block_text in edu_blocks_raw_text:
        current_entry = {
            "Institution": "N/A", "Degree": "N/A", "Major": "N/A", "Year": "N/A",
            "Start Date": "N/A", "End Date": "N/A"
        }
        doc_block = nlp(block_text)

        # 1. Dates (Use _extract_date_range_and_parse for consistency)
        dates_str, start_date_display, end_date_display, start_dt_obj, end_dt_obj = _extract_date_range_and_parse(block_text)
        current_entry['Year'] = dates_str # Raw date string
        current_entry['Start Date'] = start_date_display
        current_entry['End Date'] = end_date_display

        # 2. Institution (Use _extract_organization utility)
        institution_name = _extract_organization(doc_block, positive_keywords=edu_org_positive_keywords, negative_keywords=edu_org_negative_keywords)
        
        # Specific cleaning for the example CV's noisy institution names
        if institution_name != "N/A":
            institution_name = re.sub(r'•\s*Surpassing expectations in Personal Training coursework\s*\.?\s*', '', institution_name, flags=re.IGNORECASE).strip()
            institution_name = re.sub(r'Concordia University\s*Concordia\s*', 'Concordia University', institution_name, flags=re.IGNORECASE).strip()
            institution_name = re.sub(r'National CPR Foundation.*', 'National CPR Foundation', institution_name, flags=re.IGNORECASE).strip()
            institution_name = re.sub(r'Barlow High School.*', 'Barlow High School', institution_name, flags=re.IGNORECASE).strip()
            
        current_entry['Institution'] = institution_name
        
        # 3. Degree (Keyword matching)
        degree_name = "N/A"
        for keyword in degree_keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', block_text.lower()):
                degree_name = keyword.title()
                break
        current_entry['Degree'] = degree_name

        # 4. Major/Field of Study
        major_name = "N/A"
        major_match = re.search(r'(?:in|of|specialty|spécialité)\s+([a-zA-Z\s]+(?:science|engineering|art|design|technology|management|computing|information|systems|business|finance|physics|computer|math|statistics|data|software|informatique|exercise)\b)', block_text, re.IGNORECASE)
        if major_match:
            major_name = major_match.group(1).strip().title()
        current_entry['Major'] = major_name
        
        # Only add if it has at least an institution OR a degree OR a valid year
        if (current_entry['Institution'] != "N/A" or current_entry['Degree'] != "N/A" or current_entry['Year'] != "N/A"):
            education_entries.append(current_entry)

    # Filter out duplicate entries using a robust identifier
    final_edu_entries = []
    seen_identifiers = set()
    for entry in education_entries:
        identifier = (
            entry.get('Institution', '').lower(),
            entry.get('Degree', '').lower(),
            entry.get('Major', '').lower(),
            entry.get('Year', '').lower() # Include year in ID
        )
        if identifier not in seen_identifiers:
            final_edu_entries.append(entry)
            seen_identifiers.add(identifier)

    return final_edu_entries


# --- UPDATED extract_experience Function ---
def extract_experience(text):
    experience_entries = []
    
    # Strategy 1: Look for explicit "X years of experience" first (from entire section)
    total_years_experience = "N/A"
    years_pattern = r'(\d+\.?\d*)\+?\s*(?:years?|yrs?|y)\s+of\s+experience|\b(\d+\.?\d*)\s*(?:year|yr|y)\b'
    match = re.search(years_pattern, text, re.IGNORECASE)
    if match:
        total_years_experience = float(match.group(1) if match.group(1) else match.group(2))

    # Re-use date pattern from utility
    date_range_pattern_exp = re.compile(
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\s*[-–—to\s]+(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}|present|current)\b' # Month YYYY - Month YYYY/present
        r'|\b\d{2}/\d{4}\s*[-–—to\s]+(?:\d{2}/\d{4}|present|current)\b' # MM/YYYY - MM/YYYY/present
        r'|\b\d{4}\s*[-–—to\s]+(?:\d{4}|present|current)\b' # YYYY - YYYY/present
        , re.IGNORECASE
    )

    # Use re.finditer to get all date ranges and their positions - PRIMARY job segmenter
    date_range_matches = list(date_range_pattern_exp.finditer(text))

    job_blocks_raw_text = []
    if date_range_matches:
        for i, current_date_match in enumerate(date_range_matches):
            block_start = current_date_match.start()
            block_end = len(text) # Default to end of section

            if i + 1 < len(date_range_matches):
                block_end = date_range_matches[i+1].start()
            
            job_blocks_raw_text.append(text[block_start:block_end].strip())
    else: # Fallback if no date ranges detected, assume paragraphs are jobs (less reliable)
        job_blocks_raw_text = [b.strip() for b in text.split('\n\n') if b.strip()]


    job_keywords = ["manager", "engineer", "specialist", "developer", "analyst", "director", "architect", "lead", "consultant", "associate", "intern", "ceo", "cto", "cfo", "head", "senior", "junior", "instructor", "coach", "leader"] # Added from your example
    
    exp_org_positive_keywords = ["inc", "ltd", "corp", "llc", "group", "company", "gym", "studio", "fitness", "foundation", "institute", "solutions", "partners"] # Keywords indicating it's a company
    exp_org_negative_keywords = ["university", "college", "school", "academy", "lycée", "education", "science", "training"] # Filter out education orgs

    for job_text_block in job_blocks_raw_text:
        current_job = {
            "Dates": "N/A", "Start Date": "N/A", "End Date": "N/A", "Duration": "N/A",
            "Company": "N/A", "Title": "N/A"
        }
        
        doc_job_block = nlp(job_text_block)

        # 1. Extract Dates (Use _extract_date_range_and_parse utility)
        dates_str, start_date_display, end_date_display, start_dt_obj, end_dt_obj = _extract_date_range_and_parse(job_text_block)
        current_job['Dates'] = dates_str
        current_job['Start Date'] = start_date_display
        current_job['End Date'] = end_date_display

        if start_dt_obj != datetime.min and end_dt_obj != datetime.min:
            delta = relativedelta(end_dt_obj, start_dt_obj)
            duration_str = f"{delta.years} years, {delta.months} months"
            current_job['Duration'] = duration_str
        
        # 2. Extract Company (Use _extract_organization utility)
        company_name = _extract_organization(doc_job_block, positive_keywords=exp_org_positive_keywords, negative_keywords=exp_org_negative_keywords)
        # Additional cleaning for specific CV examples where location is appended to company
        if company_name != "N/A":
            company_name = re.sub(r'\s*(?:datteln|ponta porã|tennessee|miami)\b', '', company_name, flags=re.IGNORECASE).strip()
            # Clean up other specific terms that might sneak in
            company_name = re.sub(r'Certified Bootcamp Instructor\s*,?\s*', '', company_name, flags=re.IGNORECASE).strip()
            company_name = re.sub(r'Bootcamp Instructor\s*,?\s*', '', company_name, flags=re.IGNORECASE).strip()

            if len(company_name.split()) < 1: # If only single word left after aggressive cleaning, might be N/A
                company_name = "N/A"

        current_job['Company'] = company_name

        # 3. Extract Job Title
        title_name = "N/A"
        
        # Prioritize finding explicit job titles
        title_candidates = []
        for kw in job_keywords:
            match_iter = re.finditer(r'(?:[A-Z][a-z]+(?:\s[A-Z][a-z]+){0,4})?\s*\b' + re.escape(kw) + r'\b(?:(?:\s[A-Z][a-z]+){0,4})?', job_text_block, re.IGNORECASE)
            for m in match_iter:
                full_match = m.group(0).strip()
                if len(full_match.split()) > 1 or full_match.lower() == kw:
                    title_candidates.append(full_match)

        if title_candidates:
            title_candidates.sort(key=len, reverse=True)
            title_name = title_candidates[0].title() if title_candidates[0].isupper() else title_candidates[0]
            # Further clean if it includes company name by mistake
            if current_job['Company'] != "N/A" and current_job['Company'].lower() in title_name.lower():
                title_name = re.sub(re.escape(current_job['Company']) + r'\s*[,-]?\s*', '', title_name, flags=re.IGNORECASE).strip()
                title_name = title_name.title() # Re-title case after cleaning
        
        current_job['Title'] = title_name
        
        # Only add job if it has at least a date AND (company OR title)
        if current_job['Dates'] != "N/A" and (current_job['Company'] != "N/A" or current_job['Title'] != "N/A"):
             experience_entries.append(current_job)


    # --- Calculate Total Experience from extracted dates (Union of Intervals) ---
    calculated_total_months = 0
    parsed_job_dates_for_total = []

    for job in experience_entries:
        # Re-parse from the formatted display strings to get datetime objects for calculation
        start_dt_obj = parse_date_string(job.get('Start Date', ''))
        end_dt_obj = parse_date_string(job.get('End Date', ''))

        if start_dt_obj != datetime.min and end_dt_obj != datetime.min:
            parsed_job_dates_for_total.append({'start': start_dt_obj, 'end': end_dt_obj})

    parsed_job_dates_for_total.sort(key=lambda x: x['start'])

    if parsed_job_dates_for_total:
        current_union_start = parsed_job_dates_for_total[0]['start']
        current_union_end = parsed_job_dates_for_total[0]['end']

        for i in range(1, len(parsed_job_dates_for_total)):
            next_job_start = parsed_job_dates_for_total[i]['start']
            next_job_end = parsed_job_dates_for_total[i]['end']

            if next_job_start <= current_union_end:
                current_union_end = max(current_union_end, next_job_end)
            else:
                delta = relativedelta(current_union_end, current_union_start)
                calculated_total_months += delta.years * 12 + delta.months
                current_union_start = next_job_start
                current_union_end = next_job_end
        
        delta = relativedelta(current_union_end, current_union_start)
        calculated_total_months += delta.years * 12 + delta.months

    if calculated_total_months > 0:
        total_years_experience = round(calculated_total_months / 12, 1)

    return experience_entries, total_years_experience


# --- Main Parsing Function ---
def parse_resume(raw_text):
    # Step 1: Segment the raw text into logical sections
    sections = segment_resume_sections(raw_text)

    parsed_data = {
        "Name": "N/A",
        "Email": "N/A",
        "Phone": "N/A",
        "Skills": [],
        "Languages": [],
        "Certificates": [],
        "Education": [],
        "Experience": [],
        "Total Experience (Years)": "N/A",
        "Raw Text Snippet": raw_text[:500] + "..." if len(raw_text) > 500 else raw_text
    }

    # Step 2: Extract Name and Contact Info from the header section only
    # Pass original casing for name, lowercased for contact info patterns
    parsed_data["Name"] = extract_name(sections.get("header", "")) 
    parsed_data["Email"], parsed_data["Phone"] = extract_contact_info(clean_text(sections.get("header", "")).lower())
    
    # Step 3: Extract info from other sections, applying necessary cleaning/lowercasing
    
    # Skills can be found in multiple places (skills section, profile, accomplishments)
    skills_search_text = (clean_text(sections.get("skills", "")) + " " + 
                          clean_text(sections.get("profile", "")).lower() + " " + # Profile is usually title-cased
                          clean_text(sections.get("accomplishments", "")).lower()) # Accomplishments are descriptions
    parsed_data["Skills"] = extract_skills(skills_search_text.lower())
   # fully_cleaned_text = clean_text(raw_text).lower()
    # Languages and Certificates typically have dedicated sections
    parsed_data["Languages"] = extract_languages(clean_text(raw_text).lower())

   # parsed_data["Languages"] = extract_languages(clean_text(sections.get("languages", "")).lower())
    parsed_data["Certificates"] = extract_certificates(clean_text(sections.get("certificates", "")).lower())
    
    # For Experience and Education - use text that is only whitespace normalized (retains original casing)
    # as ORG/Title extraction benefits from it.
    employment_history_text = clean_text(sections.get("employment_history", ""))
    education_text = clean_text(sections.get("education", ""))

    parsed_data["Experience"], parsed_data["Total Experience (Years)"] = extract_experience(employment_history_text)
    parsed_data["Education"] = extract_education(education_text)

    return parsed_data

# --- Streamlit Application ---
def main():
    st.set_page_config(page_title="AI-Powered Resume Parser", layout="wide")
    st.title("📄 AI-Powered Resume Parser & Analyzer")
    st.markdown("Upload a PDF resume below to extract key information.")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        st.success("File uploaded successfully!")
        pdf_bytes = BytesIO(uploaded_file.read())
        
        st.subheader("Processing Resume...")
        
        col1, col2 = st.columns(2)

        with col1:
            with st.spinner("Extracting raw text..."):
                raw_text = extract_text_from_pdf(pdf_bytes)
            
            if raw_text:
                with st.expander("Show Raw Extracted Text (with original casing)"):
                    st.text(raw_text)
            else:
                st.warning("Could not extract text from the PDF. Please try another file.")
                st.stop()

        with col2:
            st.subheader("Extracted Information:")
            with st.spinner("Extracting structured data..."):
                parsed_info = parse_resume(raw_text)
                st.json(parsed_info)

        st.markdown("---")
        st.subheader("Text used for general extraction (lowercased & cleaned):")
        # Display the combined lowercased text for visual debugging
        # Note: Actual functions work on segmented text
        display_cleaned_text = clean_text(parsed_info["Raw Text Snippet"]).lower()
        st.text_area("Combined Cleaned Resume Content (for general keyword search - first 1000 chars)", display_cleaned_text[:1000], height=200)
        if len(display_cleaned_text) > 1000:
            st.write("*(...text truncated for display. Full text used for parsing.)*")

    else:
        st.info("Please upload a PDF file to analyze a resume.")

if __name__ == "__main__":
    main()