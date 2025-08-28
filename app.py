import streamlit as st
from io import BytesIO
import pdfplumber
import re
import spacy
from datetime import datetime
from dateutil.relativedelta import relativedelta
from spacy.matcher import PhraseMatcher
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_LEFT

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("SpaCy 'en_core_web_sm' model not found. Please run the following commands in your terminal:")
    st.code("pip install spacy")
    st.code("python -m spacy download en_core_web_sm")
    st.stop()



FALSE_ORGS = [
    "city",
      #"state",
        "inc", "ltd", "corp", "llc", "group", "team", "client", "platform",
    "studio", "health", "safety", "management", "project", "program", "class", "classes",
    "exercise", "fitness", "training", "nutrition", "system", "systems", "solutions",
    "leader", "manager", "director", "specialist", "analyst", "developer", "architect",
  #  "coach",
    "instructor",
    "bachelor", "master", "diploma", "degree", "ph.d",
    "past", "current", "present", "online", "events", "us", "uk", "usa", "canada",
    "los angeles", "tennessee", "miami", "datteln", "ponta porã"
    #, "surpassing"
    , "crossfit", "level", "varsity", "track", "athlete", "command",
    "pinterest", "linkedin", "template","administration arlington"
]

JOBS = ["manager", "engineer", "specialist", "developer", "analyst", "director", "architect", "lead", "consultant", "associate", "intern", "ceo", "cto", "cfo", "head", "senior", "junior", "instructor", "coach", "leader", "owner", "founder", "trainee", "coordinator", "assistant", "executive", "officer", "president", "vp", "vice president", "clerk", "representative", "agent", "technician", "scientist", "researcher"]

EXP_ORGS_POS = ["inc", "ltd", "corp", "llc", "group", "company", "gym", "studio", "fitness", "foundation", "institute", "solutions", "partners", "agency", "firm", "ventures", "technologies", "hospital", "clinic", "co.", "corporation", "industries", "systems"]

EXP_ORGS_NEG = ["university", "college", "school", "academy", "lycée", "education", "science", "training", "management", "project", "program", "system", "leader", "instructor", "bootcamp", "student"]

DEGREES = [
    "bachelor","accreditation", "master", "ph.d", "doctorate", "diploma", "associate",
    "b.s", "m.s", "mba", "b.a", "m.a", "engineer", "ingenieur", "dut",
    "licence", "certificat", "licentiate", "degree", "diplome", "baccalauréat",
    "high school diploma", "ged", "science", "arts", "technology", "physics", "computer", "math", "statistics", "data", "software", "informatique", "exercise", "design", "mechanical", "engineering",
    "kinesiology", "health science", "nutrition", "education","b.s","bachelor"
]

EDU_ORGS_POS = ["university", "college", "institute", "école", "ecole", "supérieure", "academy", "lycée", "lycee", "school", "foundation", "est", "ensias", "barlow", "concordia", "cpr foundation", "foundation","harvard university","wisconsin state university","high school","starfish holistic"]

EDU_ORGS_NEG = ["fitness", "gym", "inc", "ltd", "corp", "llc", "group", "company", "solutions", "partners", "technologies", "innovations"]

LANGS = [
    "English", "French", "Spanish", "German", "Arabic", "Chinese", "Japanese",
    "Portuguese", "Russian", "Italian", "Korean", "Dutch", "Swedish"
]

SKILLS_LIST = [
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
    "cardio training", "fitness routines", "hiit", "client assessments", "health & safety",
    "personal training", "crossfit", "coaching", "cycling", "nutrition", "staffing", "marketing",
    "group fitness", "program budgeting", "program statistics", "fitness assessments",
    "active listening", "strength and conditioning", "group cycling", "skiing", "hockey", "knitting",
    "customer interface", "project coordination", "product engineering", "qa experience",
    "sales support", "technical solutions", "fixture submission", "fixture design",
    "bom preparation", "material procurement", "production orders", "ul inspector",
    "product development", "etl intertek", "fixture sample kits", "architect collaboration",
    "lighting designer", "cost effective solutions", "environmentally friendly solutions",
    "product concepts", "submittal drawings", "sales & marketing support", "project timelines",
    "lighting fixtures", "quality control", "cost control", "product redesign", "lean manufacturing",
    "fabrication drawings", "vending", "material selection", "bill of material",
    "installation instructions", "specification sheets", "manufacturing staff support",
    "assembly process", "cross functional project team", "new product line launch",
    "asme", "ada",
    "customer corrections", "3d model", "technical specifications", "graphics design",
    "prototype development", "functional evaluation", "aesthetic evaluation"
]

CERTS_LIST = [
    "pmp", "cism", "cissp", "ccna", "aws certified solutions architect",
    "microsoft certified", "google cloud professional", "scrum master",
    "itil", "comptia a+", "comptia network+", "comptia security+",
    "certified ethical hacker", "prince2", "six sigma", "cfa", "cpa",
   "program",
     "coach's prep certified",

    "cpr certified"
]

SECTION_MARKERS = [
    ("PROFILE", r'^\s*PROFILE\s*$', "profile"),
    ("EMPLOYMENT HISTORY", r'^\s*(?:EMPLOYMENT\s+HISTORY|PROFESSIONAL\s+EXPERIENCE|WORK\s+EXPERIENCE|EXPERIENCE)\s*$', "exp"),
    ("EDUCATION", r'^\s*EDUCATION\s*$', "edu"),
    ("SKILLS", r'^\s*SKILLS\s*$', "skills"),
    ("LANGUAGES", r'^\s*LANGUAGES?\s*$', "langs"),
    ("CERTIFICATES", r'^\s*CERTIFICATES?\s*$', "certs"),
    ("COURSES", r'^\s*COURSES?\s*$', "certs"),


    ("ACHIEVEMENTS", r'^\s*ACHIEVEMENTS?\s*$', "achiv"),
    ("HOBBIES", r'^\s*HOBBIES\s*$', "hobbies"),
    ("ACCOMPLISHMENTS", r'^\s*ACCOMPLISHMENTS\s*$', "achieves"),
    ("LINKS", r'^\s*LINKS\s*$', "links_sec"),
    ("WORK HISTORY", r'^\s*WORK HISTORY\s*$', "exp"),
    ("PROJECTS", r'^\s*PROJECTS\s*$', "projects"),
    ("AWARDS", r'^\s*AWARDS\s*$', "awards"),
]

COMPILED_SECTION_MARKERS = []
for d_name, pat, s_key in SECTION_MARKERS:
    COMPILED_SECTION_MARKERS.append((s_key, re.compile(pat, re.MULTILINE | re.IGNORECASE)))

PHRASE_MATCHER_SKILLS = PhraseMatcher(nlp.vocab)
SKILLS_PATTERNS = [nlp.make_doc(skill) for skill in SKILLS_LIST]
PHRASE_MATCHER_SKILLS.add("SKILL_LIST", SKILLS_PATTERNS)

PHRASE_MATCHER_CERTS = PhraseMatcher(nlp.vocab)
CERTS_PATTERNS = [nlp.make_doc(cert) for cert in CERTS_LIST]
PHRASE_MATCHER_CERTS.add("CERT_LIST", CERTS_PATTERNS)


def extract_text(file):
    text=""
    try :
        with pdfplumber.open(file) as pdf:
         for page in pdf.pages:
            text+=page.extract_text() +"\n"
    except Exception as e:
        st.error(f"Error : extracting text from pdf {e}")
        return None
    return text

def clean(text):
    if not text:
        return ""
    clean_text = re.sub(r'\s+', ' ', text)
    clean_text = clean_text.strip()
    return clean_text

def parse_date(date_str):
    date_str = date_str.lower().strip()
    #print(date_str)
    if "present" in date_str or "presen" in date_str or "current" in date_str:
        return datetime.now()
    fmts = [
        "%b %Y", "%B %Y", "%m/%Y", "%Y", "%Y-%m",
    ]
    for fmt in fmts:
        try:
            if fmt == "%Y":
                return datetime.strptime(date_str, fmt).replace(month=1, day=1)
            if fmt == "%Y-%m":
                return datetime.strptime(date_str, fmt).replace(day=1)
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return datetime.min

def format_date(dt_obj):
    if not isinstance(dt_obj, datetime) or dt_obj == datetime.min:
        return "N/A"
    threshold = datetime.now() - relativedelta(months=3)
    if dt_obj >= threshold:
         return "Present"
    return dt_obj.strftime("%b %Y")

def segment_sections(txt):
    sects = {
        "head": "", "profile": "", "exp": "", "edu": "",
        "skills": "", "langs": "", "certs": "", "hobbies": "",
        "achieves": "", "links_sec": "", "projects": "", "awards": ""
    }

    bounds = []
    for s_key, pat_comp in COMPILED_SECTION_MARKERS:
        for m in pat_comp.finditer(txt):
            bounds.append((m.start(), s_key, m.end()))
    bounds.sort()

    if not bounds:
        sects["head"] = txt.strip()
        return sects

    if bounds[0][0] > 0:
        sects["head"] = txt[0 : bounds[0][0]].strip()

    for i, (start_h_idx, s_key, end_h_idx) in enumerate(bounds):
        content_start = end_h_idx
        content_end = len(txt)

        if i + 1 < len(bounds):
            content_end = bounds[i+1][0]

        sects[s_key] = txt[content_start : content_end].strip()

    return sects

def get_org(doc_block, positives=[], negatives=[]):
    best_org = None

    for ent in doc_block.ents:
        if ent.label_ == "ORG":
            #print(ent)
            org_txt = ent.text.strip()
            org_txt_lower = org_txt.lower()
            #print(f"---------------- inst {org_txt}" )
            if len(org_txt.split()) < 1 or len(org_txt) < 2:
                continue
            #print(f" {best_org} -------------------------- {org_txt}")

            if positives and not any(kw.lower() in org_txt_lower for kw in positives):
                continue
            #print(f" {best_org} -------------------------- {org_txt}")
            is_neg = False
            for word in FALSE_ORGS:
                if re.search(r'\b' + re.escape(word.lower()) + r'\b', org_txt_lower):
                    #print(f"{word}---------------uuu------{org_txt_lower} ")
                    is_neg = True
                    break
            if is_neg:
                continue
            #print(f" {best_org} -------------------------- {org_txt}")
            for word in negatives:
                 if re.search(r'\b' + re.escape(word.lower()) + r'\b', org_txt_lower):
                    is_neg = True
                    break
            if is_neg:
                continue
            #print(f" {best_org} befor the last con-------------------------- {org_txt}")
            if best_org is None or len(org_txt) > len(best_org):
                best_org = org_txt

    if best_org:
        best_org = re.sub(r'•\s*Surpassing \s*\.?\s*', '', best_org, flags=re.IGNORECASE).strip()
        best_org = re.sub(r'[,.\s]+$', '', best_org).strip()
        best_org = re.sub(r'^\s*[,.\s]+', '', best_org).strip()

        if len(best_org.split()) < 1 or len(best_org) < 2 or best_org.lower() in [o.lower() for o in FALSE_ORGS]:
            return "N/A"

        return best_org.title() if best_org.isupper() else best_org
    return "N/A"

def get_dates(txt_block):
    date_pat = re.compile(
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\s*[-–—to\s]+(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}|present|current)\b'
        r'|\b\d{2}/\d{4}\s*[-–—to\s]+(?:\d{2}/\d{4}|present|current)\b'
        r'|\b\d{4}\s*[-–—to\s]+(?:\d{4}|present|current)\b'
        , re.IGNORECASE
    )

    dates_found = date_pat.search(txt_block)

    dates_str = "N/A"
    start_display = "N/A"
    end_display = "N/A"
    start_dt = datetime.min
    end_dt = datetime.min

    if dates_found:
        dates_str = dates_found.group(0)
        parts = re.split(r'\s*[-–—to]\s*', dates_str, flags=re.IGNORECASE)
        start_part = parts[0]
        end_part = parts[1] if len(parts) > 1 else start_part

        start_dt = parse_date(start_part)
        end_dt = parse_date(end_part)

        start_display = format_date(start_dt)
        end_display = format_date(end_dt)

    return dates_str, start_display, end_display, start_dt, end_dt



def get_name(head_txt):
    #print(head_txt)
    if not head_txt:
        return "N/A"

    lines = head_txt.split('\n')
    name_cand = "N/A"
    #print(" /////////////// the name /////////")

    for line in lines[:2]:
        line=re.sub(r',.*$','',line).strip()
        words = line.split()
        if 2 <= len(words) <= 4 and (line.isupper() or line.istitle()):
            if not re.search(r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}|\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b|\b(?:profile|experience|education|skills|contact)\b', line, re.IGNORECASE):
                 name_cand = line
                 break
    if name_cand == "N/A":
        doc = nlp(clean(head_txt))
        potentials = []
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                if (ent.text.istitle() or ent.text.isupper()) and len(ent.text.split()) >= 2:
                     if not re.search(r'\d|[#@/\\]', ent.text):
                        potentials.append((ent.start_char, ent.text))

        if potentials:
            potentials.sort(key=lambda x: x[0])
            name_cand = potentials[0][1]

    if name_cand != "N/A":
        removes = JOBS + ["experience", "profile", "contact", "education", "skills"]
        removes.sort(key=len, reverse=True)

        temp_name = name_cand
        for kw in removes:
            pat = r'[,.\s-]*\b' + re.escape(kw) + r'\b\s*$'
            if re.search(pat, temp_name, re.IGNORECASE):
                clean_name = re.sub(pat, '', temp_name, flags=re.IGNORECASE).strip()
                if len(clean_name.split()) >= 2:
                    temp_name = clean_name
                else:
                    temp_name = "N/A"
                break

        if temp_name != "N/A" and len(temp_name.split()) >= 2:
            name_cand = temp_name
        else:
            name_cand = "N/A"

    if name_cand != "N/A":
        name_cand = name_cand.title() if name_cand.isupper() else name_cand

    return name_cand

def get_contact(head_txt):
    email = "N/A"
    phone = "N/A"

    head_txt_lower = head_txt

    email_pat = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pat, head_txt_lower)
    if emails:
        email = emails[0]

    phone_pat = r'\b(?:\+?\d{1,3}[-.\s]?)?(\d{10})\b|\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
    phones = re.findall(phone_pat, head_txt_lower)

    clean_phones = []
    for item in phones:
        if isinstance(item, tuple):
            clean_phones.append(''.join(filter(None, item)))
        else:
            clean_phones.append(item)

    if clean_phones:
        match_phone = clean_phones[0]
        phone = re.sub(r'[\s\.\-\(\)]+', '', match_phone)

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


def get_skills(txt,skills_only):
    #print(" --------------------- Skills ")
    #print(txt)
    skills = set()
    doc = nlp(txt)


    matches = PHRASE_MATCHER_SKILLS(doc)

    for m_id, start, end in matches:
        span = doc[start:end]
        skills.add(span.text.lower())
    #print(" //////////////////////the found skills by matching ")
    #print(skills)

    phrase_pat = re.compile(
        r'\b(?:[a-z]+(?:(?:\s|-)[a-z]+){0,3})\b',
        re.IGNORECASE
    )
    #print(" //////////////////////the found skills by regex ")

    for m in phrase_pat.finditer(skills_only):
        phrase = m.group(0)
        doc_phrase = nlp(phrase)
        if 2 <= len(doc_phrase) <= 5 and any(t.pos_ in ["NOUN", "VERB", "ADJ"] for t in doc_phrase):
             if phrase.lower() not in SKILLS_LIST and phrase.lower() not in [o.lower() for o in FALSE_ORGS]:
                skills.add(phrase.lower())
    #print(skills)
    return sorted(list(skills))


def get_langs(txt):

    found_langs = []

    langs_check = [lang.lower() for lang in LANGS]

    prof_map = {
        "native speaker": "Native", "fluent": "Fluent", "proficient": "Proficient",
        "basic": "Basic", "conversational": "Conversational", "very good command": "Very Good",
        "good command": "Good", "intermediate": "Intermediate", "advanced": "Advanced", "beginner": "Beginner", "good working": "Good"
    }

    lang_pats = [re.escape(lang) for lang in langs_check]
    prof_pats = [re.escape(prof) for prof in prof_map.keys()]

    combined_pat = re.compile(
        r'\b(' + '|'.join(lang_pats) + r')\b'
        r'(?:[\s,.-]*\b(' + '|'.join(prof_pats) + r')\b)?'
        r'|\b(' + '|'.join(prof_pats) + r')\b'
        , re.IGNORECASE
    )

    results_dict = {}
    matches = combined_pat.finditer(txt.lower())
    for m in matches:
        lang_match = m.group(1)
        prof_match2 = m.group(2)
        prof_match3 = m.group(3)


        found_lang = lang_match.lower() if lang_match else None
        found_prof_norm = None
        if prof_match2:
            found_prof_norm = prof_map.get(prof_match2.lower())
        elif prof_match3:
            found_prof_norm = prof_map.get(prof_match3.lower())

        if found_lang:
            if found_lang not in results_dict:
                results_dict[found_lang] = {'lang': found_lang.title(), 'prof': 'N/A'}

            if found_prof_norm and results_dict[found_lang]['prof'] == 'N/A':
                results_dict[found_lang]['prof'] = found_prof_norm

        elif found_prof_norm:
            current_list = list(results_dict.values())
            if current_list and current_list[-1]['prof'] == 'N/A':
                current_list[-1]['prof'] = found_prof_norm
                results_dict[current_list[-1]['lang'].lower()] = current_list[-1]

    found_langs = list(results_dict.values())
    return found_langs


def get_certs(txt):
    certs = set()
    doc = nlp(txt)

    matches = PHRASE_MATCHER_CERTS(doc)
    for m_id, start, end in matches:
        span = doc[start:end]

        certs.add(span.text.lower())

    line_pat = re.compile(
      r'\b(?:certificate|program|course|diploma|level)\b',
        re.IGNORECASE
    )

    for line_num, line in enumerate(txt.split('\n')):
        clean_line = line.strip()

        if clean_line and line_pat.search(clean_line):
            words_in_line = clean_line.split()

    filter_certs = ["high school diploma", "diploma", "degree", "bachelor", "master", "doctorate", "associate", "ph.d","diploma"]
    result=set()
    certs = {c for c in certs if c not in filter_certs}
    return sorted(list(certs))

def get_edu(txt):
    edu_entries = []
    
    date_matches = []
    date_pattern = r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\s*[—–-]\s*(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}|present|current)\b'
    for match in re.finditer(date_pattern, txt, re.IGNORECASE):
        date_matches.append(match)

    if not date_matches:
        return _fallback_parser(txt)

    search_zones = []
    date_start_positions = [match.start() for match in date_matches]
    
    start_of_zone = 0
    for i in range(len(date_start_positions)):
        end_of_zone = len(txt)
        if i + 1 < len(date_start_positions):
            end_of_zone = date_start_positions[i+1]
        
        zone_text = txt[start_of_zone:end_of_zone].strip()
        if zone_text:
            search_zones.append(zone_text)
        
        start_of_zone = end_of_zone

    for zone_txt in search_zones:
        if not zone_txt.strip(): continue
        edu_entries.append(_extract_details_from_block(zone_txt))

    return _deduplicate_entries(edu_entries)

def _extract_details_from_block(block_txt):
    entry = {"Inst": "N/A", "Degree": "N/A", "Major": "N/A", "Year": "N/A", "Start": "N/A", "End": "N/A", "Raw": block_txt}
    
    dates_str, start_disp, end_disp, _, _ = get_dates(block_txt)
    entry.update({'Year': dates_str, 'Start': start_disp, 'End': end_disp})

    inst_name = _find_institution(block_txt)
    entry['Inst'] = inst_name

    degree_name, major_name = _find_degree_and_major(block_txt, inst_name, dates_str)
    entry['Degree'] = degree_name
    entry['Major'] = major_name
    
    return entry

def _find_institution(text):
    for kw in EDU_ORGS_POS:
        match = re.search(r'((?:[A-Z][a-zÀ-ÿ-]+\s*){0,4}\b' + re.escape(kw) + r'\b(?:\s*[A-Z][a-zÀ-ÿ-]+){0,4})', text, re.IGNORECASE)
        if match:
            return match.group(1).strip(' ,')
    return "N/A"

def _find_degree_and_major(text, inst_name, dates_str):
    degree_name, major_name = "N/A", "N/A"
    
    found_degrees = []
    for kw in DEGREES:
        pattern = r'\b' + re.escape(kw) + r'(\.|\b|’s)'
        for match in re.finditer(pattern, text, re.IGNORECASE):
            found_degrees.append({'text': match.group(0), 'pos': match.start()})
    
    if found_degrees:
        found_degrees.sort(key=lambda x: x['pos'])
        first_degree_match = found_degrees[0]
        degree_name = first_degree_match['text'].title().strip(' .’s')
        
        major_text = text
        if dates_str != "N/A": major_text = major_text.replace(dates_str, "")
        if inst_name != "N/A": major_text = major_text.replace(inst_name, "")
        major_text = major_text.replace(first_degree_match['text'], "")

        major_name = re.sub(r'^(?:in|of)\s+', '', major_text, flags=re.IGNORECASE)
        noise_words = ['Arlington', 'Bossier City', 'Orlando', 'Madison', 'Miami', 'Tennessee', 'Concordia']
        for word in noise_words:
            major_name = re.sub(r'\b' + re.escape(word) + r'.*$', '', major_name, flags=re.IGNORECASE)
        
        major_name = major_name.split('•')[0].strip(' ,').replace('\n', ' ').strip()

    return degree_name, major_name

def _deduplicate_entries(entries):
    final_edu = []
    seen_ids = set()
    for entry in entries:
        identifier = (entry.get('Inst', 'N/A').lower(), entry.get('Degree', 'N/A').lower(), entry.get('Year', 'N/A').lower())
        if identifier not in seen_ids:
            final_edu.append(entry)
            seen_ids.add(identifier)
    return final_edu

def _fallback_parser(txt):
    blocks = []
    current_block = []
    for line in txt.split('\n'):
        clean_line = line.strip()
        is_new_header = (len(clean_line.split()) < 8 and (clean_line and clean_line[0].isupper()))
        
        if is_new_header and current_block:
            blocks.append('\n'.join(current_block))
            current_block = [clean_line] if clean_line else []
        elif clean_line:
            current_block.append(clean_line)
    if current_block:
        blocks.append('\n'.join(current_block))
    
    entries = []
    for block in blocks:
        entries.append(_extract_details_from_block(block))
    return _deduplicate_entries(entries)

def get_exp(txt):
    
    exp_entries = []
    
    total_exp_years = "N/A"
    years_pat = re.compile(r'(\d+\.?\d*)\+?\s*(?:years?|yrs?|y)\s+of\s+experience|\b(\d+\.?\d*)\s*(?:year|yr|y)\b', re.IGNORECASE)
    m = re.search(years_pat, txt)
    if m:
        total_exp_years = float(m.group(1) if m.group(1) else m.group(2))

    date_pat_exp = re.compile(
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\s*[-–—to\s]+(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}|present|current)\b'
        r'|\b\d{2}/\d{4}\s*[-–—to\s]+(?:\d{2}/\d{4}|present|current)\b'
        r'|\b\d{4}\s*[-–—to\s]+(?:\d{4}|present|current)\b'
        , re.IGNORECASE
    )

    date_matches = list(date_pat_exp.finditer(txt))

    blocks_raw = []
    if date_matches:
        for i, m in enumerate(date_matches):
            block_start = m.start()
            block_end = len(txt)

            if i + 1 < len(date_matches):
                block_end = date_matches[i+1].start()
            
            blocks_raw.append(txt[block_start:block_end].strip())
    else:
        blocks_raw = [b.strip() for b in txt.split('\n\n') if b.strip()]
        temp_blocks = []
        curr_block = []
        
        for line in blocks_raw:
            if re.search(r'^\s*(?:[A-Z][a-z\s]*){1,4}(?:[A-Z][a-z]+)*\s*$', line) and len(line.split()) < 5:
                if curr_block:
                    temp_blocks.append('\n'.join(curr_block))
                curr_block = [line]
            else:
                curr_block.append(line)
        if curr_block:
            temp_blocks.append('\n'.join(curr_block))
        blocks_raw = temp_blocks

    for block_txt in blocks_raw:
        curr_job = {
            "Dates": "N/A", "Start": "N/A", "End": "N/A", "Duration": "N/A",
            "Comp": "N/A", "Title": "N/A", "Raw": block_txt
        }
        
        doc_b = nlp(block_txt)

        dates_str, start_disp, end_disp, start_dt, end_dt = get_dates(block_txt)
        curr_job['Dates'] = dates_str
        curr_job['Start'] = start_disp
        curr_job['End'] = end_disp

        if start_dt != datetime.min and end_dt != datetime.min:
            delta = relativedelta(end_dt, start_dt)
            duration_str = f"{delta.years} years, {delta.months} months"
            curr_job['Duration'] = duration_str
        
        title_name = "N/A"
        title_cands = []
        for kw in JOBS:
            title_pat = re.compile(
                r'(?:[A-Z][a-z]+\s*){0,3}\b' + re.escape(kw) + r'\b(?:(?:\s|\-)[A-Z][a-z]+){0,3}',
                re.IGNORECASE
            )
            for m in title_pat.finditer(block_txt):
                full_m = m.group(0).strip()
                if kw.lower() in full_m.lower() and len(full_m.split()) <= 7:
                    if not re.search(r'\d{4}', full_m) and len(full_m.replace(kw, '').strip()) > 0:
                        title_cands.append(full_m)

        if title_cands:
            title_cands.sort(key=lambda x: (x[1], -len(x[0])))
            title_name = title_cands[0]
            title_name = title_name.title() if title_name.isupper() else title_name

        curr_job['Title'] = title_name
        comp_name = "N/A"
        
      
        spacy_org_cands = []
        for ent in doc_b.ents:
            if ent.label_ == "ORG":
                org_txt_cand = ent.text.strip()
                org_txt_cand_lower = org_txt_cand.lower()
                is_neg = False
                
                for word in FALSE_ORGS + EXP_ORGS_NEG:
                    if re.search(r'\b' + re.escape(word.lower()) + r'\b', org_txt_cand_lower):
                        is_neg = True
                        break
                
                if not is_neg and any(re.search(r'\b' + re.escape(job_kw.lower()) + r'\b', org_txt_cand_lower) for job_kw in JOBS):
                    is_neg = True

                if not is_neg and len(org_txt_cand.split()) >=1 and len(org_txt_cand) >= 2: 
                    
                    spacy_org_cands.append((org_txt_cand, block_txt.find(org_txt_cand))) 
        
        if spacy_org_cands:
            spacy_org_cands.sort(key=lambda x: (x[1], -len(x[0])))
            comp_name = spacy_org_cands[0][0]

       
        if comp_name == "N/A":
            search_start = 0
            if curr_job['Title'] != "N/A":
                title_pos = block_txt.find(curr_job['Title'])
                if title_pos != -1:
                    search_start = title_pos + len(curr_job['Title'])
            elif curr_job['Dates'] != "N/A":
                dates_pos = block_txt.find(curr_job['Dates'])
                if dates_pos != -1:
                    search_start = dates_pos + len(curr_job['Dates'])

            rem_txt = block_txt[search_start:].strip()
            comp_pat = re.search(
                r'\b'
                r'([A-Z][a-z]+'
                r'(?:\s+[A-Z][a-z]+)*)'
                r'\b',
                rem_txt, re.IGNORECASE
            )
            
            if comp_pat:
                pot_comp_raw = comp_pat.group(1).strip()
                
                pot_comp_raw = re.sub(r'[,.\s]+$', '', pot_comp_raw).strip()
                pot_comp_raw = re.sub(r'^\s*(?:at|in|of)\s*', '', pot_comp_raw, flags=re.IGNORECASE).strip()
                is_plaus = True
                if len(pot_comp_raw.split()) < 1 or len(pot_comp_raw) < 2 or len(pot_comp_raw.split()) > 7:
                    is_plaus = False
                elif re.search(r'\d{4}', pot_comp_raw):
                    is_plaus = False
                elif any(re.search(r'\b'+re.escape(word.lower())+r'\b', pot_comp_raw.lower()) for word in ( JOBS + ["city", "state", "inc", "ltd", "corp", "llc"])):
                    is_plaus = False
                if is_plaus:
                    doc_pot = nlp(pot_comp_raw)
                    org_ents = [ent.text for ent in doc_pot.ents if ent.label_ == "ORG"]
                    if pot_comp_raw.istitle() or pot_comp_raw.isupper():
                        comp_name = pot_comp_raw
                    elif org_ents:
                        comp_name = org_ents[0]
                    
        

        if comp_name == "N/A" or len(comp_name.split()) < 1:
            comp_from_org = get_org(
                doc_b,
                positives=EXP_ORGS_POS,
                negatives=EXP_ORGS_NEG
            )
            if comp_from_org != "N/A":
                comp_name = comp_from_org
        if comp_name != "N/A":
            comp_name = re.sub(r'\s*(?:datteln|ponta porã|tennessee|miami|city|state|usa|uk|inc|ltd|corp|llc|ran)\b', '', comp_name, flags=re.IGNORECASE).strip()
            comp_name = re.sub(r'[,.]+$', '', comp_name).strip()
            comp_name = re.sub(r'^\s*[,.]+', '', comp_name).strip()

            if len(comp_name.split()) < 1 or len(comp_name) < 2 or comp_name.lower() in [o.lower() for o in FALSE_ORGS]:
                comp_name = "N/A"
            else:
                comp_name = comp_name.title() if comp_name.isupper() else comp_name

        curr_job['Comp'] = comp_name

        if curr_job['Dates'] != "N/A" and (curr_job['Comp'] != "N/A" or curr_job['Title'] != "N/A"):
             exp_entries.append(curr_job)

    total_months = 0
    job_dates = []

    for job in exp_entries:
        start_dt = parse_date(job.get('Start', ''))
        end_dt = parse_date(job.get('End', ''))

        if start_dt != datetime.min and end_dt != datetime.min:
            job_dates.append({'start': start_dt, 'end': end_dt})

    job_dates.sort(key=lambda x: x['start'])

    if job_dates:
        curr_union_start = job_dates[0]['start']
        curr_union_end = job_dates[0]['end']

        for i in range(1, len(job_dates)):
            next_job_start = job_dates[i]['start']
            next_job_end = job_dates[i]['end']

            if next_job_start <= curr_union_end:
                curr_union_end = max(curr_union_end, next_job_end)
            else:
                delta = relativedelta(curr_union_end, curr_union_start)
                total_months += delta.years * 12 + delta.months
                curr_union_start = next_job_start
                curr_union_end = next_job_end
        
        delta = relativedelta(curr_union_end, curr_union_start)
        total_months += delta.years * 12 + delta.months

    if total_months > 0:
        total_exp_years = round(total_months / 12, 1)

    return exp_entries, total_exp_years
def get_links(txt):
    links = {
        "linkedin": [],
        "github": [],
        "other": []
    }
    
    linkedin_pat = r'linkedin\.com/in/[\w\d\-_]+'
    github_pat = r'github\.com/[\w\d\-_]+'
    general_url_pat = r'https?://(?:www\.)?[a-zA-Z0-9./\-_]+\.[a-zA-Z]{2,}(?:/[a-zA-Z0-9./\-_]*)*'
    
    linkedin_matches = re.findall(linkedin_pat, txt, re.IGNORECASE)
    for m in linkedin_matches:
        full_url = f"https://{m}" if not m.startswith("http") else m
        if full_url not in links["linkedin"]:
            links["linkedin"].append(full_url)

    github_matches = re.findall(github_pat, txt, re.IGNORECASE)
    for m in github_matches:
        full_url = f"https://{m}" if not m.startswith("http") else m
        if full_url not in links["github"]:
            links["github"].append(full_url)
    
    other_matches = re.findall(general_url_pat, txt, re.IGNORECASE)
    for m in other_matches:
        if "linkedin.com/in" not in m.lower() and \
           "github.com" not in m.lower() and \
           m not in links["other"]:
            links["other"].append(m)

    return links

def extract_info(txt):
    sects = segment_sections(txt)

    info = {
        "Name": "N/A", "Email": "N/A", "Phone": "N/A", "Skills": [], "Languages": [],
        "Certs": [], "Edu": [], "Exp": [],
        "TotalExp": "N/A",
        "Links": {"linkedin": [], "github": [], "other": []},
        "RawTextSnippet": txt[:2000] + "..." if len(txt) > 2000 else txt,
        "RawEdu": sects.get("edu", ""),
        "RawExp": sects.get("exp", "")
    }

    info["Name"] = get_name(sects.get("head", ""))
    info["Email"], info["Phone"] = get_contact(clean(sects.get("head", "")).lower())
    
    skills_txt = (clean(sects.get("skills", "")) + " " +
                  clean(sects.get("profile", "")) + " " +
                  clean(sects.get("achieves", "")) + " " +
                  clean(sects.get("exp", ""))).lower()
    info["Skills"] = get_skills(skills_txt,(sects.get("skills", "")))
    #print(" -------------------------------only skilss")
    #print(sects.get("skills", ""))
    #print(" -------------------------------only skilss")

    info["Languages"] = get_langs(sects.get("langs",txt))
    info["Certs"] = get_certs(sects.get("certs", ""))

    info["Exp"], info["TotalExp"] = get_exp(clean(sects.get("exp", "")))
    info["Edu"] = get_edu(clean(sects.get("edu", "")))
    
    links_txt = sects.get("head", "") + " " + sects.get("links_sec", "") + " " + txt
    info["Links"] = get_links(links_txt)

    return info
def dataframe_to_pdf(df):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(letter), rightMargin=20, leftMargin=20, topMargin=20, bottomMargin=20)
    elements = []

    data = [df.columns.tolist()] + df.values.tolist()

    col_widths = []
    for col in df.columns:
        max_len = max(df[col].astype(str).apply(len).max(), len(col))
        col_widths.append(max_len * 7)

    table = Table(data, colWidths=col_widths)

    style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4a90e2")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f5f5f5")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
    ])
    table.setStyle(style)

    elements.append(table)
    doc.build(elements)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf
def export_pdf_paragraphs(data_list):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    elements = []

    styles = getSampleStyleSheet()
    normal_style = styles['Normal']
    normal_style.fontSize = 10
    normal_style.leading = 12
    normal_style.alignment = TA_LEFT

    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading2'], fontSize=12, leading=14, spaceAfter=6)

    for idx, item in enumerate(data_list, 1):
        elements.append(Paragraph(f"Candidate {idx}: {item.get('Name','-')}", title_style))

        content_lines = []
        for key, value in item.items():
            if isinstance(value, list):
                value = ", ".join(value)
            content_lines.append(f"<b>{key}:</b> {value}")

        paragraph_text = "<br/>".join(content_lines)
        elements.append(Paragraph(paragraph_text, normal_style))
        elements.append(Spacer(1, 12))

    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

def match_cand(cand_data, reqs):
    score = 0
    max_score = 0

    req_skills = reqs["skills"]
    cand_skills = [s.lower() for s in cand_data.get("Skills", [])]
    matched_skills = set(cand_skills) & set(req_skills)
    score += len(matched_skills) * 3
    max_score += len(req_skills) * 3 if req_skills else 0

    min_exp = reqs["exp_years"]
    cand_exp = cand_data.get("TotalExp", 0)
    if isinstance(cand_exp, (int, float)) and cand_exp >= min_exp:
        score += 15
    max_score += 15

    req_edu_levels = [e.lower() for e in reqs["edu_levels"]]
    cand_edu_entries = cand_data.get("Edu", [])
    
    edu_matched = False
    if "any" in req_edu_levels or not req_edu_levels:
        edu_matched = True
    else:
        for req_level in req_edu_levels:
            for edu_entry in cand_edu_entries:
                full_edu_txt = f"{edu_entry.get('Degree', '')} {edu_entry.get('Inst', '')}".lower()
                if req_level in full_edu_txt or \
                   (req_level == "high school" and "high school diploma" in full_edu_txt):
                    edu_matched = True
                    break
            if edu_matched:
                break
    
    score += (8 if edu_matched else 0)
    max_score += 8
    
    req_majors = [m.strip().lower() for m in reqs["majors"].split(',') if m.strip()]
    cand_majors = [edu.get('Major', '').lower() for edu in cand_data.get('Edu', [])]
    
    major_matched = False
    if not req_majors:
        major_matched = True
    else:
        for req_major in req_majors:
            for cand_major in cand_majors:
                if req_major in cand_major:
                    major_matched = True
                    break
            if major_matched:
                break
    score += (6 if major_matched else 0)
    max_score += 6

    pref_insts_input = [inst.strip().lower() for inst in reqs["insts"].split(',') if inst.strip()]
    cand_insts = [edu.get('Inst', '').lower() for edu in cand_data.get('Edu', [])]
    
    inst_matched = False
    if not pref_insts_input:
        inst_matched = True
    else:
        for pref_inst in pref_insts_input:
            for cand_inst in cand_insts:
                if pref_inst in cand_inst:
                    inst_matched = True
                    break
            if inst_matched:
                break
        score += (5 if inst_matched else 0)
        max_score += 5

    req_langs_lower = [lang.strip().lower() for lang in reqs["langs"]]
    cand_langs_lower = [lang_entry.get('lang', '').lower() for lang_entry in cand_data.get('Languages', [])]

    matched_req_langs_list = []
    unmatched_req_langs_list = []
    
    langs_all_matched = True
    
    if req_langs_lower:
        for req_lang_lower in req_langs_lower:
            if req_lang_lower in cand_langs_lower:
                matched_req_langs_list.append(req_lang_lower.title())
            else:
                unmatched_req_langs_list.append(req_lang_lower.title())
                langs_all_matched = False
        
        if langs_all_matched:
            score += 7
        elif len(matched_req_langs_list) > 0:
            score += len(matched_req_langs_list) * 2
        
        max_score += 7

    add_reqs_kws = [k.strip().lower() for k in reqs["add_reqs"].split(',') if k.strip()]
    full_cand_txt_kws = clean(cand_data.get("RawTextSnippet", "")).lower()
    
    matched_add_kws = []
    if add_reqs_kws:
        for kw in add_reqs_kws:
            if kw and kw in full_cand_txt_kws:
                score += 2
                matched_add_kws.append(kw)
            max_score += 2
    
    percentage = (score / max_score * 100) if max_score > 0 else 0
    
    return {
        "score": score,
        "percent": round(percentage, 2),
        "matched_skills": list(matched_skills),
        "matched_kws": matched_add_kws,
        "edu_met": edu_matched,
        "major_met": major_matched,
        "inst_met": inst_matched,
        "langs_met_all": langs_all_matched,
        "matched_req_langs": matched_req_langs_list,
        "unmatched_req_langs": unmatched_req_langs_list,
    }


def main():
    st.set_page_config(
        page_title="RecruitMate",
        page_icon="icon.png"
    )
    
    st.markdown(
        """
     <style>
.stApp {
    background-color: #f5f5f5;
}

.stExpander > div > div > button {
    background-color: #4a90e2;
    border-radius: 5px;
    border: 1px solid #357abd;
    color: #ffffff;
    padding: 10px;
    margin-top: 5px;
    font-weight: 500;
}

.stExpander > div > div > button:hover {
    background-color: #357abd;
    border-color: #2d5f91;
    color: #ffffff;
}
</style>
        """,
        unsafe_allow_html=True
    )

    st.image("logo.png", width=200)
    st.title("Your AI Recruiting Assistant")
    st.write("An intelligent AI tool that instantly reviews resumes, evaluates qualifications, and ranks candidates so you can focus on the best.")
    st.markdown("Here is a reference resume format to follow so the system can process it correctly")
    with open("London-Resume-Template-Professional.pdf", "rb") as pdf_file:
     PDFbyte = pdf_file.read()

    st.download_button(
        label="Download PDF",
        data=PDFbyte,
        file_name="example_resume.pdf",
        mime="application/pdf",
    )
    st.header("1. Enter the job requirements:")
    
    skills = st.text_area("Required skills (comma-separated):")
    exp_years = st.number_input("Minimum experience (Years):", min_value=0.0, max_value=20.0, value=1.0, step=0.5)
    
    edu_levels = st.multiselect(
        "Required education level/degree:",
        ["Any", "High School", "Bachelor's", "Master's", "Ph.D."]
    )
    
    majors = st.text_area("Required major/field of study (comma-separated):", "")
    insts = st.text_area("Preferred institutions (comma-separated):", "")
    
    langs_req = st.multiselect(
        "Required languages:",
        [lang.title() for lang in LANGS],
        default=[]
    )
    
    add_reqs = st.text_input("Additional keywords/requirements:")
    
    req_skills_proc = [s.strip().lower() for s in skills.split(',') if s.strip()]

    st.markdown("---")
    st.subheader("2. Upload resumes:")
    uploads = st.file_uploader("Choose PDF Resume(s)", type="pdf", accept_multiple_files=True)

    candidates = []

    if uploads:
        for file in uploads:
            st.info(f"Processing: {file.name} ...")
            pdf_bytes = BytesIO(file.read())
            
            with st.spinner(f"Extracting and parsing {file.name} ...."):
                txt = extract_text(pdf_bytes)
                if txt :
                    info = extract_info(txt)
                    info["Filename"] = file.name
                    candidates.append(info)
                else:
                    st.warning(f"Could not extract text from {file.name}. Skipping this file.")
        
        if candidates:
            st.markdown("---")
            st.subheader("3. Candidate Matching Results:")

            reqs_bundle = {
                "skills": req_skills_proc,
                "exp_years": exp_years,
                "edu_levels": edu_levels,
                "majors": majors,
                "insts": insts,
                "langs": langs_req,
                "add_reqs": add_reqs,
            }

            display = []
            for cand_data in candidates:
                match_results = match_cand(cand_data, reqs_bundle)
                
                display.append({
                    "parsed_data": cand_data,
                    "match_results": match_results
                })
            
            display.sort(key=lambda x: x["match_results"]["percent"], reverse=True)
            
            summary_data = []
            for item in display:
                summary_data.append({
                    "Filename": item["parsed_data"].get("Filename", "N/A"),
                    "Name": item["parsed_data"].get("Name", "N/A"),
                    "Match %": item["match_results"]["percent"],
                    "Score": item["match_results"]["score"],
                    "Total Exp (Years)": item["parsed_data"].get("TotalExp", "N/A"),
                    "Skills Met": f"{len(item['match_results']['matched_skills'])}/{len(req_skills_proc) if req_skills_proc else 0}",
                    "Major Met": "Yes" if item["match_results"]["major_met"] else "No",
                    "Preferred institutions  Met": "Yes" if item["match_results"]["inst_met"] else "No",
                    "languages Met": f"{len(item['match_results']['matched_req_langs'])}/{len(langs_req) if langs_req else 0}",
                })
            
            if summary_data:
                df_summary = pd.DataFrame(summary_data)
                st.dataframe(df_summary.style.format({"Match %": "{:.2f}%"}), height=300, use_container_width=True)
                st.markdown("---")

                st.subheader("4. Exporting the results :")
                
                full_export = []
                for item in display:
                    p_data = item["parsed_data"]
                    m_results = item["match_results"]
                    full_export.append({
                        "Filename": p_data.get("Filename", "-"),
                        "Name": p_data.get("Name", "-"),
                        "Match Percent": m_results["percent"],
                        "Match Score": m_results["score"],
                        "Total experience (Years)": p_data.get("TotalExp", "-"),
                        "Email": p_data.get("Email", "-"),
                        "Phone": p_data.get("Phone", "-"),
                        "Matched Skills": ", ".join(m_results["matched_skills"]),
                        "Education Met": "Yes" if m_results["edu_met"] else "No",
                        "Major Met": "Yes" if m_results["major_met"] else "No",
                        "Preferred institutions": "Yes" if m_results["inst_met"] else "No",
                        "Languages Met (All)": "Yes" if m_results["langs_met_all"] else "No",
                        "Matched Required languages": ", ".join(m_results["matched_req_langs"]),
                        "Missing Required languages": ", ".join(m_results["unmatched_req_langs"]),
                        "Skills Extracted": ", ".join(p_data.get("Skills", [])),
                        "Education Extracted": "<br/>".join([f"{e.get('Degree','')} from {e.get('Inst','')} ({e.get('Year','')})" for e in p_data.get("Edu", [])]),
                        "Experience Details": "<br/>".join([f"{e.get('Title','')} at {e.get('Comp','')} ({e.get('Dates','')})" for e in p_data.get("Exp", []) if e.get('Comp','N/A') != 'N/A' or e.get('Title','N/A') != 'N/A']),
                        "Languages Extracted": " \n ".join([f"{l.get('lang','-')} ({l.get('prof','')})" for l in p_data.get('Languages', [])]),
                        "Certificates Extracted": ", ".join(p_data.get("Certs", [])),
                        "LinkedIn URL": ", ".join(p_data["Links"].get("linkedin", [])),
                        "GitHub URL": ", ".join(p_data["Links"].get("github", [])),
                        "Other URLs": ", ".join(p_data["Links"].get("other", [])),
                    })
                
                if full_export:
                    pdf_export = export_pdf_paragraphs(full_export)
                    st.download_button(
                        label="Download All Results to PDF",
                        data=pdf_export,
                        file_name="RecruitMate_results.pdf",
                        mime="application/pdf",
                        help="Exports a PDF with all extracted details and match results for all candidates."
                    )

                st.subheader("5. Detailed Breakdown for Each Candidate:")
                for i, item in enumerate(display):
                    p_data = item["parsed_data"]
                    m_results = item["match_results"]

                    exp_title = f"**{i+1}. {p_data.get('Name', p_data.get('Filename', 'Candidate'))}** - Match: {m_results['percent']}% (Score: {m_results['score']})"
                    
                    with st.expander(exp_title):
                        if p_data.get("Email", "N/A") != "N/A" or p_data.get("Phone", "N/A") != "N/A":
                            st.markdown("##### Contact Info")
                            if p_data.get("Email", "N/A") != "N/A":
                                st.write(f"**Email:** {p_data.get('Email')}")
                            if p_data.get("Phone", "N/A") != "N/A":
                                st.write(f"**Phone:** {p_data.get('Phone')}")

                        st.markdown("##### Match Details")
                        if m_results['matched_skills']:
                            st.write(f"**Matched Skills:** {', '.join(m_results['matched_skills'])}")
                        st.write(f"**Education Level Met:** {'Yes' if m_results['edu_met'] else 'No'}")
                        st.write(f"**Major/Field Met:** {'Yes' if m_results['major_met'] else 'No'}")
                        st.write(f"**Preferred Institution Met:** {'Yes' if m_results['inst_met'] else 'No'}")

                        if p_data["Links"].get("linkedin") or p_data["Links"].get("github") or p_data["Links"].get("other"):
                            st.markdown("##### Links")
                            if p_data["Links"].get("linkedin"):
                                st.write(f"**LinkedIn:** {', '.join(p_data['Links']['linkedin'])}")
                            if p_data["Links"].get("github"):
                                st.write(f"**GitHub:** {', '.join(p_data['Links']['github'])}")
                            if p_data["Links"].get("other"):
                                st.write(f"**Other URLs:** {', '.join(p_data['Links']['other'])}")

                        st.markdown("##### Extracted Information")
                        
                        if p_data.get('Skills'):
                            st.write(f"**Skills:** {', '.join(p_data.get('Skills'))}")
                        
                        st.write("**Education:**")
                        has_valid_edu = p_data.get('Edu') and any(e.get('Degree', 'N/A') != 'N/A' or e.get('Inst', 'N/A') != 'N/A' for e in p_data.get('Edu'))
                        if not has_valid_edu:
                            st.code(p_data.get('RawEdu', 'No education section found.'), language=None)
                        else:
                            for edu in p_data['Edu']:
                                edu_str = f"- {edu.get('Degree','')} in {edu.get('Major','')} from {edu.get('Inst','')} ({edu.get('Year','N/A')})"
                                st.markdown(edu_str.replace(" in N/A", "").replace(" from N/A", "").replace(" in  from", " from"))

                        st.write("**Experience:**")
                        has_valid_exp = p_data.get('Exp') and any(e.get('Title', 'N/A') != 'N/A' or e.get('Comp', 'N/A') != 'N/A' for e in p_data.get('Exp'))
                        if not has_valid_exp:
                            st.code(p_data.get('RawExp', 'No experience section found.'), language=None)
                        else:
                            if p_data.get('TotalExp', 'N/A') != 'N/A':
                                st.write(f"  *Total Years: {p_data.get('TotalExp')}*")
                            for exp in p_data['Exp']:
                                st.markdown(f"- **{exp.get('Title','N/A')}** at **{exp.get('Comp','-')}** ({exp.get('Dates','-')})")

                        if p_data.get('Languages'):
                            st.write("**Languages:**")
                            for lang_entry in p_data['Languages']:
                                lang = lang_entry.get('lang','N/A')
                                prof = lang_entry.get('prof','N/A')
                                st.markdown(f"- {lang}{f' ({prof})' if prof != 'N/A' else ''}")

                        if p_data.get('Certs'):
                            st.write(f"**Certificates:** {', '.join(p_data.get('Certs'))}")

            if not display:
                st.info("No candidates processed successfully.")
        else:
            st.info("No valid resumes were processed.")
    else:
        st.info("Upload PDF resumes to begin analysis.")

if __name__ == "__main__":
    main()