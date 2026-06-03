import re
from typing import List, Dict, Any, Set

# ==============================================================================
# Services Company Allowlist
# ==============================================================================
SERVICES_COMPANIES: Set[str] = {
    "tcs",
    "tata consultancy services",
    "infosys",
    "wipro",
    "accenture",
    "cognizant",
    "cognizant technology solutions",
    "capgemini",
    "hcl",
    "hcltech",
    "hcl technologies",
    "tech mahindra",
    "mindtree",
    "lti",
    "l&t infotech",
    "larsen & toubro infotech",
    "dxc",
    "dxc technology",
    "ntt data",
    "deloitte",
    "pwc",
    "ey",
    "kpmg",
    "l&t",
    "larsen & toubro",
    "persistent systems",
    "ust",
    "ust global",
    "virtusa",
    "coforge",
    "mphasis",
    "genpact",
    "conduent",
    "syntel",
    "hexaware",
    "tata technologies",
    "ltimindtree",
    "birlasoft"
}

def normalize_company_name(name: str) -> str:
    """
    Normalizes a company name by lowercasing, removing punctuation, 
    and stripping common legal/entity suffixes.
    """
    if not name:
        return ""
    
    # Lowercase & strip
    cleaned = name.lower().strip()
    
    # Replace common symbols with space to keep words distinct
    cleaned = re.sub(r'[-_/\.,&]', ' ', cleaned)
    
    # Remove non-alphanumeric (except spaces)
    cleaned = re.sub(r'[^a-z0-9\s]', '', cleaned)
    
    # Collapse multiple spaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    # Trailing legal entity suffixes (ordered longest/most specific first)
    suffixes = [
        r'\bprivate limited\b', r'\bpvt ltd\b', r'\blimited\b', r'\bltd\b',
        r'\bincorporated\b', r'\binc\b', r'\bcorporation\b', r'\bcorp\b',
        r'\bllc\b', r'\bpvt\b', r'\bco\b', r'\bcompany\b'
    ]
    
    # Iteratively remove trailing suffixes
    changed = True
    while changed:
        changed = False
        for suffix_pat in suffixes:
            pattern = suffix_pat + r'\s*$'
            new_cleaned = re.sub(pattern, '', cleaned).strip()
            if new_cleaned != cleaned and new_cleaned != "":
                cleaned = new_cleaned
                changed = True
                break
                
    return cleaned

def is_services_company(company_name: str) -> bool:
    """
    Checks if a company name belongs to a service-oriented/consulting company.
    """
    norm_name = normalize_company_name(company_name)
    if not norm_name:
        return False
    
    # Direct match in the services set
    if norm_name in SERVICES_COMPANIES:
        return True
    
    # Prefix or word-boundary matches for major service providers
    major_prefixes = [
        "tcs", "tata consultancy", "infosys", "wipro", "accenture", 
        "cognizant", "capgemini", "hcl", "tech mahindra", "mindtree", 
        "deloitte", "pwc", "kpmg", "persistent system", "virtusa", 
        "coforge", "mphasis", "genpact", "hexaware", "ltimindtree"
    ]
    for prefix in major_prefixes:
        if re.search(r'\b' + re.escape(prefix) + r'\b', norm_name):
            return True
            
    return False

def classify_career_services_ratio(career_history: List[Dict[str, Any]], threshold: float = 0.8) -> bool:
    """
    Heuristic: Classifies the career history as services-heavy if the ratio of 
    distinct service-oriented employers to total distinct employers is >= threshold.
    
    If the career history is empty, returns False.
    """
    if not career_history:
        return False
    
    unique_employers = set()
    service_employers = set()
    
    for role in career_history:
        company = role.get("company", "").strip()
        if not company:
            continue
            
        norm_company = normalize_company_name(company)
        if norm_company:
            unique_employers.add(norm_company)
            if is_services_company(company):
                service_employers.add(norm_company)
                
    if not unique_employers:
        return False
        
    ratio = len(service_employers) / len(unique_employers)
    return ratio >= threshold
